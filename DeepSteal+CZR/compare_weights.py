#!/usr/bin/env python3
import argparse
from typing import Dict, Tuple
import torch

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if (a==b).all() else 0.0
    return float(torch.dot(a, b).item() / denom)

def tensor_metrics(gt: torch.Tensor, rec: torch.Tensor) -> Dict[str, float]:
    diff = (gt - rec).to(torch.float32)
    mse = float((diff**2).mean().item())
    l1  = float(diff.abs().mean().item())
    linf = float(diff.abs().max().item())
    cos = cosine_sim(gt, rec)
    exact = float((gt == rec).float().mean().item()) if gt.dtype.is_floating_point is False else 0.0
    # For float (typical), exact match % isn’t very meaningful; we’ll set it later via constraints.
    return {"mse": mse, "l1": l1, "linf": linf, "cos": cos, "exact": exact}

def _clamp_int8_round(x, scale):
    q = torch.round(x / scale)
    return q.clamp_(-128, 127).to(torch.int16)

def check_constraints_per_tensor(name, gt, rec, cons):
    out = {"set1_exact_pct": float("nan"),
           "set2_in_range_pct": float("nan"),
           "set2_exact_pct": float("nan"),
           "set3_exact_pct": float("nan"),
           "cov_full": 0.0, "cov_part": 0.0, "cov_none": 0.0}

    mfull, mpart, mnone = cons["mask_full"], cons["mask_part"], cons["mask_none"]
    n = gt.numel()
    out["cov_full"] = float(mfull.sum().item() / n)
    out["cov_part"] = float(mpart.sum().item() / n)
    out["cov_none"] = 1.0 - out["cov_full"] - out["cov_part"]

    scale = float(cons["scale"])

    def q8(x):  # int8 domain projection
        return torch.round(x / scale).clamp_(-128, 127).to(torch.int16)

    if mfull.any():
        out["set1_exact_pct"] = float((q8(gt[mfull]) == q8(rec[mfull])).float().mean().item())

    if mpart.any():
        qmin = q8(cons["part_min"][mpart]); qmax = q8(cons["part_max"][mpart])
        qr   = q8(rec[mpart]); qg = q8(gt[mpart])
        in_range = (qr >= qmin) & (qr <= qmax)
        out["set2_in_range_pct"] = float(in_range.float().mean().item())
        out["set2_exact_pct"]    = float((qr == qg).float().mean().item())

    if mnone.any():
        out["set3_exact_pct"] = float((q8(gt[mnone]) == q8(rec[mnone])).float().mean().item())

    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="victim_vgg16_cifar10_int8.pt", help="Ground truth state_dict (.pt)")
    ap.add_argument("--reconstructed", default="reconstructed_czr_int8.pt", help="Reconstructed state_dict (.pt)")
    ap.add_argument("--constraints", default="leak_constraints_k2.pt", help="Constraints file (.pt)")
    ap.add_argument("--verbose", action="store_true", help="Print per-parameter details")
    args = ap.parse_args()

    gt: Dict[str, torch.Tensor]  = torch.load(args.gt, map_location="cpu")
    rec: Dict[str, torch.Tensor] = torch.load(args.reconstructed, map_location="cpu")

    # Constraints (optional but recommended)
    cons_pack = torch.load(args.constraints, map_location="cpu") if args.constraints else None
    cons_dict: Dict[str, Dict[str, torch.Tensor]] = cons_pack.get("constraints", {}) if cons_pack else {}

    # Align keys that exist in both state_dicts
    common = [k for k in gt.keys() if k in rec]
    missing_in_rec = [k for k in gt.keys() if k not in rec]
    extra_in_rec   = [k for k in rec.keys() if k not in gt]

    if missing_in_rec:
        print(f"[WARN] {len(missing_in_rec)} keys missing in reconstructed (will ignore): {missing_in_rec[:5]}{' ...' if len(missing_in_rec)>5 else ''}")
    if extra_in_rec:
        print(f"[WARN] {len(extra_in_rec)} extra keys in reconstructed (will ignore): {extra_in_rec[:5]}{' ...' if len(extra_in_rec)>5 else ''}")

    # Per-layer metrics
    rows = []
    total_elems = 0
    sum_mse = 0.0
    sum_l1 = 0.0
    max_linf = 0.0
    cos_num = 0.0
    cos_den = 0.0  # we’ll average cosine by param size

    # Set-1 & Set-2 quality trackers
    set1_exact_hits = 0
    set1_count = 0
    set2_inrange_hits = 0
    set2_count = 0

    for name in common:
        g = gt[name]
        r = rec[name]
        if g.shape != r.shape:
            print(f"[SKIP] Shape mismatch for {name}: {g.shape} vs {r.shape}")
            continue

        # Compute basic differences
        m = tensor_metrics(g, r)

        # Weight by size for global aggregates
        n = g.numel()
        total_elems += n
        sum_mse += m["mse"] * n
        sum_l1  += m["l1"]  * n
        max_linf = max(max_linf, m["linf"])

        # Cosine: average weighted by size (not perfect but reasonable)
        cos_num += cosine_sim(g, r) * n
        cos_den += n

        # Constraint checks if available for this tensor
        c_row = {}
        if name in cons_dict and g.ndim > 0:
            c = cons_dict[name]
            # sanity shape
            if tuple(c["full_min"].shape) == tuple(g.shape):
                cons_stats = check_constraints_per_tensor(name, g, r, c)
                c_row.update(cons_stats)

                # aggregate Set-1/Set-2 stats
                if not (cons_stats["set1_exact_pct"] != cons_stats["set1_exact_pct"]):  # not NaN
                    set1_exact_hits += int((c["mask_full"]).sum().item() * cons_stats["set1_exact_pct"])
                    set1_count      += int((c["mask_full"]).sum().item())
                if not (cons_stats["set2_in_range_pct"] != cons_stats["set2_in_range_pct"]):
                    set2_inrange_hits += int((c["mask_part"]).sum().item() * cons_stats["set2_in_range_pct"])
                    set2_count        += int((c["mask_part"]).sum().item())

        row = {
            "name": name,
            "shape": tuple(g.shape),
            "mse": m["mse"],
            "l1": m["l1"],
            "linf": m["linf"],
            "cos": m["cos"],
        }
        row.update(c_row)
        rows.append(row)

    # Print summary
    print("\n=== Per-parameter summary (first 30) ===")
    for r in rows[:30]:
        extras = ""
        if "set1_exact_pct" in r and r["set1_exact_pct"] == r["set1_exact_pct"]:
            extras += f" | S1 exact {r['set1_exact_pct']*100:.2f}%"
        if "set2_in_range_pct" in r and r["set2_in_range_pct"] == r["set2_in_range_pct"]:
            extras += f" | S2 in-range {r['set2_in_range_pct']*100:.2f}%"
        print(f"{r['name']:50s} {str(r['shape']):>16s}  mse={r['mse']:.4e}  l1={r['l1']:.4e}  linf={r['linf']:.4e}  cos={r['cos']:.6f}{extras}")

    # Global aggregates
    global_mse = sum_mse / max(total_elems, 1)
    global_l1  = sum_l1  / max(total_elems, 1)
    global_cos = cos_num / max(cos_den, 1)

    print("\n=== Global metrics ===")
    print(f"MSE (avg over all params) : {global_mse:.6e}")
    print(f"L1  (avg over all params) : {global_l1:.6e}")
    print(f"Linf (max abs diff)       : {max_linf:.6e}")
    print(f"Cosine similarity (avg)   : {global_cos:.6f}")

    if set1_count > 0:
        print(f"Set-1 exact match         : {set1_exact_hits/set1_count*100:.2f}% ({set1_exact_hits}/{set1_count})")
    if set2_count > 0:
        print(f"Set-2 within-range        : {set2_inrange_hits/set2_count*100:.2f}% ({set2_inrange_hits}/{set2_count})")

    if args.verbose:
        print("\n[INFO] Full per-parameter rows (all):")
        for r in rows:
            print(r)

if __name__ == "__main__":
    main()