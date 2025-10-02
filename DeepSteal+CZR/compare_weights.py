#!/usr/bin/env python3
import argparse
from typing import Dict
import torch

SUFFIX_Q = "::qint8"
SUFFIX_S = "::scale"

# ----------------- Helpers -----------------

def _load_as_float_state(path: str) -> Dict[str, torch.Tensor]:
    """
    Loads either:
      - a normal float state_dict (returned as-is), or
      - an int8-packed state (base::qint8 + base::scale) and dequantizes to float.
    """
    pack = torch.load(path, map_location="cpu")
    # Detect int8-packed format
    any_q = any(isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q) for k, v in pack.items())
    if not any_q:
        return pack  # already float

    state: Dict[str, torch.Tensor] = {}
    for k, v in pack.items():
        if not (isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q)):
            continue
        base = k[:-len(SUFFIX_Q)]
        s_key = base + SUFFIX_S
        if s_key not in pack:
            raise KeyError(f"Missing {s_key} for {k}")
        scale = float(pack[s_key].item() if isinstance(pack[s_key], torch.Tensor) else pack[s_key])
        state[base] = v.to(torch.float32) * scale
    # pass through any non-parameter buffers if you want (optional)
    return state

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32); b = b.flatten().to(torch.float32)
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if (a == b).all() else 0.0
    return float(torch.dot(a, b).item() / denom)

def tensor_metrics(gt: torch.Tensor, rec: torch.Tensor) -> Dict[str, float]:
    diff = (gt - rec).to(torch.float32)
    mse = float((diff**2).mean().item())
    l1  = float(diff.abs().mean().item())
    linf = float(diff.abs().max().item())
    cos = cosine_sim(gt, rec)
    return {"mse": mse, "l1": l1, "linf": linf, "cos": cos}

def _q8(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.round(x / scale).clamp_(-128, 127).to(torch.int16)

def check_constraints_per_tensor(gt: torch.Tensor, rec: torch.Tensor, cons: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Runs all set checks in the int8 domain using the provided per-tensor 'scale'.
    Returns coverage and set-wise exact/in-range percentages.
    """
    out = {
        "cov_full": 0.0, "cov_part": 0.0, "cov_none": 0.0,
        "set1_exact_pct": float("nan"),
        "set2_in_range_pct": float("nan"),
        "set2_exact_pct": float("nan"),
        "set3_exact_pct": float("nan"),
    }

    mfull, mpart, mnone = cons["mask_full"], cons["mask_part"], cons["mask_none"]
    n = gt.numel()
    if n == 0:
        return out
    out["cov_full"] = float(mfull.sum().item() / n)
    out["cov_part"] = float(mpart.sum().item() / n)
    out["cov_none"] = 1.0 - out["cov_full"] - out["cov_part"]

    scale = float(cons["scale"])

    # Set-1: exact
    if mfull.any():
        out["set1_exact_pct"] = float((_q8(gt[mfull], scale) == _q8(rec[mfull], scale)).float().mean().item())

    # Set-2: in-range + exact
    if mpart.any():
        qmin = _q8(cons["part_min"][mpart], scale)
        qmax = _q8(cons["part_max"][mpart], scale)
        qg   = _q8(gt[mpart], scale)
        qr   = _q8(rec[mpart], scale)
        in_range = (qr >= qmin) & (qr <= qmax)
        out["set2_in_range_pct"] = float(in_range.float().mean().item())
        out["set2_exact_pct"]    = float((qr == qg).float().mean().item())

    # Set-3: exact
    if mnone.any():
        out["set3_exact_pct"] = float((_q8(gt[mnone], scale) == _q8(rec[mnone], scale)).float().mean().item())

    return out

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="victim_vgg16_cifar10_int8.pt", help="Ground-truth weights (.pt) float or int8-packed")
    ap.add_argument("--reconstructed", default="reconstructed_czr_int8.pt", help="Reconstructed weights (.pt) float or int8-packed")
    ap.add_argument("--constraints", default="leak_constraints.pt", help="Constraints file (.pt)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gt  = _load_as_float_state(args.gt)
    rec = _load_as_float_state(args.reconstructed)

    cons_pack = torch.load(args.constraints, map_location="cpu") if args.constraints else None
    cons_dict: Dict[str, Dict[str, torch.Tensor]] = cons_pack.get("constraints", {}) if cons_pack else {}

    # Intersect keys
    common = [k for k in gt.keys() if k in rec]
    miss = [k for k in gt.keys() if k not in rec]
    extra = [k for k in rec.keys() if k not in gt]
    if miss:
        print(f"[WARN] {len(miss)} keys missing in reconstructed (ignored). Example: {miss[:5]}")
    if extra:
        print(f"[WARN] {len(extra)} extra keys in reconstructed (ignored). Example: {extra[:5]}")

    rows = []
    total_elems = 0
    sum_mse = 0.0
    sum_l1 = 0.0
    max_linf = 0.0
    cos_num = 0.0
    cos_den = 0.0

    # Global set counters
    set1_exact_hits = 0
    set1_count = 0
    set2_inrange_hits = 0
    set2_exact_hits = 0
    set2_count = 0
    set3_exact_hits = 0
    set3_count = 0

    for name in common:
        g = gt[name]; r = rec[name]
        if g.shape != r.shape:
            print(f"[SKIP] shape mismatch for {name}: {tuple(g.shape)} vs {tuple(r.shape)}")
            continue

        # Base metrics
        m = tensor_metrics(g, r)
        n = g.numel()
        total_elems += n
        sum_mse += m["mse"] * n
        sum_l1  += m["l1"]  * n
        max_linf = max(max_linf, m["linf"])
        cos_num += m["cos"] * n
        cos_den += n

        row = {"name": name, "shape": tuple(g.shape), **m}

        # Constraint checks
        if name in cons_dict and g.ndim > 0 and tuple(cons_dict[name]["full_min"].shape) == tuple(g.shape):
            stats = check_constraints_per_tensor(g, r, cons_dict[name])
            row.update(stats)

            # aggregate sets
            mfull = cons_dict[name]["mask_full"].sum().item()
            mpart = cons_dict[name]["mask_part"].sum().item()
            mnone = cons_dict[name]["mask_none"].sum().item()

            if stats["set1_exact_pct"] == stats["set1_exact_pct"]:  # not NaN
                set1_exact_hits += int(mfull * stats["set1_exact_pct"])
                set1_count      += int(mfull)

            if stats["set2_in_range_pct"] == stats["set2_in_range_pct"]:
                set2_inrange_hits += int(mpart * stats["set2_in_range_pct"])
                set2_count        += int(mpart)
            if stats["set2_exact_pct"] == stats["set2_exact_pct"]:
                set2_exact_hits   += int(mpart * stats["set2_exact_pct"])

            if stats["set3_exact_pct"] == stats["set3_exact_pct"]:
                set3_exact_hits   += int(mnone * stats["set3_exact_pct"])
                set3_count        += int(mnone)

        rows.append(row)

    # Per-parameter (first 30)
    print("\n=== Per-parameter summary (first 30) ===")
    for r in rows[:30]:
        extras = []
        if "set1_exact_pct" in r and r["set1_exact_pct"] == r["set1_exact_pct"]:
            extras.append(f"S1 exact {r['set1_exact_pct']*100:.2f}%")
        if "set2_in_range_pct" in r and r["set2_in_range_pct"] == r["set2_in_range_pct"]:
            extras.append(f"S2 in-range {r['set2_in_range_pct']*100:.2f}%")
        if "set2_exact_pct" in r and r["set2_exact_pct"] == r["set2_exact_pct"]:
            extras.append(f"S2 exact {r['set2_exact_pct']*100:.2f}%")
        if "set3_exact_pct" in r and r["set3_exact_pct"] == r["set3_exact_pct"]:
            extras.append(f"S3 exact {r['set3_exact_pct']*100:.2f}%")
        tag = (" | " + " / ".join(extras)) if extras else ""
        print(f"{r['name']:50s} {str(r['shape']):>16s}  mse={r['mse']:.4e}  l1={r['l1']:.4e}  linf={r['linf']:.4e}  cos={r['cos']:.6f}{tag}")

    # Global aggregates
    global_mse = sum_mse / max(1, total_elems)
    global_l1  = sum_l1  / max(1, total_elems)
    global_cos = cos_num / max(1, cos_den)

    print("\n=== Global metrics ===")
    print(f"MSE (avg over all params) : {global_mse:.6e}")
    print(f"L1  (avg over all params) : {global_l1:.6e}")
    print(f"Linf (max abs diff)       : {max_linf:.6e}")
    print(f"Cosine similarity (avg)   : {global_cos:.6f}")

    if set1_count > 0:
        print(f"Set-1 exact match         : {set1_exact_hits/set1_count*100:.2f}% ({set1_exact_hits}/{set1_count})")
    if set2_count > 0:
        print(f"Set-2 within-range        : {set2_inrange_hits/set2_count*100:.2f}% ({set2_inrange_hits}/{set2_count})")
        print(f"Set-2 exact match         : {set2_exact_hits/set2_count*100:.2f}% ({set2_exact_hits}/{set2_count})")
    if set3_count > 0:
        print(f"Set-3 exact match         : {set3_exact_hits/set3_count*100:.2f}% ({set3_exact_hits}/{set3_count})")

    if args.verbose:
        print("\n[INFO] Full per-parameter rows (all):")
        for r in rows:
            print(r)

if __name__ == "__main__":
    main()