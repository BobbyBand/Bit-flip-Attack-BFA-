#!/usr/bin/env python3
import argparse
import torch
from typing import Dict

# ------------------ Helpers ------------------

def clamp_int8_like(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-128, 127).to(torch.int16)

def czr_int8_from_constraints(c: Dict[str, torch.Tensor],
                              fallback_strategy: str = "keep_orig",
                              orig_q: torch.Tensor = None) -> torch.Tensor:
    """
    Returns an INT8 tensor (same shape as param).
    Operates directly in int8 domain.
    For Set-3 (mask_none):
      - "keep_orig": keep original int8 weights (requires orig_q)
      - "zeros": set to 0
      - "gaussian": sample Gaussian in float, quantize to int8
    """
    device = c["full_min"].device
    scale = float(c["scale"])
    shape = c["full_min"].shape
    q_out = torch.zeros(shape, dtype=torch.int8, device=device)

    # ---- Set-1: exact
    if c["mask_full"].any():
        q_full = torch.round(c["full_mean"] / scale).clamp(-128, 127).to(torch.int8)
        q_out[c["mask_full"]] = q_full[c["mask_full"]]

    # ---- Set-2: CZR
    mpart = c["mask_part"]
    if mpart.any():
        qmin = clamp_int8_like(torch.round(c["part_min"] / scale))
        qmax = clamp_int8_like(torch.round(c["part_max"] / scale))

        pos = mpart & (qmin >= 0) & (qmax >= 0)
        q_out[pos] = qmin[pos].to(torch.int8)

        neg = mpart & (qmin <= -1) & (qmax <= -1)
        q_out[neg] = qmax[neg].to(torch.int8)

        amb = mpart & ~(pos | neg)
        if amb.any():
            cand_pos = torch.clamp_min(qmin[amb], 0).to(torch.int16)
            cand_neg = torch.clamp_max(qmax[amb], -1).to(torch.int16)
            choose_pos = cand_pos.abs() <= cand_neg.abs()
            q_out[amb] = torch.where(choose_pos, cand_pos, cand_neg).to(torch.int8)

    # ---- Set-3: fallback
    mnone = c["mask_none"]
    if mnone.any():
        if fallback_strategy == "keep_orig":
            if orig_q is None:
                raise ValueError("keep_orig fallback needs orig_q.")
            q_out[mnone] = orig_q[mnone]
        elif fallback_strategy == "zeros":
            q_out[mnone] = torch.zeros_like(q_out[mnone])
        elif fallback_strategy == "gaussian":
            mids = []
            if c["mask_full"].any():
                mids.append(c["full_mean"][c["mask_full"]])
            if c["mask_part"].any():
                mids.append(((c["part_min"] + c["part_max"]) * 0.5)[c["mask_part"]])
            if len(mids) == 0:
                std = 0.01
            else:
                mids_cat = torch.cat(mids).to(torch.float32)
                std = float(mids_cat.std().item() if mids_cat.numel() > 1 else 0.01)
            samp = torch.normal(mean=0.0, std=std,
                                size=(mnone.sum().item(),), device=device)
            q_none = torch.round(samp / scale).clamp(-128, 127).to(torch.int8)
            q_out[mnone] = q_none
        else:
            raise ValueError(f"Unknown fallback_strategy {fallback_strategy}")

    return q_out

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim_int8", default="victim_vgg16_cifar10_int8.pt",
                    help="Path to int8-packed victim weights (.pt)")
    ap.add_argument("--constraints", default="leak_constraints.pt",
                    help="Path to leak constraints (.pt)")
    ap.add_argument("--out", default="reconstructed_czr_int8.pt",
                    help="Output path for reconstructed int8-packed weights")
    ap.add_argument("--fallback_none", default="keep_orig",
                    choices=["keep_orig", "zeros", "gaussian"],
                    help="Strategy for Set-3 weights")
    args = ap.parse_args()

    # Load int8 victim and constraints
    victim_pack: Dict[str, torch.Tensor] = torch.load(args.victim_int8, map_location="cpu")
    cons_pack = torch.load(args.constraints, map_location="cpu")
    constraints: Dict[str, Dict[str, torch.Tensor]] = cons_pack["constraints"]

    recon_pack: Dict[str, torch.Tensor] = {}
    used = set()

    for k, v in victim_pack.items():
        if not k.endswith("::qint8"):
            # copy scale or other entries
            recon_pack[k] = v.clone()
            continue

        base = k[:-8]
        if base in constraints:
            c = constraints[base]
            orig_q = v if args.fallback_none == "keep_orig" else None
            q_recon = czr_int8_from_constraints(c, args.fallback_none, orig_q=orig_q)
            recon_pack[k] = q_recon.cpu()
            recon_pack[base + "::scale"] = victim_pack[base + "::scale"].clone()
            used.add(base)
        else:
            # no constraints: keep original
            recon_pack[k] = v.clone()
            if base + "::scale" in victim_pack:
                recon_pack[base + "::scale"] = victim_pack[base + "::scale"].clone()

    torch.save(recon_pack, args.out)
    print(f"[OK] Wrote CZR-reconstructed int8-packed weights to {args.out}")

if __name__ == "__main__":
    main()