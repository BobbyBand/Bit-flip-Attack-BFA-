#!/usr/bin/env python3
import argparse, torch
from typing import Dict, Tuple

# ------------------------ Low-level helpers ------------------------

def u8_to_s8(u: torch.Tensor) -> torch.Tensor:
    return torch.where(u >= 128, u - 256, u)

def s8_to_u8(s: torch.Tensor) -> torch.Tensor:
    return (s & 0xFF).to(torch.int16)

@torch.no_grad()
def quantize_int8_per_tensor(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-tensor int8 quantization:
      q = clamp(round(w/scale), -128, 127),
      scale = max(|w|)/127 (clamped >=1e-8)
    Returns (q_int8, scale)
    """
    max_abs = w.abs().max()
    scale = (max_abs / 127.0).clamp(min=1e-8)
    q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q, scale

def bits_prefix_mask(k: int) -> int:
    return 0 if k <= 0 else ((0xFF << (8 - k)) & 0xFF)

def int8_range_from_topbits(u8_true: torch.Tensor, k_top: int):
    """
    Given true U8 and number of known top bits k, compute [min_u8, max_u8].
    """
    if k_top <= 0:
        return torch.zeros_like(u8_true), torch.full_like(u8_true, 255)
    m = bits_prefix_mask(k_top)
    top = (u8_true & m)
    low_max = (1 << (8 - k_top)) - 1
    return top, top | low_max

def dequantize_from_int8_range(min_u8, max_u8, scale):
    min_s8 = u8_to_s8(min_u8.to(torch.int16)).to(torch.float32)
    max_s8 = u8_to_s8(max_u8.to(torch.int16)).to(torch.float32)
    return min_s8 * scale, max_s8 * scale

# ------------------------ Leak simulation (per tensor) ------------------------

@torch.no_grad()
def build_leak_profile(param: torch.Tensor,
                       full_frac: float,
                       partial_frac: float,
                       partial_top_bits: int,
                       rng: torch.Generator = None) -> Dict[str, torch.Tensor]:
    """
    Produce Set-1 / Set-2 / Set-3 masks and float-domain ranges based on per-tensor int8.
      - Set-1 (full): all 8 bits known -> exact value (min=max)
      - Set-2 (partial): top-k bits known -> [min,max] consistent range
      - Set-3 (none): unconstrained
    """
    device = param.device
    n = param.numel()

    # choose indices
    if rng is None:
        idx = torch.randperm(n, device=device)
    else:
        if getattr(rng, "device", None) != device.type:
            rng = torch.Generator(device=device.type).manual_seed(int(torch.seed()))
        idx = torch.randperm(n, generator=rng, device=device)

    n_full = int(max(0, min(1.0, full_frac)) * n)
    n_part = int(max(0, min(1.0, partial_frac)) * n)
    n_part = min(n_part, max(0, n - n_full))

    mfull = torch.zeros(n, dtype=torch.bool, device=device)
    mpart = torch.zeros(n, dtype=torch.bool, device=device)
    mfull[idx[:n_full]] = True
    mpart[idx[n_full:n_full + n_part]] = True
    mnone = ~(mfull | mpart)

    # per-tensor int8 quantization
    q8, scale = quantize_int8_per_tensor(param)
    u8 = s8_to_u8(q8.to(torch.int16))  # 0..255

    # FULL
    full_u8 = u8.view(-1)[mfull]
    full_min_f, full_max_f = dequantize_from_int8_range(full_u8, full_u8, scale)
    full_mean_f = (full_min_f + full_max_f) * 0.5

    # PARTIAL
    k = int(max(0, min(8, partial_top_bits)))
    part_u8 = u8.view(-1)[mpart]
    if k == 0:
        pmin_u8 = torch.zeros_like(part_u8)
        pmax_u8 = torch.full_like(part_u8, 255)
    else:
        m = bits_prefix_mask(k)
        top = (part_u8 & m)
        low_max = (1 << (8 - k)) - 1
        pmin_u8 = top
        pmax_u8 = top | low_max
    part_min_f, part_max_f = dequantize_from_int8_range(pmin_u8, pmax_u8, scale)
    part_mean_f = (part_min_f + part_max_f) * 0.5

    # assemble
    out = {
        "mask_full": mfull.view_as(param),
        "mask_part": mpart.view_as(param),
        "mask_none": mnone.view_as(param),
        "full_min": torch.zeros_like(param),
        "full_max": torch.zeros_like(param),
        "full_mean": torch.zeros_like(param),
        "part_min": torch.zeros_like(param),
        "part_max": torch.zeros_like(param),
        "part_mean": torch.zeros_like(param),
        "scale": torch.as_tensor(scale, device=device),
    }

    out["full_min"].view(-1)[mfull] = full_min_f
    out["full_max"].view(-1)[mfull] = full_max_f
    out["full_mean"].view(-1)[mfull] = full_mean_f

    out["part_min"].view(-1)[mpart] = part_min_f
    out["part_max"].view(-1)[mpart] = part_max_f
    out["part_mean"].view(-1)[mpart] = part_mean_f
    return out

# ------------------------ Build constraints for whole state ------------------------

@torch.no_grad()
def constraints_from_state_dict(state: Dict[str, torch.Tensor],
                                full_frac=0.60,
                                partial_frac=0.30,
                                partial_top_bits=1,
                                seed=7,
                                fully_recover_first_n: int = 0,
                                fully_recover_bn: bool = True,
                                fully_recover_classifier: bool = True,
                                fully_recover_prefixes: str = "") -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build constraints and force Set-1 for:
      - first `fully_recover_first_n` parameter tensors (by state order)
      - BatchNorm / running stats (if fully_recover_bn)
      - classifier.{weight,bias} (if fully_recover_classifier)
      - any parameter whose name starts with any prefix in fully_recover_prefixes (comma-separated)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    cons = {}

    param_names = [k for k, t in state.items()
                   if isinstance(t, torch.Tensor) and t.ndim > 0 and t.numel() > 0]

    # parse prefixes
    prefix_list = []
    if isinstance(fully_recover_prefixes, str) and fully_recover_prefixes.strip():
        prefix_list = [p.strip() for p in fully_recover_prefixes.split(",") if p.strip()]

    for idx, name in enumerate(param_names):
        t = state[name]
        use_full_frac = float(full_frac)
        use_part_frac = float(partial_frac)

        # 1) first-N override
        if fully_recover_first_n > 0 and idx < fully_recover_first_n:
            use_full_frac, use_part_frac = 1.0, 0.0
        else:
            lname = name.lower()

            # 2) BN/affine/running stats override
            if fully_recover_bn and (("bn" in lname) or ("batchnorm" in lname) or
                                     ("running_mean" in lname) or ("running_var" in lname)):
                use_full_frac, use_part_frac = 1.0, 0.0

            # 3) classifier override
            if fully_recover_classifier and (
                lname == "classifier.weight" or lname == "classifier.bias" or
                lname.endswith("classifier.weight") or lname.endswith("classifier.bias")
            ):
                use_full_frac, use_part_frac = 1.0, 0.0

            # 4) explicit name-prefix override
            if prefix_list and any(name.startswith(p) for p in prefix_list):
                use_full_frac, use_part_frac = 1.0, 0.0

        cons[name] = build_leak_profile(t, use_full_frac, use_part_frac,
                                        partial_top_bits, rng=g)
    return cons

# ------------------------ Load float state from float/int8 pack ------------------------

def maybe_load_float_state(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    Accepts either:
      - a normal float state_dict (returned as-is), OR
      - an int8-packed dict with keys 'name::qint8' and 'name::scale'
        -> returns a float state_dict by dequantizing each packed tensor.
    """
    pack = torch.load(weights_path, map_location="cpu")
    any_q = [k for k in pack if isinstance(pack[k], torch.Tensor) and k.endswith("::qint8")]
    if any_q:
        state = {}
        SUFFIX = "::qint8"
        for k, v in pack.items():
            if k.endswith(SUFFIX):
                base = k[:-len(SUFFIX)]
                s_key = base + "::scale"
                if s_key not in pack:
                    raise KeyError(f"Missing scale for {base}")
                scale = float(pack[s_key].item()) if isinstance(pack[s_key], torch.Tensor) else float(pack[s_key])
                state[base] = v.to(torch.float32) * scale
            elif k.endswith("::scale"):
                continue
            else:
                state[k] = v
        return state
    return pack

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="Path to victim weights (.pt), float or int8-pack (name::qint8 + name::scale)")
    ap.add_argument("--full_frac", type=float, default=0.60,
                    help="Global fraction of elements in Set-1 (exact).")
    ap.add_argument("--partial_frac", type=float, default=0.30,
                    help="Global fraction of elements in Set-2 (top-k known).")
    ap.add_argument("--partial_top_bits", type=int, default=1,
                    help="k in top-k (MSB-first) for Set-2.")
    ap.add_argument("--fully_recover_first_n", type=int, default=0,
                    help="Force the first N parameter tensors (by state order) to Set-1.")
    ap.add_argument("--fully_recover_bn", action="store_true",
                    help="Force BatchNorm/affine/running stats to Set-1.")
    ap.add_argument("--fully_recover_classifier", action="store_true",
                    help="Force classifier.{weight,bias} to Set-1.")
    ap.add_argument("--fully_recover_prefixes", type=str, default="",
                    help="Comma-separated name prefixes forced to Set-1 (e.g. 'features.10.,features.14.,features.17.,classifier.')")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="leak_constraints.pt")
    args = ap.parse_args()

    state = maybe_load_float_state(args.weights)
    cons = constraints_from_state_dict(
        state,
        full_frac=args.full_frac,
        partial_frac=args.partial_frac,
        partial_top_bits=args.partial_top_bits,
        seed=args.seed,
        fully_recover_first_n=args.fully_recover_first_n,
        fully_recover_bn=args.fully_recover_bn,
        fully_recover_classifier=args.fully_recover_classifier,
        fully_recover_prefixes=args.fully_recover_prefixes
    )

    meta = {
        "full_frac": args.full_frac,
        "partial_frac": args.partial_frac,
        "partial_top_bits": args.partial_top_bits,
        "seed": args.seed,
        "fully_recover_first_n": args.fully_recover_first_n,
        "fully_recover_bn": args.fully_recover_bn,
        "fully_recover_classifier": args.fully_recover_classifier,
        "fully_recover_prefixes": args.fully_recover_prefixes,
    }

    torch.save({"constraints": cons, "meta": meta}, args.out)
    print(f"[OK] wrote constraints to {args.out}")

if __name__ == "__main__":
    main()