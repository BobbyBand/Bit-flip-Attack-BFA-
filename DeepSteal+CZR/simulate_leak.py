#!/usr/bin/env python3
import argparse, torch
from typing import Dict, Tuple

def u8_to_s8(u: torch.Tensor): return torch.where(u>=128,u-256,u)
def s8_to_u8(s: torch.Tensor): return (s & 0xFF).to(torch.int16)

@torch.no_grad()
def quantize_int8_per_tensor(w: torch.Tensor):
    max_abs=w.abs().max(); scale=(max_abs/127.0).clamp(min=1e-8)
    q=torch.round(w/scale).clamp(-128,127).to(torch.int8)
    return q, scale

def bits_prefix_mask(k:int)->int: return 0 if k<=0 else ((0xFF<<(8-k)) & 0xFF)

def int8_range_from_topbits(u8_true: torch.Tensor, k_top: int):
    if k_top<=0: return torch.zeros_like(u8_true), torch.full_like(u8_true,255)
    m=bits_prefix_mask(k_top); top=(u8_true & m); low_max=(1<<(8-k_top))-1
    return top, top|low_max

def dequantize_from_int8_range(min_u8,max_u8,scale):
    min_s8=u8_to_s8(min_u8.to(torch.int16)).to(torch.float32)
    max_s8=u8_to_s8(max_u8.to(torch.int16)).to(torch.float32)
    return min_s8*scale, max_s8*scale

@torch.no_grad()
def build_leak_profile(param: torch.Tensor, full_frac: float, partial_frac: float, partial_top_bits: int,
                       rng: torch.Generator=None)->Dict[str,torch.Tensor]:
    device=param.device; n=param.numel()
    if rng is None: idx=torch.randperm(n, device=device)
    else:
        if rng.device!=device.type: rng=torch.Generator(device=device.type).manual_seed(int(torch.seed()))
        idx=torch.randperm(n, generator=rng, device=device)
    n_full=int(max(0,min(1.0,full_frac))*n)
    n_part=int(max(0,min(1.0,partial_frac))*n)
    n_part=min(n_part, max(0,n-n_full))
    mfull=torch.zeros(n,dtype=torch.bool,device=device); mpart=torch.zeros(n,dtype=torch.bool,device=device)
    mfull[idx[:n_full]]=True; mpart[idx[n_full:n_full+n_part]]=True; mnone=~(mfull|mpart)

    q8,scale=quantize_int8_per_tensor(param); u8=s8_to_u8(q8.to(torch.int16))
    full_u8=u8.view(-1)[mfull]
    full_min_f,full_max_f=dequantize_from_int8_range(full_u8,full_u8,scale)
    full_mean_f=(full_min_f+full_max_f)*0.5

    k=int(max(0,min(8,partial_top_bits)))
    part_u8=u8.view(-1)[mpart]
    if k==0:
        pmin_u8=torch.zeros_like(part_u8); pmax_u8=torch.full_like(part_u8,255)
    else:
        m=bits_prefix_mask(k); top=(part_u8 & m); low_max=(1<<(8-k))-1
        pmin_u8=top; pmax_u8=top|low_max
    part_min_f,part_max_f=dequantize_from_int8_range(pmin_u8,pmax_u8,scale)
    part_mean_f=(part_min_f+part_max_f)*0.5

    out={
        "mask_full": mfull.view_as(param),
        "mask_part": mpart.view_as(param),
        "mask_none": mnone.view_as(param),
        "full_min": torch.zeros_like(param),
        "full_max": torch.zeros_like(param),
        "full_mean":torch.zeros_like(param),
        "part_min": torch.zeros_like(param),
        "part_max": torch.zeros_like(param),
        "part_mean":torch.zeros_like(param),
        "scale": torch.as_tensor(scale, device=device),
    }
    out["full_min"].view(-1)[mfull]=full_min_f; out["full_max"].view(-1)[mfull]=full_max_f; out["full_mean"].view(-1)[mfull]=full_mean_f
    out["part_min"].view(-1)[mpart]=part_min_f; out["part_max"].view(-1)[mpart]=part_max_f; out["part_mean"].view(-1)[mpart]=part_mean_f
    return out

@torch.no_grad()
def constraints_from_state_dict(state: Dict[str,torch.Tensor], full_frac=0.60, partial_frac=0.30, partial_top_bits=1, seed=7):
    g=torch.Generator(device="cpu").manual_seed(seed)
    cons={}
    for name,t in state.items():
        if torch.is_floating_point(t) and t.ndim>0 and t.numel()>0:
            cons[name]=build_leak_profile(t, full_frac, partial_frac, partial_top_bits, rng=g)
    return cons

def maybe_load_float_state(weights_path:str)->Dict[str,torch.Tensor]:
    pack=torch.load(weights_path, map_location="cpu")
    # int8-pack detection
    any_q=[k for k in pack if isinstance(pack[k],torch.Tensor) and k.endswith("::qint8")]
    if any_q:
        state={}
        for k,v in pack.items():
            SUFFIX = "::qint8"
            if k.endswith(SUFFIX):
                base = k[:-len(SUFFIX)]      # strip only "::qint8"
                s_key = base + "::scale"
                if s_key not in pack:
                    raise KeyError(f"Missing scale for {base}")
                scale = float(pack[s_key].item()) if isinstance(pack[s_key], torch.Tensor) else float(pack[s_key])
                state[base] = pack[k].to(torch.float32) * scale
            elif k.endswith("::scale"):
                continue
            else:
                state[k]=v
        return state
    # already a float state_dict
    return pack

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to victim weights (.pt) float or int8-pack")
    ap.add_argument("--full_frac", type=float, default=0.60)
    ap.add_argument("--partial_frac", type=float, default=0.30)
    ap.add_argument("--partial_top_bits", type=int, default=1)
    ap.add_argument("--out", default="leak_constraints.pt")
    args=ap.parse_args()

    state=maybe_load_float_state(args.weights)
    cons=constraints_from_state_dict(state, args.full_frac, args.partial_frac, args.partial_top_bits, seed=7)
    meta={"full_frac":args.full_frac,"partial_frac":args.partial_frac,"partial_top_bits":args.partial_top_bits,"seed":7}
    torch.save({"constraints":cons,"meta":meta}, args.out)
    print(f"[OK] wrote constraints to {args.out}")

if __name__=="__main__": main()