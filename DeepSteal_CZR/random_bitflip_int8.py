#!/usr/bin/env python3
"""
Random bit-flip (Rowhammer-style) on int8-packed weights and eval.

Usage examples (one-liners shown at the end):
  python random_bitflip_int8.py \
    --in reconstructed_czr_int8.pt --out flipped_int8.pt --mode total_count --N 50000 --seed 7

This will flip N random bits across all qint8 tensors uniformly.
"""
import argparse, math, random
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ------------------------------
# Model (same VGG_CIFAR used earlier)
# ------------------------------
cfg = {
    "VGG16": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, "M",
              512, 512, 512, "M",
              512, 512, 512, "M"]
}

class VGG_CIFAR(nn.Module):
    def __init__(self, vgg_name="VGG16", num_classes=10):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg_list):
        layers = []
        in_ch = 3
        for v in cfg_list:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------------------------
# Data loader for CIFAR-10 (test)
# ------------------------------
def make_test_loader(batch_size=256, num_workers=2):
    tfm_test = T.Compose([T.ToTensor()])
    test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)
    loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader

@torch.no_grad()
def eval_acc(model: nn.Module, loader, device="cpu"):
    model.eval()
    correct = 0; total = 0
    model.to(device)
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    return correct/total

# ------------------------------
# Helpers for int8 pack format
# ------------------------------
SUFFIX_Q = "::qint8"
SUFFIX_S = "::scale"

def u8_from_s8(s: torch.Tensor) -> torch.Tensor:
    """s: signed int8 tensor (torch.int8) -> returns uint8 as int16 tensor 0..255"""
    return (s.to(torch.int16) & 0xFF).to(torch.uint8)

def s8_from_u8(u: torch.Tensor) -> torch.Tensor:
    """u: uint8 tensor -> returns signed int8 tensor"""
    u16 = u.to(torch.int16)
    # convert unsigned 0..255 -> signed -128..127
    return torch.where(u16 >= 128, (u16 - 256).to(torch.int16), u16).to(torch.int8)

def dequantize_int8_pack_to_float_state(pack: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Given an int8-packed dict (k::qint8, k::scale), produce a normal float state_dict mapping base->float tensor.
    """
    state = {}
    for k,v in pack.items():
        if isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q):
            base = k[:-len(SUFFIX_Q)]
            s_key = base + SUFFIX_S
            if s_key not in pack:
                raise KeyError(f"Missing scale for {base}")
            scale = float(pack[s_key].item() if isinstance(pack[s_key], torch.Tensor) else pack[s_key])
            q = v.to(torch.int8).to(torch.float32)
            state[base] = q * scale
    # copy non-parameter items optionally (ignored for model loading)
    return state

# ------------------------------
# Bit-flip routines
# ------------------------------
def collect_q_elements(pack: Dict[str, torch.Tensor]) -> List[Tuple[str, int]]:
    """
    Return list of (key, num_elements) for all ::qint8 entries.
    """
    items = []
    for k,v in pack.items():
        if isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q):
            items.append((k, v.numel()))
    return items

def total_elements(pack: Dict[str, torch.Tensor]) -> int:
    return sum(n for _, n in collect_q_elements(pack))

def flip_random_bits_total(pack: Dict[str, torch.Tensor], N: int, seed: int = 7) -> Dict[str, torch.Tensor]:
    """Flip N random bit positions across all qint8 bytes uniformly."""
    rng = np.random.default_rng(seed)
    items = collect_q_elements(pack)
    keys, sizes = zip(*items) if items else ([], [])
    cum = np.cumsum([0] + list(sizes))
    total = int(cum[-1])
    if total == 0:
        return pack
    # choose N positions in [0, total-1]
    chosen = rng.integers(0, total, size=N)
    # choose bit indices 0..7 for each flip
    bit_idxs = rng.integers(0, 8, size=N)
    # For each flip, map to (key, local_index, bit)
    # Build per-key lists
    per_key_indices = {k: [] for k in keys}
    for pos, bit in zip(chosen, bit_idxs):
        # find which key contains pos using binary search on cum
        i = int(np.searchsorted(cum, pos, side='right') - 1)
        if i < 0:
            i = 0
        if i >= len(keys):
            i = len(keys)-1
        local = int(pos - cum[i])
        per_key_indices[keys[i]].append( (local, int(bit)) )

    # Make a copy of pack (shallow copy then clone tensors we modify)
    newpack = dict(pack)
    for k, flips in per_key_indices.items():
        if not flips: continue
        q = newpack[k].clone().to(torch.uint8)  # 0..255 bytes
        flat = q.view(-1)
        # apply flips (multiple flips on same index allowed)
        for idx, bit in flips:
            # XOR the chosen bit
            flat[idx] = (flat[idx].item() ^ (1 << bit))
        # convert back to int8 representation
        newpack[k] = s8_from_u8(flat.view(newpack[k].shape)).to(torch.int8)
    return newpack

def flip_random_bits_per_tensor(pack: Dict[str, torch.Tensor], flips_per_tensor: int, seed: int = 7) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    newpack = dict(pack)
    for k,v in pack.items():
        if not (isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q)): 
            continue
        q = v.clone().to(torch.uint8).view(-1)
        n = q.numel()
        if n == 0: 
            continue
        # choose flips_per_tensor indices (with replacement)
        idxs = rng.integers(0, n, size=flips_per_tensor)
        bits = rng.integers(0, 8, size=flips_per_tensor)
        for ii, b in zip(idxs, bits):
            q[ii] = (q[ii].item() ^ (1 << int(b)))
        newpack[k] = s8_from_u8(q.view(v.shape)).to(torch.int8)
    return newpack

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input int8-packed state (.pt)")
    ap.add_argument("--out", required=True, help="Output flipped int8-packed state (.pt)")
    ap.add_argument("--mode", choices=["total_count", "per_tensor_count"], default="total_count",
                    help="total_count: flip N bits across all tensors. per_tensor_count: flip N bits per tensor.")
    ap.add_argument("--N", type=int, default=10000, help="Number of flips (meaning depends on mode)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval", action="store_true", help="Evaluate model accuracy before and after flipping")
    ap.add_argument("--device", default="cpu", help="Device for evaluation (cpu or cuda:0)")
    args = ap.parse_args()

    # load pack
    pack = torch.load(args.inp, map_location="cpu")
    # quick sanity: does pack contain qint8 entries?
    q_items = collect_q_elements(pack)
    if not q_items:
        raise RuntimeError("No ::qint8 entries found in input pack.")

    print(f"[INFO] Found {len(q_items)} qint8 tensors totaling {total_elements(pack)} elements.")

    # Optional eval before
    if args.eval:
        print("[INFO] Evaluating original (dequantized) model...")
        base_state = dequantize_int8_pack_to_float_state(pack)
        model = VGG_CIFAR("VGG16")
        try:
            model.load_state_dict(base_state, strict=False)
        except Exception as e:
            # try mapping keys by name if necessary (we expect base keys to match)
            model_state = model.state_dict()
            for k,v in base_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k].copy_(v)
            model.load_state_dict(model_state)
        test_loader = make_test_loader()
        acc_before = eval_acc(model, test_loader, device=args.device)
        print(f"[BEFORE] Test accuracy = {acc_before*100:.2f}%")

    # perform flips
    if args.mode == "total_count":
        newpack = flip_random_bits_total(pack, args.N, seed=args.seed)
    else:
        newpack = flip_random_bits_per_tensor(pack, args.N, seed=args.seed)

    torch.save(newpack, args.out)
    print(f"[OK] Saved flipped int8-packed model to: {args.out}")

    # Optional eval after
    if args.eval:
        print("[INFO] Evaluating flipped (dequantized) model...")
        flipped_state = dequantize_int8_pack_to_float_state(newpack)
        model2 = VGG_CIFAR("VGG16")
        try:
            model2.load_state_dict(flipped_state, strict=False)
        except Exception:
            model_state = model2.state_dict()
            for k,v in flipped_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k].copy_(v)
            model2.load_state_dict(model_state)
        acc_after = eval_acc(model2, test_loader, device=args.device)
        print(f"[AFTER] Test accuracy = {acc_after*100:.2f}%")
        print(f"[DELTA] Accuracy change = {(acc_after-acc_before)*100:.2f}%")

if __name__ == "__main__":
    main()