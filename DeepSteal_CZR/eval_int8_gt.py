#!/usr/bin/env python3
import argparse, torch
import torch.nn as nn
import torchvision, torchvision.transforms as T

SUFFIX_Q = "::qint8"
SUFFIX_S = "::scale"

# ---------------- Model (VGG-16 for CIFAR-10) ----------------
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
        layers = []; in_ch = 3
        for v in cfg_list:
            if v == "M":
                layers += [nn.MaxPool2d(2,2)]
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_ch = v
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# -------------- Int8-pack -> float32 state_dict --------------
def dequantize_int8_pack_to_float_state(pack: dict) -> dict:
    state = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q):
            base = k[:-len(SUFFIX_Q)]
            s_key = base + SUFFIX_S
            if s_key not in pack:
                raise KeyError(f"Missing scale for {base}")
            scale = float(pack[s_key].item() if isinstance(pack[s_key], torch.Tensor) else pack[s_key])
            state[base] = v.to(torch.float32) * scale
    return state

# ---------------- CIFAR-10 loader & eval ----------------
def make_test_loader(batch_size=256, num_workers=2):
    tfm = T.Compose([T.ToTensor()])
    test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    return torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def eval_acc(model: nn.Module, loader, device="cpu"):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8_pt", required=True, help="Path to int8-packed GT model (.pt)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    print(f"[INFO] Loading int8-packed weights from: {args.int8_pt}")
    pack = torch.load(args.int8_pt, map_location="cpu")
    # dequantize to float32
    state = dequantize_int8_pack_to_float_state(pack)

    # build model & load
    model = VGG_CIFAR("VGG16")
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # fallback: copy by matching keys
        msd = model.state_dict()
        for k, v in state.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k].copy_(v)
        model.load_state_dict(msd, strict=False)

    # eval
    test_loader = make_test_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    acc = eval_acc(model, test_loader, device=args.device)
    print(f"[RESULT] CIFAR-10 top-1 accuracy = {acc*100:.2f}%")

if __name__ == "__main__":
    main()