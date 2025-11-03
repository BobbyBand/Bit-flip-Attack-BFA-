#!/usr/bin/env python3
import argparse, random
from typing import Dict, Tuple, Optional, List
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

SUFFIX_Q = "::qint8"
SUFFIX_S = "::scale"

# ----------------------------
# Utils
# ----------------------------

def set_seed(s=7):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------------------
# VGG CIFAR (original, generalized to VGG11/VGG16)
# ----------------------------

cfg = {
    "VGG11":[64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "VGG16":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]
}

class VGG_CIFAR(nn.Module):
    def __init__(self, name="VGG16", num_classes=10):
        super().__init__()
        self.name = name.upper()
        self.features=self._make_layers(cfg[self.name])
        self.classifier=nn.Linear(512, num_classes)
    def _make_layers(self, cfg_list):
        layers=[]; in_ch=3
        for v in cfg_list:
            if v=="M": layers+=[nn.MaxPool2d(2,2)]
            else:
                layers += [nn.Conv2d(in_ch,v,3,padding=1,bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch=v
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.features(x)          # 32 -> 1 spatial after 5 pools in VGG16; VGG11 also ends at 1x1
        x=torch.flatten(x,1)
        return self.classifier(x)

# ----------------------------
# ResNet CIFAR (BasicBlock, 3x3 stem, no initial maxpool)
# ----------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet_CIFAR(nn.Module):
    def __init__(self, layers: List[int], num_classes=10, width=64):
        super().__init__()
        self.inplanes = width
        self.conv1 = conv3x3(3, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(width,   layers[0], stride=1)
        self.layer2 = self._make_layer(width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(width*4, layers[2], stride=2)
        self.layer4 = self._make_layer(width*8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width*8*BasicBlock.expansion, num_classes)
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)

def create_model(name:str, num_classes:int):
    n = name.lower()
    if n == "vgg16": return VGG_CIFAR("VGG16", num_classes)
    if n == "vgg11": return VGG_CIFAR("VGG11", num_classes)
    if n == "resnet18": return ResNet_CIFAR([2,2,2,2], num_classes)
    if n == "resnet34": return ResNet_CIFAR([3,4,6,3], num_classes)
    raise ValueError(f"Unknown model: {name}")

# ----------------------------
# Data & Eval
# ----------------------------

def make_loaders(batch_size=128, num_workers=2, subset_ratio=1.0, dataset="cifar10"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tfm_train=T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean,std)])
    tfm_test=T.Compose([T.ToTensor(),T.Normalize(mean,std)])
    is_c100 = dataset.lower() == "cifar100"
    ds_train = (torchvision.datasets.CIFAR100 if is_c100 else torchvision.datasets.CIFAR10)(
        "./data",train=True,download=True,transform=tfm_train)
    ds_test = (torchvision.datasets.CIFAR100 if is_c100 else torchvision.datasets.CIFAR10)(
        "./data",train=False,download=True,transform=tfm_test)
    if subset_ratio<1.0:
        n=int(len(ds_train)*subset_ratio); idx=torch.randperm(len(ds_train))[:n]
        ds_train=torch.utils.data.Subset(ds_train, idx.tolist())
    return (torch.utils.data.DataLoader(ds_train,batch_size,shuffle=True,num_workers=num_workers,pin_memory=True),
            torch.utils.data.DataLoader(ds_test,256,shuffle=False,num_workers=num_workers,pin_memory=True),
            (100 if is_c100 else 10))

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); c=t=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        p=model(x).argmax(1); c+=(p==y).sum().item(); t+=y.numel()
    return c/t

# ----------------------------
# Int8 helpers 
# ----------------------------

@torch.no_grad()
def quantize_int8_per_tensor(w: torch.Tensor):
    max_abs=w.abs().max(); scale=(max_abs/127.0).clamp(min=1e-8)
    q=torch.round(w/scale).clamp(-128,127).to(torch.int8)
    return q, scale

@torch.no_grad()
def save_int8_state_from_state_dict(state: Dict[str,torch.Tensor], out_path:str):
    pack={}
    for k,v in state.items():
        if not torch.is_floating_point(v) or v.numel()==0:
            pack[k]=v.clone(); continue
        q,scale=quantize_int8_per_tensor(v)
        pack[k+SUFFIX_Q]=q.cpu()
        pack[k+SUFFIX_S]=torch.tensor(float(scale),dtype=torch.float32)
    torch.save(pack,out_path)
    print(f"[int8] wrote: {out_path}")

def _load_float_state_from_any(weights_path: str) -> Dict[str, torch.Tensor]:
    pack = torch.load(weights_path, map_location="cpu")
    if not isinstance(pack, dict):
        raise TypeError(f"Unsupported state type from {weights_path!r}: {type(pack)}")
    any_q = any(isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q) for k, v in pack.items())
    if not any_q:
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in pack.items()}
    state: Dict[str, torch.Tensor] = {}
    for k, v in pack.items():
        if not (isinstance(v, torch.Tensor) and k.endswith(SUFFIX_Q)):
            continue
        base = k[:-len(SUFFIX_Q)]
        s_key = base + SUFFIX_S
        if s_key not in pack:
            raise KeyError(f"Missing {s_key} for packed tensor {k}")
        scale_t = pack[s_key]
        scale = float(scale_t.item() if isinstance(scale_t, torch.Tensor) else scale_t)
        state[base] = v.to(torch.float32) * scale
    return state

# ----------------------------
# Constraint manager 
# ----------------------------

class ConstraintManager:
    """
    Handles Set-1 freezing and Set-2 projection/regularization based on leaked constraints.
    Set-1 positions are kept fixed; Set-2 positions are regularized toward a running mean and
    clipped within shifting ranges derived from leaked min/max.
    """
    def __init__(self, model: nn.Module, constraints: Dict[str, Dict[str, torch.Tensor]],
                 device: torch.device, stats_ema: float = 0.1, freeze_set1: bool = True):
        self.stats_ema = float(stats_ema)
        self.freeze_set1 = freeze_set1
        self.params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        for name, param in model.named_parameters():
            if name not in constraints:
                continue
            cons = constraints[name]
            if not isinstance(cons, dict):
                continue
            try:
                mask_full = cons["mask_full"].to(device=device, dtype=torch.bool)
                mask_part = cons["mask_part"].to(device=device, dtype=torch.bool)
                mask_none = cons["mask_none"].to(device=device, dtype=torch.bool)
            except KeyError:
                continue
            if mask_full.shape != param.shape or mask_part.shape != param.shape:
                continue

            info: Dict[str, torch.Tensor] = {
                "mask_full": mask_full,
                "mask_part": mask_part,
                "mask_none": mask_none
            }
            if freeze_set1 and mask_full.any():
                frozen = param.detach().clone()
                info["frozen"] = frozen
                hook = param.register_hook(lambda grad, m=mask_full: grad.masked_fill(m, 0))
                self.hooks.append(hook)

            if mask_part.any():
                part_min = cons["part_min"].to(device=device, dtype=param.dtype).clone()
                part_max = cons["part_max"].to(device=device, dtype=param.dtype).clone()
                part_mean = cons["part_mean"].to(device=device, dtype=param.dtype).clone()

                lower_margin = (part_mean - part_min).clamp_min(0.0)
                upper_margin = (part_max - part_mean).clamp_min(0.0)

                info.update({
                    "mean": part_mean,
                    "current_mean": part_mean.clone(),
                    "current_min": part_min.clone(),
                    "current_max": part_max.clone(),
                    "lower_margin": lower_margin,
                    "upper_margin": upper_margin,
                })
            self.params[name] = info

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def regularization(self, model: nn.Module) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        total = 0
        for name, param in model.named_parameters():
            info = self.params.get(name)
            if not info or "mean" not in info:
                continue
            mask = info["mask_part"]
            if not mask.any():
                continue
            diff = param[mask] - info["current_mean"][mask]
            reg = reg + torch.sum(diff * diff)
            total += diff.numel()
        if total == 0:
            return torch.tensor(0.0, device=reg.device)
        return reg / total

    @torch.no_grad()
    def project(self, model: nn.Module, clip_enabled: bool = True):
        for name, param in model.named_parameters():
            info = self.params.get(name)
            if not info:
                continue
            if self.freeze_set1 and "frozen" in info:
                mask_full = info["mask_full"]
                param.data[mask_full] = info["frozen"][mask_full]
            if clip_enabled and "current_min" in info:
                mask_part = info["mask_part"]
                if mask_part.any():
                    min_t = info["current_min"]
                    max_t = info["current_max"]
                    param.data[mask_part] = torch.max(torch.min(param.data[mask_part], max_t[mask_part]), min_t[mask_part])

    @torch.no_grad()
    def update_stats(self, model: nn.Module):
        for name, param in model.named_parameters():
            info = self.params.get(name)
            if not info or "current_mean" not in info:
                continue
            mask_part = info["mask_part"]
            if not mask_part.any():
                continue
            current_vals = param.data[mask_part]
            mean_t = info["current_mean"]
            if self.stats_ema > 0.0:
                ema = self.stats_ema
                mean_t_vals = mean_t[mask_part]
                mean_t_vals.mul_(1.0 - ema).add_(current_vals * ema)
                mean_t[mask_part] = mean_t_vals
            else:
                mean_t[mask_part] = current_vals

            lower = info["lower_margin"]
            upper = info["upper_margin"]
            info["current_min"][mask_part] = mean_t[mask_part] - lower[mask_part]
            info["current_max"][mask_part] = mean_t[mask_part] + upper[mask_part]

# ----------------------------
# Training
# ----------------------------

def train(epochs, device, subset_ratio, batch_size, model_name,
          dataset="cifar10", lr=0.1, momentum=0.9, wd=5e-4,
          init_state: Optional[Dict[str, torch.Tensor]] = None, strict_load: bool = False,
          constraints: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
          lambda_set2: float = 1e-4, stats_ema: float = 0.1,
          polish_iters: int = 40, polish_lr_scale: float = 0.1,
          freeze_set1: bool = True):
    # data
    train_loader,test_loader,num_classes = make_loaders(batch_size=batch_size, subset_ratio=subset_ratio, dataset=dataset)
    # model
    model = create_model(model_name, num_classes=num_classes).to(device)
    if init_state is not None:
        load_res = model.load_state_dict(init_state, strict=strict_load)
        if getattr(load_res, "missing_keys", None):
            print(f"[init] missing keys ({len(load_res.missing_keys)}): {load_res.missing_keys[:5]}")
        if getattr(load_res, "unexpected_keys", None):
            print(f"[init] unexpected keys ({len(load_res.unexpected_keys)}): {load_res.unexpected_keys[:5]}")
    init_acc = None
    if init_state is not None:
        init_acc = eval_acc(model, test_loader, device)
        print(f"[init] accuracy before training: {init_acc*100:.2f}%")
    opt=torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    constraint_mgr = None
    if constraints:
        constraint_mgr = ConstraintManager(model, constraints, torch.device(device),
                                           stats_ema=stats_ema, freeze_set1=freeze_set1)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.5*epochs), int(0.75*epochs)], gamma=0.1)
    total_iters = epochs * len(train_loader)
    polish_active = False
    step_idx = 0
    for ep in range(1,epochs+1):
        model.train()
        epoch_ce = 0.0
        epoch_reg = 0.0
        epoch_batches = 0
        for x,y in train_loader:
            step_idx += 1
            in_polish = polish_iters > 0 and step_idx > max(0, total_iters - polish_iters)
            if in_polish and not polish_active:
                for g in opt.param_groups:
                    g["lr"] = g["lr"] * polish_lr_scale
                polish_active = True
                print(f"[polish] entering final phase at step {step_idx}: lr scaled by {polish_lr_scale}, lambda=0, clipping disabled")
            opt.zero_grad(set_to_none=True)
            x,y=x.to(device),y.to(device)
            logits = model(x)
            ce_loss = F.cross_entropy(logits,y)
            loss = ce_loss
            lambda_eff = 0.0 if in_polish else lambda_set2
            reg_val = torch.tensor(0.0, device=device)
            if constraint_mgr is not None and lambda_eff > 0.0:
                reg_val = constraint_mgr.regularization(model)
                loss = loss + lambda_eff * reg_val
            loss.backward()
            opt.step()
            clip_enabled = not in_polish
            if constraint_mgr is not None:
                constraint_mgr.project(model, clip_enabled=clip_enabled)
                constraint_mgr.update_stats(model)
            epoch_ce += ce_loss.item()
            epoch_reg += reg_val.item() if constraint_mgr is not None else 0.0
            epoch_batches += 1
        sched.step()
        reg_avg = (epoch_reg / max(1, epoch_batches)) if constraint_mgr is not None else 0.0
        ce_avg = epoch_ce / max(1, epoch_batches)
        print(f"[{model_name}] epoch {ep}/{epochs}  acc={eval_acc(model,test_loader,device)*100:.2f}%  CE={ce_avg:.4f}  reg={reg_avg:.4f}")
    if constraint_mgr is not None:
        constraint_mgr.close()
    return model, test_loader, init_acc

# ----------------------------
# CLI
# ----------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="vgg16", choices=["vgg16","vgg11","resnet18","resnet34"])
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--subset_ratio",type=float,default=1.0)
    ap.add_argument("--out_float",default=None, help="Defaults to victim_{model}_{dataset}_float.pt")
    ap.add_argument("--out_int8",default=None, help="Defaults to victim_{model}_{dataset}_int8.pt")
    ap.add_argument("--init_weights",default=None,help="Optional float or int8-packed weights to initialize the model")
    ap.add_argument("--init_strict",action="store_true",help="Load init weights with strict=True (default: False)")
    ap.add_argument("--gt_weights",default=None,help="Optional ground-truth weights to report comparison accuracy")
    ap.add_argument("--constraints",default=None,help="Constraint pack (.pt) containing masks/ranges for Set-1/2/3")
    ap.add_argument("--lambda_set2",type=float,default=1e-4,help="L2 regularization weight for Set-2 deviation")
    ap.add_argument("--stats_ema",type=float,default=0.1,help="EMA factor for updating Set-2 mean (0 disables EMA)")
    ap.add_argument("--polish_iters",type=int,default=40,help="Number of final iterations to run with lambda=0 and no clipping")
    ap.add_argument("--polish_lr_scale",type=float,default=0.1,help="LR multiplier applied during the final polish phase")
    ap.add_argument("--no_freeze_set1",action="store_true",help="If set, do not freeze Set-1 weights")
    args=ap.parse_args()

    set_seed(7)
    device=args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")

    out_float = args.out_float or f"victim_{args.model}_{args.dataset}_float.pt"
    out_int8  = args.out_int8  or f"victim_{args.model}_{args.dataset}_int8.pt"

    init_state = None
    if args.init_weights:
        print(f"[init] loading from {args.init_weights}")
        init_state = _load_float_state_from_any(args.init_weights)

    constraints_dict = None
    if args.constraints:
        cons_pack = torch.load(args.constraints, map_location="cpu")
        if isinstance(cons_pack, dict) and "constraints" in cons_pack:
            constraints_dict = cons_pack["constraints"]
        elif isinstance(cons_pack, dict):
            constraints_dict = cons_pack
        else:
            raise TypeError(f"Unsupported constraints format in {args.constraints}")

    model,test_loader,init_acc=train(args.epochs, device, args.subset_ratio, args.batch_size,
                                     model_name=args.model, dataset=args.dataset,
                                     init_state=init_state, strict_load=args.init_strict,
                                     constraints=constraints_dict, lambda_set2=args.lambda_set2,
                                     stats_ema=args.stats_ema, polish_iters=args.polish_iters,
                                     polish_lr_scale=args.polish_lr_scale,
                                     freeze_set1=not args.no_freeze_set1)
    acc=eval_acc(model, test_loader, device)
    if init_acc is not None:
        print(f"Final test accuracy: {acc*100:.2f}%  (improved from {init_acc*100:.2f}%)")
    else:
        print(f"Final test accuracy: {acc*100:.2f}%")

    # Optional GT eval uses the same backbone to avoid mismatches.
    if args.gt_weights:
        print(f"[gt] evaluating {args.gt_weights}")
        gt_state = _load_float_state_from_any(args.gt_weights)
        gt_model = create_model(args.model, num_classes=model.fc.out_features if hasattr(model, "fc") else model.classifier.out_features).to(device)
        gt_res = gt_model.load_state_dict(gt_state, strict=False)
        if getattr(gt_res, "missing_keys", None):
            print(f"[gt] missing keys ({len(gt_res.missing_keys)}): {gt_res.missing_keys[:5]}")
        if getattr(gt_res, "unexpected_keys", None):
            print(f"[gt] unexpected keys ({len(gt_res.unexpected_keys)}): {gt_res.unexpected_keys[:5]}")
        gt_acc = eval_acc(gt_model, test_loader, device)
        print(f"[gt] accuracy: {gt_acc*100:.2f}%")
        if init_acc is not None:
            diff_init = (init_acc - gt_acc) * 100.0
            print(f"[delta] init - GT: {diff_init:+.2f} pts")
        diff_trained = (acc - gt_acc) * 100.0
        print(f"[delta] trained - GT: {diff_trained:+.2f} pts")

    torch.save(model.state_dict(), out_float); print(f"[float] wrote: {out_float}")
    save_int8_state_from_state_dict(model.state_dict(), out_int8)

if __name__=="__main__": 
    main()