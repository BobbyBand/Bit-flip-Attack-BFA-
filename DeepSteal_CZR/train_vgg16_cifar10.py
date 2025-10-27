#!/usr/bin/env python3
import argparse, random
from typing import Dict, Tuple, Optional, List
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

SUFFIX_Q = "::qint8"
SUFFIX_S = "::scale"

def set_seed(s=7):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

cfg = {"VGG16":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]}

class VGG_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features=self._make_layers(cfg["VGG16"])
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
        x=self.features(x); x=torch.flatten(x,1); return self.classifier(x)

def make_loaders(batch_size=128, num_workers=2, subset_ratio=1.0):
    tfm_train=T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor()])
    tfm_test=T.Compose([T.ToTensor()])
    train=torchvision.datasets.CIFAR10("./data",train=True,download=True,transform=tfm_train)
    test=torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=tfm_test)
    if subset_ratio<1.0:
        n=int(len(train)*subset_ratio); idx=torch.randperm(len(train))[:n]; train=torch.utils.data.Subset(train, idx.tolist())
    return (torch.utils.data.DataLoader(train,batch_size,shuffle=True,num_workers=num_workers,pin_memory=True),
            torch.utils.data.DataLoader(test,256,shuffle=False,num_workers=num_workers,pin_memory=True))

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); c=t=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        p=model(x).argmax(1); c+=(p==y).sum().item(); t+=y.numel()
    return c/t

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
        pack[k+"::qint8"]=q.cpu()
        pack[k+"::scale"]=torch.tensor(float(scale),dtype=torch.float32)
    torch.save(pack,out_path)
    print(f"[int8] wrote: {out_path}")

def _load_float_state_from_any(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    Accepts a float state_dict or an int8-packed dict (name::qint8 + name::scale).
    Returns a float32 state_dict suitable for nn.Module.load_state_dict.
    """
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

def train(epochs, device, subset_ratio, batch_size, lr=0.1, momentum=0.9, wd=5e-4,
          init_state: Optional[Dict[str, torch.Tensor]] = None, strict_load: bool = False,
          constraints: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
          lambda_set2: float = 1e-4, stats_ema: float = 0.1,
          polish_iters: int = 40, polish_lr_scale: float = 0.1,
          freeze_set1: bool = True):
    model=VGG_CIFAR().to(device)
    if init_state is not None:
        load_res = model.load_state_dict(init_state, strict=strict_load)
        if load_res.missing_keys:
            print(f"[init] missing keys ({len(load_res.missing_keys)}): {load_res.missing_keys[:5]}")
        if load_res.unexpected_keys:
            print(f"[init] unexpected keys ({len(load_res.unexpected_keys)}): {load_res.unexpected_keys[:5]}")
    train_loader,test_loader=make_loaders(batch_size=batch_size, subset_ratio=subset_ratio)
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
        print(f"[Victim] epoch {ep}/{epochs}  acc={eval_acc(model,test_loader,device)*100:.2f}%  CE={ce_avg:.4f}  reg={reg_avg:.4f}")
    if constraint_mgr is not None:
        constraint_mgr.close()
    return model, test_loader, init_acc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--subset_ratio",type=float,default=1.0)
    ap.add_argument("--out_float",default="victim_vgg16_cifar10_float.pt")
    ap.add_argument("--out_int8",default="victim_vgg16_cifar10_int8.pt")
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

    gt_acc = None
    if args.gt_weights:
        print(f"[gt] evaluating {args.gt_weights}")
        gt_state = _load_float_state_from_any(args.gt_weights)
        gt_model = VGG_CIFAR().to(device)
        gt_res = gt_model.load_state_dict(gt_state, strict=False)
        if gt_res.missing_keys:
            print(f"[gt] missing keys ({len(gt_res.missing_keys)}): {gt_res.missing_keys[:5]}")
        if gt_res.unexpected_keys:
            print(f"[gt] unexpected keys ({len(gt_res.unexpected_keys)}): {gt_res.unexpected_keys[:5]}")
        gt_acc = eval_acc(gt_model, test_loader, device)
        print(f"[gt] accuracy: {gt_acc*100:.2f}%")
        if init_acc is not None:
            diff_init = (init_acc - gt_acc) * 100.0
            print(f"[delta] init - GT: {diff_init:+.2f} pts")
        diff_trained = (acc - gt_acc) * 100.0
        print(f"[delta] trained - GT: {diff_trained:+.2f} pts")

    torch.save(model.state_dict(), args.out_float); print(f"[float] wrote: {args.out_float}")
    save_int8_state_from_state_dict(model.state_dict(), args.out_int8)

if __name__=="__main__": main()
