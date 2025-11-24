#!/usr/bin/env python3
import argparse, math, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
try:
    import timm
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -----------------------------
# Args
# -----------------------------
def build_argparser():
    ap = argparse.ArgumentParser("ViT/DeiT/Swin trainer (CIFAR or ImageNet)")
    ap.add_argument("--model",
                    choices=[
                        "deit_tiny_patch16_224",
                        "deit_small_patch16_224",
                        "swin_tiny_patch4_window7_224",
                        "vit_base_patch16_224",
                    ],
                    default="vit_base_patch16_224")
    ap.add_argument("--dataset", choices=["cifar10","cifar100","imagenet"], default="cifar10")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--disable_cosine", action="store_true")
    ap.add_argument("--use_mixup", action="store_true")
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--subset_ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_float", default=None)
    ap.add_argument("--out_int8", default=None)
    return ap

# -----------------------------
# Datasets
# -----------------------------
def get_dataset(name: str, root: str):
    train_tf = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    if name == "cifar10":
        ds_train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        ds_val   = datasets.CIFAR10(root, train=False, download=True, transform=val_tf)
        num_classes = 10
    elif name == "cifar100":
        ds_train = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        ds_val   = datasets.CIFAR100(root, train=False, download=True, transform=val_tf)
        num_classes = 100
    elif name == "imagenet":
        ds_train = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        ds_val   = datasets.ImageFolder(os.path.join(root, "val"), transform=val_tf)
        num_classes = len(ds_train.classes)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds_train, ds_val, num_classes

def subset_dataset(dataset, ratio, seed):
    if ratio >= 1.0: return dataset
    n = len(dataset)
    k = max(1, int(n*ratio))
    g = random.Random(seed)
    idx = list(range(n)); g.shuffle(idx)
    return Subset(dataset, idx[:k])

# -----------------------------
# Model + training helpers
# -----------------------------
def build_model(name, num_classes, pretrained):
    if not _HAS_TIMM: raise RuntimeError("timm required")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

def cosine_decay(step, total_steps, lr_min, lr_max):
    if step >= total_steps: return lr_min
    cos = (1+math.cos(math.pi*step/total_steps))/2
    return lr_min + (lr_max - lr_min)*cos

def adjust_lr(optim, lr): 
    for pg in optim.param_groups: pg["lr"]=lr

def train_one_epoch(model, loader, optim, crit, device, mixup_fn=None):
    model.train(); loss_sum=0; total=0; correct=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if mixup_fn: x,y=mixup_fn(x,y)
        logits=model(x); loss=crit(logits,y)
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        loss_sum+=loss.item()*x.size(0)
        if mixup_fn is None:
            pred=logits.argmax(1); correct+=(pred==y).sum().item()
        total+=y.size(0)
    return loss_sum/len(loader.dataset), 100.0*correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); crit=nn.CrossEntropyLoss(); loss_sum=0; total=0; correct=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        logits=model(x); loss=crit(logits,y)
        loss_sum+=loss.item()*x.size(0)
        pred=logits.argmax(1); correct+=(pred==y).sum().item(); total+=y.size(0)
    return loss_sum/len(loader.dataset), 100.0*correct/total

@torch.no_grad()
def quantize_state_dict_int8(sd):
    qsd={}; scales={}
    for k,v in sd.items():
        if not torch.is_floating_point(v): continue
        w=v.detach().cpu().float(); max_abs=w.abs().max()
        scale=(max_abs/127).clamp(min=1e-8).item()
        q=torch.round(w/scale).clamp_(-128,127).to(torch.int8)
        qsd[k]=q; scales[k]=scale
    return qsd, scales

# -----------------------------
# Main
# -----------------------------
def main():
    args=build_argparser().parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_train, ds_val, num_classes = get_dataset(args.dataset, args.data_root)
    ds_train = subset_dataset(ds_train, args.subset_ratio, args.seed)

    tr_dl=DataLoader(ds_train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    va_dl=DataLoader(ds_val,batch_size=max(256,args.batch_size),shuffle=False,num_workers=args.workers,pin_memory=True)

    model=build_model(args.model,num_classes,args.pretrained).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    mixup_fn=None
    if args.use_mixup and _HAS_TIMM:
        mixup_fn=Mixup(args.mixup_alpha,args.cutmix_alpha,args.label_smoothing,num_classes=num_classes)
        crit=SoftTargetCrossEntropy()
    else:
        crit=nn.CrossEntropyLoss()

    iters_per_epoch=len(tr_dl); total_steps=iters_per_epoch*args.epochs
    warmup_steps=args.warmup_epochs*iters_per_epoch; lr_min=1e-6; global_step=0

    for ep in range(args.epochs):
        if not args.disable_cosine:
            if global_step<warmup_steps:
                lr=args.lr*(global_step+1)/max(1,warmup_steps)
            else:
                lr=cosine_decay(global_step-warmup_steps,total_steps-warmup_steps,lr_min,args.lr)
            adjust_lr(optim,lr)
        ce_train,acc_train=train_one_epoch(model,tr_dl,optim,crit,device,mixup_fn)
        global_step+=iters_per_epoch
        ce_val,acc_val=evaluate(model,va_dl,device)
        print(f"[{args.model}] epoch {ep+1}/{args.epochs} acc={acc_val:.2f}% CE={ce_val:.4f}")

    out_f=args.out_float or f"{args.model}_{args.dataset}_float.pt"
    torch.save({"model":model.state_dict(),"num_classes":num_classes,"args":vars(args)},out_f)
    print(f"Saved float weights: {out_f}")

    if args.out_int8:
        qsd,scales=quantize_state_dict_int8(model.state_dict())
        torch.save({"q_state_dict":qsd,"scales":scales,"arch":args.model,"args":vars(args)},args.out_int8)
        print(f"Saved INT8 weights: {args.out_int8}")

if __name__=="__main__":
    main()