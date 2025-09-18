#!/usr/bin/env python3
import argparse, random
from typing import Dict, Tuple
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

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

def train(epochs, device, subset_ratio, batch_size, lr=0.1, momentum=0.9, wd=5e-4):
    model=VGG_CIFAR().to(device)
    train_loader,test_loader=make_loaders(batch_size=batch_size, subset_ratio=subset_ratio)
    opt=torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.5*epochs), int(0.75*epochs)], gamma=0.1)
    for ep in range(1,epochs+1):
        model.train()
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x),y).backward(); opt.step()
        sched.step()
        print(f"[Victim] epoch {ep}/{epochs}  acc={eval_acc(model,test_loader,device)*100:.2f}%")
    return model, test_loader

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--subset_ratio",type=float,default=1.0)
    ap.add_argument("--out_float",default="victim_vgg16_cifar10_float.pt")
    ap.add_argument("--out_int8",default="victim_vgg16_cifar10_int8.pt")
    args=ap.parse_args()

    set_seed(7)
    device=args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")
    model,_=train(args.epochs, device, args.subset_ratio, args.batch_size)
    acc=eval_acc(model, make_loaders(batch_size=args.batch_size, subset_ratio=args.subset_ratio)[1], device)
    print(f"Final test accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), args.out_float); print(f"[float] wrote: {args.out_float}")
    save_int8_state_from_state_dict(model.state_dict(), args.out_int8)

if __name__=="__main__": main()