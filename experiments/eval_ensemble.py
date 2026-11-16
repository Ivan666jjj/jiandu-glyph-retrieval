# -*- coding: utf-8 -*-
"""集成评估：5 个 wide 种子，embedding 归一化后平均。"""
import os, json, random, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE='/Users/Admin/Desktop/02_古文字/残字模型'
D1=os.path.join(HERE,'dahan1'); ENS=os.path.join(HERE,'dahan4_ens')
OUT=os.path.join(HERE,'dahan4_ens'); IMG=64
DEV=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('设备:',DEV,flush=True)

d=np.load(os.path.join(D1,'chars.npz'),allow_pickle=True)
X,Y,SP=d['X'],np.array([str(v) for v in d['y']]),np.array([str(v) for v in d['split']])

def make_net(n):
    def blk(a,b): return nn.Sequential(nn.Conv2d(a,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU(True),
        nn.Conv2d(b,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU(True),nn.MaxPool2d(2))
    net=nn.Module()
    net.f=nn.Sequential(blk(1,48),blk(48,96),blk(96,192),blk(192,384))
    net.head=nn.Linear(384,n)
    net.embed=lambda x: F.adaptive_avg_pool2d(net.f(x),1).flatten(1)
    return net

nets=[]
classes=None
for sd in [1,2,3,4,5]:
    ck=torch.load(os.path.join(ENS,f'w{sd}.pt'),map_location='cpu',weights_only=False)
    classes=ck['classes']
    m=make_net(len(classes)); m.load_state_dict(ck['state']); nets.append(m.to(DEV).eval())
print(f'{len(nets)} 个模型已加载',flush=True)

def T(a):
    t=torch.from_numpy(np.stack(a)).float().div_(255.).unsqueeze(1)
    return ((t-.5)/.5).to(DEV)

@torch.no_grad()
def emb_single(m,arrs,bs=256):
    o=[]
    for i in range(0,len(arrs),bs): o.append(F.normalize(m.embed(T(arrs[i:i+bs])),dim=1))
    return torch.cat(o)

@torch.no_grad()
def emb_ens(arrs,bs=256):
    """归一化 → 平均 → 再归一化"""
    outs=[]
    for i in range(0,len(arrs),bs):
        x=T(arrs[i:i+bs])
        es=[F.normalize(m.embed(x),dim=1) for m in nets]
        avg=torch.stack(es,0).mean(0)
        outs.append(F.normalize(avg,dim=1))
    return torch.cat(outs).cpu().numpy()

def cut(a,mode,ratio):
    im=np.array(a).copy()
    if mode in ('left','right'):
        w=max(1,int(IMG*ratio))
        if mode=='left': im[:,:w]=255
        else: im[:,IMG-w:]=255
    elif mode in ('top','bottom'):
        h=max(1,int(IMG*ratio))
        if mode=='top': im[:h,:]=255
        else: im[IMG-h:,:]=255
    else:
        c=int(IMG*(1-ratio))
        for y in range(IMG):
            if y<=c: im[y,:]=255
            else:
                b=int((y-c)/max(1,IMG-c)*IMG*0.7)
                if b<IMG:
                    if mode=='diag_h': im[y,:b]=255
                    else: im[y,IMG-b:]=255
    return im

cnt=collections.Counter(Y[np.where(SP=='train')[0]])
keep={c for c,n in cnt.items() if n>=2}
tr=[int(i) for i in np.where(SP=='train')[0] if Y[i] in keep]
te=[int(i) for i in np.where(SP=='test')[0] if Y[i] in keep][:1000]
print(f'gallery {len(tr)} 测试 {len(te)}',flush=True)

print('编码 gallery（集成）...',flush=True)
GE=emb_ens([X[i] for i in tr])
gY=[Y[i] for i in tr]; gS=[SP[i] for i in tr]
qS=[SP[i] for i in te]
print('gallery 就绪',flush=True)

# 单模型（种子1）作对照
print('编码单模型 gallery...',flush=True)
G1=emb_single(nets[0],[X[i] for i in tr]).cpu().numpy()

res={}
for ratio,tag in [(0.0,'clean'),(0.3,'cut30'),(0.5,'cut50'),(0.7,'cut70'),(0.8,'cut80')]:
    rr=random.Random(99)
    q=[X[i] if ratio==0 else cut(X[i],rr.choice(['left','right','top','bottom','diag_h','diag_v']),ratio) for i in te]
    for name,GE_,func in [('single',G1,emb_single),('ens',GE,emb_ens)]:
        if name=='single':
            Eq=func(nets[0],q).cpu().numpy()
        else:
            Eq=func(q)
        hits={k:0 for k in (1,5,10,20)}
        for i in range(len(te)):
            sim=Eq[i]@GE_.T; order=np.argsort(-sim); labs=[]
            for o in order:
                if gS[o]==qS[i]: continue
                labs.append(gY[o])
                if len(labs)>=20: break
            g=Y[te[i]]
            for k in hits:
                if g in labs[:k]: hits[k]+=1
        n=len(te); r={f'R{k}':round(v/n*100,1) for k,v in hits.items()}
        res[f'{tag}_{name}']=r
    s=res[f'{tag}_single']; e=res[f'{tag}_ens']
    print(f'{tag:8} 单模型 {s["R1"]:>5}%  →  集成 {e["R1"]:>5}%  ({e["R1"]-s["R1"]:+.1f})   R20 {s["R20"]:>5}% → {e["R20"]:>5}%',flush=True)

json.dump(res,open(os.path.join(OUT,'eval_ens.json'),'w'),ensure_ascii=False,indent=1)
print('已写',OUT)
