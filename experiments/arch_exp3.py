# -*- coding: utf-8 -*-
"""表 6 补齐：ConvNeXt-Atto / Transformer / InfoNCE 三种结构，统一口径

修复了 arch_exp.py 的 bug：
  原 evaluate() 直接把单通道图喂给 ConvNeXt（需要 3 通道），
  训练时做了 repeat(1,3,1,1)，评估时漏了 → RuntimeError。

数据划分、残损协议、排除同简、评估函数与 arch_exp.py 完全一致。

用法：
  python arch_exp3.py convnext
  python arch_exp3.py transformer
  python arch_exp3.py infonce
"""
import os
import sys
import json
import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG = 64
KS = (1, 5, 10, 15, 20)
MODE = sys.argv[1] if len(sys.argv) > 1 else 'transformer'
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

d = np.load(os.path.join(HERE, 'data', 'chars_han.npz'), allow_pickle=True)
X = np.array(d['X'])
Y = np.array([str(v) for v in d['y']])
SP = np.array([str(v) for v in d['split']])
XML = np.array([str(v) for v in d['xml']])
tr = np.where(SP == 'train')[0]
te = np.where(SP == 'test')[0]
cnt = collections.Counter(Y[tr])
keep = {c for c, n in cnt.items() if n >= 2}
tr = np.array([i for i in tr if Y[i] in keep])
te = np.array([i for i in te if Y[i] in keep])
classes = sorted(keep)
c2i = {c: i for i, c in enumerate(classes)}
print(f'{MODE}: 训练 {len(tr):,} / {len(classes)} 类', flush=True)


def T(a, ch3=False):
    t = torch.from_numpy(np.stack(a)).float().div_(255.).unsqueeze(1)
    t = (t - .5) / .5
    if ch3:
        t = t.repeat(1, 3, 1, 1)
    return t.to(DEV)


def tcut(a, mode, ratio):
    im = np.array(a).copy()
    if mode in ('left', 'right'):
        w = max(1, int(IMG * ratio))
        if mode == 'left':
            im[:, :w] = 255
        else:
            im[:, IMG - w:] = 255
    else:
        h = max(1, int(IMG * ratio))
        if mode == 'top':
            im[:h, :] = 255
        else:
            im[IMG - h:, :] = 255
    return im


class PatchViT(nn.Module):
    def __init__(s, n, d_model=192, depth=4, heads=4, patch=8):
        super().__init__()
        npatch = (IMG // patch) ** 2
        s.proj = nn.Conv2d(1, d_model, patch, patch)
        s.pos = nn.Parameter(torch.zeros(1, npatch + 1, d_model))
        s.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, 0.1,
                                           batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, depth)
        s.norm = nn.LayerNorm(d_model)
        s.head = nn.Linear(d_model, n)

    def embed(s, x):
        h = s.proj(x).flatten(2).transpose(1, 2)
        h = torch.cat([s.cls.expand(h.size(0), -1, -1), h], 1) + s.pos
        h = s.norm(s.enc(h))
        return h[:, 0]


def make_cnn(n, chs=(32, 64, 128, 256)):
    def blk(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
            nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True), nn.MaxPool2d(2))
    seq = []
    prev = 1
    for c in chs:
        seq.append(blk(prev, c))
        prev = c
    net = nn.Module()
    net.f = nn.Sequential(*seq)
    net.head = nn.Linear(prev, n)
    net.embed = lambda x: F.adaptive_avg_pool2d(net.f(x), 1).flatten(1)
    return net


CH3 = False
if MODE == 'convnext':
    import timm
    class CNext(nn.Module):
        def __init__(s, n):
            super().__init__()
            s.bb = timm.create_model('convnext_atto', pretrained=True, features_only=True,
                                     out_indices=(1, 2, 3), in_chans=3)
            s.head = nn.Linear(320, n)
        def embed(s, x):
            return F.adaptive_avg_pool2d(s.bb(x)[-1], 1).flatten(1)
    net = CNext(len(classes)).to(DEV)
    CH3 = True
elif MODE == 'transformer':
    net = PatchViT(len(classes)).to(DEV)
elif MODE == 'infonce':
    net = make_cnn(len(classes)).to(DEV)
elif MODE == 'cnn':
    net = make_cnn(len(classes)).to(DEV)
else:
    raise SystemExit('mode 只能是 convnext / transformer / infonce / cnn')

npar = sum(p.numel() for p in net.parameters())
print(f'{MODE} 参数 {npar/1e6:.2f}M  3通道={CH3}', flush=True)


def train_ce(net, epochs=8, per=16000):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(777)
    torch.manual_seed(777)
    for ep in range(1, epochs + 1):
        net.train()
        idx = list(tr)
        if len(idx) > per:
            idx = rng.sample(idx, per)
        rng.shuffle(idx)
        tot = nb = 0
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            x = T([X[j] for j in b], CH3)
            y = torch.tensor([c2i[Y[j]] for j in b], device=DEV)
            loss = F.cross_entropy(net.head(net.embed(x)), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f'  ep{ep}/{epochs} loss {tot/nb:.3f}', flush=True)


def train_infonce(net, epochs=8, per=16000, tau=0.07):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(777)
    torch.manual_seed(777)
    for ep in range(1, epochs + 1):
        net.train()
        idx = list(tr)
        if len(idx) > per:
            idx = rng.sample(idx, per)
        rng.shuffle(idx)
        tot = nb = 0
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            x = T([X[j] for j in b], CH3)
            z = F.normalize(net.embed(x), dim=1)
            sim = z @ z.T / tau
            lab = torch.tensor([c2i[Y[j]] for j in b], device=DEV)
            pos = (lab[:, None] == lab[None, :]).float()
            pos.fill_diagonal_(0)
            logp = F.log_softmax(sim, dim=1)
            denom = pos.sum(1).clamp(min=1)
            loss = (-(logp * pos).sum(1) / denom).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f'  ep{ep}/{epochs} infonce_loss {tot/nb:.4f}', flush=True)


@torch.no_grad()
def emb(net, arrs, bs=512):
    o = []
    for i in range(0, len(arrs), bs):
        o.append(F.normalize(net.embed(T(arrs[i:i + bs], CH3)), dim=1))
    return torch.cat(o).cpu().numpy()


def evaluate(net, tag):
    EG = emb(net, [X[i] for i in tr])
    gY = [Y[i] for i in tr]
    gS = [XML[i] for i in tr]
    sub = list(te)[:1000]
    qS = [XML[i] for i in sub]
    res = {}
    for ratio, t in [(0.0, 'clean'), (0.3, 'cut30'), (0.5, 'cut50'), (0.7, 'cut70')]:
        rr = random.Random(99)
        q = [X[i] if ratio == 0 else tcut(X[i], rr.choice(['left', 'right', 'top', 'bottom']), ratio)
             for i in sub]
        Eq = emb(net, q)
        hits = {k: 0 for k in KS}
        for i in range(len(sub)):
            sim = Eq[i] @ EG.T
            order = np.argsort(-sim)
            labs = []
            for o in order:
                if gS[o] == qS[i]:
                    continue
                labs.append(gY[o])
                if len(labs) >= 20:
                    break
            g = Y[sub[i]]
            for k in KS:
                if g in labs[:k]:
                    hits[k] += 1
        n = len(sub)
        r = {f'R{k}': round(v / n * 100, 1) for k, v in hits.items()}
        res[t] = r
        print(f'  {tag} {t:8} R1 {r["R1"]:>5}% R20 {r["R20"]:>5}%', flush=True)
    return res


if MODE == 'infonce':
    train_infonce(net, EPOCHS)
else:
    train_ce(net, EPOCHS)
net.eval()
out = {MODE: {'params_m': round(npar / 1e6, 2), 'epochs': EPOCHS, **evaluate(net, MODE)}}

OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, f'arch3_{MODE}.json'), 'w'), ensure_ascii=False, indent=1)
torch.save({'state': net.state_dict(), 'classes': classes},
           os.path.join(OUT, f'arch3_{MODE}.pt'))
print('DONE', MODE, flush=True)
