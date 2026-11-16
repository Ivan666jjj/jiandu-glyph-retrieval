"""消融：proj_head 与 排除同简 两个因素各自的贡献。"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.expanduser("~/Desktop/02_古文字/残字模型")
BEIDA = os.path.join(BASE, "beida")

d = np.load(os.path.join(BEIDA, "beida_labeled.npz"), allow_pickle=True)
X, y, slip = d["X"], np.array([str(v) for v in d["y"]]), np.array([str(v) for v in d["slip"]])
split = json.load(open(os.path.join(BEIDA, "split.json"), encoding="utf-8"))
tr = np.array([it["idx"] for it in split["items"] if it["split"] == "train"])
te = np.array([it["idx"] for it in split["items"] if it["split"] == "test"])


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))

    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)


def T(x):
    t = torch.from_numpy(x).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5


net = Net()
sd = torch.load(os.path.join(BASE, "model.pt"), map_location="cpu")["state"]
own = net.state_dict()
own.update({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape})
net.load_state_dict(own)
net.eval()

proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 128))
proj.load_state_dict(torch.load(os.path.join(BEIDA, "proj_head.pt"), map_location="cpu"))
proj.eval()


@torch.no_grad()
def emb(idx, use_proj):
    e = F.normalize(net.embed(T(X[idx])), dim=1)
    if use_proj:
        e = F.normalize(proj(e), dim=1)
    return e.numpy()


yg, yq, sg = y[tr], y[te], slip[tr]


def run(use_proj, mask_same_slip):
    Eg, Eq = emb(tr, use_proj), emb(te, use_proj)
    hits, r50, r10 = 0, 0, 0
    for k, qi in enumerate(te):
        sim = Eg @ Eq[k]
        if mask_same_slip:
            sim = np.where(sg == slip[qi], -2.0, sim)
        order = np.argsort(-sim)
        labs = yg[order]
        pos = np.where(labs == yq[k])[0]
        r = int(pos[0]) + 1 if len(pos) else 10**6
        hits += r == 1
        r10 += r <= 10
        r50 += r <= 50
    n = len(te)
    tag = f"proj={'on ' if use_proj else 'off'} mask={'on ' if mask_same_slip else 'off'}"
    print(f"[{tag}] T1={hits/n*100:5.1f}% R10={r10/n*100:5.1f}% R50={r50/n*100:5.1f}%")
    return {"proj": use_proj, "mask": mask_same_slip,
            "top1": hits / n, "r10": r10 / n, "r50": r50 / n}


res = [run(p, m) for p in (False, True) for m in (True, False)]
json.dump(res, open(os.path.join(BASE, "xh2_ablation.json"), "w"), ensure_ascii=False, indent=2)
print("\nwrote xh2_ablation.json")
