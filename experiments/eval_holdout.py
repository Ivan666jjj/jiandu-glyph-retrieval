"""用与 ResNet 完全相同的划分，重测小 CNN —— 检验之前的 31.9% 是否虚高。"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BEIDA = os.path.join(HERE, "beida")

d = np.load(os.path.join(BEIDA, "beida_labeled.npz"), allow_pickle=True)
X, y, slip = d["X"], d["y"], d["slip"]
labels = np.array(y); slips = np.array(slip)

# 与 train_gpu2.py 完全一致的划分
random.seed(0)
by_cls = {}
for i, l in enumerate(labels):
    by_cls.setdefault(l, []).append(i)
test_idx = set()
for l, idxs in by_cls.items():
    s_ = list({slips[i] for i in idxs})
    random.shuffle(s_)
    k = max(1, int(len(s_) * 0.2)) if len(s_) >= 2 else 0
    keep = set(s_[:k])
    for i in idxs:
        if slips[i] in keep: test_idx.add(i)
test_idx = np.array(sorted(test_idx))
train_idx = np.array([i for i in range(len(X)) if i not in test_idx])
print(f"训练 {len(train_idx)} | 测试 {len(test_idx)}")


class Net(nn.Module):
    def __init__(s, n):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n)
    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)


def T(x):
    t = torch.from_numpy(x).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5


classes = open(os.path.join(HERE, "cache", "classes_300.txt"), encoding="utf-8").read().split("\n")
net = Net(len(classes))
net.load_state_dict(torch.load(os.path.join(HERE, "model.pt"), map_location="cpu")["state"])
net.eval()

proj = None
pp = os.path.join(BEIDA, "proj_head.pt")
if os.path.exists(pp):
    proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 128))
    proj.load_state_dict(torch.load(pp, map_location="cpu"))
    proj.eval()
    print("已加载投影头")


@torch.no_grad()
def embed_all(bs=256):
    outs = []
    Xt = T(X)
    for s in range(0, len(Xt), bs):
        e = net.embed(Xt[s:s + bs])
        if proj is not None:
            e = proj(F.normalize(e, dim=1))
        outs.append(F.normalize(e, dim=1))
    return torch.cat(outs)


def evaluate(E):
    E = E.numpy()
    h1 = h5 = tot = 0
    for i in test_idx:
        sim = E[train_idx] @ E[i]
        order = np.argsort(-sim)
        gi = train_idx[order]
        order = [j for j in gi if slips[j] != slips[i]]
        if not order: continue
        tot += 1
        if labels[order[0]] == labels[i]: h1 += 1
        if labels[i] in labels[order[:5]]: h5 += 1
    return h1 / max(tot, 1) * 100, h5 / max(tot, 1) * 100, tot


E = embed_all()
a1, a5, n = evaluate(E)
print(f"【小 CNN · 正确划分】Top-1 {a1:.1f}%  Top-5 {a5:.1f}%  (测试 n={n})")

# 对照：同批数据不分划分（复现旧口径）
h1 = h5 = tot = 0
En = E.numpy()
for i in range(len(En)):
    sim = En @ En[i]; sim[i] = -9
    order = [j for j in np.argsort(-sim) if slips[j] != slips[i]]
    if not order: continue
    tot += 1
    if labels[order[0]] == labels[i]: h1 += 1
    if labels[i] in labels[order[:5]]: h5 += 1
print(f"【小 CNN · 旧口径(全量互查)】Top-1 {h1/tot*100:.1f}%  Top-5 {h5/tot*100:.1f}%")

json.dump({"split": {"train": int(len(train_idx)), "test": int(len(test_idx))},
           "cnn_holdout": {"top1": round(a1,1), "top5": round(a5,1)},
           "cnn_old_metric": {"top1": round(h1/tot*100,1), "top5": round(h5/tot*100,1)}},
          open(os.path.join(BEIDA, "cnn_holdout.json"), "w"), ensure_ascii=False, indent=2)
print("已写 cnn_holdout.json")
