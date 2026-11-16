"""在同一份 split.json 上评估现有权重（不训练），作为基线。"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BEIDA = os.path.join(HERE, "beida")
DEV = "cpu"

d = np.load(os.path.join(BEIDA, "beida_labeled.npz"), allow_pickle=True)
X, y, slip = d["X"], d["y"], d["slip"]
labels = np.array([str(v) for v in y])
slips = np.array([str(v) for v in slip])

split = json.load(open(os.path.join(BEIDA, "split.json"), encoding="utf-8"))
tr = np.array([it["idx"] for it in split["items"] if it["split"] == "train"])
te = np.array([it["idx"] for it in split["items"] if it["split"] == "test"])
print(f"训练 {len(tr)} | 测试 {len(te)} | hash {split['hash']}")


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


def load_and_eval(path, tag):
    net = Net().to(DEV)
    d2 = torch.load(path, map_location="cpu")
    sd = d2["state"]
    own = net.state_dict()
    loaded = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    own.update(loaded)
    net.load_state_dict(own)
    net.eval()
    print(f"\n[{tag}] 迁移 {len(loaded)} 个张量")

    @torch.no_grad()
    def emb(idx, bs=256):
        out = []
        for s in range(0, len(idx), bs):
            out.append(F.normalize(net.embed(T(X[idx[s:s + bs]])), dim=1).cpu())
        return torch.cat(out).numpy()

    Eg, Eq = emb(tr), emb(te)
    h1 = h5 = 0
    rec = {10: 0, 20: 0, 50: 0}
    n = 0
    per_cls = {}
    hits = []
    for k, qi in enumerate(te):
        sim = Eg @ Eq[k]
        order = [j for j in np.argsort(-sim) if slips[tr[j]] != slips[qi]]
        if not order: continue
        n += 1
        labs = labels[tr[order]]
        gold = labels[qi]
        ok = labs[0] == gold
        if ok: h1 += 1
        hits.append(ok)
        if gold in labs[:5]: h5 += 1
        for kk in rec:
            if gold in labs[:kk]: rec[kk] += 1
        c = per_cls.setdefault(gold, [0, 0])
        c[1] += 1
        if ok: c[0] += 1
    f1s = [2 * (c[0] / c[1]) / (1 + c[0] / c[1]) for c in per_cls.values() if c[1] > 0]

    rng = np.random.default_rng(0)
    hits = np.array(hits)
    bs = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(1000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])

    print(f"  Top-1 {h1/n*100:.1f}%  Top-5 {h5/n*100:.1f}%  macro-F1 {np.mean(f1s)*100:.1f}%  (n={n})")
    print(f"  Recall@10 {rec[10]/n*100:.1f}%  @20 {rec[20]/n*100:.1f}%  @50 {rec[50]/n*100:.1f}%")
    print(f"  95% CI: [{lo*100:.1f}, {hi*100:.1f}]")
    return {"top1": round(h1/n*100, 1), "top5": round(h5/n*100, 1),
            "macro_f1": round(float(np.mean(f1s))*100, 1),
            "recall": {k: round(v/n*100, 1) for k, v in rec.items()},
            "ci95": [round(lo*100, 1), round(hi*100, 1)], "n": n}


res = {}
res["baseline_300"] = load_and_eval(os.path.join(HERE, "model.pt"), "基线（300类权重，未在新数据上训练）")

v2 = os.path.join(HERE, "xiaohuan1_v2.pt")
if os.path.exists(v2):
    res["v2_parts"] = load_and_eval(v2, "v2 部件多任务")

json.dump(res, open(os.path.join(HERE, "baseline_eval.json"), "w"), ensure_ascii=False, indent=2)
print("\n已写 baseline_eval.json")
