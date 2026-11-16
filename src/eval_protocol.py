# -*- coding: utf-8 -*-
"""统一评估协议：固定口径 + 逐样本命中记录 + 显著性检验。

口径固定来源：beida/split.json（hash 1d8b7ffa1f2c62d8），排除同简。
所有对比都在同一批 250 个测试 query 上逐样本记录，供 McNemar 配对检验使用。
"""
import os, json, hashlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)
IMG = 64

# ---------- 口径 ----------
split_path = os.path.join(BEIDA, 'split.json')
proto_hash = hashlib.sha256(open(split_path, 'rb').read()).hexdigest()[:16]
d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(split_path, encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])
te = np.array([it['idx'] for it in split['items'] if it['split'] == 'test'])
classes = sorted(set(Y[tr]))
c2i = {c: i for i, c in enumerate(classes)}
tr = np.array([i for i in tr if Y[i] in c2i])

PROTOCOL = {
    'protocol_hash': proto_hash,
    'split_file': 'beida/split.json',
    'exclude_same_slip': True,
    'n_train': int(len(tr)),
    'n_test': int(len(te)),
    'n_classes': int(len(classes)),
}
print('口径', PROTOCOL)


class Net(nn.Module):
    def __init__(s, n):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                                 nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                                 nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n); s.aux = nn.Linear(256, 182)
    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)


class UNet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        def down(a, b): return nn.Sequential(nn.Conv2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.LeakyReLU(0.2, True))
        def up(a, b): return nn.Sequential(nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU(True))
        s.d1 = nn.Sequential(nn.Conv2d(1, c, 3, 1, 1), nn.ReLU(True))
        s.d2, s.d3, s.d4 = down(c, c*2), down(c*2, c*4), down(c*4, c*8)
        s.u1, s.u2, s.u3 = up(c*8, c*4), up(c*8, c*2), up(c*4, c)
        s.out = nn.Conv2d(c*2, 1, 3, 1, 1)
    def forward(s, x):
        d1 = s.d1(x); d2 = s.d2(d1); d3 = s.d3(d2); d4 = s.d4(d3)
        u1 = s.u1(d4); u2 = s.u2(torch.cat([u1, d3], 1)); u3 = s.u3(torch.cat([u2, d2], 1))
        return torch.tanh(s.out(torch.cat([u3, d1], 1)))


cls300 = open(os.path.join(HERE, 'cache', 'classes_300.txt'), encoding='utf-8').read().split('\n')
net = Net(len(cls300))
ck = torch.load(os.path.join(HERE, 'xh2_aux_parts.pt'), map_location='cpu', weights_only=False)
net.load_state_dict(ck['state'], strict=False); net.eval()
G = UNet()
G.load_state_dict(torch.load(os.path.join(HERE, 'exp4_gan_v2', 'restorer_v2.pt'),
                             map_location='cpu', weights_only=False)['G'])
G.eval()


def T(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5


@torch.no_grad()
def restore(arrs):
    out = []
    for s in range(0, len(arrs), 64):
        y = G(T(arrs[s:s+64])).numpy()
        out.append(((y * .5 + .5) * 255).clip(0, 255).astype(np.uint8)[:, 0])
    return np.concatenate(out)


@torch.no_grad()
def emb(arrs):
    out = []
    for s in range(0, len(arrs), 128):
        out.append(F.normalize(net.embed(T(arrs[s:s+128])), dim=1))
    return torch.cat(out).numpy()


rng = random.Random(42)
def damage(a, p=0.30):
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    if rng.random() < 0.5:
        bw, bh = int(IMG*p), int(IMG*p*rng.uniform(0.5, 1.5))
        x = rng.randint(0, IMG-bw); y = rng.randint(0, IMG-min(bh, IMG))
        ImageDraw.Draw(im).rectangle([x, y, x+bw, y+min(bh, IMG)], fill=255)
    else:
        bw = int(IMG*p); x = rng.randint(0, IMG-bw); y = rng.randint(0, max(1, IMG-bw))
        ImageDraw.Draw(im).ellipse([x, y, x+bw, y+bw], fill=255)
    return np.array(im)


Eg = emb([X[i] for i in tr])


def rank(arr, topk=20):
    """返回该 query 的候选顺序（已排除同简），以及每个 k 是否命中。"""
    v = emb([arr])[0]
    sim = Eg @ v
    order = np.argsort(-sim)
    labs = []
    for o in order:
        i = int(tr[o])
        labs.append((Y[i], float(sim[o]), slips[i]))
    return labs


def evaluate(queries, tag, dual=None):
    """queries: list of (test_idx, image_array, name)。返回逐样本命中 + 汇总。"""
    per = {}
    for t, img, name in queries:
        gold = Y[t]
        if dual is not None:
            if name == 'broken':
                a, b = emb([img])[0], dual[(t, 'fixed')]
            elif name == 'fixed':
                a, b = emb([img])[0], dual[(t, 'broken')]
            else:
                a, b = emb([img])[0], dual[(t, 'fixed')]
            v = 0.3 * a + 0.7 * b   # 与 exp5 一致：w=0.7 给补全图
            v = v / (np.linalg.norm(v) + 1e-9)
            sim = Eg @ v
            order = np.argsort(-sim)
            labs = [(Y[int(tr[o])], float(sim[o]), slips[int(tr[o])]) for o in order]
        else:
            labs = rank(img)
        # 排除同简
        seen, ranked = set(), []
        for lab, s, sp in labs:
            if sp == slips[t]:
                continue
            ranked.append(lab)
            if len(ranked) >= 20:
                break
        rec = {f'hit@{k}': bool(gold in ranked[:k]) for k in (1, 5, 10, 20)}
        rec['gold'] = gold
        per[str(int(t))] = rec
    n = len(per)
    agg = {f'R{k}': round(sum(v[f'hit@{k}'] for v in per.values()) / n * 100, 1) for k in (1, 5, 10, 20)}
    res = dict(PROTOCOL); res.update({'exp_id': tag, 'n_eval': n})
    res.update(agg)
    res['per_sample'] = per
    json.dump(res, open(os.path.join(OUTDIR, f'{tag}.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'{tag:22} R1 {agg["R1"]}%  R5 {agg["R5"]}%  R20 {agg["R20"]}%  (n={n})', flush=True)
    return res


# ---------- 跑四种设定 ----------
rng = random.Random(42)
broken = {int(t): damage(X[t]) for t in te}
fixed = {int(t): restore([broken[int(t)]])[0] for t in te}

evaluate([(int(t), X[t], 'clean') for t in te], 'main_clean')
evaluate([(int(t), broken[int(t)], 'broken') for t in te], 'main_broken')
evaluate([(int(t), fixed[int(t)], 'fixed') for t in te], 'main_fixed')

# 双通道
dual_cache = {(int(t), 'broken'): emb([broken[int(t)]])[0] for t in te}
dual_cache.update({(int(t), 'fixed'): emb([fixed[int(t)]])[0] for t in te})
evaluate([(int(t), broken[int(t)], 'broken') for t in te], 'main_dual', dual=dual_cache)

print('\n已写', OUTDIR)
