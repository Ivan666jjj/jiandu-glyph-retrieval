# -*- coding: utf-8 -*-
"""残损协议核对：按边长砍 vs 按墨迹比例砍，差多少？

背景：现有残缺测试用的是「砍掉 x% 的边宽」，但论文里说的「30% 残缺」
通常指「去掉 30% 的墨迹像素」。两者在字形偏一侧时差很多。

本脚本量化两种定义的实际去墨比例，判断两条曲线能否同图比较。
"""
import numpy as np
import collections, json, os

HERE = '/root/autodl-tmp/补全器训练包'
IMG = 64

d = np.load(os.path.join(HERE, 'data', 'chars_han.npz'), allow_pickle=True)
X = np.array(d['X']); Y = np.array([str(v) for v in d['y']])
SP = np.array([str(v) for v in d['split']])
te = np.array([i for i in range(len(X)) if SP[i] == 'test'])

# 墨迹 = 亮像素（黑底白字）
def ink_mask(img):
    return img > 127


def cut_by_edge(img, mode, ratio):
    """现有协议：砍掉 x% 的边宽（不管那里有没有墨迹）。"""
    im = img.copy()
    w = max(1, int(IMG * ratio))
    if mode == 'left':
        im[:, :w] = 0
    elif mode == 'right':
        im[:, IMG - w:] = 0
    elif mode == 'top':
        im[:w, :] = 0
    else:
        im[IMG - w:, :] = 0
    return im


def cut_by_ink(img, ratio, rng, mode='edge'):
    """按墨迹像素数精确去除 ratio（优先从边缘，模拟简牍真实残损）。"""
    m = ink_mask(img)
    total = m.sum()
    if total == 0:
        return img.copy()
    k = int(ratio * total)
    out = m.copy()
    H, W = IMG, IMG
    # 边缘权重：上下端与左右侧权重高
    wt = np.ones((H, W), np.float32)
    wt[:H // 4] = 3.0
    wt[3 * H // 4:] = 3.0
    wt[:, :W // 8] += 2.0
    wt[:, 7 * W // 8:] += 2.0
    removed = 0
    try:
        while removed < k:
            idx = np.flatnonzero(out.ravel())
            if len(idx) == 0:
                break
            w = wt.ravel()[idx]
            pick = rng.choice(idx, p=w / w.sum())
            y, x = divmod(int(pick), W)
            r = int(rng.integers(1, 4))
            yy, xx = np.ogrid[:H, :W]
            blob = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
            gain = np.count_nonzero(out & blob)
            if gain == 0:
                out.ravel()[pick] = False
                removed += 1
            else:
                out &= ~blob
                removed += gain
    except Exception:
        pass
    res = img.copy()
    res[~out] = 0
    return res


print(f'测试样本 {len(te)}', flush=True)
rng = np.random.default_rng(0)

print('\n=== 协议 A：按边长砍（现有）===', flush=True)
print(f'{"名义":<8}{"实际去墨比例":<16}{"标准差":<10}{"最小":<8}{"最大":<8}', flush=True)
resA = {}
sample = list(te[:2000])
for mode in ['left', 'right', 'top', 'bottom']:
    for ratio in [0.3, 0.5, 0.7]:
        vals = []
        for j in sample:
            m0 = ink_mask(X[j])
            if m0.sum() == 0:
                continue
            m1 = ink_mask(cut_by_edge(X[j], mode, ratio))
            vals.append(1 - m1.sum() / m0.sum())
        v = np.array(vals)
        key = f'{mode}_{ratio}'
        resA[key] = {'mean': float(v.mean()), 'std': float(v.std()),
                     'min': float(v.min()), 'max': float(v.max())}
        print(f'{mode+" "+str(ratio):<12}{v.mean():<16.1%}{v.std():<10.3f}{v.min():<8.1%}{v.max():<8.1%}', flush=True)

print('\n=== 协议 B：按墨迹比例精确去除 ===', flush=True)
resB = {}
for ratio in [0.3, 0.5, 0.7]:
    vals = []
    for j in sample[:600]:
        m0 = ink_mask(X[j])
        if m0.sum() == 0:
            continue
        m1 = ink_mask(cut_by_ink(X[j], ratio, rng))
        vals.append(1 - m1.sum() / m0.sum())
    v = np.array(vals)
    resB[str(ratio)] = {'mean': float(v.mean()), 'std': float(v.std())}
    print(f'名义 {ratio:.0%}  实际去墨 {v.mean():.1%}  标准差 {v.std():.3f}', flush=True)

print('\n' + '=' * 60, flush=True)
print('诊断：', flush=True)
avg = np.mean([resA[f'{m}_0.3']['mean'] for m in ['left', 'right', 'top', 'bottom']])
print(f'  协议 A 名义 30% 的实际平均去墨：{avg:.1%}', flush=True)
if abs(avg - 0.3) > 0.08:
    print(f'  → 偏差 {abs(avg-0.3)*100:.1f} 点，**协议 A 的"30%"与实际去墨差异显著**', flush=True)
    print('  → 现有残缺表格需重新标注或重跑', flush=True)
else:
    print('  → 偏差在可接受范围，两种协议可视为大致等价', flush=True)

json.dump({'A_edge': resA, 'B_ink': resB},
          open(os.path.join(HERE, 'results', 'damage_protocol.json'), 'w'),
          ensure_ascii=False, indent=1)
print('DONE', flush=True)
