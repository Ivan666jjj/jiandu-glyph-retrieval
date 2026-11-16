"""固化数据划分：按简留出，结果写 split.json，带 hash。

从今天起，所有评估一律用这份划分，不再每次随机。
"""
import os, json, random, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BEIDA = os.path.join(HERE, "beida")
SEED = 0
TEST_RATIO = 0.2

d = np.load(os.path.join(BEIDA, "beida_labeled.npz"), allow_pickle=True)
X, y, slip = d["X"], d["y"], d["slip"]
labels = [str(v) for v in y]
slips = [str(v) for v in slip]

# 按类别分组，每类按简随机抽 20% 的简作为测试
by_cls = {}
for i, l in enumerate(labels):
    by_cls.setdefault(l, []).append(i)

rng = random.Random(SEED)

# 全局按「简」抽 20% 作测试（不是每类抽，否则小类会被抽空）
all_slips = sorted({s for s in slips})
rng.shuffle(all_slips)
n_test_slips = max(1, int(len(all_slips) * TEST_RATIO))
test_slips = set(all_slips[:n_test_slips])

# 兜底：若某类全部落入测试集，把这类的简移回训练，保证每类可学
changed = True
while changed:
    changed = False
    for l, idxs in by_cls.items():
        cls_slips = {slips[i] for i in idxs}
        if cls_slips and cls_slips <= test_slips:
            test_slips -= cls_slips
            changed = True

items = []
for i in range(len(X)):
    items.append({
        "idx": i,
        "label": labels[i],
        "slip": slips[i],
        "split": "test" if slips[i] in test_slips else "train",
    })

# 页面信息可能不存在，能取就取
pages = None
try:
    pages = list(d["page"])
except Exception:
    pass
if pages is not None:
    for i, it in enumerate(items):
        it["page"] = int(pages[i])

# 计算可复现 hash（只看 idx/label/slip/split 的确定性序列）
payload = "|".join(f"{it['idx']}:{it['label']}:{it['slip']}:{it['split']}" for it in items)
h = hashlib.sha256(payload.encode()).hexdigest()[:16]

n_train = sum(1 for it in items if it["split"] == "train")
n_test = len(items) - n_train
n_cls = len(by_cls)
test_cls = len({it["label"] for it in items if it["split"] == "test"})
only_train = n_cls - test_cls

# 每类样本数分布
per_cls = {l: len(v) for l, v in by_cls.items()}
counts = sorted(per_cls.values())

out = {
    "seed": SEED,
    "rule": "按简留出，每类随机抽 20% 的简作测试（至少 1 枚）",
    "hash": h,
    "n_total": len(items),
    "n_train": n_train,
    "n_test": n_test,
    "n_classes": n_cls,
    "n_classes_in_test": test_cls,
    "n_classes_train_only": only_train,
    "per_class": {
        "min": counts[0],
        "median": counts[len(counts) // 2],
        "max": counts[-1],
        "lt3": sum(1 for c in counts if c < 3),
    },
    "items": items,
}

path = os.path.join(BEIDA, "split.json")
json.dump(out, open(path, "w"), ensure_ascii=False)

print(f"划分已固化  hash={h}")
print(f"  总样本 {len(items)} | 训练 {n_train} | 测试 {n_test}")
print(f"  类别 {n_cls} | 测试中出现 {test_cls} | 只在训练集 {only_train}")
print(f"  每类样本：最少 {counts[0]}，中位 {counts[len(counts)//2]}，最多 {counts[-1]}，"
      f"少于3个的类 {out['per_class']['lt3']} 个")
print(f"  写入 {path}")
