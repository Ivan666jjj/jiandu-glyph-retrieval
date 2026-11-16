"""训练字符 embedding 模型，并评估完整字与残字的检索能力。

输出：残字模型/model.pt, report.json
"""
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SIZE = 64
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

N_CLASS = 300
tr = np.load(os.path.join(CACHE, f"train_{N_CLASS}.npz"))
va = np.load(os.path.join(CACHE, f"val_{N_CLASS}.npz"))
Xtr, ytr = tr["X"], tr["y"]
Xva, yva = va["X"], va["y"]
classes = open(os.path.join(CACHE, f"classes_{N_CLASS}.txt"), encoding="utf-8").read().split("\n")
print(f"train {Xtr.shape}  val {Xva.shape}  classes {len(classes)}  device {DEV}")


def to_tensor(x):
    t = torch.from_numpy(x).float().div_(255.0).unsqueeze(1)
    return (t - 0.5) / 0.5


# ---------------- 残损增强：训练时就让它见残字 ----------------
def augment(batch, p_cut=0.5, p_hole=0.5):
    """batch: (N,1,64,64) tensor，就地返回增强后的副本。"""
    out = batch.clone()
    n, _, h, w = out.shape
    for i in range(n):
        if random.random() < p_cut:
            frac = random.uniform(0.2, 0.55)
            if random.random() < 0.5:          # 上下截
                hh = int(h * frac)
                if random.random() < 0.5:
                    out[i, :, :hh, :] = 0
                else:
                    out[i, :, h - hh:, :] = 0
            else:                               # 左右截
                ww = int(w * frac)
                if random.random() < 0.5:
                    out[i, :, :, :ww] = 0
                else:
                    out[i, :, :, w - ww:] = 0
        if random.random() < p_hole:
            for _ in range(random.randint(1, 3)):
                hh = random.randint(h // 10, h // 4)
                ww = random.randint(w // 10, w // 4)
                y0 = random.randint(0, h - hh)
                x0 = random.randint(0, w - ww)
                out[i, :, y0:y0 + hh, x0:x0 + ww] = 0
    return out


class Net(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.head = nn.Linear(256, n_class)

    def embed(self, x):
        x = self.f(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


model = Net(len(classes)).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
EPOCHS = int(os.environ.get("EPOCHS", "12"))
BATCH = 128
steps = (len(Xtr) + BATCH - 1) // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * steps)

Xt_tr = to_tensor(Xtr); yt_tr = torch.from_numpy(ytr).long()
Xt_va = to_tensor(Xva); yt_va = torch.from_numpy(yva).long()

t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xt_tr))
    tot = cor = 0
    for s in range(steps):
        idx = perm[s * BATCH:(s + 1) * BATCH]
        if len(idx) == 0:
            continue
        xb = augment(Xt_tr[idx]).to(DEV)
        yb = yt_tr[idx].to(DEV)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        cor += (logits.argmax(1) == yb).sum().item(); tot += len(idx)
    # 验证：干净字
    model.eval()
    with torch.no_grad():
        vc = 0
        for s in range(0, len(Xt_va), 512):
            xb = Xt_va[s:s + 512].to(DEV); yb = yt_va[s:s + 512].to(DEV)
            vc += (model(xb).argmax(1) == yb).sum().item()
    print(f"ep{ep+1:02d}  train_acc {cor/tot:.3f}  val_acc {vc/len(Xt_va):.3f}  "
          f"{time.time()-t0:.0f}s", flush=True)

torch.save({"state": model.state_dict(), "n_class": len(classes)}, os.path.join(HERE, "model.pt"))
print("模型已存")
