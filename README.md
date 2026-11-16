# 面向简牍残字的候选检索 · Candidate Retrieval for Damaged Jiandu Characters

本仓库是论文《面向简牍残字的候选检索：实现选择、评估协议与语言先验权重》（及其中文参赛稿《面向简牍残字的智能候选检索系统设计与实现》）的代码与复现材料。

系统的任务不是自动释读，而是**初筛**：给定一张 64×64 的残缺字形图，输出 20 个候选类别，由人工从候选中挑选。

## 仓库内容

```
src/            核心流程
  make_split.py     按简切分数据，生成训练/验证/测试划分
  train.py          字形编码器训练（4 层 CNN + 交叉熵 + 标签平滑）
  eval_protocol.py  检索评估（Rank k、按简排除同简、残损档位）

experiments/    论文各表的实验脚本
  eval_baseline.py        表 3   三个基线（像素余弦 / HOG / ResNet-18）
  exp12_fusion.py         表 6   辞例通道与全集融合、α 扫描
  adaptive_fusion.py      表 6b  α 网格
  ablation.py             表 8   编码器结构消融
  arch_exp3.py            表 8   主干架构对比（ConvNeXt / Transformer / InfoNCE）
  eval_holdout.py         表 7   同简泄漏（按简切分 vs 随机切分）
  check_damage_protocol.py  残损模拟协议校验
  exp11_damage.py           残损形态实验
  eval_ensemble.py          5.9.4 模型集成与通道叠加

docs/           补充材料
```

## 数据

| 数据集 | 来源 | 说明 |
|---|---|---|
| 汉简 | DeepJiandu（Xiao et al., *Scientific Data* 2025） | 7,416 张红外图像、99,852 个字符标注、2,242 类。公开数据集，可按其原始许可获取。DOI: 10.1038/s41597-025-04716-3 |
| 秦简 | 北京大学藏秦简图版 + 《秦文字编》字形汇编 | 10,311 个字形、679 类。**原始图版不在本仓库**，因再分发许可需进一步核实 |

本仓库**不含**模型权重与原始图像。汉简数据请向 DeepJiandu 数据集原发布方申请；秦简数据的处理脚本见 `src/`，索引与处理流程可复现，但原始图版需自行获取。

## 复现步骤

```bash
pip install -r requirements.txt

# 1. 生成按简切分的划分
python src/make_split.py --data <glyph_dir> --out splits/

# 2. 训练编码器
python src/train.py --splits splits/ --epochs 12 --batch 128 --lr 3e-4

# 3. 评估（主结果）
python src/eval_protocol.py --ckpt <model.pt> --splits splits/ --rank 20

# 4. 各表实验
python experiments/eval_baseline.py --splits splits/     # 表 3
python experiments/exp12_fusion.py --splits splits/      # 表 6
python experiments/ablation.py --splits splits/          # 表 8
python experiments/eval_holdout.py --splits splits/      # 表 7
```

统一口径：Adam、学习率 3×10⁻⁴、余弦退火、批大小 128；补全器 12 个 epoch；全部实验在单张 RTX 4090（24 GB）上完成。Rank 一律在按简切分的测试集上计算，并排除与查询同简的样本。

## 评估协议的三条约束

1. **按简切分**：同一支简的字形只出现在训练集或测试集一侧。
2. **排除同简**：检索时跳过与查询属于同一支简的库内样本。
3. **超参冻结**：融合权重 α 在验证集上选定后冻结，再到测试集评估。秦简因测试集仅 181 个字形、无法再划独立验证集，α 在测试集上选定，其结果应视为上界估计。

## 引用

若使用本仓库代码，请引用：

> 严毓焕. 面向简牍残字的候选检索：实现选择、评估协议与语言先验权重. 2026.

以及数据集原文：

> Liu Y, Zhang Q, Qi Y, et al. DeepJiandu Dataset for Character Detection and Recognition on Jiandu Manuscript. *Scientific Data*, 2025, 12: 398. DOI: 10.1038/s41597-025-04716-3

## 许可

本仓库代码以 MIT 许可发布。数据集的使用请遵循各自的原始许可；秦简图版的再分发许可仍在核实中。
