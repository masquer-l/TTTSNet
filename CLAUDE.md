# TTTSNet 项目：数据筛选与标注工作指南

> 本文件用于在另一环境开展 TTTS 胎儿镜血管分割数据筛选、标注与实验复现。  
> 仓库地址：`https://github.com/masquer-l/TTTSNet.git`  
> 更新时间：2026-07-04

---

## 1. 仓库内容说明

本仓库已按“代码 + 关键文档 + 关键实验产物”整理，目录结构如下：

```text
.
├── CLAUDE.md                          # 本说明文件
├── src/                               # TTTSNet 源代码（模型、数据集、工具函数）
├── configs/                           # 训练/推理配置文件
├── scripts/                           # Shell 启动脚本
├── tools/                             # 可直接运行的 Python 脚本（训练、评测、伪标签）
├── experiments/                       # 关键实验目录（仅含两个关键实验，不含 .pth 权重）
│   ├── tttsnet_single_baseline_20260623_222644/   # TTTSNet CNN baseline
│   └── tttsnet_vit_layerwise_lr_20260627_083345/  # 当前最强 ViT + Layerwise LR
├── project_material/                  # 论文与实验相关文档
│   ├── 11_paper_direction_guidance.md # 当前论文方向与防发散指导
│   ├── 10_current_experimental_conclusions.md
│   ├── 09_experimental_conclusions_and_redo_plan.md
│   ├── README.md
│   ├── experiment_summary.csv
│   └── ...
├── README.md                          # TTTSNet 代码仓库说明
├── RUNNING_SUMMARY.md                 # 实验进展摘要
├── EXPERIMENTS.md                     # 完整实验索引与结果总表
└── EXPERIMENT_LAYOUT.md               # 实验目录规范
```

---

## 2. 关键实验与本地权重下载路径

> **注意**：`.pth` 权重文件未提交到仓库（GitHub 单文件限制 100MB，ViT 权重约 1.1GB）。  
> 请从下方原服务器路径手动复制到对应实验目录的 `checkpoints/` 下。

### 2.1 TTTSNet CNN Baseline（EXP-001）

- **实验目录**：`experiments/tttsnet_single_baseline_20260623_222644/`
- **模型版本**：`TTTSNet-v1.0.0`
- **说明**：修复 augmentation bug 后的单帧全监督 CNN baseline
- **关键结果**：
  - 内部 Val mIoU：`0.6049`
  - 内部 Val Dice：`0.7382`
  - SFY mIoU：`0.6275`
  - SFY Dice：`0.7495`
- **权重原路径**：
  ```text
  /autodl-fs/data/masquer.li/code/TTTSNet/experiments/tttsnet_single_baseline_20260623_222644/checkpoints/best_model.pth
  ```
- **目标路径**：`./experiments/tttsnet_single_baseline_20260623_222644/checkpoints/best_model.pth`
- **文件大小**：约 62MB

### 2.2 TTTSNet ViT + Layerwise LR（EXP-011，当前 SOTA）

- **实验目录**：`experiments/tttsnet_vit_layerwise_lr_20260627_083345/`
- **模型版本**：`TTTSNet-v2.1.0`
- **说明**：SAM ViT-B 替换原 Init_Block，并对 `vit_encoder` 使用 0.1x 学习率
- **关键结果**：
  - 内部 Val mIoU：`0.6405`
  - 内部 Val Dice：`0.7670`
  - SFY mIoU：`0.6466`
  - SFY Dice：`0.7644`
- **权重原路径**：
  ```text
  /autodl-fs/data/masquer.li/code/TTTSNet/experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/best_model.pth
  /autodl-fs/data/masquer.li/code/TTTSNet/experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/model_epoch_100.pth
  ```
- **目标路径**：
  ```text
  ./experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/best_model.pth
  ./experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/model_epoch_100.pth
  ```
- **文件大小**：每个约 1.1GB

### 2.3 下载示例（rsync）

假设你已从原服务器拿到 SSH 访问权限，可在本仓库根目录执行：

```bash
# baseline 权重
rsync -avP \
  <user>@<host>:/autodl-fs/data/masquer.li/code/TTTSNet/experiments/tttsnet_single_baseline_20260623_222644/checkpoints/best_model.pth \
  ./experiments/tttsnet_single_baseline_20260623_222644/checkpoints/

# ViT SOTA 权重（best + epoch_100）
rsync -avP \
  <user>@<host>:/autodl-fs/data/masquer.li/code/TTTSNet/experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/ \
  ./experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/
```

---

## 3. 数据筛选与标注工作流

### 3.1 生成候选数据预测

若已有未标注图像，可用 baseline 或 ViT 模型生成初始预测，作为标注起点：

```bash
# 使用 CNN baseline 生成伪标签/预测
python tools/eval_sfy.py \
  --config experiments/tttsnet_single_baseline_20260623_222644/config.json \
  --checkpoint experiments/tttsnet_single_baseline_20260623_222644/checkpoints/best_model.pth \
  --dataset_path /path/to/your/unlabeled/images \
  --output_csv experiments/baseline_candidate_predictions.csv

# 使用 ViT SOTA 生成伪标签/预测
python tools/eval_sfy.py \
  --config experiments/tttsnet_vit_layerwise_lr_20260627_083345/config.json \
  --checkpoint experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/best_model.pth \
  --dataset_path /path/to/your/unlabeled/images \
  --output_csv experiments/vit_candidate_predictions.csv
```

> 提示：`eval_sfy.py` 默认在 SFY 数据集上评测，但也可用于任意图像目录，只需保证 `--dataset_path` 下为 `images/` 与 `masks/`（或参考 `eval_sfy.py` 的目录约定）。

### 3.2 筛选高价值样本

建议优先筛选以下类型样本进行人工标注/校验：

- **Baseline 与 ViT 预测 IoU 差异大**的样本（说明模型对该样本不确定）。
- **细血管、低对比、激光光斑、器械遮挡**等困难样本。
- **前景面积过小或过大**的样本（面积分布 < 5% 或 > 30%）。
- **跨帧一致性差**的视频片段（若有时序数据）。
- **SFY 外部数据**中模型失败案例。

### 3.3 伪标签生成与清洗

```bash
python tools/generate_pseudo_labels.py \
  --config configs/config.json \
  --checkpoint experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/best_model.pth \
  --unlabeled_path /path/to/unlabeled/images \
  --output_dir pseudo_labels/new_batch \
  --confidence_threshold 0.9
```

生成后建议按置信度、前景面积、与相邻帧一致性等指标分层，只保留高质量伪标签进入训练。

---

## 4. 快速复现主实验

### 4.1 环境准备

```bash
pip install -r requirements.txt  # 若存在；否则参考 configs/ 与 src/ 的 import 安装
```

### 4.2 运行 baseline

```bash
./scripts/train.sh configs/config.json
```

### 4.3 运行 ViT + Layerwise LR

```bash
./scripts/train.sh configs/config_vit_layerwise_lr.json
```

### 4.4 在 SFY 上评测已有 checkpoint

```bash
python tools/eval_sfy.py \
  --config experiments/tttsnet_vit_layerwise_lr_20260627_083345/config.json \
  --checkpoint experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/best_model.pth \
  --dataset_path /path/to/SFY \
  --output_csv experiments/sfy_vit_evaluation.csv
```

---

## 5. 配置与路径迁移注意事项

1. **绝对路径**：`configs/*.json` 中 `dataset_config.train_paths/val_paths` 以及工具脚本中的默认路径均为原服务器绝对路径。迁移后请通过命令行参数或环境变量覆盖。
2. **SAM 预训练权重**：ViT 模型需要 `sam_vit_b_01ec64.pth`，默认路径为 `/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth`。新环境需下载并更新 `sam_checkpoint` 配置。
3. **不要提交 `.pth`**：本仓库 `.gitignore` 已忽略所有 `*.pth`、`*.pt`、完整 `experiments/` 目录。如需新增实验目录，请用 `git add -f` 显式添加结果文件，并排除权重。
4. **数据保密**：SFY 与手术视频数据不要上传至公开仓库。

---

## 6. 参考文档优先级

1. `project_material/11_paper_direction_guidance.md` — 当前论文方向与防发散规则。
2. `EXPERIMENTS.md` — 完整实验结果、版本定义、产物路径索引。
3. `EXPERIMENT_LAYOUT.md` — 实验目录命名与归档规范。
4. `RUNNING_SUMMARY.md` — 实验进展与关键发现。
5. `project_material/10_current_experimental_conclusions.md` — 当前阶段结论。

---

## 7. 后续建议

- 在筛选/标注过程中，优先记录**失败模式 taxonomy**（如细血管断裂、激光光斑误分割、器械遮挡漏分等）。
- 对新增标注数据，建议按视频/场景分层，避免训练集与验证集来自同一段视频。
- 若新数据规模较大，可重新运行 baseline 与 ViT 锚点实验，验证跨域泛化。

如有问题，可在本仓库提交 Issue 或在本地 `CLAUDE.md` 中继续补充环境特定说明。
