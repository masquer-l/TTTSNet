# TTTSNet 实验目录与版本管理规范

> 本文件定义 TTTSNet 论文实验的命名规范、版本号体系和目录结构。
> 所有新实验必须遵循本规范；历史实验已通过符号链接和 `VERSION` 文件补录。

## 1. 版本号体系

采用 **模型版本 + 实验编号** 双层体系。

### 1.1 模型版本 `TTTSNet-vX.Y.Z`

用于标识模型架构与训练策略的变体，在论文中可直接引用。

| 位置 | 含义 | 取值示例 |
|---|---|---|
| `X` (Major) | 架构代际 | `1`=原版 TTTSNet(CNN); `2`=ViT backbone; `3`=Transformer decoder; `4`=ViT+Transformer decoder |
| `Y` (Minor) | 组件/训练策略变体 | `0`=基线; `1`=分层学习率; `2`=clDice loss; `3`=时序模块; `4`=半监督 |
| `Z` (Patch) | 同配置下的修复或超参数微调 | `0`=首次; `1`=修复 bug 后重训; `2`=超参数微调 |

**示例**：
- `TTTSNet-v1.0.0`：修复 augmentation bug 后的 CNN baseline
- `TTTSNet-v2.0.0`：初代 ViT backbone
- `TTTSNet-v2.1.0`：ViT + 分层学习率
- `TTTSNet-v3.0.1`：Transformer decoder v2（4 层 + pos embed）
- `TTTSNet-v4.1.0`：ViT + Transformer decoder + 分层学习率
- `TTTSNet-v4.2.0`：ViT + Transformer decoder + clDice loss
- `TTTSNet-v4.4.0`：ViT + Transformer decoder + 半监督

### 1.2 实验编号 `EXP-XXX`

用于唯一标识一次具体训练运行（包括不同 seed、重跑等）。

- 按有价值实验的梳理顺序递增，从 `EXP-001` 开始。
- 同一模型版本的不同 seed/重跑使用不同 EXP 编号。
- 废弃或早期探索性实验不分配 EXP 编号。

## 2. 实验目录命名

### 2.1 历史实验（保留原始目录）

历史实验目录保持原始时间戳名称不变，通过**符号链接**提供规范入口：

```
experiments/EXP-{XXX}_{version}_seed{seed}_{YYYYMMDD-HHMMSS} -> {原始目录}/
```

例如：
```
experiments/EXP-010_TTTSNet-v2.1.0_seed42_20260626-095855 ->
    tttsnet_vit_layerwise_lr_20260626_095855/
```

### 2.2 未来实验（训练脚本自动生成）

修改后的训练脚本将直接生成规范化目录名：

```
experiments/EXP-{XXX}_{version}_seed{seed}_{YYYYMMDD-HHMMSS}/
```

其中 `version` 从配置文件的 `runtime_config.version` 字段读取，`EXP-{XXX}` 由脚本自动递增（或从配置读取）。

## 3. 实验目录内容标准

每个实验目录必须包含以下文件/子目录：

```
experiments/EXP-{XXX}_{version}_seed{seed}_{YYYYMMDD-HHMMSS}/
├── VERSION                     # 模型版本与 EXP 编号
├── config.json                 # 训练配置副本
├── training.log                # stdout/stderr 日志（自动捕获）
├── epoch_history.csv           # 每 epoch 指标
├── step_history.csv            # 每 step 指标（可选）
├── summary.json                # 最终汇总（best mIoU/Dice、epoch、总时间）
├── experiment_meta.json        # ExperimentTracker 元数据
├── checkpoints/
│   ├── best_model.pth          # 最佳模型（保留）
│   └── model_epoch_xxx.pth     # 中间 checkpoint（受 keep_last_n 限制）
├── tb_logs/                    # TensorBoard 事件
└── sfy_results/                # 如有 SFY 外部评测
    ├── metrics.csv
    ├── summary.json
    └── predictions/            # 预测图（可选）
```

### 3.1 VERSION 文件格式

```text
EXP-ID: EXP-010
Model-Version: TTTSNet-v2.1.0
Seed: 42
Status: completed
Description: ViT backbone + layerwise LR, seed 42
```

### 3.2 summary.json 关键字段

```json
{
  "best_val_miou": 0.6360,
  "best_val_dice": 0.7626,
  "best_epoch": 83,
  "total_time_h": 3.86
}
```

## 4. 训练脚本规范

所有训练脚本（`tools/train.py`、`tools/train_semi.py`、`tools/train_temporal.py`、`tools/train_sam_random.py`）必须：

1. 在创建 `exp_dir` 后，将 `sys.stdout`/`sys.stderr` 重定向到 `exp_dir/training.log`（使用 `TeeLogger` 保留控制台输出）。
2. 在训练结束时恢复 stdout/stderr。
3. 统一 `keep_last_n_checkpoints` 清理逻辑。
4. 在 `exp_dir` 下生成 `VERSION` 文件（如配置中提供版本号）。
5. 避免 shell 脚本与 Python 脚本叠加创建双层目录。

## 5. 主索引文档

所有实验的元数据汇总在 `TTTSNet/EXPERIMENTS.md` 中维护，包括：
- 模型版本到 EXP 编号的映射
- 内部验证指标
- SFY 外部评测指标
- 产物路径
- 论文用途标注（main result / ablation / baseline / deprecated）

## 6. 废弃实验

不满足以下条件的实验列为废弃/参考，不分配 EXP 编号：
- 无 `epoch_history.csv` 或训练未完成
- 指标明显异常（如 SFY mIoU < 0.1）
- 早期探索性尝试，已被后续版本完全替代

废弃实验仍在 `EXPERIMENTS.md` 的附录中记录，便于追溯失败原因。
