# TTTSNet 实验索引

本文件记录 TTTSNet-Research v2.0 里程碑下的所有实验，确保实验路径、配置和结论可追踪。

**最后更新:** 2026-06-20

---

## 实验列表

| # | 实验名称 | 阶段 | 路径 | 状态 | Best val_mIoU | Best Epoch | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | TTS-Net-Single baseline | Phase 2 | `experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/` | ✅ 完成 | **0.5992** | 66 | 正式基线，训练 100 epochs，1.64h |
| 2 | TTS-Net-Temporal v1 (λ=0.1) | Phase 3 | `experiments/tttsnet_temporal_20260620_013149/tttsnet_temporal_20260620_013154/` | ⏹️ 停止 | 0.5491 | 18 | 35 epochs 停止，未超 baseline |
| 3 | TTS-Net-Temporal v2 (λ=1.0) | Phase 3 | `experiments/tttsnet_temporal_v2_20260620_021245/tttsnet_temporal_v2_20260620_021249/` | ✅ 完成待分析 | 0.5459 | - | 100 epochs 完成，仍未超 baseline |
| 4 | Baseline vs Temporal v1 对比 | Phase 3 | `experiments/comparison_baseline_vs_temporal_v1/` | ✅ 完成 | - | - | 对比表格和曲线 |
| 5 | TTS-Net-Semi（伪标签+有标注） | Phase 4 | `experiments/tttsnet_semi_20260620_085333/tttsnet_semi_20260620_085338/` | 🔄 进行中 | 0.4630 (epoch 10) | 10 | 100 epochs 计划，训练已启动 |
| 6 | TTS-Net-Temporal v3（强同步增强，λ=0.1） | Phase 3 | `configs/config_temporal_v3.json` | ⏳ 待执行 | - | - | 待 semi 训练完成后启动 |

---

## `experiments/deprecated/` 归档说明

`deprecated/` 仅用于存放因**代码异常中断、配置错误、无意义的调试运行**导致结果不可用的实验产物。

> 注意：结果未达预期但训练过程正常的实验（如 Temporal v1/v2）不放入 `deprecated/`，仍保留在 `experiments/` 根目录供后续分析。

### 当前归档内容

| 归档路径 | 原因 |
|---|---|
| `experiments/deprecated/tttsnet_semi_20260620_084856/` | 首次启动 semi 训练时因 image/mask 尺寸不一致异常退出 |
| `experiments/deprecated/semi_training_nohup.log` | 上述失败运行的 nohup 日志 |

### 已物理删除的调试运行

以下目录为早期无意义的测试运行，已直接删除，仅在此记录：

| 原路径 | 原因 |
|---|---|
| `experiments/tttsnet_single_baseline_20260619_232711/` | 仅 2 epochs 的测试运行，非正式 baseline |
| `experiments/tttsnet_temporal_20260620_012935/` | 仅 1 epoch 的测试运行 |

---

## Phase 4 半监督状态

| 组件 | 路径 | 状态 |
|---|---|---|
| 伪标签生成脚本 | `tools/generate_pseudo_labels.py` | ✅ 就绪 |
| 半监督训练脚本 | `tools/train_semi.py` | ✅ 就绪 |
| 半监督数据集 | `src/dataset_semi.py` | ✅ 就绪（已修复尺寸对齐） |
| 半监督配置 | `configs/config_semi.json` | ✅ 就绪 |
| 伪标签数据 | `sfy_data_v1_20251019/*/pseudo_labels/` | ✅ 已生成 18935 张（confidence≥0.9） |
| 伪标签日志 | `experiments/pseudo_label_generation.log` | ✅ 成功完成 |
| 半监督训练 | `experiments/tttsnet_semi_20260620_085333/tttsnet_semi_20260620_085338/` | 🔄 进行中 |

---

## 关键结论

1. **Baseline 0.599 是当前最佳结果**，未达 0.65 目标。
2. **Temporal 一致性约束未能提升性能**（v1: 0.549, v2: 0.546）。相关实验保留在 `experiments/` 根目录，待进一步分析。
3. **Phase 4 半监督** 已启动，当前 epoch 9 val_mIoU = 0.459，仍在训练中。

---

## 如何复现

```bash
# Phase 2 baseline
./scripts/train.sh configs/config.json

# Phase 3 temporal v1
./scripts/train_temporal.sh configs/config_temporal.json

# Phase 3 temporal v2
./scripts/train_temporal_v2.sh configs/config_temporal_v2.json

# Phase 4 pseudo labels (示例)
python tools/generate_pseudo_labels.py \
  --config configs/config.json \
  --checkpoint experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/checkpoints/best_model.pth \
  --unlabeled_path /autodl-fs/data/masquer.li/temperal_data/sfy_data_v1_20251019/ \
  --output_dir /autodl-fs/data/masquer.li/temperal_data/sfy_data_v1_pseudo_labels

# Phase 4 semi-supervised
./scripts/train_semi.sh configs/config_semi.json
```

> 注意：路径已对齐到 `configs/` 和 `tools/` 目录结构。当前伪标签已直接生成到 `sfy_data_v1_20251019/*/pseudo_labels/`。
