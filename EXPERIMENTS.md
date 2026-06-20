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

---

## 已清理/归档的实验

以下目录因是早期测试或无效运行，已从 `experiments/` 中移除：

| 原路径 | 原因 |
|---|---|
| `experiments/tttsnet_single_baseline_20260619_232711/` | 仅 2 epochs 的测试运行，非正式 baseline |
| `experiments/tttsnet_temporal_20260620_012935/` | 仅 1 epoch 的测试运行 |

> 这些目录中的数据未纳入结论。如需恢复，可从 git 历史或备份中找回。

---

## Phase 4 半监督准备

| 组件 | 路径 | 状态 |
|---|---|---|
| 伪标签生成脚本 | `tools/generate_pseudo_labels.py` | ✅ 就绪 |
| 半监督训练脚本 | `tools/train_semi.py` | ✅ 就绪 |
| 半监督数据集 | `src/dataset_semi.py` | ✅ 就绪 |
| 半监督配置 | `configs/config_semi.json` | ✅ 就绪 |
| 伪标签输出目录 | `pseudo_labels/` | ⏸️ 当前为空，生成中断/未完成 |
| 伪标签日志 | `experiments/pseudo_label_generation.log` | ⏸️ 需检查 |

---

## 关键结论

1. **Baseline 0.599 是当前最佳结果**，未达 0.65 目标。
2. **Temporal 一致性约束未能提升性能**（v1: 0.549, v2: 0.546）。
3. **Phase 4 半监督** 需要先解决伪标签生成问题。

---

## 如何复现

```bash
# Phase 2 baseline
./scripts/train.sh configs/config.json

# Phase 3 temporal v1
./scripts/train_temporal.sh configs/config_temporal.json

# Phase 3 temporal v2
./scripts/train_temporal_v2.sh configs/config_temporal_v2.json

# Phase 4 pseudo labels
python tools/generate_pseudo_labels.py \
  --config configs/config.json \
  --checkpoint experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/checkpoints/best_model.pth \
  --output_dir pseudo_labels/sfy_data_v1

# Phase 4 semi-supervised
./scripts/train_semi.sh configs/config_semi.json
```

> 注意：路径已对齐到 `configs/` 和 `tools/` 目录结构。
