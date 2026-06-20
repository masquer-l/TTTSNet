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
| 5 | TTS-Net-Semi（伪标签+有标注） | Phase 4 | `experiments/tttsnet_semi_20260620_085333/tttsnet_semi_20260620_085338/` | ⏸️ 已暂停 | 0.4630 (epoch 10) | 10 | 按诊断建议暂停，优先做实 baseline |
| 6 | TTS-Net-Temporal v3（强同步增强，λ=0.1） | Phase 3 | `configs/config_temporal_v3.json` | ⏳ 待执行 | - | - | 待 AUFL quick check 完成后决定 |
| 7 | B3 AUFL loss 30ep quick check | Phase 2 | `experiments/tttsnet_aufl_30ep_20260620_103645/tttsnet_aufl_30ep_20260620_103650/` | 🔄 进行中 | - | - | 验证 AUFL 是否优于当前 Dice+BCE+CE |

---

## 诊断实验结果

### B1 标签唯一值统计

| 数据集 | 标签值 | 像素占比 | 出现图片数 |
|---|---|---|---|
| Train | 0 | 88.72% | 2060 |
| Train | 1 | 8.80% | 2043 |
| Train | 2 | 1.11% | 581 |
| Train | 3 | 1.37% | 293 |
| Val | 0 | 86.83% | 658 |
| Val | 1 | 9.53% | 648 |
| Val | 2 | 1.95% | 320 |
| Val | 3 | 1.69% | 83 |

**结论**: 当前 `label > 1 -> 0` 的处理方式是正确的。把 value=2,3 也当前景反而会显著降低 val_mIoU（最佳 0.497 vs 当前 0.601）。value=2,3 不是目标血管区域。

### B2 验证集阈值扫描（Single best checkpoint）

| 标签处理方式 | best threshold | best val_mIoU | val_mIoU @ 0.5 |
|---|---|---|---|
| 只保留 value=1 | 0.4 | **0.6014** | 0.5984 |
| 保留 value=1,2,3 | 0.3 | 0.4973 | 0.4897 |

**结论**: 0.5 阈值不是主要瓶颈；标签处理正确。

### S1 伪标签质量审计（100 张随机样本）

| 指标 | 结果 |
|---|---|
| 总伪标签数 | 18745 |
| 面积占比 | mean=11.43%, median=11.53%, max=29.01% |
| 连通区域数 | mean=3.71, median=3, max=10 |
| 空 mask | 0 |
| 近全图 mask (>95%) | 0 |
| 启发式 good | 90% |
| 启发式 medium | 10% |
| 启发式 bad | 0% |

**结论**: 伪标签质量较好，不像之前担心的"噪声主导"。但保留率 94% 仍偏宽，后续可尝试更严格的筛选策略。

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
2. **标签处理无误**：value=2,3 不是目标血管，当前 `label > 1 -> 0` 正确。
3. **阈值 0.5 接近最优**：最佳阈值 0.4 仅比 0.5 高 0.003 mIoU。
4. **Temporal 一致性约束未能提升性能**（v1: 0.549, v2: 0.546），待 AUFL 实验后启动诊断实验 T1。
5. **伪标签质量较好**：100 张样本中 90% good, 10% medium, 0% bad。
6. **AUFL loss 30ep quick check 已启动**，用于判断 loss 设计是否是 baseline 瓶颈。

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
