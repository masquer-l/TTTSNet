# TTTSNet-Research v2.0 实验进展摘要

**更新时间:** 2026-06-20  
**当前阶段:** Phase 3 已完成训练，进入结果分析；Phase 4 准备中  

---

## 已完成实验

### Phase 2: TTS-Net-Single（单帧全监督基线）

- **状态**: ✅ 完成
- **实验目录**: `TTTSNet/experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/`
- **最佳结果**: val_mIoU = **0.5992** (epoch 66), val_Dice = 0.7324
- **训练时间**: 1.64 小时 / 100 epochs
- **关键配置**:
  - 模型：原始 TTTSNet (custom lightweight, 5.3M params)
  - 输入：448×448
  - Loss：Dice + BCE + 0.5×CE
  - 增强：完整 custom augmentations（激光/灰尘/结构缺陷/光纤）
- **结论**: 训练稳定，但未达周计划 0.65 目标。
- **交付物**: `EXPERIMENT.md`, `data_split.json`, `training_curves.png`, `experiment_meta.json`, `summary.json`, best checkpoint

---

## 已完成但待分析的实验

### Phase 3: TTS-Net-Temporal v1（λ_temp=0.1）

- **状态**: ⏹️ 已停止（epoch 35）
- **实验目录**: `TTTSNet/experiments/tttsnet_temporal_20260620_013149/tttsnet_temporal_20260620_013154/`
- **最佳结果**: val_mIoU = **0.5491** (epoch 18)
- **停止原因**: 未超过 baseline，且 epoch 18 后下滑饱和
- **分析**: 时序数据集缺少与 baseline 同等强度的几何/custom 增强，λ_temp 可能过小
- **交付物**: `experiment_meta.json`, `epoch_history.csv`, `summary.json` 待创建

### Phase 3: TTS-Net-Temporal v2（λ_temp=1.0）

- **状态**: ✅ 训练完成（100 epochs），结果待分析
- **实验目录**: `TTTSNet/experiments/tttsnet_temporal_v2_20260620_021245/tttsnet_temporal_v2_20260620_021249/`
- **最佳结果**: val_mIoU = **0.5459**, val_Dice = 0.6886
- **关键发现**: 即使 λ_temp 提升到 1.0，仍未超过 baseline 0.599，且略低于 v1 的 0.549
- **待确认问题**:
  1. 时序 loss 实现是否存在 bug？
  2. temporal dataset 的增强策略是否过弱？
  3. 时序约束是否本身对该任务价值有限？

---

## 进行中实验

### Phase 4: TTS-Net-Semi 准备

- **状态**: ⏸️ 伪标签生成中断/未完成
- **代码**: `tools/train_semi.py`, `src/dataset_semi.py`, `tools/generate_pseudo_labels.py` 已就绪
- **输出目录**: `TTTSNet/pseudo_labels/`（当前为空）
- **日志**: `TTTSNet/experiments/pseudo_label_generation.log`
- **阻塞点**: 需要确认伪标签生成是否真的完成，以及置信度阈值 0.9 是否过高导致保存率为 0
- **下一步**: 重新运行伪标签生成，或降低 confidence_threshold 验证

---

## 关键发现

1. **原论文 TTTSNet 不是 ResNet-50 backbone**：源码显示为 custom lightweight network，448×448 输入。
2. **Custom augmentations 已修复 Albumentations 2.x 兼容性**：使用自定义 `CustomDefectsAugmentation` 组合实现。
3. **Baseline 0.599 未达 0.65**：可能原因包括 loss 组合、增强策略、输入尺寸等。
4. **Temporal 一致性约束目前未能提升性能**：
   - v1 (λ=0.1): 0.549 < 0.599
   - v2 (λ=1.0): 0.546 < 0.599
   - 需要先排查实现 bug，再决定是否继续调参
5. **数据分布**：FetReg2021 train 前景比例均值 8.85%，血管分割是典型类别不平衡问题。

---

## 下步计划

### 立即（阶段 A 整理后续）
1. ✅ 清理废弃实验目录（2-epoch 测试 baseline、1-epoch 测试 temporal）
2. ✅ 删除 `src/__pycache__/`
3. ⏳ 建立 `EXPERIMENTS.md` 统一索引
4. ⏳ 更新 GSD `STATE.md` 和 `ROADMAP.md`

### 短期（分析与决策）
1. 深入分析 temporal v1/v2 的训练曲线，确认时序 loss 行为是否正常
2. 可视化 temporal dataset 的输出，确认增强策略是否足够强
3. 重新运行伪标签生成，确认 Phase 4 可行性

### 中期（代码重构）
1. 合并 `train.py` / `train_temporal.py` / `train_semi.py`
2. 统一 dataset 基类
3. 统一 config 管理

---

## 资源使用

- GPU: NVIDIA GeForce RTX 4080 SUPER / 32GB
- 当前无训练进程在跑
- 内存：503GB 总量，使用 19GB

---

## 实验路径索引

| 实验 | 路径 | 状态 |
|---|---|---|
| Single baseline | `experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/` | ✅ 完成 |
| Temporal v1 (λ=0.1) | `experiments/tttsnet_temporal_20260620_013149/tttsnet_temporal_20260620_013154/` | ⏹️ 停止 |
| Temporal v2 (λ=1.0) | `experiments/tttsnet_temporal_v2_20260620_021245/tttsnet_temporal_v2_20260620_021249/` | ✅ 完成待分析 |
| Baseline vs Temporal v1 对比 | `experiments/comparison_baseline_vs_temporal_v1/` | ✅ 完成 |
| Pseudo-label 日志 | `experiments/pseudo_label_generation.log` | ⏸️ 中断 |
