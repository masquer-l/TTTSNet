# TTTSNet-Research v2.0 实验进展摘要

**更新时间:** 2026-06-20  
**当前阶段:** 按诊断建议优先做实 Single baseline；AUFL quick check 运行中；Semi 已暂停  

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

### Phase 4: TTS-Net-Semi（已暂停）

- **状态**: ⏸️ 已暂停（按诊断建议优先做实 baseline）
- **实验目录**: `TTTSNet/experiments/tttsnet_semi_20260620_085333/tttsnet_semi_20260620_085338/`
- **当前结果**: val_mIoU = **0.4630** (epoch 10)
- **代码修复**: `src/dataset_semi.py` 已修复 image/mask 尺寸对齐问题
- **数据**: 2060 有标注 + 18935 伪标签（confidence≥0.9），共 20995 训练样本
- **伪标签质量审计**: 100 张随机样本中 90% good, 10% medium, 0% bad
- **阻塞点**: 已解除
- **下一步**: 待 baseline 做实、伪标签筛选策略优化后，再重启 semi 实验

### B3: AUFL loss 30ep Quick Check（进行中）

- **状态**: 🔄 训练中
- **实验目录**: `TTTSNet/experiments/tttsnet_aufl_30ep_20260620_103645/tttsnet_aufl_30ep_20260620_103650/`
- **目的**: 验证 AsymmetricUnifiedFocalLoss 是否优于当前 Dice+BCE+CE
- **配置**: `configs/config_aufl_30ep.json`
- **下一步**: 30 epochs 后比较 val_mIoU 曲线

### 已归档的失败尝试

- **首次 semi 启动**: `experiments/deprecated/tttsnet_semi_20260620_084856/`
  - 失败原因: `dataset_semi.py` 中伪标签尺寸（448×448）与原图尺寸不一致，Albumentations shape check 报错
  - 修复: 在 `__getitem__` 中把 label resize 到 image 尺寸后再 transform

---

## 关键发现

1. **原论文 TTTSNet 不是 ResNet-50 backbone**：源码显示为 custom lightweight network，448×448 输入。
2. **Custom augmentations 已修复 Albumentations 2.x 兼容性**：使用自定义 `CustomDefectsAugmentation` 组合实现。
3. **Baseline 0.599 未达 0.65**：可能原因包括 loss 组合、增强策略、输入尺寸等。
4. **标签处理已验证无误**:
   - Train/Val masks 存在 value=0,1,2,3
   - 当前 `label > 1 -> 0` 只保留 value=1 是正确的
   - 把 value=2,3 也当前景会使 val_mIoU 从 0.599 降到 0.497
5. **阈值扫描**:
   - Single best checkpoint 最佳阈值 0.4，mIoU=0.6014
   - 0.5 阈值 mIoU=0.5984，差距极小
6. **伪标签质量较好**:
   - 100 张随机样本：90% good, 10% medium, 0% bad
   - 面积分布合理（mean 11.4%），无空 mask 或全图 mask
7. **AUFL loss quick check** 已启动，30 epochs 后评估是否优于当前 loss
8. **Semi 训练已暂停**，等待 baseline 做实和伪标签策略优化
### 立即
1. 等待 AUFL 30ep quick check 完成，判断 loss 是否为主要瓶颈
2. 若 AUFL 有效，跑完整 100ep AUFL baseline；若无效，继续探索其他因素（输入尺寸、数据划分等）
3. 执行 T1 Temporal no-loss 诊断实验

### 短期
1. 根据 AUFL 结果决定是否引入更严格的伪标签筛选（mean confidence、area filtering）
2. 设计 3 seeds baseline 实验，报告 mean±std
3. 尝试输入尺寸 512/640

### 中期
1. 合并 `train.py` / `train_temporal.py` / `train_semi.py`
2. 统一 dataset 基类
3. 统一 config 管理

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
| Pseudo-label 日志 | `experiments/pseudo_label_generation.log` | ✅ 成功完成 |
| AUFL 30ep quick check | `experiments/tttsnet_aufl_30ep_20260620_103645/tttsnet_aufl_30ep_20260620_103650/` | 🔄 进行中 |
| Semi-supervised | `experiments/tttsnet_semi_20260620_085333/tttsnet_semi_20260620_085338/` | ⏸️ 已暂停 |
