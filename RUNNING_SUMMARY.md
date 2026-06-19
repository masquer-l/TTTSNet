# TTTSNet-Research v2.0 实验进展摘要

**更新时间:** 2026-06-20 02:30  
**用户状态:** 睡眠中，Claude 自主执行 10 小时工作计划

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
- **交付物**: `EXPERIMENT.md`, `data_split.json`, `training_curves.png`, best checkpoint

---

## 进行中实验

### Phase 3: TTS-Net-Temporal v1（λ_temp=0.1）

- **状态**: ⏹️ 已停止（epoch 34）
- **最佳结果**: val_mIoU = **0.5491** (epoch 18)
- **停止原因**: 未超过 baseline，且 epoch 18 后下滑饱和
- **分析**: 时序数据集缺少与 baseline 同等强度的几何/custom 增强，λ_temp 可能过小
- **交付物**: `EXPERIMENT_INCOMPLETE.md`

### Phase 3: TTS-Net-Temporal v2（λ_temp=1.0）

- **状态**: 🔄 运行中（当前 epoch 4+, val_mIoU ~0.52）
- **目标**: 验证时序约束强度是否不足
- **预计完成**: ~03:50

### Phase 4: TTS-Net-Semi 准备

- **状态**: 🔄 伪标签生成中（~12,779 / 20,133 帧）
- **代码**: `train_semi.py`, `src/dataset_semi.py`, `generate_pseudo_labels.py` 已就绪
- **计划**: 使用 baseline 模型为 sfy 无标注视频生成伪标签，然后训练 semi 模型

---

## 关键发现

1. **原论文 TTTSNet 不是 ResNet-50 backbone**：源码显示为 custom lightweight network，448×448 输入。
2. **Custom augmentations 已修复 Albumentations 2.x 兼容性**：使用自定义 `CustomDefectsAugmentation` 组合实现。
3. **Baseline 0.599 未达 0.65**：可能原因包括 loss 组合、增强策略、输入尺寸等。
4. **Temporal v1 未达预期**：在缺少强增强的情况下，时序一致性反而略降。
5. **数据分布**：FetReg2021 train 前景比例均值 8.85%，血管分割是典型类别不平衡问题。

---

## 下步计划

1. 等待 Temporal v2 完成，评估 λ_temp=1.0 的效果
2. 伪标签生成完成后启动 Semi 训练
3. 根据 Temporal v2 和 Semi 结果决定是否需要进一步调参 ablation
4. 生成跨实验对比表格和可视化
5. 更新 GSD 文档并提交最终代码

---

## 资源使用

- GPU: RTX 4080 / 32GB，当前 Temporal v2 占用 ~2GB
- 内存：503GB 总量，使用 19GB
- 无并行 GPU 训练，保证可靠性
