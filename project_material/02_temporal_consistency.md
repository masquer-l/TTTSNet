# 02 时序一致性约束（Temporal Consistency）

## 2.1 动机

胎儿镜视频具有天然时序连续性：相邻帧在血管结构和图像内容上高度相关。时序一致性约束的目标是让模型在相邻帧上产生稳定的预测，从而减少单帧分割中的抖动和孤立错误。

核心假设：
- 相邻帧的血管分割结果应该在空间上保持一致；
- 时序约束可以帮助模型学习更鲁棒的特征，尤其在前景边界和细血管区域。

## 2.2 模型设计

### 2.2.1 数据组织

构建连续 3 帧片段 `[t-1, t, t+1]`：
- 输入：3 帧图像 stacked，形状 `[3, 3, H, W]`
- 监督：中间帧 `t` 的 GT mask，形状 `[1, H, W]`
- 时序约束：相邻帧 `t-1` 和 `t+1` 的预测应与中间帧预测保持一致

数据集实现：`TTTSNet/src/dataset_temporal.py`

### 2.2.2 时序同步增强

为保证时序 loss 的时空对齐，3 帧共享同一组增强参数：

- 使用 `A.ReplayCompose` 记录中间帧 image+mask 的增强参数
- 通过 `A.ReplayCompose.replay(params, image=...)` 将同一组参数重放到左右相邻帧
- 颜色增强（ColorJitter、RandomBrightnessContrast、CLAHE）在三帧间同步
- 水平翻转在 numpy 阶段同步完成

### 2.2.3 时序损失函数

对中间帧和相邻帧的预测概率计算一致性损失：

```python
# pred: [B, 3, 2, H, W]，3 帧的预测 logits
probs = F.softmax(pred, dim=2)[:, :, 1, :, :]  # [B, 3, H, W]

# 相邻帧与中间帧的预测差异
loss_t1 = (probs[:, 0] - probs[:, 1]).pow(2) * confidence_weight
loss_t2 = (probs[:, 2] - probs[:, 1]).pow(2) * confidence_weight
loss_temporal = (loss_t1 + loss_t2).mean()
```

其中 confidence weight 为 `p * (1 - p) * 4`，在预测不确定区域（p≈0.5）权重最大。

### 2.2.4 总损失

```
L_total = L_supervision + λ_temp × L_temporal
```

监督损失与 baseline 相同：Dice + BCE + 0.5×CE。

## 2.3 实验配置

| 实验 | λ_temp | 数据增强 | Epochs | 配置 |
|---|---|---|---|---|
| Temporal v1 | 0.1 | 同步颜色增强 + 水平翻转 | 35（提前停止） | `configs/config_temporal.json` |
| Temporal v2 | 1.0 | 同步颜色增强 + 水平翻转 | 100 | `configs/config_temporal_v2.json` |
| Temporal no-loss (T1) | 0.0 | 同上 | 100 | `configs/config_temporal_no_loss.json` |

**注意**: Temporal v1/v2 的增强弱于 baseline（缺少 VerticalFlip、CustomDefects、ShiftScaleRotate、Blur）。

## 2.4 实验结果

### 2.4.1 主要结果

| 实验 | Best val_mIoU | Best Epoch | 最终 val_mIoU | 结论 |
|---|---|---|---|---|
| Baseline | **0.5992** | 66 | 0.5906 | 当前最佳 |
| Temporal v1 (λ=0.1) | 0.5491 | 18 | - | 低于 baseline，epoch 18 后下滑 |
| Temporal v2 (λ=1.0) | 0.5459 | 24 | 0.5217 | 低于 baseline，后期过拟合 |
| Temporal no-loss (T1) | 0.5340 | 87 | 0.5309 | **关闭 temporal loss 后仍远低于 baseline**，说明 dataset/增强是主因 |

### 2.4.2 Temporal v2 曲线分析

- train/loss 从 1.109 降至 0.062，train/loss_temporal 约 0.0004
- val/loss 在 0.574 附近最佳后逐渐升至 0.99
- val_mIoU 在 epoch 24 达峰后长期停留在 0.52 左右

判断：**训练集拟合增强，验证集泛化不提升**。

### 2.4.3 可能原因分析

| 因素 | 说明 | 待验证 |
|---|---|---|
| 增强过弱 | Temporal 缺少 baseline 的 CustomDefects/几何增强 | T1 no-loss |
| temporal loss 权重设计反向 | `p*(1-p)` 强调不确定区域，而非高置信区域 | T1 no-loss + corrected-confidence |
| temporal loss 信号太弱 | train/loss_temporal 仅约 0.0004 | gradient/logging |
| 时序约束本身价值有限 | 胎儿镜血管在相邻帧间运动/形变较大 | - |

## 2.5 诊断计划

### T1: Temporal No-Loss

- **配置**: `loss_temporal_weight=0`
- **目的**: 排除 temporal loss 本身，单独看 TemporalDataset/增强的影响
- **结果**: 100 epochs 完成，best val_mIoU = **0.5340** (epoch 87)，final = **0.5309**
- **判断**: no-loss 结果 ≈ temporal v1/v2 (0.55)，**dataset/增强差异是 Temporal 低于 Single 的主因**

### Temporal v3（建议执行）

T1 结果支持引入与 baseline 对齐的强同步增强：
- VerticalFlip
- Blur/MotionBlur
- CustomDefectsAugmentation
- ShiftScaleRotate

所有增强仍通过 `ReplayCompose` 在三帧间同步。配置已准备：`configs/config_temporal_v3.json` + `scripts/train_temporal_v3.sh`。

## 2.6 当前结论

- 当前时序一致性约束**未能提升** val_mIoU。
- **T1 诊断关键发现**: 关闭 temporal loss 后性能仍只有 0.534，与 temporal v1/v2 相当，说明 **dataset/增强差异是主因**，而非 temporal loss 本身。
- 建议执行 **Temporal v3**（强同步增强），验证增强对齐后能否恢复/超越 baseline。

### 论文定位

时序一致性目前**不适合作为论文主贡献**。若 Temporal v3 能证明是增强差异导致，可作为消融实验；若仍不提升，可作为负向分析。

---

**相关文件**: `TTTSNet/src/dataset_temporal.py`, `TTTSNet/tools/train_temporal.py`, `TTTSNet/configs/config_temporal*.json`

**实验目录**:
- `TTTSNet/experiments/tttsnet_temporal_20260620_013149/`
- `TTTSNet/experiments/tttsnet_temporal_v2_20260620_021245/`
- `TTTSNet/experiments/tttsnet_temporal_no_loss_20260620_114901/tttsnet_temporal_no_loss_20260620_114905/`
