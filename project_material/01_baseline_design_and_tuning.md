# 01 TTTSNet 单帧基线设计与调优

## 1.1 任务背景

本研究面向双胎输血综合征（TTTS）胎儿镜血管分割。输入为胎儿镜视频帧，输出为二值血管 mask。由于血管细长、前景占比低（约 8-9%）、图像受激光/灰尘/光纤等伪影干扰，属于典型的类别不平衡医学图像分割问题。

## 1.2 模型设计

### 1.2.1 网络架构

采用原论文 TTTSNet 的 custom lightweight network，而非 ResNet-50 backbone：

- **输入尺寸**: 448×448
- **输出**: 2 类 logits（背景 + 血管）
- **模型参数量**: 5,308,465
- **核心结构**: 双分支编码器 + 特征融合解码器（详见 `TTTSNet/src/models/TTTSNet.py`）

### 1.2.2 数据增强

训练时使用 Albumentations 2.x 兼容的增强管线，包含针对胎儿镜图像的自定义缺陷增强：

| 增强 | 参数 | 目的 |
|---|---|---|
| Resize | 448×448 | 统一输入尺寸 |
| HorizontalFlip | p=0.5 | 增加姿态多样性 |
| VerticalFlip | p=0.5 | 增加姿态多样性 |
| Blur / MotionBlur | p=0.2 | 模拟失焦/运动模糊 |
| CustomDefectsAugmentation | p=0.5 | 模拟激光、灰尘、结构缺陷、光纤 |
| ShiftScaleRotate | p=0.2 | 几何形变 |
| ColorJitter | p=0.3 | 颜色扰动 |
| RandomBrightnessContrast | p=0.3 | 亮度对比度扰动 |
| CLAHE | p=0.15 | 局部对比度增强 |
| Normalize + ToTensorV2 | - | 标准化 |

自定义缺陷增强通过组合 `AddLaserPointer`、`AddDustParticles`、`AddStructuralDefects`、`AddOpticalFiber` 实现（见 `TTTSNet/src/utils/custom_augmentations.py`）。

### 1.2.3 损失函数

基础损失为 Dice + BCE + 0.5×CE：

```
L_total = 1.0 × L_dice + 1.0 × L_bce + 0.5 × L_ce
```

- `DiceLoss`: 来自 `segmentation_models.pytorch` 风格的 binary Dice，直接优化分割重叠度
- `BCEWithLogitsLoss`: pixel-wise 二分类交叉熵
- `CrossEntropyLoss`: 多类 CE，作为正则

### 1.2.4 优化器与训练设置

| 配置 | 值 |
|---|---|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Weight decay | 0.01 |
| 学习率调度 | CosineAnnealingLR, T_max=100, eta_min=1e-7 |
| Batch size | 4 |
| Epochs | 100 |
| Seed | 42 |
| AMP | 启用 |
| Gradient clipping | max_norm=1.0 |

## 1.3 基线实验结果

### 1.3.1 主要结果

| 指标 | 值 | Epoch |
|---|---|---|
| Best val_mIoU | **0.5992** | 66 |
| Best val_Dice | 0.7324 | 66 |
| Final val_mIoU | 0.5906 | 100 |
| 训练时间 | 1.64 h | - |

训练曲线显示：前 20 epochs 快速上升到约 0.57，之后进入 0.58-0.60 平台期，未达周计划 0.65 目标。

### 1.3.2 曲线特征

- train/loss 从 1.06 稳定下降到 0.26
- val_mIoU 在 epoch 66 达峰后不再提升
- 判断：训练流程通，但泛化平台约 0.60，需要进一步诊断瓶颈

## 1.4 基线调优诊断

### 1.4.1 B1 标签唯一值统计

| 数据集 | value=0 | value=1 | value=2 | value=3 |
|---|---|---|---|---|
| Train pixel% | 88.72% | 8.80% | 1.11% | 1.37% |
| Val pixel% | 86.83% | 9.53% | 1.95% | 1.69% |

**处理逻辑**: `np.where(label > 1, 0, label)` 后 `(label > 0).astype(uint8)`，即只保留 value=1 作为前景。

**验证**: 若把 value=1,2,3 全部作为前景，val_mIoU 从 0.599 降到 0.497。

**结论**: 当前标签处理正确，value=2,3 不是目标血管区域。

### 1.4.2 B2 验证集阈值扫描

对 Single best checkpoint 在验证集上扫描阈值 0.1-0.9：

| 处理方式 | best threshold | best val_mIoU | val_mIoU @ 0.5 |
|---|---|---|---|
| 只保留 value=1 | 0.4 | **0.6014** | 0.5984 |
| 保留 value=1,2,3 | 0.3 | 0.4973 | 0.4897 |

**结论**: 0.5 阈值接近最优，不是主要瓶颈。

### 1.4.3 B3 AsymmetricUnifiedFocalLoss 快速验证

30 epochs quick check，默认参数（weight=0.5, delta=0.6, gamma=0.2）：

| 方法 | 30ep best val_mIoU | 对应 epoch |
|---|---|---|
| AUFL | 0.5647 | 27 |
| Baseline (Dice+BCE+0.5CE) | 0.5720 | 24 |
| Baseline 完整 100ep | 0.5992 | 66 |

**结论**: 默认参数 AUFL 未超过当前 loss，loss 不是当前 baseline 的主要瓶颈。

## 1.5 当前结论与下一步

### 已排除的瓶颈

- ❌ 标签处理错误
- ❌ 评估阈值不合适
- ❌ 基础 loss 组合明显劣于 AUFL

### 仍待探索的瓶颈

| 优先级 | 方向 | 说明 |
|---|---|---|
| 高 | 输入分辨率 512/640 | 细血管在 448×448 下可能断裂 |
| 高 | 多 seeds 基线 | 报告 mean±std，确认 0.60 是否为稳定上限 |
| 中 | 数据划分 / 域差异 | train/test 间存在胎儿镜视角差异 |
| 中 | 更大 backbone / 预训练 | 当前 5.3M 参数模型容量有限 |

### 论文定位

单帧 baseline 是后续所有对比的基石。当前 0.599 是 TTTSNet 在该数据集上的可复现结果，论文中应明确说明训练设置、数据增强和评估方式。

---

**相关文件**: `TTTSNet/configs/config.json`, `TTTSNet/tools/train.py`, `TTTSNet/src/models/TTTSNet.py`, `TTTSNet/src/utils/custom_augmentations.py`

**实验目录**: `TTTSNet/experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/`
