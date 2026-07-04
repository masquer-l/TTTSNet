# 05 TTTSNet 使用 SAM ViT-B 作为预训练 Backbone

## 5.1 动机

TTTSNet 原版的 `Init_Block` 是一个轻量的 3 层卷积 stem（无预训练），参数规模小、感受野有限。胎儿镜血管分割面临细血管、低对比度、手术器械遮挡等挑战，预训练的大规模视觉 backbone 可能提供更丰富的特征表达。

本实验在**最小改动**原则下，将 TTTSNet 的 `Init_Block` 替换为 SAM（Segment Anything Model）的 ViT-B image encoder，保留 TTTSNet 的后续模块（RFFM-A/B、SEM-B blocks、MAD）不变，以隔离评估 "ViT backbone / 预训练" 带来的收益。

## 5.2 模型设计

### 5.2.1 改动点

- **Backbone 替换**：`src/models/tttsnet_vit.py`
  - 加载 SAM ViT-B image encoder（`sam_vit_b_01ec64.pth`）
  - 冻结/微调可配置（本实验 `freeze_vit=false`，端到端微调）
  - 将 ViT-B 输出 `[B, 256, H/16, W/16]` 通过 `ViTAdapter`（1×1 conv + 8× 上采样）映射到 `[B, 64, H/2, W/2]`
- **后续结构不变**：`RFFM_A`、`RFFM_B`、`SEM_B_Block`、`MAD` 均复用原版实现
- **输入尺寸不变**：448×448，与 baseline 完全一致
- **预处理适配**：TTTSNet dataset 输出 `[0,1]`，模型 forward 内转换为 SAM 的 mean/std 归一化空间

### 5.2.2 位置编码插值

SAM ViT-B 的位置编码在 1024×1024 上预训练。448×448 输入下 patch 数为 28×28，运行时将预训练 pos_embed 双线性插值到当前尺寸，保证 forward 可运行。

### 5.2.3 总损失

与 baseline 保持一致：

```
L_total = 1.0 × DiceLoss + 1.0 × BCELoss + 0.5 × CrossEntropyLoss
```

## 5.3 实验配置

| 配置项 | 值 |
|---|---|
| 模型 | `TTTSNetViT` |
| Backbone checkpoint | `/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth` |
| 冻结 backbone | false |
| 输入尺寸 | 448×448 |
| Batch size | 4 |
| Optimizer | AdamW |
| LR | 1e-4 |
| Scheduler | CosineAnnealingLR，T_max=100 |
| Epochs | 100 |
| Seed | 42 |
| Loss | Dice + BCE + 0.5×CE |

配置文件：`TTTSNet/configs/config_vit_backbone.json`

训练入口：`TTTSNet/tools/train.py --config configs/config_vit_backbone.json`

## 5.4 实验结果

### 5.4.1 主要结果

| 实验 | Best val_mIoU | Best Epoch | Best val_Dice | 总耗时 |
|---|---|---|---|---|
| TTTSNet baseline | 0.5992 | 66 | 0.732 | ~1.6h |
| **TTTSNetViT (SAM ViT-B backbone)** | **0.6191** | 60 | **0.7489** | **2.8h** |

### 5.4.2 关键观察

- **正向收益**：ViT backbone 相对 baseline 提升约 **2.0% mIoU** 和 **1.7% Dice**。
- **收敛更快**：best epoch 为 60，早于 baseline 的 66。
- **训练稳定**：100 epochs 无 NaN/Inf，train loss 从 0.95 降至 0.14。
- **最终轻微过拟合**：epoch 60 后 val_mIoU 稳定在 0.61 附近，未继续上升。

### 5.4.3 曲线摘要

| Epoch | Train Loss | Val mIoU | Val Dice | Val Pixel Acc |
|---|---|---|---|---|
| 1 | 0.953 | 0.312 | 0.436 | 0.935 |
| 10 | 0.419 | 0.554 | 0.693 | 0.956 |
| 20 | 0.341 | 0.573 | 0.708 | 0.958 |
| 30 | 0.298 | 0.597 | 0.728 | 0.958 |
| 40 | 0.254 | 0.598 | 0.731 | 0.960 |
| 50 | 0.223 | 0.611 | 0.741 | 0.961 |
| **60** | **0.195** | **0.619** | **0.749** | **0.962** |
| 70 | 0.172 | 0.612 | 0.742 | 0.962 |
| 80 | 0.155 | 0.611 | 0.742 | 0.962 |
| 90 | 0.147 | 0.612 | 0.742 | 0.962 |
| 100 | 0.145 | 0.612 | 0.742 | 0.962 |

完整曲线见实验目录：`TTTSNet/experiments/tttsnet_vit_backbone_20260620_155115/epoch_history.csv`

![TTTSNetViT 训练曲线](vit_backbone_results/vit_backbone_curves.png)

## 5.5 与 Baseline 的对比分析

| 维度 | Baseline | TTTSNetViT |
|---|---|---|
| Backbone | 3 层卷积 Init_Block（无预训练） | SAM ViT-B（ImageNet-1K + SA-1B 预训练） |
| 可训练参数量 | ~5.3M | ~91M（ViT-B 89M + TTTSNet 解码器 ~2M） |
| 输入尺寸 | 448×448 | 448×448 |
| Loss | Dice+BCE+CE | Dice+BCE+CE |
| Best val_mIoU | 0.5992 | 0.6191 (+0.0199) |
| Best val_Dice | 0.732 | 0.749 (+0.017) |
| 训练时间/epoch | ~55s | ~95s |

**判断**：ViT backbone 带来了正向、稳定的提升，证明预训练 + 更大感受野对细血管分割有帮助。但提升幅度有限（约 2 个点），说明 backbone 不是唯一瓶颈。

## 5.6 结论与下一步

### 当前结论

1. **TTTSNetViT 是有效的改进方向**：best val_mIoU 达到 **0.6191**，超过 baseline。
2. **收益主要来自网络容量和预训练权重**，而非 TTTSNet 架构本身的问题。
3. **提升空间有限**：2% 的增益说明仍需在数据增强、解码器设计、输入分辨率等方面继续优化。

### 待验证方向

| 方向 | 说明 | 配置 |
|---|---|---|
| 冻结 ViT-B 只训练解码器 | 验证微调 vs frozen 的差异 | `freeze_vit=true` |
| 更大输入分辨率 | 448 → 512/640，看细血管收益 | 调整 `img_size` |
| 输入分辨率对齐 SAM | 1024×1024 + 调整 TTTSNet 解码器 | 需重新设计 adapter |
| 多 seeds | 验证 0.619 是否稳定 | seed=2024, 3407 |

### 论文定位

TTTSNetViT 可作为**消融实验**或**改进点之一**：
- 若后续 semi/temporal 能在此基础上进一步提升，可共同构成论文贡献；
- 若单独作为贡献，2% 的增益建议配合分析（冻结/微调、分辨率、可视化）增强说服力。

---

**相关文件**:
- `TTTSNet/src/models/tttsnet_vit.py`
- `TTTSNet/tools/train.py`
- `TTTSNet/configs/config_vit_backbone.json`

**实验目录**: `TTTSNet/experiments/tttsnet_vit_backbone_20260620_155115/`

---

**最后更新**: 2026-06-20
