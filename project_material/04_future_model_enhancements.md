# 04 后续模型改进方向

## 4.1 当前状态总结

| 方向 | 当前结果 | 是否瓶颈 | 下一步 |
|---|---|---|---|
| Baseline | 0.5992 | 基石 | 继续提升到 0.64-0.66 |
| Temporal loss | 0.546 | 未提升 | T1 完成，执行 Temporal v3 |
| Semi-supervised | 0.463@10ep | 待优化 | 等待 baseline 稳定 |
| Loss (AUFL) | 0.565@30ep | 否 | 不再投入 |
| TTTSNetViT (SAM encoder) | **0.6191@60ep** | **有效提升** | **已跑满 100 epochs，见 05_vit_backbone_sam_encoder.md** |
| SAM random points | 0.5754@84ep | 未超过 TTTSNet | 已完成 100 epochs，见 08_sam_random_points.md |
| Temporal v3 | 0.5978@52ep | 接近 baseline | 已完成 100 epochs |

核心目标：先把 TTTSNet Single baseline 从 ~0.60 抬到可信、稳定、可复现的水平，再决定半监督/时序是否作为主贡献。

## 4.2 输入分辨率提升

### 4.2.1 动机

当前输入 448×448 可能对细血管分割不足。胎儿镜血管通常只有几个像素宽，resize 后容易断裂或模糊。

### 4.2.2 计划

| 实验 | 输入尺寸 | 预期 |
|---|---|---|
| Baseline-512 | 512×512 | 细血管保留更好 |
| Baseline-640 | 640×640 | 进一步提升细节，但计算成本增加 |

### 4.2.3 实施

- 修改 `dataset_config.img_size` 为 512 或 640
- 相应调整 batch size 以避免 OOM
- 保持其他设置与 baseline 一致

## 4.3 多 Seeds 基线

### 4.3.1 动机

当前 baseline 仅 seed=42。单 seed 结果可能受随机性影响，需要 mean±std 来确认稳定上限。

### 4.3.2 计划

训练 3 seeds（42, 2024, 3407），每个 100 epochs：
- 报告 mean ± std 的 val_mIoU 和 val_Dice
- 确认 0.60 附近是否为稳定平台

## 4.4 模型容量与预训练

### 4.4.1 动机

当前 TTTSNet 仅 5.3M 参数，且无预训练。在 2060 张有标注数据上可能容量不足。

### 4.4.2 方向

| 方向 | 说明 | 当前状态 |
|---|---|---|
| 更大 backbone | 尝试 ResNet-50 / EfficientNet-B3 作为编码器 | 未开始 |
| 预训练 ViT backbone | 使用 SAM 的 ViT-B encoder 替代原编码器 | **已完成 100 epochs，best val_mIoU=0.6191，见 05_vit_backbone_sam_encoder.md** |
| SAM 提示分割 | 使用 Segment Anything Model + 随机点提示 | 完整 100 epochs 训练中，2ep val_mIoU=0.343 |
| 更深的解码器 | 增加特征融合层数 | 未开始 |

### 4.4.3 观察

- TTTSNetViT 使用 SAM ViT-B 编码器，**100 epochs 后 best val_mIoU 达到 0.6191**，相对 baseline 提升约 2%，证明预训练 ViT backbone 有正向收益。
- SAM random points 完整 100 epochs 后 best val_mIoU 为 **0.5754**，低于 TTTSNet baseline 和 TTTSNetViT，说明完整 SAM 架构 + 弱随机点提示不适合此任务。
- Temporal v3 使用强同步增强后 best val_mIoU 达到 **0.5978**，接近 baseline，但仍未超越。

### 4.5.1 观察

- pixel acc 约 0.96 但 mIoU 约 0.60，说明背景分类容易，前景边界/细血管错误明显；
- 不同 video 间可能存在胎儿镜视角、光照、手术器械差异。

### 4.5.2 方向

| 方向 | 说明 |
|---|---|
| Video-aware 划分 | 确保 train/val 来自不同 video，减少数据泄露 |
| 域适应 | 对抗域适应或自监督预训练减少 video 间差异 |
| Hard negative mining | 针对边界和细血管样本加重训练 |

## 4.6 半监督方向深化

若 baseline 稳定后 semi 仍不提升，可考虑：

| 方向 | 方法 |
|---|---|
| 严格伪标签筛选 | mean confidence / top-k mean confidence / area filtering |
| 课程学习 | 早期只用 high-confidence pseudo labels，逐步放宽 |
| 一致性正则 | 对伪标签数据施加强/弱增强一致性约束 |
| FixMatch / Mean Teacher | 更先进的半监督框架 |

## 4.7 时序方向深化

若 T1 显示 dataset/增强是主因，可执行 temporal v3：

| 改进 | 说明 |
|---|---|
| 强同步增强 | 加入 CustomDefects、ShiftScaleRotate、Blur |
| 修正 confidence weight | 从 `p*(1-p)` 改为高置信权重 `abs(p-0.5)` |
| 多帧长度 | 尝试 5 帧或 7 帧 clip |
| 显式时序建模 | 在模型中加入 ConvLSTM 或时序注意力 |

## 4.8 推荐优先级

### 立即执行（T1 完成后）

1. 根据 T1 结果决定 temporal v3 是否值得跑
2. 尝试输入分辨率 512/640
3. 跑 3 seeds baseline

### 短期

1. 若分辨率/多 seeds 无显著提升，尝试更大 backbone + 预训练
2. 优化伪标签筛选策略

### 中期

1. 统一训练入口和 dataset 基类
2. 建立更系统的实验跟踪和论文图表流程

## 4.9 论文叙事建议

当前最稳妥的论文主线：

> **Robust Semi-supervised Fetoscopic Vessel Segmentation for TTTS with Pseudo-label Quality Control**

前提：
- baseline 稳定达到 0.64-0.66；
- semi-supervised 在严格质量控制后稳定超过 baseline 1-2 个点。

若 semi 无法超过 baseline，可转向：
- 强基线复现 + 医学增强分析；
- 或时序一致性的负结果诊断。

---

**相关规划文档**: `TTTSNet/.planning/ROADMAP.md`, `TTTSNet/.planning/STATE.md`
