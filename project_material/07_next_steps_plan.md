# 07 下一步实验计划

> **注意**：2026-06-23 发现 `custom_augmentations.py` 概率解析 bug，导致此前 baseline、ViT、Temporal、Semi 实验均未真正启用自定义缺陷增强。完整结论与重训优先级已整理到 [`09_experimental_conclusions_and_redo_plan.md`](09_experimental_conclusions_and_redo_plan.md)。本文档保留原始计划作为参考。

## 7.1 当前结果总览

| 实验 | Best val_mIoU | Best val_Dice | 状态 | 结论 |
|---|---|---|---|---|
| TTTSNet Single baseline | 0.5992 | 0.7324 | ⚠️ 需重训（augmentation bug） | 当前基准 |
| TTTSNetViT (SAM ViT-B backbone) | **0.6191** | **0.7489** | ⚠️ 建议验证（augmentation bug） | 有效提升 (+2.0% mIoU) |
| Temporal v3 | 0.5978 | 0.7320 | ⚠️ 需重训（augmentation bug） | 接近 baseline，仍未超越 |
| SAM random points | 0.5754 / 0.6794* | 0.7147 / 0.8032* | 🔄 v3 对齐中 | 0.575 为真正随机点；0.679 为 TTTS_SAM GT-mask 点监督上界 |
| AUFL 30ep | 0.5647 | 0.7037 | ✅ 完成 | 未超过 baseline |
| Temporal no-loss (T1) | 0.5541 | 0.6948 | ✅ 完成 | dataset/增强差异是主因 |
| Temporal v1 (λ=0.1) | 0.5491 | 0.6917 | ✅ 完成 | 低于 baseline |
| Temporal v2 (λ=1.0) | 0.5459 | 0.6886 | ✅ 完成 | 低于 baseline |
| Semi-supervised (10ep) | 0.4630 | 0.5950 | ⏸️ 暂停 / 需重设计 | 筛选过宽，噪声主导 |

\* 0.6794/0.8032 来自 TTTS_SAM A0.2.5 复现，实为 GT mask 采样点监督上界。

完整曲线见：`project_material/all_experiments_curves.png`

---

## 7.2 推荐执行顺序

基于当前证据和 tokens/时间成本，推荐按以下优先级推进：

### P0：半监督优化（伪标签质量控制）

- **动机**：当前所有架构探索中，**TTTSNetViT（0.6191）是最佳起点**。半监督有潜力在此基础上进一步提升。
- **关键改进**：
  1. **严格筛选策略**：
     - 用 `mean_confidence ≥ 0.85` 替代 `max_conf ≥ 0.9`
     - 或 `top-5% mean confidence ≥ 0.9`
  2. **面积过滤**：
     - 过滤空 mask
     - 过滤血管面积占比 < 2% 或 > 30% 的异常样本
  3. **固定 batch 比例**：
     - 每个 batch 中 labeled:pseudo = 1:1 或 1:2
  4. **Weighted Dice**：
     - 让 Dice loss 也支持 per-sample weighting，与 BCE/CE 一致
- **配置**：新增 `configs/config_semi_v2.json`
- **训练入口**：`tools/train_semi.py --config configs/config_semi_v2.json`
- **成功标准**：在 TTTSNetViT backbone（0.619）基础上再提升 1-2 个点。

### P1：多 Seeds 基线

- **动机**：当前所有结果基于 seed=42，需要确认稳定性。
- **计划**：对 TTTSNetViT 训练 3 seeds（42, 2024, 3407）。
- **输出**：mean ± std 的 val_mIoU / val_Dice。
- **成本**：3 × 2.8h = 8.4h，可在夜间并行跑。

### P2：输入分辨率提升

- **动机**：细血管在 448×448 下可能丢失细节。
- **方案**：
  - Baseline-512：img_size=512，batch size 降至 2-3
  - Baseline-640：img_size=640，batch size 降至 1-2
- **优先级**：P1 之后，若多 seeds 显示稳定平台再尝试。

### P3：统一训练入口与 dataset 基类

- **动机**：当前 `train.py`, `train_temporal.py`, `train_semi.py`, `train_sam_random.py` 有大量重复代码。
- **方案**：
  1. 抽象 `BaseTrainer` 基类
  2. 统一 `create_dataloaders`, `create_optimizer`, `create_scheduler`
  3. 各任务继承基类，只覆盖 `create_model`, `train_step`, `val_step`
- **收益**：减少维护成本，降低新实验接入门槛。

### 已完成的 Temporal v3 / SAM random points

- **Temporal v3**：best val_mIoU = 0.5978，接近 baseline 但未超越。强同步增强有效恢复了 Temporal 性能，但时序 loss 本身未证明能超越单帧 baseline。作为消融/负向分析。
- **SAM random points**：best val_mIoU = 0.5754，完整 SAM 架构 + 随机点提示未超过 TTTSNet 系列。作为 negative result / 消融对比。

---

## 7.3 论文叙事建议

当前最有潜力的主线：

> **Robust Semi-supervised Fetoscopic Vessel Segmentation for TTTS with Pseudo-Label Quality Control**

前提条件：
1. TTTSNetViT 多 seeds 下稳定在 0.61-0.62；
2. 优化后的半监督（P0）在 TTTSNetViT 上稳定提升 1-2 个点；
3. Temporal v3 能作为消融实验说明时序/增强的影响。

若半监督无法提升，可转向：
- 强 baseline 复现 + 医学图像增强分析；
- 或时序一致性的负结果诊断。

---

## 7.4 资源与时间估算

| 实验 | 时间 | 备注 |
|---|---|---|
| Semi v2（严格筛选） | ~4h | 含伪标签重新生成 |
| 多 seeds ViT (×3) | ~8h | 可并行 |
| 分辨率 512/640 | ~3-5h each | 视 batch size |
| 统一训练入口 | ~2h | 开发 + 验证 |

---

**最后更新**: 2026-06-23
