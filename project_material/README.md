# TTTSNet 论文实验材料索引

本目录按研究 feature 整理 TTTSNet-Research v2.0 的模型设计与实验结论，用于支撑后续论文写作。每个文件独立成章，包含设计动机、方法、配置、结果和结论。

## 文件列表

| 编号 | 文件 | 主题 | 状态 |
|---|---|---|---|
| 01 | [`01_baseline_design_and_tuning.md`](01_baseline_design_and_tuning.md) | TTTSNet 单帧基线设计与调优 | 已完成主要诊断 |
| 02 | [`02_temporal_consistency.md`](02_temporal_consistency.md) | 时序一致性约束 | 诊断中 |
| 03 | [`03_pseudo_label_supervision.md`](03_pseudo_label_supervision.md) | 伪标签半监督学习 | 暂停，等待 baseline 做实 |
| 04 | [`04_future_model_enhancements.md`](04_future_model_enhancements.md) | 后续模型改进方向 | 规划中 |
| 05 | [`05_vit_backbone_sam_encoder.md`](05_vit_backbone_sam_encoder.md) | TTTSNet 使用 SAM ViT-B backbone | 已完成 100 epochs |
| 06 | [`06_aufl_experiment.md`](06_aufl_experiment.md) | AUFL 损失函数消融 | 已完成 30ep |
| 07 | [`07_next_steps_plan.md`](07_next_steps_plan.md) | 下一步实验计划 | 持续更新 |
| 08 | [`08_sam_random_points.md`](08_sam_random_points.md) | SAM + 随机点提示 | 已对齐 TTTS_SAM，v3 训练中 |
| 09 | [`09_experimental_conclusions_and_redo_plan.md`](09_experimental_conclusions_and_redo_plan.md) | 实验结论整理与重训计划 | 持续更新 |
| 09 | [`08_code_refactor_plan.md`](08_code_refactor_plan.md) | 代码重构计划 | 规划中 |
| 13 | [`13_data_resources_and_annotation_summary.md`](13_data_resources_and_annotation_summary.md) | 数据资源与标注方案总结 | 持续更新 |

## 当前关键结论

1. **Baseline** 当前 best val_mIoU = **0.5992**（基于旧 augmentation，需重训）。
2. **TTTSNetViT（SAM ViT-B backbone）** 最佳，best val_mIoU = **0.6191**，方向有效（建议修复 augmentation 后验证）。
3. **Temporal v3** best val_mIoU = **0.5978**，接近 baseline 但仍未超越（需重训确认）。
4. **SAM random points** 原全随机点 best val_mIoU = **0.5754**；TTTS_SAM A0.2.5 实为 GT mask 点监督上界 **0.6794**；TTTSNet v3 对齐训练进行中。
5. **AUFL** 0.5647，**loss 形式不是瓶颈**。
6. **Semi-supervised** 早期暂停，需按 Semi v2 方案重新设计伪标签质量控制。
7. **`custom_augmentations.py` 概率解析 bug 已修复**，此前 baseline/ViT/Temporal/Semi 实验均未真正启用自定义缺陷增强，需选择性重训。

完整结果对比见 [`experiment_summary.csv`](experiment_summary.csv)、[`all_experiments_curves.png`](all_experiments_curves.png) 和 [`09_experimental_conclusions_and_redo_plan.md`](09_experimental_conclusions_and_redo_plan.md)。

## 实验根目录

所有原始实验数据位于 `TTTSNet/experiments/`，配置文件位于 `TTTSNet/configs/`。

---

**最后更新**: 2026-06-23
