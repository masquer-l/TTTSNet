# 当前实验结论与续接状态

**最后更新**: 2026-06-25

## 1. 已完成的监督/消融实验

| 实验 | 配置/模型 | Best val_mIoU | Best val_Dice | 最佳 epoch | 备注 |
|---|---|---|---|---|---|
| Baseline redo | `configs/config.json`, seed=42 | **0.6049** | **0.7382** | 87 | 修复 augmentation bug 后的 anchor |
| TTTSNetViT 50ep | `config_vit_backbone.json` | 0.5961 | 0.7310 | 44 | 仅 50 epochs 快速验证 |
| Temporal v3 重训 | `config_temporal_v3.json` | 0.6038 | 0.7377 | 72 | 时序 loss 未超越 baseline |
| Transformer decoder ablation | `config_transformer_decoder.json` | 0.5989 | 0.7328 | 48 | 50 epochs |
| clDice final candidate | `config_vit_transformer_cldice.json` | 0.5935 | 0.7293 | 49 | 50 epochs，clDice weight=0.2 |
| Semi-supervised v2 | `config_semi_v2.json` | 0.5955 | 0.7306 | 69 | 100 epochs，39k 伪标签，未超越 baseline |

## 2. 多 seeds 稳定性验证（进行中）

由外部脚本 PID `647508` 顺序执行：

| 模型 | seed 42 | seed 2024 | seed 3407 | 状态 |
|---|---|---|---|---|
| Baseline (TTTSNet) | 0.6049 / 0.7382 | 0.6038 / 0.7371 | 0.5990 / 0.7317 | ✅ 完成 |
| TTTSNetViT | 0.5929 / 0.7284 | 🔄 进行中 | ⏳ 待执行 | 🔄 序列执行中 |

## 3. 关键结论

1. **修复 augmentation bug 后，baseline 稳中有升**：从旧 0.5992 提升到 0.6049。
2. **ViT backbone 方向性有效，但 50ep 未完全收敛**：50ep 0.5961 < baseline，旧 100ep 结果为 0.6191。
3. **Temporal v3 仍未证明超越单帧**：0.6038 接近 baseline 但略低。
4. **Transformer decoder 与 clDice 在该配置下未带来提升**：均低于 baseline。
5. **Semi-supervised v2 未能在当前伪标签策略下超越纯监督 baseline**：best 0.5955 < 0.6049。
6. **多 seeds 显示 baseline 较稳定**：3 seeds 在 0.5990–0.6049 之间波动。

## 4. 当前进行中的工作

- **多 seeds 序列**: PID `647508`，当前运行 ViT seed 2024（约 epoch 3/100，预计还需 3–3.5 小时）。
- 已安排 **18:42** 自动检查多 seeds 进度。

## 5. 待完成工作

- 完成 ViT seeds 2024、3407（多 seeds 验证）。
- SFY 0923 外部评测集泛化性评估（Task #6，状态显示已完成，但需确认是否有输出文件）。
- 模型效率统计（Task #9，状态显示已完成，但需确认输出文件）。
- 更新论文实验表格与叙事。

## 6. 续接指南

新 session 启动后：
1. 读取本文件获取当前结论。
2. 检查 PID `647508` 是否仍在运行。
3. 读取 `experiments/tttsnet_vit_backbone_20260625_153352/epoch_history.csv` 确认 ViT seed 2024 进度。
4. 按 cron 或手动继续检查多 seeds 序列，直至完成。
5. 验证 SFY 评测与效率统计输出是否存在；如缺失，补跑。

---
**相关文件**:
- 主计划：`project_material/09_experimental_conclusions_and_redo_plan.md`
- Paper pipeline：`TTTSNet/experiments/paper_pipeline_status.md`
- 任务列表：Claude task list (#1–#9)
