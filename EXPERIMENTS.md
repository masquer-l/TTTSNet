# TTTSNet 实验结果总表

> 版本号规范：`TTTSNet-vX.Y.Z` + `EXP-XXX`
> 目录规范见 [EXPERIMENT_LAYOUT.md](EXPERIMENT_LAYOUT.md)
> 最后更新：2026-06-27

## 1. 关键结论速览

- **当前最强模型**：`TTTSNet-v2.1.0`（ViT backbone + 分层学习率）
  - 内部 Val：mIoU 0.6360–0.6405，Dice 0.7626–0.7670
  - SFY 测试：mIoU 0.6582–0.6625，Dice 0.7759–0.7766
- **最佳单一结果**：`EXP-010`（ViT + 分层学习率，seed 2024）内部 mIoU 0.6401
- **最有前景组合**：`EXP-015`（ViT + Transformer decoder + 分层学习率）内部 mIoU 0.6409，SFY 待评测
- **clDice 修正实验**：`EXP-017` 进行中

## 2. 主要结果总表

| EXP | 模型版本 | 描述 | Seed | Best Val mIoU | Best Val Dice | Best Epoch | SFY mIoU | SFY Dice | SFY clDice | 论文用途 | 原始目录 | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | TTTSNet-v1.0.0 | CNN baseline | 42 | 0.6049 | 0.7382 | 87 | 0.6275 | 0.7495 | 0.8080 | main baseline | [tttsnet_single_baseline_20260623_222644](experiments/tttsnet_single_baseline_20260623_222644) | completed |
| EXP-002 | TTTSNet-v1.0.0 | CNN baseline | 2024 | 0.6038 | 0.7371 | 81 | - | - | - | stability | [tttsnet_single_baseline_20260625_050905](experiments/tttsnet_single_baseline_20260625_050905) | completed |
| EXP-003 | TTTSNet-v1.0.0 | CNN baseline | 3407 | 0.5990 | 0.7316 | 93 | - | - | - | stability | [tttsnet_single_baseline_20260625_080901](experiments/tttsnet_single_baseline_20260625_080901) | completed |
| EXP-004 | TTTSNet-v1.3.0 | CNN + Temporal v3 | 42 | 0.6038 | 0.7377 | 72 | 0.5982 | 0.7271 | 0.7879 | ablation | [tttsnet_temporal_v3_20260624_043531](experiments/tttsnet_temporal_v3_20260624_043531) | completed |
| EXP-005 | TTTSNet-v2.0.0 | ViT backbone | 42 | 0.6191 | 0.7489 | 60 | 0.5707¹ | 0.7023¹ | 0.7850¹ | ablation | [tttsnet_vit_backbone_20260620_155115](experiments/tttsnet_vit_backbone_20260620_155115) | completed |
| EXP-006 | TTTSNet-v2.0.0 | ViT backbone quick (50ep) | 42 | 0.5961 | 0.7310 | 44 | 0.5707 | 0.7023 | 0.7850 | ablation | [tttsnet_vit_backbone_20260624_020542](experiments/tttsnet_vit_backbone_20260624_020542) | completed |
| EXP-007 | TTTSNet-v2.0.0 | ViT backbone | 42 | 0.5929 | 0.7284 | 67 | - | - | - | stability | [tttsnet_vit_backbone_20260625_112206](experiments/tttsnet_vit_backbone_20260625_112206) | completed |
| EXP-008 | TTTSNet-v2.0.0 | ViT backbone | 2024 | 0.6208 | 0.7510 | 63 | - | - | - | stability | [tttsnet_vit_backbone_20260625_153352](experiments/tttsnet_vit_backbone_20260625_153352) | completed |
| EXP-009 | TTTSNet-v2.1.0 | ViT + Layerwise LR | 42 | 0.6360 | 0.7626 | 83 | 0.6625 | 0.7766 | 0.8374 | **main result** | [tttsnet_vit_layerwise_lr_20260626_095855](experiments/tttsnet_vit_layerwise_lr_20260626_095855) | completed |
| EXP-010 | TTTSNet-v2.1.0 | ViT + Layerwise LR | 2024 | 0.6401 | 0.7660 | 44 | 0.6582 | 0.7759 | 0.8255 | **main result** | [tttsnet_vit_layerwise_lr_20260626_203144](experiments/tttsnet_vit_layerwise_lr_20260626_203144) | completed |
| EXP-011 | TTTSNet-v2.1.0 | ViT + Layerwise LR | 3407 | 0.6405 | 0.7670 | 61 | 0.6466 | 0.7644 | 0.8372 | stability | [tttsnet_vit_layerwise_lr_20260627_083345](experiments/tttsnet_vit_layerwise_lr_20260627_083345) | completed |
| EXP-012 | TTTSNet-v3.0.0 | Transformer decoder (50ep) | 42 | 0.5989 | 0.7328 | 48 | 0.5999 | 0.7312 | 0.8082 | ablation | [tttsnet_transformer_decoder_20260624_080532](experiments/tttsnet_transformer_decoder_20260624_080532) | completed |
| EXP-013 | TTTSNet-v3.0.1 | Transformer decoder v2 | 42 | 0.6162 | 0.7479 | 77 | 0.6218 | 0.7455 | 0.8123 | ablation | [tttsnet_transformer_decoder_v2_20260626_095857](experiments/tttsnet_transformer_decoder_v2_20260626_095857) | completed |
| EXP-014 | TTTSNet-v4.2.0 | ViT + Transformer + clDice (50ep, bug) | 42 | 0.5935 | 0.7293 | 49 | 0.5904 | 0.7206 | 0.7927 | deprecated | [tttsnet_vit_transformer_cldice_20260624_100608](experiments/tttsnet_vit_transformer_cldice_20260624_100608) | completed |
| EXP-015 | TTTSNet-v4.1.0 | ViT + Transformer decoder + Layerwise LR | 42 | 0.6409 | 0.7669 | 61 | 0.6433 | 0.7614 | 0.8324 | main result | [tttsnet_vit_transformer_decoder_layerwise_lr_20260627_083221](experiments/tttsnet_vit_transformer_decoder_layerwise_lr_20260627_083221) | completed |
| EXP-016 | TTTSNet-v4.4.0 | ViT + Transformer + Semi-supervised v2 | 42 | 0.5955 | 0.7306 | 69 | 0.0106 | 0.0106 | 0.5178 | deprecated | [tttsnet_semi_v2_20260624_130832](experiments/tttsnet_semi_v2_20260624_130832) | completed |
| EXP-017 | TTTSNet-v2.2.0 | ViT + Layerwise LR + clDice (fixed) | 42 | 0.6417 | 0.7682 | 71 | 0.6512 | 0.7673 | 0.8413 | ablation | [tttsnet_vit_layerwise_lr_cldice_20260627_162809](experiments/tttsnet_vit_layerwise_lr_cldice_20260627_162809) | completed |

## 3. 模型版本说明

| 模型版本 | 架构 | 关键改动 | 代表性 EXP |
|---|---|---|---|
| TTTSNet-v1.0.0 | CNN baseline | 原版 TTTSNet，修复 augmentation bug 后重训 | EXP-001 ~ EXP-003 |
| TTTSNet-v1.3.0 | CNN + Temporal | 3 帧时序一致性 | EXP-004 |
| TTTSNet-v2.0.0 | ViT backbone | SAM ViT-B 替换 Init_Block | EXP-005 ~ EXP-008 |
| TTTSNet-v2.1.0 | ViT + Layerwise LR | 对 `vit_encoder` 使用 0.1x 学习率 | EXP-009 ~ EXP-011 |
| TTTSNet-v2.2.0 | ViT + Layerwise LR + clDice | 修正 clDice 各向异性腐蚀后测试 | EXP-017 |
| TTTSNet-v3.0.0 | Transformer decoder | 在 CNN 后接轻量 Transformer | EXP-012 |
| TTTSNet-v3.0.1 | Transformer decoder v2 | 4 层 + pos embed + pooled_size=56 | EXP-013 |
| TTTSNet-v4.1.0 | ViT + Transformer + Layerwise LR | 组合最强组件 | EXP-015 |
| TTTSNet-v4.2.0 | ViT + Transformer + clDice | 含 bug 的 clDice 早期尝试 | EXP-014 |
| TTTSNet-v4.4.0 | ViT + Transformer + Semi | 半监督 v2 | EXP-016 |

## 4. 多 Seed 稳定性（ViT + Layerwise LR）

| Seed | 内部 Val mIoU | 内部 Val Dice | SFY mIoU | SFY Dice |
|---|---|---|---|---|
| 42 | 0.6360 | 0.7626 | 0.6625 | 0.7766 |
| 2024 | 0.6401 | 0.7660 | 0.6582 | 0.7759 |
| 3407 | 0.6405 | 0.7670 | 0.6466 | 0.7644 |
| **均值 ± 标准差** | **0.6389 ± 0.0025** | **0.7652 ± 0.0024** | **0.6524 ± 0.0080** | **0.7723 ± 0.0062** |

## 5. 效率对比

数据来源：`experiments/efficiency_statistics.csv`

| 模型 | 配置 | 参数量 (M) | 推理延迟 (ms) | 输入尺寸 |
|---|---|---|---|---|
| TTTSNet | config.json | 5.31 | 12.12 | 448 |
| TTTSNetViT | config_vit_backbone.json | 94.92 | 17.88 | 448 |
| TTTSNet (Temporal) | config_temporal_v3.json | 5.31 | 6.36 | 448 |
| TTTSNetTransformerDecoder | config_transformer_decoder.json | 5.57 | 6.90 | 448 |
| TTTSNetViTTransformerDecoder | config_vit_transformer_cldice.json | 95.19 | 18.50 | 448 |
| TTTSNetViTTransformerDecoder | config_semi_v2.json | 95.19 | 18.45 | 448 |

## 6. 废弃/参考实验

以下实验无完整 `epoch_history.csv` 或已被后续版本完全替代，不分配 EXP 编号，仅作追溯参考：

| 目录 | 说明 | 状态 |
|---|---|---|
| `tttsnet_single_20260619_233051` | 最早 CNN baseline 尝试 | 无记录 |
| `tttsnet_aufl_30ep_20260620_103645` | AUFL loss 30ep 快速尝试 | 无完整记录 |
| `tttsnet_temporal_20260620_013149` | Temporal 早期尝试 | 无完整记录 |
| `tttsnet_temporal_v2_20260620_021245` | Temporal v2 | 无完整记录 |
| `tttsnet_temporal_no_loss_20260620_114901` | Temporal 消融（loss=0） | 无完整记录 |
| `tttsnet_temporal_v3_20260620_213745` | Temporal v3 早期尝试 | 无完整记录 |
| `tttsnet_semi_20260620_085333` | 早期半监督尝试 | 无完整记录 |
| `tttsnet_semi_v2_20260624_130909` | Semi v2 重复/失败 | 无 epoch_history |
| `sam_random_points_20260620_140230` | SAM random points (TTTSNet 早期对齐) | 仅 2 epoch |
| `A0_2_5_random_points_aligned_*` | TTTS_SAM A0.2.5 对齐实验 | 见 TTTS_SAM 文档 |

## 7. 产物路径索引

### 7.1 主结果模型

- **EXP-009 (TTTSNet-v2.1.0, seed 42)**
  - 目录：`experiments/tttsnet_vit_layerwise_lr_20260626_095855/`
  - 最佳 checkpoint：`checkpoints/best_model.pth`
  - SFY 结果：`sfy_results/summary.json`
  - 日志：`training.log`

- **EXP-010 (TTTSNet-v2.1.0, seed 2024)**
  - 目录：`experiments/tttsnet_vit_layerwise_lr_20260626_203144/`
  - 最佳 checkpoint：`checkpoints/best_model.pth`
  - SFY 结果：`sfy_results/summary.json`
  - 日志：`training.log`

- **EXP-015 (TTTSNet-v4.1.0, seed 42)**
  - 目录：`experiments/tttsnet_vit_transformer_decoder_layerwise_lr_20260627_083221/`
  - 最佳 checkpoint：`checkpoints/best_model.pth`
  - 日志：`training.log`

## 8. 后续待办

- [x] 完成 EXP-017（ViT + Layerwise LR + clDice fixed）训练与 SFY 评测
- [x] 为 EXP-015 运行 SFY 评测
- [x] 为 EXP-011（seed 3407）运行 SFY 评测
- [ ] 生成效率对比表与可视化
- [ ] 根据最终指标更新论文实验表格

## 脚注

¹ SFY 结果来自 `vit_backbone_50ep` checkpoint，100ep 模型未单独在 SFY 上评测。
