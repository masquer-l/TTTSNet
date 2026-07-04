# 09 实验结论整理与重训计划

**最后更新**: 2026-06-23

## 1. executive summary

最近两周围绕 TTTSNet 的实验可以归纳为三类：

| 类别 | 代表实验 | 核心结论 |
|---|---|---|
| 架构/预训练 | TTTSNetViT | **预训练 ViT backbone 有效**，方向正确 |
| 损失/提示 | AUFL、SAM random points | **损失形式不是瓶颈**；**真正无提示的随机点太弱** |
| 数据/增强 | Temporal v1/v2/v3、Semi-supervised | **增强 pipeline 和数据质量是关键**；时序 loss 本身未证明超越单帧 |
| 外部基准 | TTTS_SAM A0.2.5 random points | **实为 GT mask 点监督上界（0.679）**，不是无人工提示结果 |

同时发现 `custom_augmentations.py` 存在概率解析 bug，导致此前所有 TTTSNet 实验（baseline、ViT、Temporal、Semi）**都没有真正启用自定义缺陷增强**。因此部分实验需要基于修复后的 pipeline 重训，才能获得公平、可复现的结论。

---

## 2. 各实验结果与结论

### 2.1 TTTSNet Single baseline

| 指标 | 当前值 | 状态 |
|---|---|---|
| Best val_mIoU | 0.5992 | 完成（基于旧 augmentation） |
| Best val_Dice | 0.7324 | 完成（基于旧 augmentation） |

**结论**：当前 baseline 是后续所有对比的锚点。

**是否需要重做**：**需要**。原因：
- `custom_augmentations.py` 的 bug 使得 `AddDustParticles / AddLaserPointer / AddOpticalFiber / AddStructuralDefects` 实际未生效。
- 修复后增强强度可能变化，baseline 上限需要重新确认。
- 只有 baseline 做实了，才能判断 ViT / Temporal / Semi 的真实增益。

**建议**：
- 使用 `configs/config.json`（建议后续改名为 `config_baseline.json`）跑满 100 epochs。
- 可同时跑 3 seeds（42, 2024, 3407）确认稳定性。

---

### 2.2 TTTSNetViT（SAM ViT-B backbone）

| 指标 | 当前值 | 状态 |
|---|---|---|
| Best val_mIoU | **0.6191** | 完成（基于旧 augmentation） |
| Best val_Dice | 0.7489 | 完成（基于旧 augmentation） |

**结论**：SAM ViT-B backbone 相对 baseline 提升约 **2.0% mIoU / 1.7% Dice**，是截至目前最有效的改进方向。

**是否需要重做**：**建议验证，视情况决定是否重跑完整 100 epochs**。原因：
- 同样受 augmentation bug 影响，真实收益可能略有浮动。
- 但 ViT backbone 的有效性逻辑很硬，方向性结论大概率成立。

**建议**：
- 先跑 30-50 epochs 快速验证修复 augmentation 后的趋势；
- 若仍稳定超过 baseline，再决定是否补跑完整 100 epochs 替换论文数据。

---

### 2.3 Temporal v3

| 指标 | 当前值 | 状态 |
|---|---|---|
| Best val_mIoU | 0.5978 | 完成（基于旧 augmentation） |
| Best val_Dice | 0.7320 | 完成（基于旧 augmentation） |

**结论**：强同步增强把 Temporal 性能拉回到接近 baseline，但时序一致性 loss 本身仍未证明能超越单帧。

**是否需要重做**：**需要**。原因：
- 受 augmentation bug 影响，且 Temporal 对增强同步非常敏感。
- 修复后需要确认“时序不提升”的结论是否仍然成立。

**建议**：
- 基于修复后的 `custom_augmentations.py` 重跑 `configs/config_temporal_v3.json`。
- 若仍不超越 baseline，可作为消融/负向结果写入论文。

---

### 2.4 SAM random points

| 实验 | Best val_mIoU | Best val_Dice | 说明 |
|---|---|---|---|
| TTTSNet SAM random（原始全随机点） | 0.5754 | 0.7147 | 真正的无 GT 提示随机点 |
| TTTS_SAM A0.2.5 "random points" | **0.6794** | **0.8032** | 实为 GT mask 采样点监督上界 |
| TTTSNet SAM random aligned v3 | 进行中 | 进行中 | GT mask 采样 + TTTS_SAM scheduler 模式 |

**结论**：
- TTTS_SAM A0.2.5 的 `points_sample_mode=RANDOM` + `prompt_with_gt=1` 是从 GT mask 随机采样 10 前景 + 10 背景点，标签 0/1 混合。
- 因此 **0.679 应视为点监督上界**，不是“无人工提示”结果。
- 真正的整图随机点（TTTSNet 原始实现）只能到 0.575，说明**高质量提示对 SAM 很关键**。

**是否需要重做**：
- **v3 正在跑**（PID 124913），用于在 TTTSNet 中复现该上界。
- 原始全随机点实验（0.575）结论可复用，作为 negative result。

---

### 2.5 AUFL

| 指标 | 当前值 | 状态 |
|---|---|---|
| Best val_mIoU | 0.5647 | 30 epochs（基于旧 augmentation） |
| Best val_Dice | 0.7037 | 30 epochs（基于旧 augmentation） |

**结论**：AUFL 30 epochs 未超过 baseline 30 epochs 水平，损失函数形式不是主要瓶颈。

**是否需要重做**：**可选，低优先级**。原因：
- 结论方向性（loss 不是瓶颈）大概率成立。
- 但若论文要引用精确数字，建议用修复后的 pipeline 补跑 30 epochs 确认。

---

### 2.6 Semi-supervised

| 指标 | 当前值 | 状态 |
|---|---|---|
| Best val_mIoU | 0.4630 | 10 epochs 暂停（基于旧 augmentation + 宽松筛选） |
| Best val_Dice | 0.5950 | 10 epochs 暂停 |

**结论**：早期 Semi 因伪标签筛选过宽、噪声主导而失败。

**是否需要重做**：**需要，且需重新设计**。原因：
- augmentation bug 影响训练稳定性。
- 需要引入更严格的伪标签质量控制（Semi v2）。

**建议**：
- 基于修复后的 `custom_augmentations.py` 实现 Semi v2：
  1. `mean_confidence ≥ 0.85` 或 `top-5% mean confidence ≥ 0.9`
  2. 过滤空 mask 和异常面积（<2% 或 >30%）
  3. labeled:pseudo = 1:1 或 1:2
  4. Dice loss 支持 per-sample weighting
- 若 Semi v2 在 TTTSNetViT 0.619 基础上再提升 1-2 个点，可作为主贡献。

---

## 3. 哪些结论可以直接复用

| 结论 | 依据 | 复用方式 |
|---|---|---|
| TTTSNetViT 优于 baseline | 0.6191 vs 0.5992 | 方向性结论可用；精确数字建议修复 augmentation 后确认 |
| 损失函数不是瓶颈 | AUFL 未超越 baseline | 可直接作为消融/负向结论 |
| 真正无提示随机点太弱 | SAM fully-random 0.575 < TTTSNetViT | 可作为 negative result |
| TTTS_SAM A0.2.5 是点监督上界 | 代码分析 + 0.679 结果 | 直接写入论文作为基准讨论 |
| 时序 loss 本身未证明超越单帧 | Temporal v1/v2/v3 均 ≤ baseline | 方向性可用；建议修复 augmentation 后确认 |
| 数据增强/伪标签质量是关键 | Temporal no-loss 与 v3 的差距、Semi 失败 | 可直接支撑论文叙事 |

---

## 4. 建议的重训与后续计划

### 4.1 立即执行

| 优先级 | 实验 | 命令/配置 | 目的 | 预计时间 |
|---|---|---|---|---|
| P0 | SAM random aligned v3 | `tools/train_sam_random.py --config configs/config_sam_random.json` | 得到 TTTSNet 中的点监督上界 | ~13h（已启动 PID 124913） |
| P1 | TTTSNet baseline 重训 | `tools/train.py --config configs/config.json` | 确认修复 augmentation 后的 baseline | ~1.6h |
| P2 | TTTSNetViT 快速验证 | `tools/train.py --config configs/config_vit_backbone.json --num_epochs 50` | 确认 ViT 优势是否仍成立 | ~1.4h |

### 4.2 短期执行

| 优先级 | 实验 | 说明 | 预计时间 |
|---|---|---|---|
| P3 | Temporal v3 重训 | 修复 augmentation 后确认时序结论 | ~2.7h |
| P4 | Semi v2 | 严格伪标签质量控制 | ~4h |
| P5 | Multi-seed baseline / ViT | 42, 2024, 3407 | ~8h（可并行） |

### 4.3 中期可选

| 优先级 | 实验 | 说明 |
|---|---|---|
| P6 | 输入分辨率 512/640 | 若 baseline 稳定后仍想提升细血管 |
| P7 | AUFL 30ep 补跑 | 若需要修复 augmentation 后的精确数字 |
| P8 | 统一训练入口/BaseTrainer | 见 `08_code_refactor_plan.md` |

---

## 5. 对论文叙事的影响

当前最稳妥的论文主线仍为：

> **Robust Semi-supervised Fetoscopic Vessel Segmentation for TTTS with Pseudo-Label Quality Control**

但需要把以下发现整合进去：

1. **预训练 ViT backbone 是有效的上游改进**（TTTSNetViT）。
2. **点监督上界很高**（SAM GT-mask random points 0.679），说明任务本身在有好提示时可达较高精度；但临床场景下无法依赖 GT 点，因此需要半监督/质量控制来逼近这个上界。
3. **真正无提示的随机点无法达到上界**（0.575），进一步说明自动/半自动提示生成的必要性。
4. **时序一致性 loss 不是灵丹妙药**，数据增强和伪标签质量才是。

如果 Semi v2 无法在 ViT backbone 上稳定提升，备选叙事：
- 强 baseline（ViT backbone + 医学增强分析）+ 点监督上界分析；
- 或时序一致性的系统负结果诊断。

---

## 6. 需要注意的代码问题

- `TTTSNet/src/utils/custom_augmentations.py` 中所有自定义 defect transform 的 `super().__init__` 调用已修复。
  - 旧代码：`super().__init__(always_apply, p)` 会把 `always_apply` 传给 `ImageOnlyTransform.__init__(self, p=0.5)`，导致 `p=False`，自定义缺陷增强实际被禁用。
  - 新代码：`super().__init__(p=1.0 if always_apply else p)`。
- 所有旧实验（baseline、ViT、Temporal、Semi）都是在“未启用自定义缺陷增强”的条件下跑的，重训后才能评估真实效果。

---

**相关文档**：
- `05_vit_backbone_sam_encoder.md`
- `06_aufl_experiment.md`
- `07_next_steps_plan.md`
- `08_sam_random_points.md`
- `08_code_refactor_plan.md`

---

# 附录 A：SFY 外部评测集（0923）规划

## A.1 数据集概况

- **路径**：`/autodl-fs/data/masquer.li/SFY_Training_Dataset/0923`
- **规模**：94 张图像 + 94 张 mask（另有 94 张可视化 jpg 和 labelme json）
- **来源**：单个视频 `12_VID001`，帧号 00300–00525，每 5 帧取一张
- **图像尺寸**：660×660
- **标签格式**：{0, 255} 二值 mask，前景占比约 7.8%
- **标注方式**：labelme 多边形标注，mask 相对较粗（血管边缘为折线）

## A.2 作为独立评测集的优势与局限

**优势**：
- 与 FetReg2021 训练集来自不同医院/设备/视频，可用于**跨中心泛化性验证**。
- 包含真实手术器械（如激光光斑），能检验模型对干扰的鲁棒性。
- 前景占比与 FetReg 接近（~8%），类别分布相对一致。

**局限**：
- 样本量小（n=94），且来自同一视频，帧间相关性强，统计显著性有限。
- 标注为 polygons 粗 mask，与 FetReg 的像素级精细标注不完全对齐，直接比较 mIoU 可能偏低。
- 仅一个子集（0923），无法代表全部 SFY 数据分布。

## A.3 建议的评测协议

1. **预处理**：
   - 读取 `labels/*.png`，将 255 视为前景、0 视为背景。
   - TTTSNet 模型：将 660×660 图像 resize 到模型训练尺寸（448 或 512），预测后再 resize 回 660×660 与 GT 对齐。
   - SAM 模型：使用 SAM 原生 longest-side resize 到 1024，postprocess 回 660×660。

2. **指标**：
   - 主指标：mIoU、Dice（与 FetReg 验证集一致，便于对比 drop）。
   - 辅助指标：Pixel Accuracy、Surface Dice（tolerance=2/5 px，缓解粗标注边界惩罚）。
   - 建议报告 **FetReg val → SFY 的相对下降幅度**，而非只看 SFY 绝对值。

3. **Prompt 设置（针对 SAM random）**：
   - 若评估“点监督上界”，从 SFY GT mask 采样 10 fg + 10 bg（与训练时一致）。
   - 若评估“无提示泛化”，使用整图随机点，与 TTTSNet 原始实验对应。

4. **结果解读**：
   - 由于标注粗细差异，SFY 绝对 mIoU 通常会比 FetReg val 低，重点看**模型间相对排序**和**下降幅度**。
   - 若 TTTSNetViT 在 SFY 上仍优于 baseline，可强化“预训练 ViT backbone 提升泛化性”的论点。
   - 若 Semi-supervised 在 SFY 上下降更小，可说明伪标签质量控制对域迁移有帮助。

## A.4 对论文的指导意义

- **外部验证**：把 SFY 0923 作为 independent test set，能显著提升论文的临床可信度。
- **域迁移分析**：对比 FetReg val 与 SFY 的指标 gap，可量化 domain shift，支撑“增强/半监督/域适应”等贡献。
- **Negative/消融支持**：若 SAM random points 在 SFY 上大幅下降，可进一步说明弱提示在真实跨域场景下不可靠。

## A.5 执行时机

建议在所有重训实验完成后统一跑一次：
1. Baseline（修复 augmentation）
2. TTTSNetViT（修复 augmentation）
3. Temporal v3（修复 augmentation）
4. Semi v2
5. SAM random aligned v3

输出一份 `sfy_generalization_results.csv`，包含每个模型在 FetReg val 和 SFY 上的 mIoU/Dice/Surface Dice，以及相对下降率。
