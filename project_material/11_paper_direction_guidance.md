# TTTSNet 论文方向与防发散指导文档

**创建时间**: 2026-07-04  
**适用阶段**: 从“广泛尝试模型改进”切换到“收敛论文问题与证据链”  
**核心目的**: 避免后续继续在模型设计、轻量化、时序、半监督、外部验证之间发散；把精力集中到能支撑论文发表的主问题、数据资产和实验闭环上。

---

## 1. 当前最重要的判断

当前工作的主要瓶颈不是“实验数量不够”，而是**论文方向尚未充分收敛**。

已经完成的实验说明：

- `TTTSNet-v1.0.0` CNN baseline 稳定，但性能有限。
- `TTTSNet-v2.1.0` ViT + Layerwise LR 是目前最强且最稳定的模型：
  - 内部 Val mIoU: `0.6389 ± 0.0025`
  - SFY mIoU: `0.6524 ± 0.0080`
- `TTTSNet-v2.2.0` clDice fixed 在拓扑指标上有价值，但不是全面超越主模型。
- Temporal、Transformer decoder、Semi-supervised v2 当前都没有形成稳定正收益。
- ViT 模型参数量约 `94.92M`，CNN baseline 约 `5.31M`，性能提升可能被质疑为主要来自更大预训练 backbone。

因此，短期内不建议继续以“探索更复杂模型设计”为主线。更稳妥的方向是：

> **围绕真实 TTTS 胎儿镜数据，定义低资源、跨域、临床可用的血管分割问题；用 baseline 与强 ViT 模型作为锚点，证明数据、泛化、稳定性和业务价值。**

---

## 2. 推荐论文主问题

建议把论文主问题收敛为：

> **在低资源 TTTS 胎儿镜血管分割场景中，如何利用真实手术数据与预训练视觉表征提升跨域泛化和临床可用性？**

这个问题比“提出一个新网络模块”更适合当前条件，原因是：

1. **你的差异化资产是自建数据和具体临床场景**，不是大规模模型设计能力。
2. 医学分割模型设计领域非常拥挤，单个 decoder、temporal block 或 loss 很难构成强贡献。
3. 当前实验已经显示复杂模块未稳定优于 ViT + Layerwise LR。
4. 审稿人更容易接受“真实场景系统分析 + 外部验证 + 业务指标”的贡献，而不是只看一个小幅 mIoU 提升。

---

## 3. 建议的论文贡献定位

当前更可行的贡献组合是：

### 贡献 1：真实 TTTS 胎儿镜外部数据集与跨域评估

把 SFY 数据从“附属测试集”提升为论文核心资产。

需要强调：

- 数据来自真实手术视频/临床环境。
- 与训练数据存在设备、场景、标注风格、干扰因素差异。
- 可用于评估跨域泛化、真实手术鲁棒性和临床可用性。

注意：

- 如果 SFY 当前只有 `n=94` 且来自同一视频，必须诚实说明局限。
- 尽量扩展到更多视频、更多片段、更多场景；如果不能扩展，也要做分层分析和 bootstrap CI。

### 贡献 2：低资源 TTTS 场景下 CNN baseline 与预训练 ViT/SAM 表征的系统比较

不要只说“ViT 提升了指标”，而要回答：

- 它提升在哪些类型样本上？
- 是否改善细血管、断裂、低对比、激光干扰、器械遮挡？
- 是否只是参数量变大带来的结果？
- 精度提升是否值得额外参数量和延迟？

必须报告：

- mIoU / Dice
- clDice 或中心线相关指标
- Surface Dice
- 参数量
- 推理延迟
- trainable parameters
- per-image paired comparison 或 bootstrap CI

### 贡献 3：面向真实手术视频的业务/临床可用性指标

单帧 mIoU 不足以体现真实业务价值。建议引入：

- 跨帧 mask area jitter
- frame-to-frame Dice / IoU
- centerline continuity
- vessel break count
- flicker rate
- failure mode taxonomy

这些指标可以让 Temporal、clDice、pseudo-label 方向获得更合理的评价场景。

---

## 4. 当前不建议作为主线的方向

### 4.1 不建议继续堆复杂模型结构

包括：

- 更深 Transformer decoder
- 更复杂 temporal block
- 随机组合 ViT + Transformer + clDice + semi
- 没有清晰假设的新 loss

理由：

- 当前已完成实验没有支持复杂结构稳定提升。
- 复杂模型会增加解释成本和审稿风险。
- 如果没有公平强 baseline，对小幅提升的说服力很弱。

允许继续做的条件：

- 该实验能直接回答论文主问题。
- 预先写清楚假设、成功标准和停止条件。
- 最多作为消融，不作为主贡献前提。

### 4.2 Semi-supervised 暂不作为主贡献

Semi 的潜力在于利用自建未标注数据，但当前 v2 结果失败，说明伪标签质量、训练配比或 domain split 没有闭环。

Semi 可以重新进入主线的条件：

- 有清晰的 unlabeled SFY / 真实手术视频来源。
- 有独立 holdout 视频或片段验证。
- pseudo-label 有质量分层，能证明高质量伪标签带来泛化提升。
- 至少在外部/跨域指标上超过 supervised ViT anchor。

否则，Semi 只作为“当前策略未证明有效”的负结果或未来工作。

### 4.3 Temporal 暂不按单帧分割指标评价

Temporal 的价值不一定体现在单帧 mIoU。继续用单帧 mIoU 判断 Temporal，可能会误杀方向。

Temporal 只有在以下评价指标下才值得继续：

- 跨帧一致性
- 预测闪烁
- 血管中心线连通性
- 连续视频上的临床可用性

如果短期无法建立这些指标，Temporal 不应作为主贡献。

---

## 5. 下一阶段推荐工作路线

### Phase A：论文问题冻结

目标：停止发散，明确论文故事线。

产出：

- 一页纸论文定位：
  - 暂定标题
  - 核心问题
  - 3 条贡献
  - 主实验表
  - 必须完成的图
  - 不做事项清单

建议标题方向：

> Data-Centric Evaluation of Pretrained Visual Representations for Low-Resource Fetoscopic Vessel Segmentation in Twin-to-Twin Transfusion Syndrome

或：

> Toward Robust TTTS Fetoscopic Vessel Segmentation: External Validation, Pretrained Adaptation, and Clinical Video Consistency

### Phase B：baseline 与 ViT anchor 做实

目标：用两个锚点建立稳定证据链。

必须完成：

- CNN baseline 3 seeds 汇总。
- ViT + Layerwise LR 3 seeds 汇总。
- 内部 Val 与 SFY 外部评测统一表格。
- 参数量、延迟、trainable parameters。
- per-image 统计检验。
- 失败案例和成功案例可视化。

判断标准：

- 如果 ViT 在多数 per-image 样本上稳定优于 baseline，且置信区间支持提升，可以作为强 anchor。
- 如果提升集中在少数样本，需要转向 failure mode / subgroup 分析，而不是夸大整体性能。

### Phase C：把 SFY 数据变成核心贡献

目标：不要只把 SFY 当 94 张测试图，而要充分挖掘业务价值。

建议整理维度：

- 视频来源
- 帧号范围
- 前景面积比例
- 低对比/高反光/激光光斑/器械遮挡
- 血管稀疏/密集
- 标注粗细与边界不确定性

优先分析：

- baseline vs ViT 在不同子群上的差异。
- clDice fixed 是否改善细血管连通。
- SFY 相比内部 Val 的 domain gap。
- 哪些场景导致模型失败。

### Phase D：轻量化/参数高效作为风险缓解分支

目标：回应“ViT 只是参数量更大”的质疑。

优先级从低成本到高成本：

1. **冻结 ViT encoder，只训练 decoder**。
2. **当前 Layerwise LR 与 full finetune 对比**。
3. **LoRA / adapter 版本**，只训练少量参数。
4. **TinyViT / MobileSAM / EfficientViT / RepViT encoder 替换**。
5. **知识蒸馏**，从强 ViT 到轻量模型。

注意：

- 轻量化不是当前主线的前置条件。
- 但至少需要一个参数高效消融，避免审稿人认为提升完全来自 18 倍参数量。

### Phase E：Temporal 和 Semi 只在满足条件后重启

Temporal 重启条件：

- 已实现跨帧稳定性指标。
- 有连续视频片段预测结果。
- 能证明 temporal 方法改善稳定性，而不是只看单帧 mIoU。

Semi 重启条件：

- 有明确 unlabeled 数据来源。
- 有伪标签质量评分。
- 有 holdout 域外验证。
- 有 pseudo-label quality ablation。

---

## 6. 防发散规则

以后每个新实验必须先回答以下 7 个问题。答不出来就不做。

1. 这个实验服务哪一个论文贡献？
2. 它比较的 anchor 是哪个：CNN baseline、ViT + Layerwise LR，还是 SFY domain analysis？
3. 预期能解决什么审稿质疑？
4. 成功标准是什么？
5. 失败后能否形成有价值的负结果？
6. 预计耗时是否小于当前阶段可承受范围？
7. 它会不会引入新的主线，导致论文故事更复杂？

推荐停止条件：

- 单个新方向连续 2 个实验没有超过 anchor，停止。
- 新方向需要超过 3 天工程改造才能验证，暂缓。
- 如果结果只能带来小于 1 个点 mIoU 提升，但解释成本很高，暂缓。
- 如果不能写进论文主表或补充表，暂缓。

---

## 7. 风险清单与应对

### 风险 1：ViT 提升被认为只是参数量优势

应对：

- 报告参数量、延迟、trainable parameters。
- 加 frozen encoder / LoRA / adapter 消融。
- 与更强公开 baseline 比较，例如 SegFormer-B0/B1、DeepLabV3+ 或 U-Net variants。
- 强调预训练表征在低资源真实手术场景中的泛化价值，而不是声称结构创新。

### 风险 2：SFY 外部验证不够大

应对：

- 尽量扩展更多视频或片段。
- 若不能扩展，做 bootstrap CI、per-image paired test 和分层分析。
- 明确写作时称为 independent external pilot set 或 real-world validation subset，不夸大为全面多中心验证。

### 风险 3：主贡献不够方法学

应对：

- 将论文定位为 data-centric / clinical validation / low-resource adaptation。
- 方法贡献保持克制：ViT/SAM adaptation + layerwise LR + optional parameter-efficient adaptation。
- 用数据集、业务指标、外部验证和失败模式补足价值。

### 风险 4：Temporal/Semi 失败削弱论文故事

应对：

- 不把它们列为主贡献。
- 作为负结果或补充材料，说明在当前评价协议下未证明有效。
- 只有在新指标或新数据闭环下成功，才提升为贡献。

### 风险 5：继续发散导致无法按时写论文

应对：

- 每周只允许一个主线实验方向。
- 每周必须产出一个可写进论文的 artifact：表格、图、统计结果、错误分析、方法段落。
- 所有新想法先进 backlog，不直接开跑。

---

## 8. 建议的 2 周执行计划

### 第 1 周：收敛与证据整理

任务：

- 冻结论文主问题和贡献草案。
- 整理 baseline vs ViT + Layerwise LR 的最终表格。
- 生成 per-image metrics 对比。
- 对 SFY 做分层标签或最小手工分类。
- 选 8-12 个 qualitative cases。

验收：

- 有一张主结果表。
- 有一张效率-性能表。
- 有一张 baseline vs ViT qualitative figure 草稿。
- 有一页 failure mode taxonomy。

### 第 2 周：补关键质疑实验

任务：

- 做 ViT frozen / layerwise / full finetune 对比，或至少补 trainable parameter 统计。
- 如果工程成本可控，做一个 LoRA/adapter quick experiment。
- 做 SFY bootstrap CI 或 paired test。
- 判断 clDice fixed 是否值得保留为拓扑消融。

验收：

- 能回答“为什么不是只靠大模型参数量”的质疑。
- 能回答“SFY 结果是否稳定”的质疑。
- 能决定 Temporal/Semi 是否继续进入下一阶段。

---

## 9. 论文写作骨架

### Introduction

核心叙事：

- TTTS 胎儿镜血管分割具有临床价值。
- 现有方法多在受控数据集上评估，真实手术域外泛化不足。
- 低资源条件下，预训练视觉模型可能有帮助，但参数量、泛化和临床可用性仍需系统评估。
- 本文围绕真实手术外部数据集，比较 CNN baseline、ViT/SAM 适配和业务指标。

### Methods

建议包括：

- 数据集与标注协议。
- CNN baseline。
- ViT/SAM encoder adaptation。
- Layerwise LR。
- 可选：clDice / parameter-efficient adaptation。
- 评估指标：mIoU、Dice、clDice、Surface Dice、temporal/clinical metrics。

### Experiments

主实验：

- 内部 validation。
- SFY external validation。
- 参数量与延迟。
- per-image statistical analysis。
- subgroup / failure mode analysis。

消融：

- ViT backbone vs ViT + Layerwise LR。
- clDice fixed。
- Frozen/LoRA/adapter，如果完成。
- Temporal/Semi 作为补充或负结果。

### Discussion

必须诚实讨论：

- ViT 参数量较大。
- SFY 样本量和视频来源限制。
- 标注粗细差异影响 mIoU。
- Temporal 和 Semi 当前未形成稳定收益。
- 未来需要多中心、更大规模真实视频验证。

---

## 10. 最终建议

短期内，把工作重心从“探索更好的模型设计”切换为：

> **定义清楚业务问题，做实自建数据集价值，用 baseline 与 ViT anchor 建立可信证据链。**

模型仍然重要，但它现在应该服务于论文问题，而不是驱动论文发散。

后续所有实验都应围绕三件事展开：

1. **证明真实 TTTS 数据上的泛化问题存在。**
2. **证明 ViT/SAM 预训练适配在低资源场景中有稳定价值，并说明代价。**
3. **证明你的自建数据和临床/视频指标能揭示单帧 mIoU 看不到的问题。**

如果某个实验不能服务这三件事之一，就不要做，至少不要在当前投稿压力下做。
