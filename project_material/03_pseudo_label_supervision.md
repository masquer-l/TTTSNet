# 03 伪标签半监督学习（Pseudo-Label Supervision）

## 3.1 动机

FetReg2021 有标注训练集仅 2060 帧，而无标注胎儿镜视频数据丰富。伪标签半监督学习旨在利用无标注视频扩充训练数据，缓解标注稀缺问题。

核心假设：
- 预训练 baseline 对高置信度无标注帧的预测可靠；
- 将这些高置信度预测作为伪标签，与有标注数据共同训练，可提升模型泛化能力。

## 3.2 模型设计

### 3.2.1 伪标签生成

使用训练好的 Single baseline 模型对无标注视频逐帧推理：

```python
# 模型输出 vessel 概率图
prob = softmax(logits)[:, 1, :, :]  # [B, H, W]
max_conf = prob.max()

# 筛选条件：整张图中最大置信度 >= 0.9
if max_conf >= 0.9:
    pred_mask = (prob > 0.5).astype(uint8) * 255
    save(pred_mask)
```

实现：`TTTSNet/tools/generate_pseudo_labels.py`

### 3.2.2 伪标签质量

| 指标 | 数值 |
|---|---|
| 无标注总帧数 | 20,133 |
| 保留伪标签帧数 | 18,745（原日志 18,935，最终目录统计 18,745） |
| 保留率 | ~93% |
| 筛选阈值 | max_conf ≥ 0.9 |

**质量审计**（100 张随机样本）：

| 指标 | 结果 |
|---|---|
| 面积占比 | mean=11.43%, median=11.53%, max=29.01% |
| 连通区域数 | mean=3.71, median=3, max=10 |
| 空 mask | 0 |
| 近全图 mask (>95%) | 0 |
| 启发式 good | 90% |
| 启发式 medium | 10% |
| 启发式 bad | 0% |

**结论**: 伪标签整体质量较好，但 `max_conf` 筛选条件偏宽——只要单个像素置信度高就保留整张图，导致保留率过高，可能引入噪声。

### 3.2.3 半监督数据集

`TTTSNet/src/dataset_semi.py` 混合有标注数据和伪标签数据：

```python
samples = labeled_samples + pseudo_samples
# labeled: (img_path, lbl_path, weight=1.0)
# pseudo:  (img_path, lbl_path, weight=0.5)
```

- 有标注样本：2,060
- 伪标签样本：18,745
- 总量：20,805
- 数量比例约 1:9

**注意**: 当前实现中 BCE/CE 做了 per-sample weight，但 Dice loss 未按 labeled/pseudo 区分，可能导致伪标签对 Dice 项影响过大。

### 3.2.4 数据对齐问题

无标注原图分辨率与模型输出 448×448 不一致，导致伪标签与原图尺寸不匹配。已在 `dataset_semi.py` 中修复：

```python
if label.shape[:2] != image.shape[:2]:
    label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
```

同时 `generate_pseudo_labels.py` 已改进为**直接生成原图尺寸**的伪标签，未来可避免该问题。

### 3.2.5 训练设置

| 配置 | 值 |
|---|---|
| 学习率 | 1e-4 |
| Batch size | 4 |
| Epochs | 100（计划） |
| pseudo weight | 0.5 |
| 有标注 weight | 1.0 |

## 3.3 实验结果

### 3.3.1 半监督训练

| Epoch | val_mIoU | val_Dice |
|---|---|---|
| 1 | 0.3273 | 0.4662 |
| 5 | 0.4397 | 0.5705 |
| 7 | 0.4569 | 0.5890 |
| 9 | 0.4594 | 0.5915 |
| 10 | 0.4630 | - |

**状态**: 训练至 epoch 10 后按诊断建议暂停，优先做实 baseline。

### 3.3.2 当前判断

- 早期 epoch 持续上升，但远低于 baseline 0.599。
- 有标注:pseudo 比例 1:9，即使 weight=0.5，噪声仍可能主导训练。
- 筛选策略过宽，需要更严格的质量控制。

## 3.4 后续改进方向

| 方向 | 方法 | 预期效果 |
|---|---|---|
| 严格筛选 | 用 mean confidence / top-k mean confidence 替代 max_conf | 降低噪声伪标签比例 |
| 面积过滤 | 过滤空 mask、极小 mask、异常大 mask | 去除明显错误样本 |
| 固定 batch 比例 | 每个 batch 中 labeled:pseudo = 1:1 或 1:2 | 防止 pseudo 主导 |
| pseudo weight sweep | 0.1 / 0.25 / 0.5 | 找到最优噪声权重 |
| Weighted Dice | Dice loss 也支持 per-sample weighting | 与 BCE/CE 一致 |

## 3.5 当前结论

- 伪标签生成流程已跑通，伪标签整体质量较好。
- 当前半监督实验早期表现不及 baseline，主要风险在于筛选过宽和数量比例失衡。
- 待 baseline 稳定后，按上述方向系统优化伪标签质量控制，再重启 semi 实验。

### 论文定位

半监督方向有潜力作为论文主贡献，但需要满足：
- 相比强化后的 Single baseline，val_mIoU 稳定提升 1-2 个点；
- 3 seeds 下趋势一致；
- 有清晰的伪标签质量控制方法作为创新点。

建议论文方向暂定为：**面向 TTTS 胎儿镜血管分割的伪标签质量控制半监督方法**。

---

**相关文件**: `TTTSNet/tools/generate_pseudo_labels.py`, `TTTSNet/src/dataset_semi.py`, `TTTSNet/tools/train_semi.py`, `TTTSNet/configs/config_semi.json`

**实验目录**:
- 伪标签：`/autodl-fs/data/masquer.li/temperal_data/sfy_data_v1_20251019/*/pseudo_labels/`
- 半监督训练：`TTTSNet/experiments/tttsnet_semi_20260620_085333/`
- 失败尝试：`TTTSNet/experiments/deprecated/tttsnet_semi_20260620_084856/`
