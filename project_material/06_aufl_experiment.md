# 06 Asymmetric Unified Focal Loss (AUFL) 消融

## 6.1 动机

TTTSNet 基线使用 Dice + BCE + CE 的组合损失。胎儿镜血管分割是典型前景占比低（约 8-9%）的类别不平衡问题，因此测试了专门设计用于不平衡分割的 **Asymmetric Unified Focal Loss (AUFL)**，看是否能替代或增强现有损失。

AUFL 结合了：
- Focal loss：降低易分类背景样本的权重
- Asymmetric loss：对假阴性和假阳性施加不同惩罚
- Unified formulation：同时处理 foreground/background 不平衡

## 6.2 实现

实现文件：`TTTSNet/src/utils/losses.py` 中的 `AsymmetricUnifiedFocalLoss`

训练入口：`TTTSNet/tools/train.py --config configs/config_aufl_30ep.json`

### 配置参数

| 参数 | 值 |
|---|---|
| `loss_type` | `aufl` |
| `aufl_weight` | 0.5 |
| `aufl_delta` | 0.6 |
| `aufl_gamma` | 0.2 |
| `n_classes` | 2 |
| Epochs | 30 |

其他设置（lr=1e-4, batch=4, 448×448, AdamW, Cosine scheduler）与 baseline 完全一致。

## 6.3 实验结果

| 指标 | AUFL 30ep | Baseline 30ep（参考） | Baseline 100ep |
|---|---|---|---|
| Best val_mIoU | **0.5647** | ~0.5720 | **0.5992** |
| Best Epoch | 27 | - | 66 |
| Best val_Dice | 0.7037 | - | 0.7324 |
| 训练时间 | 0.42 h | - | 1.64 h |

**结论**：AUFL 30 epochs 的 best val_mIoU（0.5647）**未超过** baseline 30 epochs 水平（约 0.5720），更低于 baseline 100 epochs（0.5992）。

## 6.4 判断

1. **Loss 不是当前主要瓶颈**：在相同 30 epoch 对比下，AUFL 没有明显优势。
2. **AUFL 超参数可能未充分调优**：`delta=0.6, gamma=0.2` 基于通用推荐，未针对胎儿镜血管任务专门搜索。
3. **继续投入 AUFL 的性价比低**：baseline 在 100 epochs 仍显著更高，优先从数据增强、backbone、输入分辨率等方向寻找突破。

## 6.5 论文定位

AUFL 可作为 **negative/消融结果** 简要提及：
> "我们也尝试了专门的不平衡损失 AUFL，但未超过标准的 Dice+BCE+CE 组合，说明在当前设定下损失函数形式不是主要瓶颈。"

不建议作为论文主贡献或重点消融方向。

---

**相关文件**: `TTTSNet/src/utils/losses.py`, `TTTSNet/configs/config_aufl_30ep.json`

**实验目录**: `TTTSNet/experiments/tttsnet_aufl_30ep_20260620_103645/tttsnet_aufl_30ep_20260620_103650/`

---

**最后更新**: 2026-06-20
