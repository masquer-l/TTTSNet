# 08 SAM + 随机点提示（人工点监督上界）

## 8.1 动机

Segment Anything Model (SAM) 在大规模视觉数据上预训练，具有强大的零样本分割能力。本实验原意是验证：在胎儿镜血管分割任务上，**不依赖人工提示**，仅使用**完全随机的点提示**（整张图内随机采样 20 个点，label 全为 1），SAM 能否通过端到端微调达到有竞争力的性能。

但在对齐 TTTS_SAM A0.2.5 代码的过程中发现，TTTS_SAM 的 `points_sample_mode=RANDOM` + `prompt_with_gt=1` **并非“整图随机点”**，而是从 GT mask 中随机采样前景/背景点（标签 0/1 混合）。因此：

- **TTTS_SAM A0.2.5 的 0.679 mIoU 应被视为“基于 GT mask 采样的点监督上界”**，而不是无人工提示的随机点结果。
- **TTTSNet 原始全随机点实现（20 点全 label 1）才是真正的“无提示/弱提示 negative result”**，结果为 0.575 mIoU。
- 本实验在当前阶段的定位改为：**在 TTTSNet 中复现 TTTS_SAM 的点监督上界**，作为后续 TTTSNet 系列实验（ViT backbone、Temporal、Semi 等）可对比的 strong baseline。

这与 TTTSNetViT 形成对照：TTTSNetViT 是把 SAM 的 ViT-B encoder 作为 TTTSNet 的 backbone；而本实验使用完整 SAM 架构（image encoder + prompt encoder + mask decoder）。

## 8.2 关键发现：TTTS_SAM A0.2.5 的 "random points" 到底做了什么

### 8.2.1 配置层面

A0.2.5 配置为：

```json
"prompt_config": {
    "box_prompt": 0,
    "points_sample_mode": "RANDOM",
    "prompt_points_num": 20,
    "prompt_with_gt": 1
}
```

由于 `prompt_with_gt=1`，数据集不会走“测试模式/无 GT 提示”分支，而是进入 `generate_positive_prompts`。

### 8.2.2 代码路径

`TTTS_SAM/src/data/dataset.py:343-371`：

```python
if self.mode == 'test' or not self.prompt_with_gt:
    point_list, point_label_list, box = self._generate_test_prompts(image_ori_size)
else:
    # 正样本：使用 PromptManager 生成提示
    point_list, point_label_list, box, gt2D = self.prompt_manager.generate_positive_prompts(
        label_org=label_org,
        label_list=label_list,
        image_ori_size=image_ori_size,
        prompt_points_num=self.prompt_points_num,
        points_sample_mode=self.points_sample_mode   # "RANDOM"
    )
```

`TTTS_SAM/src/prompts/prompt_utils.py:28-111` 中的真正采样逻辑：

```python
def generate_points_from_mask(mask, num_points=10, fg_ratio=0.5, sample_mode='random'):
    h, w = mask.shape
    num_fg = max(1, int(num_points * fg_ratio))  # 10
    num_bg = num_points - num_fg                  # 10

    # 从 mask>0 的前景像素中随机采样
    fg_coords = np.where(mask > 0)
    indices = np.random.choice(len(fg_coords[0]), min(num_fg, len(fg_coords[0])), replace=False)
    fg_points = np.column_stack([fg_coords[1][indices], fg_coords[0][indices]])

    # 从 mask==0 的背景像素中随机采样
    bg_coords = np.where(mask == 0)
    indices = np.random.choice(len(bg_coords[0]), min(num_bg, len(bg_coords[0])), replace=False)
    bg_points = np.column_stack([bg_coords[1][indices], bg_coords[0][indices]])

    points = np.concatenate([fg_points, bg_points], axis=0)
    labels = np.concatenate([
        np.ones(len(fg_points), dtype=np.float32),
        np.zeros(len(bg_points), dtype=np.float32),
    ])
    return points, labels
```

**结论**：`RANDOM` 只是“在 GT mask 的前景/背景像素中随机采样”，得到的是 **10 个前景点 + 10 个背景点，标签 0/1 混合**。这与文档/配置字面含义“20 个整图随机点、label 全 1”完全不同。

### 8.2.3 如果把 `prompt_with_gt` 关掉会怎样？

`prompt_with_gt=0` 会进入 `_generate_test_prompts`，它调用 `pointSample` → `generate_grid_points`，生成的是**均匀网格点 + label 全 1**，仍然不是“整图随机”。因此 TTTS_SAM 当前代码里实际上**不存在真正的整图随机点实现**。

## 8.3 模型与训练设置

| 配置项 | 值 |
|---|---|
| 模型 | SAM ViT-B |
| Checkpoint | `/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth` |
| 输入尺寸 | 1024×1024 |
| Prompt | **从 GT mask 随机采样 10 前景点 + 10 背景点**（`points_sample_mode=RANDOM`, `prompt_with_gt=1`） |
| Box prompt | 否 |
| Freeze image encoder | 否（端到端微调） |
| Batch size | 2 |
| LR | 1e-4 |
| Optimizer | AdamW |
| Loss | DiceCELoss + 0.1 × IoU MSE loss |
| Epochs | 100 |
| Scheduler step | 仅在验证 epoch step：epoch 0、4、5、9、10…（约 40 次/100 epoch） |

实现文件：
- `TTTSNet/src/dataset_sam_random.py`
- `TTTSNet/tools/train_sam_random.py`
- `TTTSNet/configs/config_sam_random.json`

## 8.4 实验结果

### 8.4.1 TTTS_SAM A0.2.5 复现结果（目标上界）

| 指标 | 数值 | Epoch |
|---|---|---|
| Best val_mIoU | **0.6794** | 90 |
| Best val_Dice | **0.8032** | 90 |
| Final val_mIoU | 0.6713 | 94 |
| Final val_Dice | 0.7967 | 94 |
| 总耗时 | ~13.5 h | - |

**实验目录**: `TTTS_SAM/work_dir_b/20260622_200100_train_A0_2_5_random_points_vit_b_bs2_lr0.0001/`

### 8.4.2 TTTSNet 移植版历史结果

| 实验 | Best val_mIoU | Best val_Dice | 备注 |
|---|---|---|---|
| TTTSNet SAM random（原始全随机点） | 0.5754 | 0.7147 | 真正的“无 GT 提示”随机点 |
| TTTSNet SAM random aligned v2 | 0.5428 | - | 仅修复 scheduler，仍用全随机点；中断于 ep62 |
| **TTTSNet SAM random aligned v3** | **进行中** | **进行中** | GT mask 采样 + TTTS_SAM scheduler 模式 |

v3 训练信息：
- PID: `124913`
- 日志: `TTTSNet/experiments/A0_2_5_random_points_aligned_v3_nohup.log`
- 启动时间: 2026-06-23

## 8.5 与 TTTSNet 系列对比

| 实验 | Best val_mIoU | Best val_Dice | 架构 / 提示 |
|---|---|---|---|
| TTTSNetViT | **0.6191** | **0.7489** | TTTSNet + SAM ViT-B backbone |
| TTTSNet baseline | 0.5992 | 0.7324 | 原版 TTTSNet |
| Temporal v3 | 0.5978 | 0.7320 | TTTSNet + 3帧时序 + 强增强 |
| **SAM GT-mask random points（上界）** | **0.6794** | **0.8032** | 完整 SAM + GT mask 采样点 |
| SAM fully-random points | 0.5754 | 0.7147 | 完整 SAM + 整图随机点（label 全 1） |
| AUFL 30ep | 0.5647 | 0.7037 | TTTSNet + AUFL loss |

说明：SAM GT-mask random points（0.679）由于训练/验证都使用 GT mask 生成点提示，指标高于 TTTSNet 系列是合理的，应作为**强点监督上界**参考，而不是与 TTTSNet 直接公平竞争的结果。

## 8.6 结论

1. **TTTS_SAM A0.2.5 的 "random points" 实际上是 GT mask 监督点提示**：10 前景 + 10 背景，标签 0/1 混合。其 0.679 mIoU 应理解为**点监督上界**。
2. **真正的整图随机点（label 全 1）无法达到 0.679**：TTTSNet 原始实现 0.575 mIoU 更接近“无人工提示”条件下的真实表现。
3. **TTTSNet SAM random aligned v3 的目标是复现这个上界**：只有数据集和 scheduler 都按 TTTS_SAM 实现，才能得到可对比的 0.679 基准。
4. **1024×1024 输入本身不是决定因素**：在相同点监督条件下，SAM 完整架构可以显著超过 448×448 的 TTTSNetViT；而在真正无提示条件下则不如 TTTSNet 系列，说明提示质量是主要变量。

## 8.7 论文定位

- **SAM GT-mask random points（0.679）**：作为**点监督上界 / strong baseline**。它说明：只要给出足够好的点提示（从 GT 采样），SAM 在胎儿镜血管分割上可以达到很高性能。
- **SAM fully-random points（0.575）**：作为 **negative result / 消融对照**，说明单纯整图随机弱提示不足以充分发挥 SAM 能力。
- 论文主线仍应围绕 **TTTSNet + 预训练 ViT backbone + 半监督质量控制** 展开，因为 TTTSNetViT（0.619）在无点监督、仅 backbone 迁移的情况下已接近甚至超过 SAM 完整架构的上界（若把 SAM 上界视为需要 GT 点监督的代价）。

若需要进一步提升 SAM 结果，可尝试：
- 使用中心线/骨架点代替完全随机点
- 加入 box prompt 或从 GT 采样的点提示
- 冻结 image encoder 只微调 mask decoder

---

**相关文件**: `TTTSNet/src/dataset_sam_random.py`, `TTTSNet/tools/train_sam_random.py`, `TTTSNet/configs/config_sam_random.json`

**历史实验目录**:
- `TTTSNet/experiments/sam_random_points_20260620_211639/`（原始全随机点）
- `TTTSNet/experiments/A0_2_5_random_points_aligned_20260622_113150/`（aligned v2）
- `TTTS_SAM/work_dir_b/20260622_200100_train_A0_2_5_random_points_vit_b_bs2_lr0.0001/`

---

**最后更新**: 2026-06-23
