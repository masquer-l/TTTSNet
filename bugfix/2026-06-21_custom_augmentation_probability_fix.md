# 2026-06-21 自定义缺陷增强概率修复记录

## 背景

在排查 `SAM random points` 复现实验结果偏低时，新增了 overfit/debug 可视化脚本，对数据链路进行冒烟测试。测试过程中发现 `A.OneOf([...])` 报出 `ZeroDivisionError: float division by zero`，说明参与 `OneOf` 的自定义增强概率被解析为 0。

该问题会影响论文实验中依赖 laser / dust / fiber / structural defects 这类自定义缺陷增强的训练结果。

## 根因

`TTTSNet/src/utils/custom_augmentations.py` 中的自定义 Albumentations transform 使用了旧版初始化写法：

```python
super(AddDustParticles, self).__init__(always_apply, p)
```

当前环境使用 Albumentations 2.x，该写法会将参数位置错误地传入父类，导致 `p` 未按预期设置。结果是：

- 单独使用 `CustomDefectsAugmentation(p=0.5)` 时，自定义缺陷增强概率不可信。
- 在 `A.OneOf([...])` 中直接使用 `AddLaserPointer / AddDustParticles / AddStructuralDefects / AddOpticalFiber` 时，所有 transform 权重可能被解析为 0，触发 `ZeroDivisionError`。

## 修复内容

已修复文件：

- `TTTSNet/src/utils/custom_augmentations.py`
- `TTTSNet/src/dataset_sam_random.py`
- `TTTSNet/configs/config_sam_random.json`
- `TTTSNet/tools/train_sam_random.py`
- `TTTSNet/tools/overfit_sam_random_debug.py`

核心修复：

```python
super(AddDustParticles, self).__init__(p=1.0 if always_apply else p)
```

同类修复已应用到：

- `AddDustParticles`
- `AddStructuralDefects`
- `AddLaserPointer`
- `AddOpticalFiber`
- `CustomDefectsAugmentation`

## 受影响实验范围

明确受影响：

- `TTTSNet baseline / Single`
- `TTTSNet Semi`
- `TTTSNet Temporal` 中 `use_strong_aug=True` 的配置
- `TTTSNet SAM random points` 复现实验

不属于该 bug 影响范围：

- 未启用自定义缺陷增强的实验
- `Temporal strong_aug=False` 中只使用基础颜色/亮度/CLAHE/normalize 的部分

仍然正常生效的普通增强包括：

- `HorizontalFlip`
- `VerticalFlip`
- `Blur`
- `MotionBlur`
- `ShiftScaleRotate`
- `ColorJitter`
- `RandomBrightnessContrast`
- `CLAHE`
- `Normalize`

## 额外对齐修复

为复现历史 `TTTS_SAM A0.2.5 random points`，同步做了以下对齐：

- `dataset_sam_random.py` 的图像读取改回旧 `TTTS_SAM` 行为：`cv2.imread` 后不再强制 BGR -> RGB。
- 补齐旧链路中的几何增强：`ElasticTransform`、`GridDistortion`、`OpticalDistortion`。
- `config_sam_random.json` 对齐历史配置：`seed=null`、`num_workers=8`、`validation_frequency=5`、`eta_min=1e-6`。
- 新增 `tools/overfit_sam_random_debug.py`，用于导出原图、增强后图、random prompt points、预测结果、GT 和 overlay。

## 验证

已执行：

```bash
python -m py_compile \
  src/utils/custom_augmentations.py \
  src/dataset_sam_random.py \
  tools/train_sam_random.py \
  tools/overfit_sam_random_debug.py
```

已执行 overfit 冒烟测试：

```bash
python tools/overfit_sam_random_debug.py \
  --config configs/config_sam_random.json \
  --sample_count 1 \
  --steps 1 \
  --visualize_every 1 \
  --output_dir experiments/overfit_sam_random_debug_smoke
```

输出文件：

- `experiments/overfit_sam_random_debug_smoke/step_0000_0_Video001_frame00500.png.png`
- `experiments/overfit_sam_random_debug_smoke/step_0001_0_Video001_frame00500.png.png`
- `experiments/overfit_sam_random_debug_smoke/overfit_history.csv`

## 当前重新启动的实验

已重新启动对齐后的 SAM random points 训练：

- 日志：`experiments/A0_2_5_random_points_aligned_nohup.log`
- 实验目录：`experiments/A0_2_5_random_points_aligned_20260621_224621`

启动检查结果：

- 训练已进入 `Epoch 1`
- GPU 正常占用约 19.5GB
- 未出现 CUDA OOM
- 未出现 Albumentations 无效增强参数警告

## 实验记录建议

建议将修复前依赖自定义缺陷增强的实验统一标注为：

> augmentation bug affected

建议优先重跑：

1. `TTTSNet Single baseline`
2. `SAM random points aligned`
3. `Semi-supervised`
4. `Temporal strong augmentation`

在论文实验表中，修复前结果不应作为最终对比结果使用，只能作为 bug 诊断前的历史记录。
