# TTS-Net-Temporal v1 (λ_temp=0.1)

**日期:** 2026-06-20  
**执行人:** 李明（Claude 辅助）  
**实验目标:** 在 TTTSNet 上加入相邻帧时序一致性约束，验证时序信息价值

---

## 配置

| 项目 | 值 |
|------|-----|
| 模型 | TTTSNet (classes=2, num_features=64) |
| 输入 | 3 帧连续片段 [t-1, t, t+1]，448×448 |
| 推理 | 单帧输入（与 baseline 相同） |
| 训练数据 | FetReg2021 Train 连续片段 (2024 clips) |
| 验证数据 | FetReg2021 Test 单帧 (658 frames) |
| Temporal Loss | uncertainty-weighted L1: `conf = p*(1-p)*4`, `w = min(conf_t, conf_tp1)` |
| λ_temp | 0.1 |
| 数据增强 | resize + normalize + photometric (无几何增强，为保证时序对齐) |

---

## 结果（训练至 epoch 34 后停止）

| 指标 | 值 | 备注 |
|------|-----|------|
| best_val_mIoU | **0.5491** | epoch 18 |
| final_val_mIoU (epoch 34) | 0.5252 | - |
| 对比 baseline (0.599) | **-5.0%** | 未达 +2% 目标 |

---

## 停止原因

1. 训练至 epoch 34 仍未超过 baseline 的 0.599。
2. 模型在 epoch 18 后性能下滑并饱和，未见继续提升迹象。
3. 推测主要原因：
   - 时序数据集缺少与 baseline 同等强度的数据增强（为保证 3 帧空间对齐，未使用几何增强和 custom defects）
   - λ_temp=0.1 可能过小，时序约束对模型影响微弱

---

## 结论

- **是否达到预期**: 否。当前配置下时序一致性未带来提升。
- **下步行动**:
  1. 重启 Temporal v2：λ_temp 提升至 1.0，增强时序约束强度。
  2. 若 v2 仍无效，考虑引入一致的几何增强（如一致的 horizontal flip）或调整时序 loss 形式。

---

## 文件位置

- 训练脚本: `TTTSNet/train_temporal.py`
- 配置文件: `TTTSNet/config_temporal.json`
- 训练日志: `experiments/tttsnet_temporal_20260620_013149/training.log`
- 指标 CSV: `epoch_history.csv`
