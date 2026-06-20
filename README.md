# TTTSNet-Research v2.0

Repository for open access paper for Medical Image Analysis submission related to Twin-to-Twin Transfusion Syndrome.

本仓库用于 TTTSNet 聚焦实验：单帧全监督基线 → 时序一致性约束 → 半监督学习。

---

## 项目结构

```text
TTTSNet/
├── config*.json              # 各类实验配置
├── train.py                  # 单帧 baseline 训练
├── train_temporal.py         # 时序一致性训练
├── train_semi.py             # 半监督训练
├── generate_pseudo_labels.py # 伪标签生成
├── scripts/                  # 启动脚本
│   ├── train.sh
│   ├── train_temporal.sh
│   ├── train_temporal_v2.sh
│   ├── train_semi.sh
│   └── compare_experiments.py
├── src/                      # 源代码
│   ├── data_loader.py
│   ├── dataset_tttsnet.py
│   ├── dataset_temporal.py
│   ├── dataset_semi.py
│   ├── models/TTTSNet.py
│   └── utils/
├── experiments/              # 实验产物（大文件被 gitignore）
├── pseudo_labels/            # 半监督伪标签输出
├── RUNNING_SUMMARY.md        # 实验进展摘要
├── EXPERIMENTS.md            # 实验路径索引
└── README.md                 # 本文件
```

---

## 快速开始

```bash
# 单帧 baseline
./scripts/train.sh config.json

# 时序一致性 v1 (λ=0.1)
./scripts/train_temporal.sh config_temporal.json

# 时序一致性 v2 (λ=1.0)
./scripts/train_temporal_v2.sh config_temporal_v2.json

# 伪标签生成
python generate_pseudo_labels.py \
  --checkpoint experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/checkpoints/best_model.pth \
  --output_dir pseudo_labels/sfy_data_v1

# 半监督训练
./scripts/train_semi.sh config_semi.json
```

---

## 当前状态

详见：
- `RUNNING_SUMMARY.md` — 实验进展和结论
- `EXPERIMENTS.md` — 实验路径索引
- `.planning/STATE.md` — GSD 项目状态
- `.planning/ROADMAP.md` — 阶段规划

**关键结论（2026-06-20）**:
- Baseline: val_mIoU = 0.599 (epoch 66)
- Temporal v1: val_mIoU = 0.549 (epoch 18, stopped at epoch 35)
- Temporal v2: val_mIoU = 0.546 (100 epochs)
- 时序一致性约束目前未能超过 baseline，需要进一步分析根因。

---

## 注意事项

- `experiments/` 目录下的大文件（checkpoints、tb_logs、step_history.csv）被 `.gitignore` 忽略
- 有价值的小文件（`EXPERIMENT.md`、`summary.json`、`config.json`、`comparison_*.csv/png`）应显式 `git add -f` 提交
- 代码重构计划中：后续将合并 `train.py` / `train_temporal.py` / `train_semi.py` 为统一入口
