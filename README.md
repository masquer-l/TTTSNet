# TTTSNet-Research v2.0

Repository for open access paper for Medical Image Analysis submission related to Twin-to-Twin Transfusion Syndrome.

本仓库用于 TTTSNet 聚焦实验：单帧全监督基线 → 时序一致性约束 → 半监督学习。

---

## 项目结构

```text
TTTSNet/
├── configs/                  # 各类实验配置
│   ├── config.json           # 单帧 baseline 配置
│   ├── config_temporal.json  # 时序一致性 v1 配置 (λ=0.1)
│   ├── config_temporal_v2.json # 时序一致性 v2 配置 (λ=1.0)
│   └── config_semi.json      # 半监督训练配置
├── tools/                    # 可执行 Python 脚本（训练、伪标签生成）
│   ├── train.py              # 单帧 baseline 训练
│   ├── train_temporal.py     # 时序一致性训练
│   ├── train_semi.py         # 半监督训练
│   └── generate_pseudo_labels.py # 伪标签生成
├── scripts/                  # Shell 启动脚本
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

所有训练都通过 `scripts/` 下的 shell wrapper 启动，配置统一从 `configs/` 读取。

```bash
# 单帧 baseline
./scripts/train.sh configs/config.json

# 时序一致性 v1 (λ=0.1)
./scripts/train_temporal.sh configs/config_temporal.json

# 时序一致性 v2 (λ=1.0)
./scripts/train_temporal_v2.sh configs/config_temporal_v2.json

# 伪标签生成
python tools/generate_pseudo_labels.py \
  --config configs/config.json \
  --checkpoint experiments/tttsnet_single_20260619_233051/tttsnet_single_baseline_20260619_233056/checkpoints/best_model.pth \
  --output_dir pseudo_labels/sfy_data_v1

# 半监督训练
./scripts/train_semi.sh configs/config_semi.json
```

也可以直接调用 `tools/` 下的 Python 脚本：

```bash
python tools/train.py --config configs/config.json --work_dir experiments/debug
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
- Semi-supervised: 🔄 训练中，当前 epoch 9 val_mIoU = 0.459
- 时序一致性约束目前未能超过 baseline，半监督实验正在验证中。

---

## 代码开发规范

1. **目录约定**
   - `configs/`：所有实验配置 JSON，禁止在根目录放置 `config*.json`。
   - `tools/`：所有可直接执行的 Python 脚本（训练、推理、工具）。
   - `scripts/`：Shell 启动脚本和一次性分析脚本。
   - `src/`：模块代码，禁止在 `src/` 中放置可直接运行的脚本。

2. **配置管理**
   - 配置文件中禁止写死绝对路径；机器相关路径使用环境变量或命令行参数覆盖。
   - 新增实验应先复制一份 `configs/config_xxx.json`，不要直接修改正在运行的基准配置。
   - 每个实验目录会自动复制一份 `config.json` 作为快照，确保结果可复现。

3. **代码提交**
   - 提交信息遵循 `<type>(<scope>): <subject>`，例如 `fix(tttsnet): ...`。
   - 重构、bug 修复、配置调整分开提交，保持原子性。
   - 修改 `src/` 后至少通过 `python -m py_compile TTTSNet/src/**/*.py TTTSNet/tools/*.py` 语法检查。

4. **代码质量**
   - 重复的 transform / loss / 工具函数优先提取到 `src/utils/`。
   - 训练脚本通过 `Path(__file__).resolve().parent` 自动定位项目根目录，不依赖执行路径。
   - 新增数据集/模型时，在 `__getitem__` 中保持 image 与 mask 的同步增强。

---

## 实验管理规范

1. **实验命名**
   - 由 `scripts/*.sh` 自动生成：`{task}_{timestamp}`，例如 `tttsnet_temporal_20260620_013149`。
   - 如需自定义名称，修改对应 shell 脚本中的 `RUN_NAME`，或在调用 `tools/` 脚本时通过 `--work_dir` 指定。

2. **实验产物归档**
   - `experiments/` 下的大文件（checkpoints、tb_logs、step_history.csv）被 `.gitignore` 忽略。
   - 有价值的小文件（`EXPERIMENT.md`、`summary.json`、`config.json`、`comparison_*.csv/png`）应显式 `git add -f` 提交。
   - 每个实验目录应包含 `EXPERIMENT.md`（或 `EXPERIMENT_INCOMPLETE.md`），记录配置、结论、待办。

3. **废弃实验归档**
   - `experiments/deprecated/` 仅用于存放因**代码异常中断、配置错误、无意义的调试运行**导致结果不可用的实验。
   - 结果未达预期但训练过程正常的实验，仍保留在 `experiments/` 根目录供后续分析。
   - 归档前在对应 `EXPERIMENT.md`（或实验目录根）注明归档原因和关键日志片段。
   - `experiments/` 根目录只保留：正式基线、已完成且纳入结论的实验、当前正在进行的实验、结果未达预期但可供分析的实验。

4. **可复现性**
   - 每次运行 shell wrapper 会自动将配置快照复制到实验目录。
   - 提交代码变更前，确保 `configs/` 中的默认配置与最近一次成功实验一致。
   -  seeds 固定：所有训练脚本默认读取 `training_config.seed`，并设置 `torch.backends.cudnn.deterministic=True`。

5. **分支与阶段**
   - 新实验/重构在独立分支进行，验证成功后再合并到主分支。
   - 主分支上的 `configs/` 应始终保持可运行的默认配置。

---

## 已知漏洞与注意事项

1. **路径硬编码**：`configs/*.json` 中 `dataset_config.train_paths/val_paths` 以及 `generate_pseudo_labels.py` 的 `--unlabeled_path` 默认值为服务器绝对路径。迁移环境时必须覆盖。
2. **统一入口未完成**：`train.py` / `train_temporal.py` / `train_semi.py` 仍有大量重复代码（loss、优化器、可视化、checkpoint 逻辑），后续计划抽象为 `tools/train.py` 统一入口 + `--mode` 参数。
3. **时序一致性效果未达预期**：Temporal v1/v2 的 val_mIoU 均低于 baseline，需进一步分析：
   - 时序 loss 权重与监督 loss 的平衡；
   - 时序数据集中相邻帧的选择策略；
   - 三帧同步增强是否真正有助于时序一致性学习。
4. **伪标签质量**：半监督性能高度依赖伪标签质量，当前阈值固定，建议增加置信度分析和伪标签筛选策略。
5. **实验目录膨胀**：`experiments/` 下的大文件不被 git 追踪，需定期手动清理失败的实验目录以释放磁盘。

---

## 下一步计划

- 抽象统一训练入口，减少 `tools/` 下脚本重复。
- 引入基于环境变量或 CLI 参数的数据路径配置，消除绝对路径。
- 针对时序一致性任务设计更细粒度的消融实验。
- 增加数据加载单元测试，覆盖 image/mask 同步增强和 ReplayCompose 参数一致性。
