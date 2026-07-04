# 08 代码重构计划

## 8.1 当前代码债务

### 8.1.1 训练脚本高度重复

当前有 4 个独立训练脚本，总计约 1875 行：

| 文件 | 行数 | 主要差异 |
|---|---|---|
| `tools/train.py` | 505 | 单帧 baseline |
| `tools/train_temporal.py` | 465 | 3 帧时序输入 + temporal loss |
| `tools/train_semi.py` | 432 | 混合有标注/伪标签 + sample weight |
| `tools/train_sam_random.py` | 473 | SAM + 随机点提示 |

共同代码占比估计 >60%，包括：
- `set_seed`, `load_config`
- `create_optimizer`, `create_scheduler`
- `save_checkpoint`
- `main()` 中的实验目录创建、配置保存、tracker 初始化、训练循环 orchestration

### 8.1.2 Dataset 实现分散

| 文件 | 用途 |
|---|---|
| `src/dataset_tttsnet.py` | 单帧数据集 |
| `src/dataset_temporal.py` | 3 帧时序数据集 |
| `src/dataset_semi.py` | 半监督混合数据集 |
| `src/dataset_sam_random.py` | SAM 随机点数据集 |
| `src/data_loader.py` | 早期 FetoscopicDataset（部分被 dataset_tttsnet.py 替代） |

问题：
- `data_loader.py` 与 `dataset_tttsnet.py` 功能重叠，可能可以合并或删除
- 自定义增强定义在 `src/utils/custom_augmentations.py`，但每个 dataset 的增强组合重复
- 没有统一的 base dataset class

### 8.1.3 配置文件命名不统一

部分配置有清晰后缀（`config_temporal_v3.json`），部分没有（`config.json` 是 baseline）。建议：
- `config_baseline.json`
- `config_temporal_v1.json`, `config_temporal_v2.json`, `config_temporal_v3.json`
- `config_semi_v1.json`, `config_semi_v2.json`
- `config_vit_backbone.json`
- `config_sam_random.json`

## 8.2 重构目标

1. **减少训练脚本重复**：提取公共函数到 `tools/training_utils.py` 或 `src/training/base_trainer.py`
2. **统一 dataset 基类**：创建 `src/datasets/base_dataset.py`，让单帧/时序/semi 继承
3. **统一配置命名**：为所有已有配置添加语义化后缀
4. **保留向后兼容**：已有实验配置和训练命令继续可用

## 8.3 重构方案

### Phase 1：提取公共工具函数（低风险）

新增 `tools/training_utils.py`：

```python
def set_seed(seed: int): ...
def load_config(config_path: str) -> Dict[str, Any]: ...
def create_optimizer(model, cfg): ...
def create_scheduler(optimizer, cfg): ...
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, is_best=False): ...
def setup_experiment_dir(cfg, args) -> Path: ...
```

修改：
- `tools/train.py`
- `tools/train_temporal.py`
- `tools/train_semi.py`
- `tools/train_sam_random.py`

风险：低。只需验证每个脚本仍能跑 1 epoch。

### Phase 2：创建 BaseTrainer（中风险）

新增 `src/training/base_trainer.py`：

```python
class BaseTrainer(ABC):
    def __init__(self, cfg, device): ...
    def create_model(self, cfg): ...
    def create_dataloaders(self, cfg): ...
    def train_one_epoch(self, ...): ...
    def validate(self, ...): ...
    def fit(self, num_epochs): ...
```

各任务继承 `BaseTrainer`，只覆盖差异部分：
- `SingleTrainer`
- `TemporalTrainer`
- `SemiTrainer`
- `SAMRandomTrainer`

新增统一入口：

```bash
python tools/train.py --config configs/config_baseline.json
python tools/train.py --config configs/config_temporal_v3.json
python tools/train.py --config configs/config_semi_v2.json
python tools/train.py --config configs/config_sam_random.json
```

风险：中。需要确保各 trainer 的 forward/loss 逻辑完全一致。

### Phase 3：统一 Dataset 基类（中风险）

新增 `src/datasets/base_dataset.py`：

```python
class BaseFetoscopicDataset(Dataset):
    def __init__(self, data_path, mode, img_size, binary=True): ...
    def _load_image(self, path): ...
    def _load_gt(self, path): ...
    def _create_base_transform(self): ...
```

各 dataset 继承并扩展：
- `TTTSNetDataset(BaseFetoscopicDataset)`
- `TemporalDataset(BaseFetoscopicDataset)`
- `SemiDataset(BaseFetoscopicDataset)`

风险：中。需要验证增强行为一致。

### Phase 4：配置重命名（低风险）

```
configs/config.json -> configs/config_baseline.json
configs/config_temporal.json -> configs/config_temporal_v1.json
configs/config_temporal_no_loss.json -> configs/config_temporal_no_loss_v1.json
configs/config_semi.json -> configs/config_semi_v1.json
```

保留旧文件作为软链接或副本，直到所有脚本迁移完成。

## 8.4 验证策略

每完成一个 phase，执行：
1. `python -m py_compile` 检查语法
2. 跑 1-2 epochs 短周期验证，确认指标与重构前一致
3. 不删除旧脚本，直到新流程稳定运行 3 个以上完整实验

## 8.5 优先级

| Phase | 优先级 | 建议时机 |
|---|---|---|
| Phase 1 提取公共函数 | P2 | SAM 训练完成后 |
| Phase 2 BaseTrainer | P2 | Phase 1 验证后 |
| Phase 3 统一 Dataset | P3 | Temporal v3 / Semi v2 实验前 |
| Phase 4 配置重命名 | P3 | Phase 2 完成后 |

## 8.6 当前已完成的清理

- ✅ 删除 `__pycache__`
- ✅ 添加 `.gitignore`
- ✅ 移动早期 debug 实验目录到 `experiments/deprecated/debug_runs/`
- ✅ 创建 `tools/compile_experiment_results.py` 统一整理实验结果
- ✅ 整理 `project_material/` 实验报告

---

**最后更新**: 2026-06-20
