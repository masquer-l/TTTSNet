# SFY 视频筛选与标注流程

本目录包含 TTTSNet 项目用于 SFY 胎儿镜手术视频数据筛选、标注与导出的命令行工具。

## 目录结构

```text
/mnt/d/torch_project/dataset/sfy_screening/
├── previews/          # 2 秒间隔低分辨率预览帧 (JPEG, ≤960px 边长)
├── overlays/          # CNN baseline 二值预测掩码 (PNG)
├── segment_frames/    # 按需缓存的段级原分辨率帧
├── annotations/       # LabelMe v3 JSON / PNG 精修标注
├── exports/           # 训练就绪导出目录
├── db/
│   └── review.db      # SQLite 单源数据库
└── logs/              # 运行日志
```

## 初始化

首次使用或重置数据库：

```bash
cd /mnt/d/torch_project/TTTSNet
python3 -m tools.screening.db
```

> 该命令会创建 `review.db` 并启用 WAL 模式，以支持预览抽取与 overlay 生成并发运行。

### 测试数据库隔离

自动化测试、CI 和功能调试必须使用独立的测试数据库，禁止直接写入生产 `review.db`：

```bash
# 初始化测试库（schema only，不含数据）
python3 -m tools.screening.db --test
```

在测试代码中，**必须在导入任何 screening 模块前**设置环境变量：

```python
import os
os.environ["SFY_USE_TEST_DB"] = "1"

# 然后才能导入 screening 模块
from tools.screening import db
```

生产代码默认连接 `review.db`；设置 `SFY_USE_TEST_DB=1` 后连接 `review_test.db`。

## 标准流程

### 1. 载入视频元数据

```bash
python3 tools/screening/build_segment_registry.py --metadata experiments/sfy_video_metadata.csv
```

此脚本会：
- 将 `sfy_video_metadata.csv` 载入 `videos` 表；
- 为每个 `status='ok'` 的视频切分 30 秒 segment；
- 注册所有 frames，并标记 2 秒间隔的预览帧。

### 2. 抽取 2 秒预览帧

```bash
python3 tools/screening/extract_previews.py --workers 4
```

- 多视频并行（默认 4 个线程），每个视频内部顺序读取，避免随机 seek 开销。
- 抽取前会检查磁盘空间是否低于 20GB。
- 大约需要 6–8 小时处理全部 331 个视频 / ~96k 预览帧。

### 3. 生成 CNN baseline 辅助 overlay

```bash
python3 tools/screening/generate_cnn_overlays.py --batch-size 32
```

- 自动检测 CUDA，无 GPU 时加 `--cpu`。
- 已做按 batch 写入 + SQLite lock 重试，可与 `extract_previews.py` 并发运行，但**推荐在预览抽取完成后统一运行**，以避免 overlay 漏掉后续生成的预览。

### 4. 启动 Web 筛选界面

```bash
python3 web_screening/app.py
```

浏览器打开 `http://127.0.0.1:5000/`：
- Dashboard：查看整体进度与每例进度，可在每个视频旁点击"标记视频无效"将整个视频（含所有段/帧）排除。
- `/segments`：段级粗筛（valid / invalid / uncertain / split）。
- `/frames/<segment_id>`：帧级精筛，可开关 CNN overlay、抽取原分辨率帧。

> **注意**：标记视频无效时默认原因可选 `not_ttts`，该视频下所有段/帧会同步置为 invalid，且后续 `export_annotations.py` 会自动排除 `review_status='invalid'` 或 `'not_ttts'` 的视频。

### 5. 按需抽取段级原分辨率帧

在 Frame Reviewer 页面点击“Extract full-res frames”，或命令行：

```bash
python3 tools/screening/extract_segment_frames.py --segment-id 01_2019-11-27_175549_VID003_00000000_00000900
```

### 6. 导出训练集

```bash
# 默认：仅导出带有人工像素 mask 的帧（label_source = manual）
python3 tools/screening/export_annotations.py --name sfy_manual --split-ratio 0.8

# 显式允许无人工 mask 的帧回退到 CNN overlay（label_source = cnn_fallback）
python3 tools/screening/export_annotations.py --name sfy_cnn_pseudo --split-ratio 0.8 --allow-cnn-fallback
```

输出目录位于 `sfy_screening/exports/<name>_<timestamp>/`：
- `images/`：训练图像（仅当 `--copy-images` 时复制，默认使用 manifest 中的绝对路径）。
- `labels/`：PNG 标签（默认仅使用人工标注 mask；`--allow-cnn-fallback` 时无人工 mask 的帧回退到 CNN overlay，尺寸自动对齐到图像）。
- `json/`：LabelMe v3 JSON（占位，可用于桌面精修）。
- `frame_manifest.csv`：训练清单，可被 `VideoFrameDataset` 直接读取，包含 `label_source` 字段（`manual` 或 `cnn_fallback`）。
- `dataset_config.json`：数据配置摘要。

> **注意**：默认导出不会把 CNN overlay 当作 GT。若需要导出伪标签，必须显式使用 `--allow-cnn-fallback`，并在下游训练/评测中清楚区分 `label_source`。

### 7. 清理缓存（可选）

导出后可删除 `segment_frames/` 中已确认的原分辨率缓存以释放空间：

```bash
# 删除已完成标注段的全分辨率缓存
python3 tools/screening/cleanup_cache.py --keep-exported
```

> `cleanup_cache.py` 如不存在，可手动删除对应 `segment_frames/<case>/<video_stem>/<segment_id>/` 目录。

## 训练对接

`src/dataset_video_manifest.py` 中的 `VideoFrameDataset` 可直接读取 `frame_manifest.csv`，并按 `split` 字段自动划分 train/valid，同时按 `case` 检查是否有数据泄漏：

```python
from src.dataset_video_manifest import VideoFrameDataset

# 仅使用人工标注（manual GT）
dataset = VideoFrameDataset(
    manifest_path="/mnt/d/torch_project/dataset/sfy_screening/exports/sfy_manual_20260705_123456/frame_manifest.csv",
    mode="train",
    img_size=448,
    binary=True,
    label_source="manual",  # 可选：仅加载 manual，排除 cnn_fallback
)
```

> **数据泄漏防护**：`VideoFrameDataset` 初始化时会校验同一 `case` 是否同时出现在 train/valid；若发现泄漏会抛出 `ValueError`。训练脚本应使用 `frame_manifest.csv` 中的 `split` 字段，而不是自行按帧随机划分。

## 常用查询

查看数据库汇总：

```bash
python3 - <<'PY'
import json
from tools.screening.db import get_summary
print(json.dumps(get_summary(), indent=2, ensure_ascii=False))
PY
```

导出 CSV 快照：

```bash
python3 - <<'PY'
from pathlib import Path
from tools.screening.db import export_snapshots
export_snapshots(Path("/mnt/d/torch_project/dataset/sfy_screening/db/snapshots"))
PY
```

## 标签规范

- 段/帧状态：`valid` / `invalid` / `uncertain` / `split`（仅段）。
- 无效原因：`no_vessel`, `out_of_focus`, `overexposure`, `underexposure`, `laser_glare`, `instrument_occlusion`, `blood_debris`, `motion_blur`, `outside_body`, `corrupted`, `other`。
- 困难标签：`fine_vessel`, `low_contrast`, `laser_spot`, `instrument`, `small_foreground`, `large_foreground`, `boundary_artifact`。

## 注意事项

1. **不要删除源视频**：无效视频/段仅通过状态标记排除。
2. **源视频路径**：`configs/*.json` 与数据库中保存的仍是当前机器路径；迁移环境时请更新 `tools/screening/config.py` 中的 `UNIFIED_DIR`。
3. **磁盘空间**：预览 ~7GB + overlay ~2GB；原分辨率帧按需生成，导出后及时清理。
4. **病例分层**：导出时按 `case` 随机划分训练/验证，避免同病例泄漏。
