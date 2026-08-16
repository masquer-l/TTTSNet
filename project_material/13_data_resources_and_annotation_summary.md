# TTTSNet 数据资源与标注方案总结

> 本文档汇总 TTTSNet 项目当前的数据资源、标注方案、配套代码工具及业务目标对齐情况。
> 创建时间：2026-07-18
> 对应代码版本：TTTSNet-v2.1.0（ViT + Layerwise LR）

---

## 1. 数据资源总览

### 1.1 SFY 院内胎儿镜视频数据

| 指标 | 数值 | 备注 |
|---|---|---|
| 病例数 | **69 例含可读视频** | case 01–70，其中 16/17/18/20/21/24 已全病例排除 |
| 视频文件数 | 364 个 | `ok` 349 个，`unreadable` 15 个 |
| 总大小 | **301.38 GB** | — |
| 总时长 | **54.06 小时** | — |
| 总帧数 | **4,419,105 帧** | 1920×1080，~22.7 fps |
| 30 秒 segment | **6,618 个** | — |
| 时间跨度 | 2019-11 至 2024-11 | — |
| 人工像素 mask | **0 帧** | 当前尚未开始真实人工像素标注 |
| 数据库 `valid` 帧 | **2,583 帧** | 真实人工可用性筛选结果（早期 967 帧测试污染已被覆盖） |
| 数据库 `invalid` 帧 | 275,608 帧 | 含整病例排除和帧级不可用标记 |
| 数据库 `uncertain` 帧 | 8 帧 | 待进一步判定 |
| 已审 segment | 0 valid / 368 invalid / 2 uncertain | 共 6,618 个 segment；segment 级 valid 待确认 |
| 审计记录 `frame_labels` | 279,209 条 | 含 `manual` / `screening` / `cnn_fallback` 等来源 |

### 1.2 数据目录结构

```text
/mnt/d/torch_project/dataset/
├── sfy_raw/                        # 原始视频（按病例名组织）
├── sfy_source_unified/             # 统一编号后的视频（case 01–70）
└── sfy_screening/                  # 筛选与标注工作区
    ├── previews/                   # 2 秒间隔低分辨率预览帧
    ├── overlays/                   # CNN baseline 二值预测掩码
    ├── segment_frames/             # 按需缓存的原分辨率帧
    ├── annotations/                # LabelMe v3 JSON / PNG 精修标注
    ├── exports/                    # 训练就绪导出目录
    └── db/review.db                # SQLite 状态库
```

### 1.3 公开数据集

| 数据集 | 训练集 | 测试集 | 用途 |
|---|---|---|---|
| FetReg2021 | 2,060 帧 | 658 帧 | 主训练 + 内部验证 |

### 1.4 关键元数据文件

| 文件 | 路径 | 内容 |
|---|---|---|
| 视频元数据 | `experiments/sfy_video_metadata.csv` | 每视频分辨率、帧率、时长、大小、可读状态 |
| 视频清单（完整） | `experiments/sfy_source_manifest_full.csv` | 原始路径 ↔ 统一路径映射 |
| 视频清单（节选） | `experiments/sfy_source_manifest.csv` | 早期子集清单 |
| 标注报告 | `annotations_report.csv` | 帧级标注记录（case/video/segment/frame/label/tags） |

---

## 2. 标注工作现状

### 2.1 当前进度

- **数据库状态**（`review.db`，截至 2026-08-09）：
  - 视频：349 ok / 15 unreadable
  - segment：6,618 个（0 valid / 368 invalid / 2 uncertain / 6,248 pending）
  - 帧：4,419,105 帧（2,583 valid / 275,608 invalid / 8 uncertain / 4,140,906 pending）
  - 审计记录 `frame_labels`：279,209 条
- **重要说明**：早期数据库中的 967 个 `valid` 帧来自自动化测试会话 `claude_annotation_workflow_test_2026-07-05T06-05-50.929921+00-00`（约 2 分钟内写入 984 次 `valid` 操作）。这些记录已被后续真实筛选操作覆盖；当前 frames 表中已无源于该测试会话的 `valid` 帧，不再影响进度统计。
- **真实人工进度**：
  - 人工像素 mask：**0 帧**
  - 真实人工可用性筛选：**2,583 帧 valid**、275,608 帧 invalid、8 帧 uncertain
- **标注报告**：`annotations_report.csv` 中的历史记录需重新从数据库生成，旧记录不能作为真实人工进度。

### 2.1a 病例级筛选结论

**已全病例排除**（所有 segment invalid）：

- case 16、17、18、20、21、24：无有效数据或整病例不可用。

**非 TTTS 病例需排除**：

- case 02、03、04、05：不属于 TTTS 队列；02/03/05 已部分标记 invalid，04 尚未处理。

**高质量病例（优先保留/标注）**：

- case 27：清晰度好，但胎儿镜有效圆形视野明显更大（crop_size≈1080），训练时需注意尺度差异。
- case 31：清晰度很高，数据质量好；VID008 文件损坏已排除。
- case 74：清晰度很高，预览完整无损坏。

**特殊质量病例**：

- case 22：清晰度低，非分辨率/码率问题，可能是镜头/光学/光照退化；标注需谨慎。

**已部分筛选的病例**：

- case 07、08、09、12、22、23、26、27、28、29、30、31、69、74 等已有不同程度的帧级 valid/invalid 标记。

### 2.2 已做工作

1. 视频数据统一化（`sfy_raw/` → `sfy_source_unified/`，case 01–70）
2. 元数据盘点（`sfy_video_metadata.csv`、`sfy_source_manifest_full.csv`）
3. 筛选/标注系统搭建（命令行工具 + Web 界面 + SQLite 数据库）
4. segment 切分（每 30 秒一段，共 6,618 段）
5. 预览帧/overlay 生成流程
6. 全量高清原图预提取（fullres），截至 2026-08-02 完成约 96.0%
7. 修复 full frame 放大图准确性：OpenCV `cap.set()` 改为完全顺序读取
8. case 29 VID001 等 OpenCV 解码失败视频的 preview 已用 ffmpeg 补齐
9. case 31 VID008 损坏视频已标记为 `unreadable` / `invalid`
10. 自适应视频 crop 参数估计（case 22/23/27/07 等）
11. 识别并记录高质量病例（27、31、74）和低质量病例（22）
12. 启动 batch_001 pixel 精修：10 例病例、200 帧，已生成 crop 图像与 ViT 伪标签，进入 X-AnyLabeling 人工精修阶段
13. 筛选/标注工作流已启动，尚未产出真实人工像素 GT

### 2.3 标注规范（当前定义）

**段/帧状态**：
- `valid`：可用
- `invalid`：不可用
- `uncertain`：不确定
- `split`：仅段级，需要拆分

**无效原因**：
`no_vessel`, `out_of_focus`, `overexposure`, `underexposure`, `laser_glare`, `instrument_occlusion`, `blood_debris`, `motion_blur`, `outside_body`, `corrupted`, `other`

**困难标签**：
`fine_vessel`, `low_contrast`, `laser_spot`, `instrument`, `small_foreground`, `large_foreground`, `boundary_artifact`

---

## 3. 标注方案设计

### 3.1 分层标注体系

| 层级 | 内容 | 成本 | 优先级 | 用途 |
|---|---|---|---|---|
| 第一层 | 血管像素 mask | 低 | **P0 必做** | 分割主线、外部验证 |
| 第二层 | 中心线、端点、分叉点、低可信区域 | 中 | **P1 推荐** | 拓扑感知、基金提示层 |
| 第三层 | 血管赤道、疑似交通支、A-A/A-V/V-V 类型、专家置信度 | 高 | P2 后置 | 基金后两年 |

### 3.2 质量分层（四级）

按申请书定义：
- **清晰**
- **轻度退化**
- **重度退化**
- **不可判读**

落地方式：
- 给每帧打亮度、对比度、反光比例、模糊程度、遮挡面积、血管可见性、器械干扰等维度分数
- 先人工标 100–200 帧作为金标准，再考虑规则/小模型自动分级
- 产出 `quality_labels.csv`
- 评测时按质量等级分别报指标

### 3.3 病例级划分

- 按病例/视频划分 train/val/test，**同一台手术连续帧不跨集**
- 固定划分并存成 `data_split.json`
- 每个 split 记录：病例数、帧数、前景占比分布、质量等级分布

### 3.4 关键帧筛选

- 从连续视频中抽关键帧：固定间隔 + 去重 + 剔除不可判读帧
- 保留帧的时间戳/帧号/来源视频
- 首批标注建议 300–500 帧覆盖尽量多病例

### 3.5 标注质控

- 标注文件版本管理
- 随机抽 10–20% 由第二位标注者/专家复核，计算标注者间一致性
- 困难情况统一处理原则并写进 `annotation_spec.md`
- 记录"不确定/不可判读"标签

---

## 4. 代码与工具链

### 4.1 命令行工具（`tools/screening/`）

| 脚本 | 功能 |
|---|---|
| `db.py` | 初始化/操作 SQLite 数据库 `review.db` |
| `build_segment_registry.py` | 载入视频元数据，切分 30 秒 segment |
| `extract_previews.py` | 抽取 2 秒间隔低分辨率预览帧 |
| `generate_cnn_overlays.py` | 用 CNN baseline 生成辅助预测掩码 |
| `extract_fullres_previews.py` | 抽取原分辨率预览帧 |
| `extract_segment_frames.py` | 按需抽取段级原分辨率帧 |
| `export_annotations.py` | 导出训练就绪数据集 |
| `estimate_video_crop.py` | 估计视频裁剪参数 |
| `crop.py` | 图像/掩码裁剪工具 |

### 4.2 Web 筛选界面（`web_screening/`）

```bash
python3 web_screening/app.py
```

- Dashboard：整体进度与每例进度
- `/segments`：段级粗筛
- `/frames/<segment_id>`：帧级精筛，可开关 CNN overlay、抽取原分辨率帧

### 4.3 训练对接

`src/dataset_video_manifest.py` 中的 `VideoFrameDataset` 可直接读取 `frame_manifest.csv`：

```python
from src.dataset_video_manifest import VideoFrameDataset

dataset = VideoFrameDataset(
    manifest_path="/mnt/d/torch_project/dataset/sfy_screening/exports/.../frame_manifest.csv",
    mode="train",
    img_size=448,
    binary=True,
)
```

### 4.4 伪标签相关代码

| 文件 | 功能 |
|---|---|
| `tools/generate_pseudo_labels.py` | 为无标注数据生成伪标签 |
| `src/dataset_semi.py` | labeled + pseudo 混合训练数据集 |
| `tools/train_semi.py` | 半监督训练入口 |

历史伪标签结果：20,133 帧 → 保留 18,745 帧（max_conf ≥ 0.9），半监督 v2 未超越 baseline，已暂停。

---

## 5. 实验资产

### 5.1 当前最强模型

- **TTTSNet-v2.1.0**：ViT backbone + 分层学习率
- 内部 Val：mIoU 0.6389±0.0025，Dice 0.7652
- SFY 跨域：mIoU 0.6524±0.0080，Dice 0.7723
- 参数量 94.92M，延迟 17.9ms

### 5.2 关键实验目录

| 实验 | 目录 | 说明 |
|---|---|---|
| CNN baseline | `experiments/tttsnet_single_baseline_20260623_222644/` | v1.0.0，修复 augmentation bug 后 |
| ViT SOTA | `experiments/tttsnet_vit_layerwise_lr_20260627_083345/` | v2.1.0，当前最强 |

> 权重文件（.pth）未提交仓库，需从原服务器复制。详见 `CLAUDE.md` 第 2 节。

---

## 6. 业务目标映射

### 6.1 论文线（短期 2–4 周）

**主问题**：低资源 TTTS 胎儿镜血管分割中，预训练视觉表征的跨域泛化与临床可用性系统评估。

**数据角色**：
- FetReg2021 是主要训练/验证基础
- 历史 SFY 外部评测集（92/94 帧口径需核对并冻结）可作为外部验证
- 院内 70 病例大规模数据尚未产生真实人工 GT，当前不能直接用于主实验
- 需要补齐：SFY 外部评测集口径核对、bootstrap CI、per-image paired test、failure mode taxonomy
- 可选：SFY 最小手工分层（高反光/器械/细血管）做 subgroup 分析

**所需标注量**：当前真实人工像素 mask 为 0 帧。若要在论文中使用院内数据，需先完成首批 300–500 帧像素标注并通过质控；短期内仍以历史 SFY 外部评测集为主要外部验证。

### 6.2 基金线（三年愿景）

**主线**：拓扑感知分割 → 骨架/图谱 → 短窗口配准 → 交通支候选 → 专家确认。

**数据角色**：
- 第一年：FetReg + SFY 70 病例数据作为"已建立稳定基线"
- 第二年/第三年：图谱/配准/交通支目前无代码、无标注，需从零开始

**必须补齐的文档/产物**：
- 伦理批件编号、纳入时间范围、脱敏流程
- `data_inventory.csv`
- `data_split.json`（病例级划分）
- `quality_labels.csv`（质量分层）
- `annotation_spec.md`（标注规范）
- `eval_protocol.md`（评测协议）
- `DATASET.md`（数据集说明）

---

## 7. 风险与注意事项

1. **数据泄漏风险**：必须按病例/视频划分 train/val/test，同一手术连续帧不跨集。
2. **绝对路径问题**：`configs/*.json`、数据库、manifest 中均为本地绝对路径，迁移环境需更新。
3. **SFY 统计力**：虽然数据总量大，但当前已标注帧极少，小样本处理需 bootstrap CI + paired test。
4. **不可读视频**：15 个 `unreadable` 视频需排查原因（部分为 macOS 资源 forks `._*` 文件）。
5. **伦理与脱敏**：院内数据未脱敏前不要上传/公开，论文配图需过审。

---

## 8. 推荐下一步（按优先级）

### 8.1 论文优先路径

- [ ] 冻结论文主问题与 3 条贡献
- [ ] 统一指标口径（mIoU/Dice/clDice/surface_dice 成对出现）
- [ ] 对现有 SFY 已标注帧做 bootstrap CI + paired test
- [ ] 挑 8–12 个 qualitative case，出 baseline vs ViT 对比图
- [ ] 写一页 failure mode taxonomy

### 8.2 数据/基金优先路径

- [ ] 确认伦理批件与脱敏流程
- [ ] 产出台账 `data_inventory.csv`
- [ ] 做病例级划分 `data_split.json`
- [ ] 启动质量分层标注（先 100–200 帧金标准）
- [ ] 写 `annotation_spec.md` 和 `eval_protocol.md`
- [ ] 首批 300–500 帧 pixel mask 标注

---

## 9. 相关文件索引

| 类型 | 文件 | 说明 |
|---|---|---|
| 项目说明 | `CLAUDE.md` | 整体工作指南、权重路径、复现命令 |
| 实验材料 | `project_material/12_deep_review_and_guidance.md` | 深度复盘、数据实施 checklist |
| 论文方向 | `project_material/11_paper_direction_guidance.md` | 论文主线与防发散规则 |
| 伪标签 | `project_material/03_pseudo_label_supervision.md` | 半监督历史与改进方向 |
| 筛选流程 | `tools/screening/README.md` | 标准筛选/标注/导出流程 |
| 视频元数据 | `experiments/sfy_video_metadata.csv` | 视频技术参数 |
| 视频清单 | `experiments/sfy_source_manifest_full.csv` | 原始↔统一路径映射 |
| 标注记录 | `annotations_report.csv` | 当前帧级标注记录 |

---

**最后更新**：2026-08-16
