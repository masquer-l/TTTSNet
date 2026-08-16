# TTTSNet 数据筛选与标注推进计划

> 创建时间：2026-07-18  
> 最近审查：2026-08-16（已更新数据筛选现状、batch_001 进展与远程同步）  
> 用途：记录当前真实状态、待办任务、验收标准和后续标注进度。
> 原则：`valid` 只表示“筛选可用”，只有经过人工确认的像素 mask 才能称为 GT。

---

## 1. 当前真实状态

### 1.1 数据资源

- 院内数据：69 个含可读视频的病例（case 01–70，部分病例缺失或全部无效）。
- 可读视频：349 个；`unreadable` 15 个。
- 总时长：约 54.06 小时。
- 总帧数：4,419,105 帧。
- 30 秒 segment：6,618 个。
- 历史 SFY 外部评测集：当前存在 92/94 帧两种口径，必须核对并冻结。
- 当前人工像素 mask：**0 帧**（仅完成可用性粗筛，尚未开始像素级标注）。

### 1.2 筛选进度（截至 2026-08-09）

数据库 `review.db` 当前真实状态：

| 级别 | valid | invalid | uncertain | pending | 已 review |
|------|-------|---------|-----------|---------|----------|
| 帧   | **2,583** | 275,608 | 8 | 4,140,906 | 278,199 |
| 片段 | 0 | 368 | 2 | 6,248 | 370 |

**说明**：

- 帧级 `valid` 2,583 为真实人工可用性筛选结果（不再是早期测试污染）。
- 早期自动化测试会话 `claude_annotation_workflow_test_2026-07-05T06-05-50.929921+00-00` 写入的 984 条记录已被后续真实操作覆盖；当前 frames 表中已无源于该测试会话的 `valid` 帧。
- 片段级 `valid` 仍为 0，说明目前主要在做帧级快速筛选，尚未批量确认 segment 有效。

### 1.3 病例级筛选结论

**已全病例排除（所有 segment invalid）**：

- case 16、17、18、20、21、24：无有效数据或整病例不可用。

**非 TTTS 病例需排除**：

- case 02、03、04、05：不属于 TTTS 队列；02/03/05 已开始标记 invalid，04 尚未处理。

**高质量病例（优先保留/标注）**：

- case 27：清晰度好，但胎儿镜有效圆形视野明显更大（crop_size≈1080），训练时需注意尺度差异。
- case 31：清晰度很高，数据质量好；VID008 文件损坏已排除。
- case 74：清晰度很高，预览完整无损坏。

**特殊质量病例**：

- case 22：清晰度低，非分辨率/码率问题，可能是镜头/光学/光照退化；标注需谨慎。

**已部分筛选的病例**：

- case 07、08、09、12、22、23、26、27、28、29、30、31、69、74 等已有不同程度的帧级 valid/invalid 标记。

### 1.4 最近工具修复

- **full frame 放大图准确性**：发现 OpenCV `cap.set()` 在 H.264 视频上 seek 不准且会导致解码错误，已改为完全顺序读取；清除了旧缓存并全量重新预提取高清原图。
- **高清原图预提取**：截至 2026-08-02 已完成约 96.0%（94,208 / 98,176 张 preview 帧有 fullres 缓存）。
- **case 29 preview 补全**：VID001 因 OpenCV 解码失败缺失 330 张 preview，已用 ffmpeg 批量补齐。
- **视频损坏处理**：case 31 VID008 `moov atom not found` 已标记为 `unreadable` / `invalid`。
- **自适应 crop 参数**：已为 case 22、23、27、07 等估计 crop 参数；case 27 因视野过大 crop_size 达 1080（图像边界）。
- **batch_001 像素精修启动**：2026-08-09 起从 10 例病例中抽取 200 帧，生成 crop 图像与 ViT 伪标签，进入 X-AnyLabeling 人工精修阶段。

### 1.5 当前工具能力

已经支持：

- 视频和 segment 注册；
- segment/帧级可用性筛选；
- 整视频 invalid 标记及级联更新；
- invalid 原因和困难标签；
- CNN overlay 查看；
- 原分辨率帧按需抽取 + 全量预提取；
- 自适应视频 crop 参数估计；
- 操作历史和 session 记录；
- manifest、图像和标签导出框架。

尚未完整支持：

- 浏览器内像素级 mask 绘制；
- LabelMe/CVAT 标注结果导入和数据库回写；
- 人工 GT 与 CNN overlay 的严格隔离；
- 四级质量分层；
- 双人复核和标注者一致性统计；
- 固定、可复现的病例级数据划分；
- 标注版本和标签来源追踪。

---

## 2. 推荐工作流

不建议自行开发完整的浏览器像素绘制工具。后续采用：

```text
院内视频
  → Web 工具做 segment 粗筛
  → 候选帧抽取、去重和分层抽样
  → LabelMe 或 CVAT 做像素标注
  → 导入脚本回写 review.db
  → 人工复核与版本冻结
  → 导出训练/验证数据
  → CNN 与 ViT 统一评测
```

职责划分：

- 当前 Web 工具：筛选、抽样、查看模型辅助结果；
- LabelMe/CVAT：人工像素标注；
- `review.db`：状态、来源、标注者、版本和质控记录；
- 导出工具：只导出符合要求的人工 GT。

---

## 3. 任务清单

### P0：修正现状与保护数据

- [ ] 备份 `review.db`，文件名带日期和用途。
- [ ] 生成测试污染明细，确认 961 个测试帧的范围。
- [ ] 将测试写入的帧状态恢复为 `pending`，但保留原始审计记录。
- [ ] 核对其余 6 个 `valid` 帧是否也属于功能测试。
- [ ] 重建真实人工进度统计。
- [ ] 从数据库重新生成 `annotations_report.csv`。
- [ ] 建立独立测试数据库 `review_test.db`。
- [ ] 禁止测试代码连接正式 `review.db`。
- [x] 修订 `project_material/13_data_resources_and_annotation_summary.md` 中“967 已标注/已筛选”的错误表述。
- [ ] 修复 `set_frame_paths_batch`：传入 `None` 时不得清空已有
  `preview_path`、`overlay_path` 或 `fullres_path`。
- [x] 为数据库连接统一启用 `busy_timeout` 和 `foreign_keys=ON`。
- [ ] 增加 schema 版本或迁移记录，避免继续使用临时 `ALTER TABLE` 补丁。
- [ ] 为 P0 数据完整性问题建立最小自动化测试。

验收标准：

- Dashboard 只显示真实人工筛选进度；
- 自动化测试不会再改写正式数据库；
- 测试污染可以通过 session/source 完整追溯。
- 按“preview → overlay → fullres”执行后，重跑任一步骤均不会丢失其他路径。
- 数据库并发写入时不出现未处理的 `database is locked`。

### P1：冻结术语和标注规范

- [ ] 创建 `annotation_spec.md`。
- [ ] 明确定义 `pending/valid/invalid/uncertain`。
- [ ] 明确血管边界、高光、遮挡、交叉和模糊区域的 mask 规则。
- [ ] 统一中文 UI 与代码中的英文标签。
- [ ] 定义四级质量标准：清晰、轻度退化、重度退化、不可判读。
- [ ] 明确 `uncertain` 的处理和仲裁机制。

最低规范：

- `valid`：适合进入后续人工标注候选池；
- `manual_gt`：已完成人工像素标注并通过最低质控；
- `reviewed_gt`：已完成第二人复核或专家确认；
- `model_overlay`：模型辅助结果，不是 GT；
- `uncertain`：不可直接进入训练或主评测集。

验收标准：

- 两位标注者面对同一困难样本时有可执行规则；
- 文档明确哪些标签可以进入训练、验证和测试集。

### P2：完善数据库与导入/导出链路

- [x] 为帧增加或规范化 `label_source`：
  `manual`、`model`、`imported`、`workflow_test`。
- [ ] 记录 `reviewer`、`review_status`、`annotation_version`。
- [ ] 增加 LabelMe/CVAT 导入脚本。
- [ ] 导入时校验图像尺寸、mask 尺寸、二值范围和空 mask。
- [ ] 修复 `update_frame_label`：人工 mask 路径必须同步写入
  `frames.annotation_mask_path` 当前态，不能只写 `frame_labels` 审计表。
- [ ] 为每次导入生成审计报告。
- [ ] 修改 `export_annotations.py`，默认只允许导出人工 GT。
- [ ] 禁止人工 mask 缺失时自动回退到 CNN overlay。
- [ ] 如果保留 overlay 导出能力，必须使用显式参数，并在 manifest 中标记
  `label_source=model`。
- [ ] manifest 增加 `label_source`、`reviewer`、`annotation_version`。
- [ ] 固定随机种子，或直接读取冻结的 `data_split.json`。
- [ ] 修复 segment 与 frame 状态冲突时的导出行为。
- [ ] 导出前生成审计摘要：人工 GT、模型标签、缺失标签、状态冲突分别计数。
- [ ] 修复 segment split：拆分时保留帧路径、状态、标签和审计关系。
- [ ] 统一 mask 二值化规则，禁止不同 Dataset 使用不同前景定义。
- [ ] 明确 fullres 图像与 preview overlay 的分辨率配对规则，禁止静默拉伸后当作 GT。

验收标准：

- 可完成“一帧 LabelMe 标注 → 导入数据库 → 导出训练数据”的端到端测试；
- 导出的每个标签都能追溯来源；
- CNN overlay 不会混入人工 GT 数据集。
- 准备“人工 mask / 仅 overlay / 无标签”三帧时，默认只导出人工 mask 帧。
- segment 拆分前后的帧路径、状态和标签保持一致。

### P2.1：修复训练数据接线与泄漏风险

- [ ] 为 `tools/train.py` 增加 manifest 数据模式，或新增受统一配置管理的
  manifest 训练入口。
- [ ] 修复 `VideoFrameDataset`：必须按 manifest 的 `split` 列过滤
  `train/val/test`。
- [ ] 加载数据后检查 train/val/test 的
  `(case_id, video_id, frame_idx)` 交集必须为零。
- [ ] 移除 `dataset_video_manifest.py` 中对 1080×1920 的 crop 硬编码，
  使用每个视频的真实原始分辨率。
- [x] 将 `estimate_video_crop.py` 纳入标准数据流程（已可用于估计，待接入导出流程）。
- [ ] 对齐 manifest Dataset 与 FetReg Dataset 的 mask 语义和必要增广。
- [ ] 用 10 帧人工 GT manifest 训练 1 epoch，验证 loss、checkpoint 和评测流程。

验收标准：

- 相同 `data_split.json` 重复导出时 split 完全一致；
- `VideoFrameDataset(mode="train")` 与 `mode="valid")` 样本交集为零；
- 10 帧人工 GT 可以完成“加载 → 训练 1 epoch → 保存 → 重新加载 → 评测”；
- 非 1080p 视频的图像与 mask crop 后仍严格对齐。

### P2.2：修复 Web 筛选工具

- [ ] 修复 `/segments` 页面缺失的 `ensureSession()` 函数及 JavaScript 语法错误。
- [ ] 修复 `/frames_old/<segment_id>` 的箭头函数语法和 API 响应解析。
- [ ] 统一 `/frames`、`/frames_old/<segment_id>` 与段列表中的跳转路由。
- [ ] 对 `sort` 参数使用白名单，不直接拼接到 SQL。
- [ ] 对 preview/overlay 文件路径执行 `resolve()` 后的根目录边界检查。
- [ ] 服务端校验 segment/frame 是否存在以及状态是否属于允许枚举。
- [ ] 非 `invalid` 状态必须清空旧 `invalid_reason`。
- [ ] 修复帧级 `invalid_reason` 和 `notes` 保存后无法回显的问题。
- [ ] 统一标签词表：数据库存英文 key，界面显示中文 label。
- [ ] 明确“整段有效”是否批量更新帧；界面和导出必须使用同一语义。
- [ ] 增加撤销功能，利用审计表恢复上一状态。
- [ ] 限制分页 `limit`，并为 segment API 返回总数。
- [ ] 所有写操作统一显示成功或失败，不允许静默失败。
- [ ] 生产运行关闭 Flask debug，并保持仅本机访问或增加 API token。

验收标准：

- `/segments`、`/frames` 和段到帧跳转均无控制台错误或 404；
- 非法状态、非法 sort 和不存在的 ID 返回 400/404；
- `../` 路径请求不能访问筛选目录外文件；
- 保存 invalid 原因后刷新页面仍可正确回显；
- 一次撤销能恢复上一条状态。

### P2.3：多人复核前的扩展任务

首批单人标注可以暂不实现，增加第二位标注者前必须完成：

- [ ] reviewer 身份由服务端绑定，不能完全信任客户端参数。
- [ ] 建立病例或 segment 级任务分配/占用机制。
- [ ] 增加 `updated_at/version` 乐观锁，发生并发修改时返回冲突。
- [ ] session 结束和工作量统计由服务端根据审计记录计算。
- [ ] 增加 primary/reviewer/adjudicated 三阶段复核状态。

### P3：首批关键帧筛选

目标：首批选择 300–500 帧，覆盖至少 8 个病例。

- [ ] 先做 segment 级粗筛，不逐帧遍历 440 万帧。
- [ ] 每个病例选择多个相互独立的短片段。
- [ ] 按 2–5 秒间隔建立候选帧池。
- [ ] 对候选帧做相似度去重。
- [ ] 按质量、干扰类型和血管结构分层抽样。
- [ ] 避免从单个 30 秒片段连续选择数百帧。
- [ ] 保存候选清单 `annotation_batch_001.csv`。

首批建议覆盖：

- 清晰常规血管；
- 细小血管；
- 低对比；
- 激光光斑或高反光；
- 器械遮挡；
- 羊水浑浊、血液或组织碎片；
- 前景面积过小或过大；
- 边界模糊和运动模糊。

建议字段：

```text
batch_id
case_id
video_id
segment_id
frame_idx
time_sec
quality_level
difficulty_tags
selection_method
selection_reason
image_path
annotation_status
reviewer
```

验收标准：

- 覆盖不少于 8 个病例；
- 没有单病例或单片段主导样本量；
- 每帧保留病例、视频、时间戳和选择原因。

### P4：首批像素标注与质控

- [ ] 先标 20–30 帧进行规范试运行。
- [ ] 集中讨论边界、高光、遮挡等分歧。
- [ ] 修订 `annotation_spec.md`。
- [ ] 再完成剩余 300–500 帧。
- [ ] 随机抽取 10%–20% 进行第二人复核。
- [ ] 计算标注者间 Dice/IoU。
- [ ] 记录所有仲裁和修订。
- [ ] 冻结首批 GT 版本。

验收标准：

- 所有 mask 能通过自动完整性检查；
- 复核子集的一致性达到预先设定标准；
- 不确定样本未被强行作为确定 GT；
- 首批 GT 有不可变版本号和清单。

### P5：数据划分与评测协议

- [ ] 创建 `data_split.json`。
- [ ] 按病例划分 train/val/test。
- [ ] 同一病例和同一手术视频不跨 split。
- [ ] 冻结历史 SFY 外部评测集，核对 92/94 帧差异。
- [ ] 创建 `eval_protocol.md`。
- [ ] 统一 resize、mask 二值化、阈值和后处理规则。
- [ ] 主指标使用 mIoU、Dice、clDice、Surface Dice。
- [ ] 报告 bootstrap 置信区间和 per-image paired test。
- [ ] 按质量和困难类型做 subgroup 分析。

验收标准：

- 所有模型共用相同数据划分和评测代码；
- 评测集不包含模型生成的伪标签；
- 论文结果可从冻结清单复现。

### P6：结构标注与基金扩展

仅在 P0–P5 跑通后启动：

- [ ] 从首批 GT 中选 50–100 帧。
- [ ] 自动骨架化后人工修正中心线。
- [ ] 标注端点、分叉点和低可信区域。
- [ ] 建立骨架召回、分叉召回和连通组件错误指标。
- [ ] 交通支和 A-A/A-V/V-V 临床语义暂不大规模铺开。

---

## 4. 两周推荐安排

### 第 1–2 天

- 完成数据库备份和测试污染清理；
- 修订真实进度；
- 建立独立测试数据库；
- 修复批量路径更新清空已有路径的问题；
- 禁止导出工具默认回退 CNN overlay；
- 写完 `annotation_spec.md` v0.1。

### 第 3–5 天

- 完成 LabelMe/CVAT 导入脚本；
- 修复人工 mask 当前态回写；
- 修改导出逻辑，严格区分人工 GT 与模型 overlay；
- 修复 Dataset split 过滤和训练 manifest 接线；
- 跑通 5–10 帧端到端流程。

### 第 6–7 天

- 建立跨病例候选池；
- 去重并生成 `annotation_batch_001.csv`；
- 试标 20–30 帧并修订规范。

### 第 2 周

- 推进 300–500 帧像素标注；
- 同时完成 10%–20% 双阅；
- 冻结 `data_split.json` 和 `eval_protocol.md`；
- 生成第一版数据质量报告。

---

## 5. 暂缓事项

在首批人工 GT 闭环完成前，暂缓：

- 继续扩大连续帧 `valid` 数量；
- 使用 CNN overlay 作为主训练或评测 GT；
- 大规模中心线、分叉点和交通支标注；
- 重启无明确数据闭环的 Temporal/Semi 实验；
- 增加新的复杂模型模块或损失函数；
- 把院内数据直接混入训练集后再称其为独立外部验证。

---

## 6. 每周跟进模板

### 周期

```text
日期：
负责人：
本周目标：
```

### 数据进度

```text
已粗筛病例数：
已粗筛 segment 数：
候选帧数：
人工 pixel mask 数：
已复核 mask 数：
质量分层帧数：
中心线/分叉点标注数：
```

### 质量与风险

```text
标注者间 Dice/IoU：
待仲裁样本数：
数据泄漏检查：
标签来源检查：
新增问题：
```

### 本周产出

- [ ] 数据清单
- [ ] 标注规范更新
- [ ] GT 版本
- [ ] 质控报告
- [ ] 评测结果
- [ ] 失败案例

### 下周任务

1. 
2. 
3. 

---

## 7. 关键文件

```text
项目根目录：
  ANNOTATION_WORK_PLAN.md

当前状态总结：
  project_material/13_data_resources_and_annotation_summary.md

数据库：
  /mnt/d/torch_project/dataset/sfy_screening/db/review.db

筛选工具：
  tools/screening/
  web_screening/

重点待修改：
  tools/screening/db.py
  tools/screening/export_annotations.py
  web_screening/routes/api.py

计划新增：
  annotation_spec.md
  eval_protocol.md
  data_split.json
  data_inventory.csv
  quality_labels.csv
  DATASET.md
  annotation_batch_001.csv
```

---

## 8. 当前第一行动

用户已完成阶段性数据粗筛，并于 2026-08-09 启动 batch_001 pixel 精修（10 例病例、200 帧）。当前应优先完成：

1. **推进 batch_001 人工精修**：使用 X-AnyLabeling 完成 200 帧 polygon 精修，标记 `reviewed` / `unreviewable`，定期运行 `tools/screening/sync_and_filter.sh` 刷新工作集；
2. **完成剩余有效病例的帧级粗筛**：重点把 case 04、06–15、19、25、32–33、35–70 等待审病例快速过一遍，标记明显 invalid 和高质量 valid；
3. **整病例排除**：把 case 02、03、04、05 全部标记为非 TTTS / invalid；
4. **segment 级确认**：对已产生大量 valid 帧的病例（27、29、31、74 等）批量把 segment 标为 valid；
5. **产出 `data_inventory.csv`**：记录每个病例的可用性结论、质量等级、排除原因；
6. **修订 `annotation_spec.md` v0.1**：基于 batch_001 试标结果，明确血管边界、高光、遮挡、细血管等绘制规则；
7. **打通“候选帧 → 像素标注 → 导入 review.db → 导出训练数据”闭环**：验证 batch_001 精修后的 LabelMe JSON 能正确导入数据库并导出训练样本。

完成 batch_001 精修与规范闭环后，再视情况扩大 pixel 标注批次。
