# YOLO 配置与后端参数对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 YOLO 项目中 train/validate/predict/track/export 配置与后端实际支持参数不一致的问题，确保 YAML 不含无效参数、不遗漏后端已支持的任务专用参数，并通过静态校验防止回归。

**Architecture:** 增加一个静态配置审计脚本作为可重复验证入口；先用它暴露 YAML 与后端 schema 的差异，再修正 train 后端的错误透传和各模式示例 YAML。业务配置保持精简，example YAML 负责完整展示对应模式/任务可配置项。

**Tech Stack:** Python 3、PyYAML、Ultralytics YOLO、现有 `commands/*.py` 配置读取体系、YAML 配置文件。

---

## File Structure

- Create: `scripts/audit_config_schema.py`
  - 负责静态审计所有 `configs/{train,predict,validate,export}` YAML。
  - 内置按模式和任务划分的允许字段、任务专用字段、禁止字段。
  - 输出所有 unknown、wrong-task、missing-example、invalid-train-forwarding 问题。

- Modify: `commands/train.py`
  - 修正训练阶段 validation 参数透传白名单。
  - 删除会导致 Ultralytics `model.train()` invalid argument 的 `topk`。
  - 按任务过滤 `end2end` 等不适用参数。

- Modify: `configs/train/**/*.yaml`
  - 清理训练配置中无效或错任务字段。
  - 补齐 example 中后端支持但未展示的任务专用字段。

- Modify: `configs/validate/**/*.yaml`
  - 保留独立 `val` 支持的 `topk`/`kpt_thres`，但只放到对应 classify/pose 示例。
  - 补齐 validate 后端支持的常用参数示例。

- Modify: `configs/predict/**/*.yaml`
  - 按 predict/track 后端支持字段对齐。
  - 任务专用字段只放到对应示例。

- Modify: `configs/export/**/*.yaml`
  - 检查并对齐格式专用参数：engine workspace、saved_model keras、int8 校准、nms/end2end 互斥。

---

### Task 1: 添加配置静态审计脚本

**Files:**
- Create: `scripts/audit_config_schema.py`

- [ ] **Step 1: 创建审计脚本**

写入 `scripts/audit_config_schema.py`：

```python
#!/usr/bin/env python3
"""Audit YOLO YAML configs against project-supported config schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOTS = ("train", "predict", "validate", "export")
TASKS = {"detect", "segment", "pose", "obb", "classify"}

COMMON_TRAIN = {
    "mode",
    "cfg",
    "tracker",
    "model.name",
    "model.yaml",
    "model.pretrained",
    "model.task",
    "model.reg_max",
    "model.classes",
    "data.config",
    "data.split",
    "train.epochs",
    "train.time",
    "train.batch",
    "train.imgsz",
    "train.device",
    "train.workers",
    "train.resume",
    "train.patience",
    "train.cache",
    "train.save",
    "train.amp",
    "train.rect",
    "train.single_cls",
    "train.fraction",
    "train.freeze",
    "train.multi_scale",
    "train.compile",
    "train.nbs",
    "train.profile",
    "train.optimizer",
    "train.lr0",
    "train.lrf",
    "train.momentum",
    "train.weight_decay",
    "train.cos_lr",
    "train.warmup_epochs",
    "train.warmup_momentum",
    "train.warmup_bias_lr",
    "train.box",
    "train.cls",
    "train.cls_pw",
    "train.dfl",
    "train.close_mosaic",
    "train.seed",
    "train.deterministic",
    "augmentation.hsv_h",
    "augmentation.hsv_s",
    "augmentation.hsv_v",
    "augmentation.degrees",
    "augmentation.translate",
    "augmentation.scale",
    "augmentation.shear",
    "augmentation.perspective",
    "augmentation.flipud",
    "augmentation.fliplr",
    "augmentation.bgr",
    "augmentation.mosaic",
    "augmentation.mixup",
    "augmentation.cutmix",
    "augmentation.copy_paste",
    "augmentation.copy_paste_mode",
    "augmentation.auto_augment",
    "augmentation.erasing",
    "validation.val",
    "validation.conf",
    "validation.iou",
    "validation.max_det",
    "validation.half",
    "validation.plots",
    "validation.dnn",
    "validation.agnostic_nms",
    "validation.augment",
    "validation.save_conf",
    "validation.save_json",
    "validation.int8",
    "validation.save_txt",
    "validation.save_crop",
    "validation.show",
    "validation.show_labels",
    "validation.show_conf",
    "validation.show_boxes",
    "validation.line_width",
    "validation.retina_masks",
    "validation.visualize",
    "validation.embed",
    "validation.vid_stride",
    "output.project",
    "output.name",
    "output.save_period",
    "output.exist_ok",
    "output.verbose",
}

TRAIN_TASK_KEYS = {
    "detect": {"train.end2end", "validation.end2end"},
    "segment": {"train.overlap_mask", "train.mask_ratio"},
    "pose": {"train.pose", "train.kobj", "train.rle", "validation.kpt_thres"},
    "obb": {"train.angle"},
    "classify": {"train.dropout"},
}

VALIDATE_COMMON = {
    "mode",
    "model.name",
    "model.task",
    "model.imgsz",
    "model.batch",
    "model.device",
    "model.classes",
    "model.embed",
    "model.vid_stride",
    "model.source",
    "data.config",
    "data.split",
    "train.imgsz",
    "train.batch",
    "train.device",
    "validation.conf",
    "validation.iou",
    "validation.max_det",
    "validation.half",
    "validation.plots",
    "validation.save_json",
    "validation.dnn",
    "validation.agnostic_nms",
    "validation.augment",
    "validation.rect",
    "validation.save_conf",
    "validation.int8",
    "validation.fraction",
    "validation.save_txt",
    "validation.save_crop",
    "validation.show",
    "validation.show_labels",
    "validation.show_conf",
    "validation.show_boxes",
    "validation.line_width",
    "validation.retina_masks",
    "validation.visualize",
    "validation.workers",
    "validation.cache",
    "output.project",
    "output.name",
    "output.save_period",
    "output.exist_ok",
    "output.verbose",
}

VALIDATE_TASK_KEYS = {
    "detect": {"validation.end2end"},
    "segment": set(),
    "pose": {"validation.kpt_thres", "model.kpt_thres"},
    "obb": set(),
    "classify": {"validation.topk", "model.topk"},
}

PREDICT_COMMON = {
    "mode",
    "model.path",
    "model.imgsz",
    "model.device",
    "model.batch",
    "model.classes",
    "model.stream",
    "model.half",
    "model.augment",
    "model.vid_stride",
    "model.retina_masks",
    "model.visualize",
    "model.embed",
    "model.int8",
    "model.line_width",
    "model.save_frames",
    "model.stream_buffer",
    "model.save_conf",
    "model.dnn",
    "model.end2end",
    "model.show",
    "model.show_boxes",
    "io.input",
    "io.output",
    "io.save_vis",
    "io.save_json",
    "io.save_txt",
    "io.save_crop",
    "nms.conf",
    "nms.iou",
    "nms.max_det",
    "nms.agnostic_nms",
    "nms.end2end",
    "visualization.box_thickness",
    "visualization.font_scale",
    "visualization.show_labels",
    "visualization.show_conf",
    "visualization.mask_alpha",
    "visualization.kpt_radius",
    "visualization.kpt_line",
    "visualization.pose_fill",
    "visualization.pose_fill_alpha",
    "visualization.obb_show_direction",
    "visualization.obb_direction_color",
    "visualization.obb_direction_length",
    "visualization.obb_direction_thickness",
    "video.fps",
    "video.codec",
    "output.verbose",
}

PREDICT_TASK_KEYS = {
    "detect": set(),
    "segment": set(),
    "pose": {"model.kpt_thres", "nms.kpt_thres", "visualization.skeleton", "visualization.kpt_names"},
    "obb": set(),
    "classify": {"model.topk", "nms.topk"},
}

TRACK_KEYS = PREDICT_COMMON | {
    "tracker",
    "model.tracker",
    "model.persist",
    "model.topk",
    "nms.topk",
    "model.kpt_thres",
    "nms.kpt_thres",
    "visualization.skeleton",
    "visualization.kpt_names",
}

EXPORT_KEYS = {
    "mode",
    "model.path",
    "model.format",
    "model.imgsz",
    "model.batch",
    "model.device",
    "export.opset",
    "export.simplify",
    "export.dynamic",
    "export.half",
    "export.nms",
    "export.optimize",
    "export.int8",
    "export.data",
    "export.fraction",
    "export.split",
    "export.workspace",
    "export.keras",
    "export.conf",
    "export.iou",
    "export.max_det",
    "export.agnostic_nms",
    "export.end2end",
    "output.path",
    "output.verbose",
    "verify.enabled",
    "verify.source",
}

TRAIN_FORBIDDEN = {"validation.topk", "model.topk"}

EXAMPLE_REQUIRED = {
    ("train", "classify"): {"train.dropout", "augmentation.auto_augment", "augmentation.erasing"},
    ("train", "segment"): {"train.overlap_mask", "train.mask_ratio", "augmentation.copy_paste", "augmentation.copy_paste_mode"},
    ("train", "pose"): {"train.pose", "train.kobj", "train.rle", "validation.kpt_thres"},
    ("train", "obb"): {"train.angle"},
    ("train", "detect"): {"train.end2end", "validation.end2end"},
    ("validate", "classify"): {"validation.topk"},
    ("validate", "pose"): {"validation.kpt_thres"},
    ("predict", "classify"): {"model.topk"},
    ("predict", "pose"): {"model.kpt_thres", "visualization.skeleton", "visualization.kpt_names"},
}

@dataclass(frozen=True)
class Issue:
    path: Path
    kind: str
    key: str
    message: str


def flatten(data: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            keys.add(current)
            if isinstance(value, dict):
                keys.update(flatten(value, current))
    return keys


def leaf_keys(data: dict[str, Any]) -> set[str]:
    all_keys = flatten(data)
    parents = {key.rsplit(".", 1)[0] for key in all_keys if "." in key}
    return {key for key in all_keys if key not in parents}


def infer_mode(path: Path, data: dict[str, Any]) -> str:
    mode = data.get("mode")
    if mode == "val":
        return "validate"
    if mode in {"train", "predict", "track", "export"}:
        return mode
    parts = path.parts
    if "validate" in parts:
        return "validate"
    if "train" in parts:
        return "train"
    if "predict" in parts:
        return "predict"
    if "export" in parts:
        return "export"
    return "unknown"


def infer_task(path: Path, data: dict[str, Any]) -> str | None:
    task = data.get("model", {}).get("task")
    if task in TASKS:
        return task
    name = path.name
    for candidate in TASKS:
        if candidate in name:
            return candidate
    return None


def allowed_keys(mode: str, task: str | None) -> set[str]:
    if mode == "train":
        return COMMON_TRAIN | TRAIN_TASK_KEYS.get(task or "", set())
    if mode == "validate":
        return VALIDATE_COMMON | VALIDATE_TASK_KEYS.get(task or "", set())
    if mode == "predict":
        return PREDICT_COMMON | PREDICT_TASK_KEYS.get(task or "", set())
    if mode == "track":
        return TRACK_KEYS
    if mode == "export":
        return EXPORT_KEYS
    return set()


def audit_file(path: Path) -> list[Issue]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return [Issue(path, "invalid-yaml", "", "YAML root must be a mapping")]

    mode = infer_mode(path.relative_to(ROOT), data)
    task = infer_task(path, data)
    keys = leaf_keys(data)
    allowed = allowed_keys(mode, task)
    issues: list[Issue] = []

    for key in sorted(keys):
        if key not in allowed:
            issues.append(Issue(path, "unknown", key, f"{key} is not allowed for mode={mode}, task={task}"))

    if mode == "train":
        for key in sorted(keys & TRAIN_FORBIDDEN):
            issues.append(Issue(path, "invalid-train", key, f"{key} must not be present in train configs"))

    if "example" in path.parts and task:
        required = EXAMPLE_REQUIRED.get((mode, task), set())
        for key in sorted(required - keys):
            issues.append(Issue(path, "missing-example", key, f"example for mode={mode}, task={task} should document {key}"))

    if mode == "export":
        export_cfg = data.get("export", {})
        fmt = data.get("model", {}).get("format")
        if export_cfg.get("nms") is True and export_cfg.get("end2end") is True:
            issues.append(Issue(path, "invalid-export", "export.nms/export.end2end", "nms=true and end2end=true are mutually exclusive"))
        if "workspace" in export_cfg and fmt != "engine":
            issues.append(Issue(path, "wrong-format", "export.workspace", "workspace only applies to TensorRT engine export"))
        if "keras" in export_cfg and fmt != "saved_model":
            issues.append(Issue(path, "wrong-format", "export.keras", "keras only applies to saved_model export"))

    return issues


def main() -> int:
    config_files = []
    for root_name in CONFIG_ROOTS:
        config_files.extend((ROOT / "configs" / root_name).rglob("*.yaml"))

    issues: list[Issue] = []
    for path in sorted(config_files):
        issues.extend(audit_file(path))

    if not issues:
        print(f"OK: audited {len(config_files)} config files")
        return 0

    for issue in issues:
        rel = issue.path.relative_to(ROOT)
        print(f"{rel}: {issue.kind}: {issue.key}: {issue.message}")
    print(f"FAILED: {len(issues)} issue(s) across {len(config_files)} config files")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行审计，确认当前失败**

Run:

```bash
python scripts/audit_config_schema.py
```

Expected: FAIL，至少包含 `configs/train/anquandai/belt.yaml` 的 `validation.topk` 和相关 example 缺失/错任务问题。

---

### Task 2: 修正 train 后端透传白名单

**Files:**
- Modify: `commands/train.py`

- [ ] **Step 1: 修改 `commands/train.py` 的训练期间 validation 透传列表**

在 `train(config)` 中找到：

```python
for key in ("val", "conf", "iou", "max_det", "half", "plots", "dnn",
            "agnostic_nms", "augment", "save_conf", "save_json", "int8",
            "save_txt", "save_crop", "show", "show_labels", "show_conf",
            "show_boxes", "line_width", "retina_masks", "visualize",
            "kpt_thres", "topk", "embed", "vid_stride"):
```

替换为：

```python
# Validation 节参数 -> 传给 train。
# 注意: train 阶段的 validation 参数最终进入 Ultralytics model.train()，
# 不能照搬独立 val/predict 的后处理参数。比如 topk 仅适用于 classify val/predict，
# 传给 model.train() 会触发 "'topk' is not a valid YOLO argument"。
train_validation_keys = [
    "val", "conf", "iou", "max_det", "half", "plots", "dnn",
    "agnostic_nms", "augment", "save_conf", "save_json", "int8",
    "save_txt", "save_crop", "show", "show_labels", "show_conf",
    "show_boxes", "line_width", "retina_masks", "visualize",
    "embed", "vid_stride",
]

if task == "pose":
    train_validation_keys.append("kpt_thres")
if task in {"detect", "segment", "pose", "obb"}:
    train_validation_keys.append("end2end")

for key in train_validation_keys:
```

并删除后面单独的 `end2end_val` 重复传入风险，如果保留，要确保 classify 不传：

```python
if end2end_val is not None and task in {"detect", "segment", "pose", "obb"}:
    train_args["end2end"] = end2end_val
```

- [ ] **Step 2: 运行当前报错配置的最小验证**

Run:

```bash
python -m commands.train --config configs/train/anquandai/belt.yaml --epochs 1 --batch 2 --workers 0 --exist-ok
```

Expected: 不再出现 `'topk' is not a valid YOLO argument`。如果数据路径不存在或训练资源不足导致后续失败，记录后续错误，但确认 `topk` invalid 已消失。

---

### Task 3: 修正 train YAML 与训练示例

**Files:**
- Modify: `configs/train/anquandai/belt.yaml`
- Modify: `configs/train/example/classify_example.yaml`
- Modify: `configs/train/example/segment_example.yaml`
- Modify: `configs/train/example/pose_example.yaml`
- Modify: `configs/train/example/obb_example.yaml`
- Modify: `configs/train/example/detect_example.yaml`

- [ ] **Step 1: 删除 classify train 配置中的 `validation.topk`**

从以下文件删除或注释 `validation.topk`：

```text
configs/train/anquandai/belt.yaml
configs/train/example/classify_example.yaml
```

删除内容：

```yaml
  topk: 5                                                 # 分类 Top-K (classify 任务专用, 默认 5)
```

- [ ] **Step 2: 精简 classify train 示例中的检测专用验证项**

在 `configs/train/example/classify_example.yaml` 和 `configs/train/anquandai/belt.yaml` 中，保留 classify 训练真正相关项：

```yaml
validation:
  val: true                                               # 训练期间是否运行验证
  half: false                                             # FP16 半精度验证
  plots: true                                             # 生成分类验证图表/混淆矩阵
  augment: false                                          # 测试时增强 TTA (慢但可能提升精度)
```

删除 classify train 中的：

```yaml
  conf:
  iou: 0.7
  max_det: 300
  save_json: false
  dnn: false
  agnostic_nms: false
  save_conf: false
  int8: false
  end2end:
```

- [ ] **Step 3: 确保任务专用字段只出现在对应 train example**

检查并调整：

- `configs/train/example/detect_example.yaml` 包含：

```yaml
  end2end:                                                # 端到端检测头 (YOLO26/YOLOv10, 免 NMS)
```

- `configs/train/example/segment_example.yaml` 包含：

```yaml
  overlap_mask: true                                      # 分割任务专用: 训练时合并实例掩码
  mask_ratio: 4                                           # 分割任务专用: 掩码下采样比
```

以及 augmentation 中：

```yaml
  copy_paste: 0.0                                         # 分割任务专用: 复制粘贴增强概率
  copy_paste_mode: flip                                   # 分割任务专用: 复制粘贴策略 flip/mixup
```

- `configs/train/example/pose_example.yaml` 包含：

```yaml
  pose: 12.0                                              # 姿态任务专用: 关键点损失增益
  kobj: 1.0                                               # 姿态任务专用: 关键点目标损失增益
  rle: 1.0                                                # 姿态任务专用: RLE 损失增益
```

validation 中包含：

```yaml
  kpt_thres: 0.5                                          # 姿态任务专用: 关键点置信度阈值
```

- `configs/train/example/obb_example.yaml` 包含：

```yaml
  angle: 1.0                                              # OBB 任务专用: 角度损失增益
```

- `configs/train/example/classify_example.yaml` 包含：

```yaml
  dropout: 0.0                                            # 分类任务专用: 分类头 Dropout 概率
```

augmentation 中包含：

```yaml
  auto_augment: randaugment                               # 分类任务专用: randaugment/autoaugment/augmix
  erasing: 0.4                                            # 分类任务专用: 随机擦除概率
```

- [ ] **Step 4: 运行审计**

Run:

```bash
python scripts/audit_config_schema.py
```

Expected: train 相关 `validation.topk` 和缺失任务专用字段问题消失。

---

### Task 4: 修正 validate/predict/track/export 示例缺失与错任务字段

**Files:**
- Modify: `configs/validate/example/*.yaml`
- Modify: `configs/predict/example/*.yaml`
- Modify: `configs/predict/example/track_example.yaml`
- Modify: `configs/export/**/*.yaml`

- [ ] **Step 1: validate classify 示例保留 `topk`**

在 `configs/validate/example/classify_example.yaml` 中确保存在：

```yaml
validation:
  topk: 5                                                 # 分类任务专用: Top-K 准确率中的 K
```

并确保 detect/segment/pose/obb validate 示例不包含 `topk`。

- [ ] **Step 2: validate pose 示例保留 `kpt_thres`**

在 `configs/validate/example/pose_example.yaml` 中确保存在：

```yaml
validation:
  kpt_thres: 0.5                                          # 姿态任务专用: 关键点置信度阈值
```

并确保 classify/detect/segment/obb validate 示例不包含 `kpt_thres`。

- [ ] **Step 3: predict classify 示例暴露 `topk`**

在 `configs/predict/example/classify_example.yaml` 中确保存在：

```yaml
model:
  topk: 5                                                 # 分类任务专用: 返回 Top-K 分类结果
```

或：

```yaml
nms:
  topk: 5                                                 # 分类任务专用: 返回 Top-K 分类结果
```

二选一即可，推荐 `model.topk`。

- [ ] **Step 4: predict pose 示例暴露 `kpt_thres` 和关键点可视化**

在 `configs/predict/example/pose_example.yaml` 中确保存在：

```yaml
model:
  kpt_thres: 0.5                                          # 姿态任务专用: 关键点置信度阈值

visualization:
  skeleton:                                               # 姿态任务专用: 关键点连线
    - [0, 1]
    - [1, 2]
  kpt_names: []                                           # 姿态任务专用: 关键点名称，可按数据集填写
```

- [ ] **Step 5: track 示例明确 track 模式参数**

在 `configs/predict/example/track_example.yaml` 中确保存在：

```yaml
mode: track

model:
  tracker: botsort.yaml                                   # 跟踪器配置: botsort.yaml/bytetrack.yaml
  persist: false                                          # 是否跨帧保留跟踪状态

tracker: botsort.yaml                                     # 根级 tracker，兼容 commands/track.py
```

- [ ] **Step 6: export 示例检查格式专用参数**

确保：

- `configs/export/example/engine/*.yaml` 可以包含：

```yaml
export:
  workspace: 4                                            # TensorRT 工作区大小 GB，仅 engine 生效
```

- 非 engine 示例不包含 `export.workspace`。
- `configs/export/example/saved_model/*.yaml` 可以包含：

```yaml
export:
  keras: false                                            # saved_model 专用: 导出 Keras 模型
```

- 非 saved_model 示例不包含 `export.keras`。
- 任意 export YAML 不允许：

```yaml
export:
  nms: true
  end2end: true
```

- [ ] **Step 7: 运行审计**

Run:

```bash
python scripts/audit_config_schema.py
```

Expected: 输出 `OK: audited ... config files`。

---

### Task 5: 最终验证

**Files:**
- Test: `scripts/audit_config_schema.py`
- Test: `commands/train.py`

- [ ] **Step 1: 运行静态审计**

Run:

```bash
python scripts/audit_config_schema.py
```

Expected:

```text
OK: audited <N> config files
```

- [ ] **Step 2: 运行当前用户报错配置的最小训练参数验证**

Run:

```bash
python -m commands.train --config configs/train/anquandai/belt.yaml --epochs 1 --batch 2 --workers 0 --exist-ok
```

Expected: 不出现：

```text
'topk' is not a valid YOLO argument
```

如果因数据集路径、GPU、权重下载等外部环境失败，在结果中明确说明失败点，并附上确认 `topk` 已不再报错的日志片段。

- [ ] **Step 3: 查看 git diff**

Run:

```bash
git diff -- commands/train.py scripts/audit_config_schema.py configs
```

Expected: diff 只包含配置对齐、后端透传过滤和审计脚本相关改动。

- [ ] **Step 4: 提交前总结**

不要自动 commit，除非用户明确要求。最终回复中列出：

- 修了哪些后端参数透传问题
- 修了哪些 YAML 问题
- 审计脚本结果
- 最小训练验证结果

---

## Self-Review

- Spec coverage: 覆盖了用户要求的“不要有无效的”“不要有缺少的”“任务专用不要忘记”，并包括 train/validate/predict/track/export。
- Placeholder scan: 无 TBD/TODO/implement later。
- Type consistency: 脚本函数名、路径和 YAML key 与现有项目结构一致。
