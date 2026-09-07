#!/usr/bin/env python3
"""审计 YOLO 配置文件与当前前端/Ultralytics 后端参数 schema 的一致性。

重点检查三类问题：
1. YAML 中存在当前命令后端不会读取或不应传递的字段。
2. 字段虽然被后端支持，但出现在了错误的任务配置中。
3. 示例配置缺少当前后端已支持且该任务应展示的字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"

TASKS = {"detect", "segment", "pose", "obb", "classify"}
DETECT_FAMILY = {"detect", "segment", "pose", "obb"}

TRAIN_FORBIDDEN = {
    "model.topk",
    "model.kpt_thres",
    "validation.topk",
    "validation.kpt_thres",
}

COMMON_TRAIN = {
    "mode",
    "model.name",
    "model.yaml",
    "model.task",
    "model.pretrained",
    "model.classes",
    "model.reg_max",
    "data.config",
    "data.split",
    "cfg",
    "tracker",
    "train.epochs",
    "train.time",
    "train.batch",
    "train.imgsz",
    "train.device",
    "train.workers",
    "train.patience",
    "train.cache",
    "train.fraction",
    "train.freeze",
    "train.multi_scale",
    "train.nbs",
    "train.optimizer",
    "train.lr0",
    "train.lrf",
    "train.momentum",
    "train.weight_decay",
    "train.close_mosaic",
    "train.seed",
    "train.warmup_epochs",
    "train.warmup_momentum",
    "train.warmup_bias_lr",
    "train.box",
    "train.cls",
    "train.cls_pw",
    "train.dfl",
    "train.save",
    "train.amp",
    "train.rect",
    "train.single_cls",
    "train.profile",
    "train.deterministic",
    "train.resume",
    "train.cos_lr",
    "train.compile",
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
    "segment": {"train.end2end", "validation.end2end", "train.overlap_mask", "train.mask_ratio", "augmentation.copy_paste", "augmentation.copy_paste_mode"},
    "pose": {"train.end2end", "validation.end2end", "train.pose", "train.kobj", "train.rle"},
    "obb": {"train.end2end", "validation.end2end", "train.angle"},
    "classify": {"train.dropout", "augmentation.auto_augment", "augmentation.erasing"},
}

WRONG_TASK_KEYS = {
    "classify": {
        "model.reg_max",
        "train.box",
        "train.dfl",
        "train.cls_pw",
        "train.end2end",
        "validation.end2end",
        "validation.conf",
        "validation.iou",
        "validation.max_det",
        "validation.agnostic_nms",
        "validation.dnn",
        "validation.save_conf",
        "validation.save_json",
        "validation.int8",
        "validation.retina_masks",
        "train.overlap_mask",
        "train.mask_ratio",
        "augmentation.copy_paste",
        "augmentation.copy_paste_mode",
        "train.pose",
        "train.kobj",
        "train.rle",
        "train.angle",
    },
    "detect": {"train.dropout", "augmentation.auto_augment", "augmentation.erasing", "train.pose", "train.kobj", "train.rle", "train.angle", "train.overlap_mask", "train.mask_ratio"},
    "segment": {"train.dropout", "augmentation.auto_augment", "augmentation.erasing", "train.pose", "train.kobj", "train.rle", "train.angle"},
    "pose": {"train.dropout", "augmentation.auto_augment", "augmentation.erasing", "train.angle", "train.overlap_mask", "train.mask_ratio"},
    "obb": {"train.dropout", "augmentation.auto_augment", "augmentation.erasing", "train.pose", "train.kobj", "train.rle", "train.overlap_mask", "train.mask_ratio"},
}

VALIDATE_COMMON = {
    "mode",
    "model.name",
    "model.task",
    "model.classes",
    "model.imgsz",
    "model.batch",
    "model.device",
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
    "validation.fraction",
    "validation.line_width",
    "validation.workers",
    "validation.cache",
    "validation.half",
    "validation.plots",
    "validation.save_json",
    "validation.dnn",
    "validation.agnostic_nms",
    "validation.augment",
    "validation.rect",
    "validation.save_conf",
    "validation.int8",
    "validation.end2end",
    "validation.save_txt",
    "validation.save_crop",
    "validation.show",
    "validation.show_labels",
    "validation.show_conf",
    "validation.show_boxes",
    "validation.retina_masks",
    "validation.visualize",
    "output.project",
    "output.name",
    "output.save_period",
    "output.exist_ok",
    "output.verbose",
}
VALIDATE_TASK_KEYS = {
    "classify": {"model.topk", "validation.topk"},
    "pose": {"model.kpt_thres", "validation.kpt_thres"},
    "detect": set(),
    "segment": set(),
    "obb": set(),
}

PREDICT_COMMON = {
    "mode",
    "model.path",
    "model.task",
    "model.imgsz",
    "model.device",
    "model.batch",
    "model.classes",
    "model.vid_stride",
    "model.embed",
    "model.line_width",
    "model.stream",
    "model.half",
    "model.augment",
    "model.retina_masks",
    "model.visualize",
    "model.int8",
    "model.save_frames",
    "model.stream_buffer",
    "model.save_conf",
    "model.dnn",
    "model.end2end",
    "model.show",
    "model.show_boxes",
    "model.tracker",
    "model.persist",
    "io.source",
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
    "visualization.show_labels",
    "visualization.show_conf",
    "visualization.box_thickness",
    "visualization.font_scale",
    "visualization.show_boxes",
    "visualization.line_width",
    "video.fps",
    "video.codec",
    "output.save",
    "output.save_txt",
    "output.save_conf",
    "output.save_crop",
    "output.project",
    "output.name",
    "output.exist_ok",
    "output.verbose",
    "tracker",
}
PREDICT_TASK_KEYS = {
    "classify": {"model.topk", "nms.topk"},
    "pose": {"model.kpt_thres", "nms.kpt_thres", "visualization.skeleton", "visualization.kpt_names", "visualization.kpt_names.*", "visualization.kpt_line", "visualization.kpt_radius"},
    "segment": {"visualization.mask_alpha"},
    "detect": set(),
    "obb": set(),
}

EXPORT_KEYS = {
    "mode",
    "model.path",
    "model.format",
    "model.imgsz",
    "model.batch",
    "model.device",
    "export.opset",
    "export.data",
    "export.split",
    "export.fraction",
    "export.workspace",
    "export.conf",
    "export.iou",
    "export.max_det",
    "export.simplify",
    "export.dynamic",
    "export.half",
    "export.nms",
    "export.optimize",
    "export.int8",
    "export.keras",
    "export.agnostic_nms",
    "export.end2end",
    "output.path",
    "output.verbose",
    "verify.enabled",
    "verify.source",
}

EXAMPLE_REQUIRED = {
    ("train", "classify"): {"train.dropout", "augmentation.auto_augment", "augmentation.erasing"},
    ("train", "segment"): {"train.overlap_mask", "train.mask_ratio", "augmentation.copy_paste", "augmentation.copy_paste_mode"},
    ("train", "pose"): {"train.pose", "train.kobj", "train.rle"},
    ("train", "obb"): {"train.angle"},
    ("predict", "classify"): {"model.topk"},
    ("predict", "pose"): {"model.kpt_thres", "visualization.skeleton", "visualization.kpt_names"},
    ("validate", "classify"): {"validation.topk"},
    ("validate", "pose"): {"validation.kpt_thres"},
}


@dataclass(frozen=True)
class Issue:
    path: Path
    kind: str
    detail: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def flatten(data: Any, prefix: str = "") -> set[str]:
    if not isinstance(data, dict):
        return {prefix} if prefix else set()
    keys: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if path == "visualization.kpt_names":
                keys.add("visualization.kpt_names")
                keys.add("visualization.kpt_names.*")
            else:
                keys |= flatten(value, path)
        else:
            keys.add(path)
    return keys


def infer_mode(path: Path) -> str | None:
    parts = path.relative_to(CONFIG_ROOT).parts
    return parts[0] if parts and parts[0] in {"train", "predict", "validate", "export"} else None


def infer_task(path: Path, data: dict[str, Any]) -> str:
    task = data.get("model", {}).get("task")
    if task in TASKS:
        return task
    name = path.stem.lower()
    parent_bits = " ".join(p.lower() for p in path.parts)
    for candidate in ("classify", "segment", "pose", "obb", "detect"):
        if candidate in name or candidate in parent_bits:
            return candidate
    model_name = str(data.get("model", {}).get("name") or data.get("model", {}).get("path") or "").lower()
    if "-cls" in model_name or "cls" in model_name:
        return "classify"
    if "-seg" in model_name or "seg" in model_name:
        return "segment"
    if "-pose" in model_name or "pose" in model_name:
        return "pose"
    if "-obb" in model_name or "obb" in model_name:
        return "obb"
    return "detect"


def allowed_keys(mode: str, task: str) -> set[str]:
    if mode == "train":
        return COMMON_TRAIN | TRAIN_TASK_KEYS.get(task, set())
    if mode == "validate":
        return VALIDATE_COMMON | VALIDATE_TASK_KEYS.get(task, set())
    if mode == "predict":
        return PREDICT_COMMON | PREDICT_TASK_KEYS.get(task, set())
    if mode == "export":
        return EXPORT_KEYS
    return set()


def audit_file(path: Path) -> list[Issue]:
    mode = infer_mode(path)
    if mode is None:
        return []

    data = load_yaml(path)
    task = infer_task(path, data)
    keys = flatten(data)
    allowed = allowed_keys(mode, task)
    issues: list[Issue] = []

    for key in sorted(keys - allowed):
        issues.append(Issue(path, "unknown-key", f"{key} 不在 {mode}/{task} schema 中"))

    if mode == "train":
        for key in sorted(keys & TRAIN_FORBIDDEN):
            issues.append(Issue(path, "forbidden-train-key", f"{key} 不是当前 Ultralytics train() 支持参数"))
        for key in sorted(keys & WRONG_TASK_KEYS.get(task, set())):
            issues.append(Issue(path, "wrong-task-key", f"{key} 不适用于 {task} train 配置"))

    if mode in {"validate", "predict"}:
        task_keys = VALIDATE_TASK_KEYS if mode == "validate" else PREDICT_TASK_KEYS
        foreign_task_keys = set().union(*(v for t, v in task_keys.items() if t != task))
        for key in sorted(keys & foreign_task_keys):
            issues.append(Issue(path, "wrong-task-key", f"{key} 不适用于 {task} {mode} 配置"))

    if "example" in path.parts:
        required = EXAMPLE_REQUIRED.get((mode, task), set())
        for key in sorted(required - keys):
            issues.append(Issue(path, "missing-example-key", f"示例缺少后端支持的 {mode}/{task} 字段 {key}"))

    if mode == "export":
        fmt = data.get("model", {}).get("format") or path.parent.name
        export = data.get("export", {})
        if export.get("nms") is True and export.get("end2end") is True:
            issues.append(Issue(path, "invalid-combination", "export.nms=true 与 export.end2end=true 互斥"))
        if "workspace" in export and fmt != "engine":
            issues.append(Issue(path, "format-specific-key", "export.workspace 只适用于 TensorRT engine"))
        if "keras" in export and fmt != "saved_model":
            issues.append(Issue(path, "format-specific-key", "export.keras 只适用于 saved_model"))
        if export.get("int8") is True and not export.get("data"):
            issues.append(Issue(path, "missing-calibration-data", "export.int8=true 时应提供 export.data 用于校准"))

    return issues


def main() -> int:
    files = sorted(
        p for root in ("train", "predict", "validate", "export")
        for p in (CONFIG_ROOT / root).rglob("*.yaml")
    )
    issues = [issue for path in files for issue in audit_file(path)]

    if issues:
        for issue in issues:
            rel = issue.path.relative_to(ROOT)
            print(f"{rel}: [{issue.kind}] {issue.detail}")
        print(f"\n发现 {len(issues)} 个配置 schema 问题。")
        return 1

    print(f"配置 schema 审计通过，共检查 {len(files)} 个 YAML 文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
