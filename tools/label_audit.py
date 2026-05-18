#!/usr/bin/env python3
"""使用已训练模型审计 YOLO 标注质量。

这个独立脚本会对比模型预测框和已有 YOLO 标签，并生成常见检测标注问题的
人工复核队列：

- 高置信预测框没有匹配标签：疑似漏标
- 已有标签没有匹配预测框：疑似误标、难例或模型漏检
- 预测框和标签框 IoU 偏低：疑似框偏移、框太松或框太紧
- 重复标签、非法框、极小框等格式/质量问题

脚本默认只读原始数据，不会自动修改标签文件。它只输出 CSV/JSON 报告、
可视化审核图，以及可选的待复核数据拷贝。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engine import YOLOInference
from core.types import NMSConfig
from utils.config import setup_ultralytics_path
from utils.constants import IMG_EXTENSIONS
from utils.io import read_text_robust

setup_ultralytics_path()


COLOR_GT_OK = (0, 180, 0)
COLOR_GT_BAD = (128, 128, 128)
COLOR_PRED_MISSING = (0, 255, 255)
COLOR_SHIFT = (0, 165, 255)
COLOR_DUP = (255, 0, 255)
COLOR_TEXT_BG = (30, 30, 30)
COLOR_WHITE = (255, 255, 255)

# ---------------------------------------------------------------------------
# 用户默认配置
#
# 这个旧配置块保留兼容；下方“实际生效的默认配置和调参说明”会覆盖这里。
# 日常修改请优先改下方中文说明更完整的配置块。
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [r"D:\best.pt"]
DEFAULT_DATA = r"D:\新建文件夹"
DEFAULT_OUTPUT = r"D:\label_audit_result"
DEFAULT_EXTRACT_DIR = r"D:\label_audit_extract"

# None 表示扫描所有自动识别到的 split/文件夹。
# 标准 YAML 数据集可使用 ["train"]、["val"] 或 ["train", "val"]。
DEFAULT_SPLITS = None

# 稳健性设置。单模型时，翻转 TTA 会提供“原图”和“翻转图”两个预测来源；
# min_support=2 表示只保留两个来源都支持的候选。
DEFAULT_TTA_FLIP = True
DEFAULT_MIN_SUPPORT = 2
DEFAULT_CONSENSUS_IOU = 0.60

# 模型预测阈值。
DEFAULT_IMGSZ = 1280
DEFAULT_DEVICE = "auto"
DEFAULT_CONF = 0.25
DEFAULT_NMS_IOU = 0.70
DEFAULT_MAX_DET = 300

# 标注审计阈值。
DEFAULT_MISSING_CONF = 0.75
DEFAULT_MISSING_IOU = 0.30
DEFAULT_SHIFT_CONF = 0.50
DEFAULT_SHIFT_MIN_IOU = 0.10
DEFAULT_GOOD_IOU = 0.60
DEFAULT_UNMATCHED_GT_IOU = 0.30
DEFAULT_DUPLICATE_IOU = 0.85
DEFAULT_MIN_AREA = 1e-6

# 提取设置。DEFAULT_EXTRACT_ISSUES 为 None 表示提取所有问题类型；
# 也可以只列出需要提取的问题类型，例如：
# ["possible_missing_label", "possible_box_shift"]
DEFAULT_EXTRACT_ISSUES = None
DEFAULT_EXTRACT_MAX_PRIORITY = 3
DEFAULT_VIS_LIMIT = 1000
DEFAULT_SAVE_VIS = True
DEFAULT_KEEP_CLEAN_VIS = False
DEFAULT_CLASS_AGNOSTIC = False

# ---------------------------------------------------------------------------
# 实际生效的默认配置和调参说明
#
# 上面的配置块仅保留兼容。日常使用时主要修改这一块；
# 这里的赋值会覆盖前面的同名默认值。
# ---------------------------------------------------------------------------

# 已训练好的权重路径。可以添加多个权重做多模型/多 checkpoint 投票，例如：
# [r"D:\best.pt", r"D:\last.pt"]。模型越多越慢，但候选会更稳。
DEFAULT_MODELS = [r"D:\best.pt"]

# 数据集根目录或数据集 YAML。
# 这里用 Unicode 转义是为了避免 Windows 终端/编辑器编码问题；
# 运行时实际路径就是：D:\新建文件夹。
DEFAULT_DATA = "D:\新建文件夹"

# 主审计输出目录：保存 issues.csv、review_lists/*.csv、summary.json 和可视化图。
DEFAULT_OUTPUT = r"D:\label_audit_result"

# 提取出来给人工复核的数据目录。会保留原始相对子路径：
# extract\labels\0105...\x.txt 修好后可按相同路径覆盖回原始数据。
DEFAULT_EXTRACT_DIR = r"D:\label_audit_extract"

# None 表示扫描自动识别到的所有目录/split。
# 如果是标准 YAML 数据集，可以改成 ["train"]、["val"] 或 ["train", "val"]。
DEFAULT_SPLITS = None

# True 表示同时跑原图和水平翻转图 TTA。
# 好处是降低偶发误报；代价是单模型时推理时间大约翻倍。
DEFAULT_TTA_FLIP = True

# 一个预测至少需要多少个独立来源支持才会被采用。
# 单模型 + 开启翻转 TTA 时，2 表示原图和翻转图都要检出并匹配。
# 改成 1 可以找出更多候选，但噪声会更多；保持 2 更适合提取给人工复核。
DEFAULT_MIN_SUPPORT = 2

# 多模型/TTA 预测框合并时使用的 IoU 阈值。
# 调低会把更松的框合并在一起；调高则要求框位置更一致。
DEFAULT_CONSENSUS_IOU = 0.60

# 推理图片尺寸。通常建议和训练尺寸一致。
# 调大可能更利于小目标，但会更慢、更占显存。
DEFAULT_IMGSZ = 1280

# 推理设备。"auto" 表示自动选择。
# 也可以按环境改成 "cpu"、"0" 或 "cuda:0"。
DEFAULT_DEVICE = "auto"

# 模型预测的基础置信度阈值，先过滤一遍预测结果，再进入清洗规则。
# 调低会保留更多预测用于分析，但噪声会增加。
DEFAULT_CONF = 0.25

# 推理阶段 NMS 的 IoU 阈值。
# 调高会保留更多重叠预测；调低会更强地抑制重复框。
DEFAULT_NMS_IOU = 0.70

# 每张图最多保留多少个模型检测框。
DEFAULT_MAX_DET = 300

# possible_missing_label：疑似漏标候选的预测置信度下限。
# 调到 0.85 队列会更干净；调低能找更多疑似漏标，但误报也会增加。
DEFAULT_MISSING_CONF = 0.75

# possible_missing_label：预测框和同类 GT 的最大 IoU 必须低于该值。
# 0.30 表示和已有标注几乎不重合时，才当作疑似漏标。
DEFAULT_MISSING_IOU = 0.30

# possible_box_shift：疑似框偏移候选的预测置信度下限。
DEFAULT_SHIFT_CONF = 0.50

# possible_box_shift：弱匹配框的 IoU 下限。
# 低于这个值通常更像漏标，而不是已有标注框偏移。
DEFAULT_SHIFT_MIN_IOU = 0.10

# possible_box_shift：弱匹配框的 IoU 上限。
# 预测框和 GT 匹配但 IoU 低于该值，会进入框质量问题队列。
# 调高会抓出更多框太松/太紧/偏移的问题。
DEFAULT_GOOD_IOU = 0.60

# unmatched_gt：GT 没有任何同类预测框 IoU 超过该值时，会进入队列。
# 这类要谨慎看：可能是误标，也可能是难例或当前模型漏检。
DEFAULT_UNMATCHED_GT_IOU = 0.30

# duplicate_label：同类 GT 框之间 IoU 高于该值时，认为疑似重复标注。
DEFAULT_DUPLICATE_IOU = 0.85

# invalid_box：归一化 GT 面积低于该值时，认为疑似极小/异常框。
DEFAULT_MIN_AREA = 1e-6

# None 表示提取所有问题类型。如果只想提取某几类，可以改成例如：
# ["possible_missing_label", "possible_box_shift"].
DEFAULT_EXTRACT_ISSUES = None

# 只提取 priority <= 该值的问题：
# 1 = 高价值/高概率标注问题，2 = 框质量问题，3 = 需要谨慎复核的问题。
DEFAULT_EXTRACT_MAX_PRIORITY = 3

# 最多保存多少张可视化 JPG。CSV 报告始终是全量，不受这个限制。
DEFAULT_VIS_LIMIT = 1000

# True 表示保存可视化 JPG；False 表示只写 CSV/JSON 报告。
DEFAULT_SAVE_VIS = True

# False 表示只保存有问题图片的可视化；True 表示干净图片也保存。
DEFAULT_KEEP_CLEAN_VIS = False

# False 表示预测框只和同类别 GT 匹配。
# 你当前类别基本可靠，所以建议保持 False。
DEFAULT_CLASS_AGNOSTIC = False


@dataclass
class LabelBox:
    index: int
    class_id: int
    xywhn: tuple[float, float, float, float]
    xyxy: tuple[float, float, float, float]
    raw: str
    valid: bool = True
    error: str = ""


@dataclass
class PredBox:
    index: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    support: int = 1
    sources: str = ""


@dataclass
class Issue:
    issue_type: str
    priority: int
    split: str
    image: str
    label: str
    class_id: int | str = ""
    class_name: str = ""
    confidence: float | str = ""
    iou: float | str = ""
    support: int | str = ""
    gt_index: int | str = ""
    pred_index: int | str = ""
    gt_xywhn: str = ""
    pred_xyxy: str = ""
    area_ratio: float | str = ""
    center_shift: float | str = ""
    note: str = ""
    vis_path: str = ""


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def xywhn_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    return (
        (x - w / 2.0) * img_w,
        (y - h / 2.0) * img_h,
        (x + w / 2.0) * img_w,
        (y + h / 2.0) * img_h,
    )


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fmt_box(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_shift_norm(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    denom = max((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2, 1e-6) ** 0.5
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / denom


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"dataset YAML must contain a mapping: {path}")
    return data


def resolve_path(base: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels" / image_dir.name


def collect_pairs_from_image_dir(image_dir: Path, label_dir: Path, split: str) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for img_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS):
        rel = img_path.relative_to(image_dir)
        pairs.append((img_path, label_dir / rel.with_suffix(".txt"), split))
    return pairs


def collect_dataset_pairs(data_arg: str, splits: list[str] | None) -> tuple[list[tuple[Path, Path, str]], dict[int, str]]:
    data_path = Path(data_arg).resolve()
    names: dict[int, str] = {}
    pairs: list[tuple[Path, Path, str]] = []

    if data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        cfg = load_dataset_yaml(data_path)
        names = normalize_names(cfg.get("names"))
        root = resolve_path(data_path.parent, cfg.get("path", data_path.parent))
        selected = splits or [s for s in ("train", "val", "test") if s in cfg]
        for split in selected:
            if split not in cfg:
                continue
            split_value = cfg[split]
            if isinstance(split_value, list):
                image_dirs = [resolve_path(root, item) for item in split_value]
            else:
                image_dirs = [resolve_path(root, split_value)]
            for image_dir in image_dirs:
                if image_dir.is_file():
                    image_dirs_from_txt = [
                        resolve_path(root, line.strip())
                        for line in read_text_robust(image_dir).splitlines()
                        if line.strip()
                    ]
                    for img_path in image_dirs_from_txt:
                        label_path = label_dir_from_image_dir(img_path.parent) / f"{img_path.stem}.txt"
                        pairs.append((img_path, label_path, split))
                elif image_dir.exists():
                    pairs.extend(collect_pairs_from_image_dir(image_dir, label_dir_from_image_dir(image_dir), split))
        return pairs, names

    if not data_path.exists():
        raise FileNotFoundError(f"data path not found: {data_path}")

    selected = splits or ["train", "val", "test"]
    found_split = False
    for split in selected:
        candidates = [
            (data_path / "images" / split, data_path / "labels" / split),
            (data_path / split / "images", data_path / split / "labels"),
        ]
        for image_dir, label_dir in candidates:
            if image_dir.exists():
                pairs.extend(collect_pairs_from_image_dir(image_dir, label_dir, split))
                found_split = True

    if not found_split:
        if (data_path / "images").exists():
            pairs.extend(collect_pairs_from_image_dir(data_path / "images", data_path / "labels", "root"))
        else:
            pairs.extend(collect_pairs_from_image_dir(data_path, data_path, "root"))

    for yaml_path in data_path.rglob("*.yaml"):
        try:
            names = normalize_names(load_dataset_yaml(yaml_path).get("names"))
            if names:
                break
        except Exception:
            continue
    return pairs, names


def parse_label_file(label_path: Path, img_w: int, img_h: int, min_area: float) -> tuple[list[LabelBox], list[Issue]]:
    boxes: list[LabelBox] = []
    issues: list[Issue] = []
    if not label_path.exists():
        return boxes, [
            Issue(
                issue_type="missing_label_file",
                priority=2,
                split="",
                image="",
                label=str(label_path),
                note="图片没有对应的 .txt 标签文件",
            )
        ]

    text = read_text_robust(label_path)
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            issues.append(
                Issue("invalid_label_format", 1, "", "", str(label_path), gt_index=line_no, note=f"标签列数少于 5 列: {line}")
            )
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            issues.append(
                Issue("invalid_label_format", 1, "", "", str(label_path), gt_index=line_no, note=f"标签中存在非数字值: {line}")
            )
            continue

        xyxy = xywhn_to_xyxy(x, y, w, h, img_w, img_h)
        error = ""
        if class_id < 0:
            error = "类别 id 为负数"
        elif w <= 0 or h <= 0:
            error = "宽度或高度小于等于 0"
        elif x < 0 or x > 1 or y < 0 or y > 1 or w > 1 or h > 1:
            error = "归一化坐标值超出 [0, 1]"
        elif x - w / 2 < -0.001 or x + w / 2 > 1.001 or y - h / 2 < -0.001 or y + h / 2 > 1.001:
            error = "标注框超出图片边界"
        elif w * h < min_area:
            error = f"标注框面积过小 {w * h:.8f} < {min_area}"

        box = LabelBox(
            index=len(boxes),
            class_id=class_id,
            xywhn=(x, y, w, h),
            xyxy=xyxy,
            raw=line,
            valid=not error,
            error=error,
        )
        boxes.append(box)
        if error:
            issues.append(
                Issue(
                    issue_type="invalid_box",
                    priority=1,
                    split="",
                    image="",
                    label=str(label_path),
                    class_id=class_id,
                    gt_index=box.index,
                    gt_xywhn=fmt_box(box.xywhn),
                    note=error,
                )
            )
    return boxes, issues


def find_duplicate_gt(labels: list[LabelBox], threshold: float) -> list[tuple[int, int, float]]:
    duplicates: list[tuple[int, int, float]] = []
    valid = [b for b in labels if b.valid]
    for i, a in enumerate(valid):
        for b in valid[i + 1 :]:
            if a.class_id != b.class_id:
                continue
            iou = box_iou(a.xyxy, b.xyxy)
            if iou >= threshold:
                duplicates.append((a.index, b.index, iou))
    return duplicates


def run_inference(engine: YOLOInference, img: np.ndarray, source: str, start_index: int = 0) -> list[PredBox]:
    result = engine(img)
    predictions: list[PredBox] = []
    for idx, det in enumerate(result.detections):
        predictions.append(
            PredBox(
                index=start_index + idx,
                class_id=int(det.class_id),
                class_name=str(det.class_name),
                confidence=float(det.confidence),
                xyxy=tuple(float(v) for v in det.bbox),
                support=1,
                sources=source,
            )
        )
    return predictions


def flip_predictions_back(predictions: list[PredBox], img_w: int) -> list[PredBox]:
    flipped: list[PredBox] = []
    for pred in predictions:
        x1, y1, x2, y2 = pred.xyxy
        flipped.append(
            PredBox(
                index=pred.index,
                class_id=pred.class_id,
                class_name=pred.class_name,
                confidence=pred.confidence,
                xyxy=(img_w - x2, y1, img_w - x1, y2),
                support=pred.support,
                sources=pred.sources,
            )
        )
    return flipped


def cluster_predictions(predictions: list[PredBox], iou_thresh: float, class_agnostic: bool) -> list[PredBox]:
    if not predictions:
        return []

    ordered = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    used: set[int] = set()
    clusters: list[list[PredBox]] = []
    for i, pred in enumerate(ordered):
        if i in used:
            continue
        cluster = [pred]
        used.add(i)
        for j, other in enumerate(ordered[i + 1 :], start=i + 1):
            if j in used:
                continue
            if not class_agnostic and pred.class_id != other.class_id:
                continue
            if box_iou(pred.xyxy, other.xyxy) >= iou_thresh:
                cluster.append(other)
                used.add(j)
        clusters.append(cluster)

    merged: list[PredBox] = []
    for idx, cluster in enumerate(clusters):
        weights = np.array([max(p.confidence, 1e-6) for p in cluster], dtype=np.float64)
        boxes = np.array([p.xyxy for p in cluster], dtype=np.float64)
        avg_box = tuple(float(v) for v in np.average(boxes, axis=0, weights=weights))
        best = max(cluster, key=lambda p: p.confidence)
        sources = sorted({s for p in cluster for s in p.sources.split("+") if s})
        merged.append(
            PredBox(
                index=idx,
                class_id=best.class_id,
                class_name=best.class_name,
                confidence=max(p.confidence for p in cluster),
                xyxy=avg_box,
                support=len(sources),
                sources="+".join(sources),
            )
        )
    return merged


def run_consensus_inference(
    engines: list[YOLOInference],
    img: np.ndarray,
    args: argparse.Namespace,
) -> list[PredBox]:
    all_predictions: list[PredBox] = []
    next_index = 0
    img_w = img.shape[1]
    for model_idx, engine in enumerate(engines):
        preds = run_inference(engine, img, source=f"m{model_idx}", start_index=next_index)
        all_predictions.extend(preds)
        next_index += len(preds)
        if args.tta_flip:
            flipped_img = cv2.flip(img, 1)
            flip_preds = run_inference(engine, flipped_img, source=f"m{model_idx}_flip", start_index=next_index)
            flip_preds = flip_predictions_back(flip_preds, img_w)
            all_predictions.extend(flip_preds)
            next_index += len(flip_preds)

    predictions = cluster_predictions(all_predictions, args.consensus_iou, args.class_agnostic)
    return [p for p in predictions if p.support >= args.min_support]


def class_name_for(class_id: int, model_names: dict[int, str], data_names: dict[int, str]) -> str:
    return data_names.get(class_id) or model_names.get(class_id) or f"cls{class_id}"


def best_gt_for_pred(pred: PredBox, labels: list[LabelBox], class_agnostic: bool) -> tuple[LabelBox | None, float]:
    best: LabelBox | None = None
    best_iou = 0.0
    for gt in labels:
        if not gt.valid:
            continue
        if not class_agnostic and gt.class_id != pred.class_id:
            continue
        iou = box_iou(pred.xyxy, gt.xyxy)
        if iou > best_iou:
            best = gt
            best_iou = iou
    return best, best_iou


def best_pred_for_gt(gt: LabelBox, predictions: list[PredBox], class_agnostic: bool) -> tuple[PredBox | None, float]:
    best: PredBox | None = None
    best_iou = 0.0
    for pred in predictions:
        if not class_agnostic and pred.class_id != gt.class_id:
            continue
        iou = box_iou(pred.xyxy, gt.xyxy)
        if iou > best_iou:
            best = pred
            best_iou = iou
    return best, best_iou


def audit_image(
    img_path: Path,
    label_path: Path,
    split: str,
    img: np.ndarray,
    labels: list[LabelBox],
    predictions: list[PredBox],
    parse_issues: list[Issue],
    args: argparse.Namespace,
    model_names: dict[int, str],
    data_names: dict[int, str],
) -> list[Issue]:
    issues: list[Issue] = []

    for issue in parse_issues:
        issue.split = split
        issue.image = str(img_path)
        issue.label = str(label_path)
        issues.append(issue)

    for a_idx, b_idx, iou in find_duplicate_gt(labels, args.duplicate_iou):
        gt = labels[a_idx]
        issues.append(
            Issue(
                issue_type="duplicate_label",
                priority=1,
                split=split,
                image=str(img_path),
                label=str(label_path),
                class_id=gt.class_id,
                class_name=class_name_for(gt.class_id, model_names, data_names),
                iou=round(iou, 4),
                gt_index=f"{a_idx},{b_idx}",
                gt_xywhn=fmt_box(gt.xywhn),
                note="同类别 GT 标注框高度重叠",
            )
        )

    valid_labels = [gt for gt in labels if gt.valid]

    for pred in predictions:
        best_gt, best_iou = best_gt_for_pred(pred, valid_labels, args.class_agnostic)
        if pred.confidence >= args.missing_conf and best_iou < args.missing_iou:
            issues.append(
                Issue(
                    issue_type="possible_missing_label",
                    priority=1,
                    split=split,
                    image=str(img_path),
                    label=str(label_path),
                    class_id=pred.class_id,
                    class_name=class_name_for(pred.class_id, model_names, data_names),
                    confidence=round(pred.confidence, 4),
                    iou=round(best_iou, 4),
                    support=pred.support,
                    pred_index=pred.index,
                    gt_index=best_gt.index if best_gt else "",
                    pred_xyxy=fmt_box(pred.xyxy),
                    note=f"高置信预测框没有匹配 GT，疑似漏标；来源={pred.sources}",
                )
            )
        elif pred.confidence >= args.shift_conf and args.shift_min_iou <= best_iou < args.good_iou and best_gt is not None:
            gt_area = box_area(best_gt.xyxy)
            pred_area = box_area(pred.xyxy)
            issues.append(
                Issue(
                    issue_type="possible_box_shift",
                    priority=2,
                    split=split,
                    image=str(img_path),
                    label=str(label_path),
                    class_id=pred.class_id,
                    class_name=class_name_for(pred.class_id, model_names, data_names),
                    confidence=round(pred.confidence, 4),
                    iou=round(best_iou, 4),
                    support=pred.support,
                    pred_index=pred.index,
                    gt_index=best_gt.index,
                    gt_xywhn=fmt_box(best_gt.xywhn),
                    pred_xyxy=fmt_box(pred.xyxy),
                    area_ratio=round(pred_area / max(gt_area, 1e-6), 4),
                    center_shift=round(center_shift_norm(pred.xyxy, best_gt.xyxy), 4),
                    note=f"预测框和 GT 弱匹配，疑似框偏移/过松/过紧；来源={pred.sources}",
                )
            )

    for gt in valid_labels:
        best_pred, best_iou = best_pred_for_gt(gt, predictions, args.class_agnostic)
        if best_iou < args.unmatched_gt_iou:
            issues.append(
                Issue(
                    issue_type="unmatched_gt",
                    priority=3,
                    split=split,
                    image=str(img_path),
                    label=str(label_path),
                    class_id=gt.class_id,
                    class_name=class_name_for(gt.class_id, model_names, data_names),
                    confidence=round(best_pred.confidence, 4) if best_pred else "",
                    iou=round(best_iou, 4),
                    support=best_pred.support if best_pred else "",
                    gt_index=gt.index,
                    pred_index=best_pred.index if best_pred else "",
                    gt_xywhn=fmt_box(gt.xywhn),
                    pred_xyxy=fmt_box(best_pred.xyxy) if best_pred else "",
                    note="GT 没有匹配预测框；删除前需谨慎复核",
                )
            )

    return issues


def draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(img, (x, y0), (x + tw + 6, y0 + th + baseline + 4), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 3, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_audit_visualization(
    img: np.ndarray,
    labels: list[LabelBox],
    predictions: list[PredBox],
    issues: list[Issue],
    model_names: dict[int, str],
    data_names: dict[int, str],
) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    issue_gt = {int(i.gt_index) for i in issues if str(i.gt_index).isdigit() and i.issue_type in {"unmatched_gt", "invalid_box"}}
    shift_gt = {int(i.gt_index) for i in issues if str(i.gt_index).isdigit() and i.issue_type == "possible_box_shift"}
    dup_gt: set[int] = set()
    missing_pred = {int(i.pred_index) for i in issues if str(i.pred_index).isdigit() and i.issue_type == "possible_missing_label"}
    shift_pred = {int(i.pred_index) for i in issues if str(i.pred_index).isdigit() and i.issue_type == "possible_box_shift"}
    for issue in issues:
        if issue.issue_type == "duplicate_label":
            for part in str(issue.gt_index).split(","):
                if part.isdigit():
                    dup_gt.add(int(part))

    for gt in labels:
        x1, y1, x2, y2 = gt.xyxy
        p1 = (clamp_int(x1, 0, w - 1), clamp_int(y1, 0, h - 1))
        p2 = (clamp_int(x2, 0, w - 1), clamp_int(y2, 0, h - 1))
        color = COLOR_GT_OK
        tag = "GT"
        if gt.index in issue_gt:
            color = COLOR_GT_BAD
            tag = "GT?"
        if gt.index in shift_gt:
            color = COLOR_SHIFT
            tag = "SHIFT"
        if gt.index in dup_gt:
            color = COLOR_DUP
            tag = "DUP"
        cv2.rectangle(vis, p1, p2, color, 2)
        draw_label(vis, f"{tag} {class_name_for(gt.class_id, model_names, data_names)} #{gt.index}", p1[0], p1[1], color)

    for pred in predictions:
        if pred.index not in missing_pred and pred.index not in shift_pred:
            continue
        x1, y1, x2, y2 = pred.xyxy
        p1 = (clamp_int(x1, 0, w - 1), clamp_int(y1, 0, h - 1))
        p2 = (clamp_int(x2, 0, w - 1), clamp_int(y2, 0, h - 1))
        color = COLOR_PRED_MISSING if pred.index in missing_pred else COLOR_SHIFT
        tag = "MISS?" if pred.index in missing_pred else "PRED"
        cv2.rectangle(vis, p1, p2, color, 2)
        draw_label(
            vis,
            f"{tag} {pred.class_name} {pred.confidence:.2f} s{pred.support} #{pred.index}",
            p1[0],
            max(15, p1[1] - 2),
            color,
        )

    summary = Counter(i.issue_type for i in issues)
    text = " | ".join(f"{k}:{v}" for k, v in summary.most_common())
    if text:
        draw_label(vis, text, 8, 24, COLOR_WHITE, scale=0.6, thickness=1)
    return vis


def write_csv(path: Path, issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(Issue)]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def write_review_lists(output_dir: Path, issues: list[Issue]) -> None:
    by_type: dict[str, list[Issue]] = {}
    for issue in issues:
        by_type.setdefault(issue.issue_type, []).append(issue)
    for issue_type, rows in by_type.items():
        rows.sort(key=lambda x: (x.priority, -float(x.confidence or 0), str(x.image)))
        write_csv(output_dir / f"{issue_type}.csv", rows)


def issue_selected_for_extract(issue: Issue, issue_types: set[str] | None, max_priority: int) -> bool:
    if issue_types is not None and issue.issue_type not in issue_types:
        return False
    return int(issue.priority) <= max_priority


def safe_relative_path(path: Path, roots: list[Path]) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve())
        except ValueError:
            continue
    return Path(path.name)


def unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return sorted(result, key=lambda p: len(p.parts))


def copy_file_preserve_tree(src: Path, dst_root: Path, rel: Path) -> Path | None:
    if not src.exists():
        return None
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def extract_review_dataset(
    extract_dir: Path,
    rows: list[dict[str, Any]],
    data_roots: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    images_dir = extract_dir / "images"
    labels_dir = extract_dir / "labels"
    visuals_dir = extract_dir / "visualizations"
    copied_images = 0
    copied_labels = 0
    copied_visuals = 0
    manifest_rows: list[dict[str, Any]] = []

    for row in rows:
        img_path = Path(row["image"])
        label_path = Path(row["label"])
        rel = safe_relative_path(img_path, data_roots)
        label_rel = rel.with_suffix(".txt")

        image_dst = copy_file_preserve_tree(img_path, images_dir, rel)
        label_dst = copy_file_preserve_tree(label_path, labels_dir, label_rel)
        copied_images += 1 if image_dst else 0
        copied_labels += 1 if label_dst else 0

        vis_dst = None
        vis_path_text = row.get("vis_path") or ""
        if vis_path_text:
            vis_path = Path(vis_path_text)
            if vis_path.exists():
                vis_rel = safe_relative_path(vis_path, [output_dir / "visualizations", output_dir])
                vis_dst = copy_file_preserve_tree(vis_path, visuals_dir, vis_rel)
                copied_visuals += 1 if vis_dst else 0

        manifest_rows.append(
            {
                "relative_image": str(rel),
                "relative_label": str(label_rel),
                "source_image": str(img_path),
                "source_label": str(label_path),
                "extracted_image": str(image_dst or ""),
                "extracted_label": str(label_dst or ""),
                "extracted_visualization": str(vis_dst or ""),
                "issue_types": ";".join(sorted(row["issue_types"])),
                "max_confidence": row["max_confidence"],
                "min_iou": row["min_iou"],
                "priority": row["priority"],
                "issue_count": row["issue_count"],
            }
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = extract_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        field_names = [
            "relative_image",
            "relative_label",
            "source_image",
            "source_label",
            "extracted_image",
            "extracted_label",
            "extracted_visualization",
            "issue_types",
            "max_confidence",
            "min_iou",
            "priority",
            "issue_count",
        ]
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "extract_dir": str(extract_dir),
        "manifest": str(manifest_path),
        "samples": len(rows),
        "copied_images": copied_images,
        "copied_labels": copied_labels,
        "copied_visualizations": copied_visuals,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用已训练模型审计 YOLO 检测标签。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="已训练权重路径；可重复传入多个权重做投票",
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="YOLO 数据集 YAML 或数据集根目录")
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], default=DEFAULT_SPLITS, help="要审计的 split；可重复传入")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="审计结果输出目录")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理图片尺寸")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="推理设备")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="模型预测基础置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, help="模型预测 NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET, help="每张图最多保留的检测框数量")
    parser.add_argument("--class-agnostic", action="store_true", default=DEFAULT_CLASS_AGNOSTIC, help="跨类别匹配预测框和标签框")
    parser.add_argument("--tta-flip", action="store_true", default=DEFAULT_TTA_FLIP, help="启用水平翻转 TTA，并记录/要求预测来源支持")
    parser.add_argument("--no-tta-flip", action="store_false", dest="tta_flip", help="关闭水平翻转 TTA")
    parser.add_argument("--consensus-iou", type=float, default=DEFAULT_CONSENSUS_IOU, help="多模型/TTA 预测框合并使用的 IoU")
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT, help="预测框被采用所需的最小来源支持数")
    parser.add_argument("--missing-conf", type=float, default=DEFAULT_MISSING_CONF, help="疑似漏标候选的最低预测置信度")
    parser.add_argument("--missing-iou", type=float, default=DEFAULT_MISSING_IOU, help="疑似漏标候选允许的最大 GT IoU")
    parser.add_argument("--shift-conf", type=float, default=DEFAULT_SHIFT_CONF, help="疑似框偏移候选的最低预测置信度")
    parser.add_argument("--shift-min-iou", type=float, default=DEFAULT_SHIFT_MIN_IOU, help="疑似框偏移候选的最低 IoU")
    parser.add_argument("--good-iou", type=float, default=DEFAULT_GOOD_IOU, help="达到该 IoU 视为框匹配良好")
    parser.add_argument("--unmatched-gt-iou", type=float, default=DEFAULT_UNMATCHED_GT_IOU, help="GT 没有预测框超过该 IoU 时进入队列")
    parser.add_argument("--duplicate-iou", type=float, default=DEFAULT_DUPLICATE_IOU, help="同类 GT 超过该 IoU 时视为疑似重复标注")
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA, help="标注框最小归一化面积，低于该值视为异常小框")
    parser.add_argument("--no-vis", action="store_true", default=not DEFAULT_SAVE_VIS, help="不保存可视化图片")
    parser.add_argument("--vis-limit", type=int, default=DEFAULT_VIS_LIMIT, help="最多保存多少张可视化图；0 表示不限制")
    parser.add_argument("--keep-clean-vis", action="store_true", default=DEFAULT_KEEP_CLEAN_VIS, help="同时保存无问题图片的可视化")
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR, help="把选中的复核样本复制到此目录，并保留相对子路径")
    parser.add_argument(
        "--extract-issue",
        action="append",
        dest="extract_issues",
        default=DEFAULT_EXTRACT_ISSUES,
        help="要提取的问题类型；可重复传入。默认提取所有问题类型。",
    )
    parser.add_argument("--extract-max-priority", type=int, default=DEFAULT_EXTRACT_MAX_PRIORITY, help="只提取 priority 小于等于该值的问题")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    model_paths = [str(Path(p)) for p in (args.model or DEFAULT_MODELS)]
    if not model_paths:
        print("未配置模型路径。请修改 DEFAULT_MODELS，或通过 --model 传入。")
        return 2
    if not args.data:
        print("未配置数据路径。请修改 DEFAULT_DATA，或通过 --data 传入。")
        return 2
    output_stem = Path(model_paths[0]).stem if len(model_paths) == 1 else f"{Path(model_paths[0]).stem}_consensus{len(model_paths)}"
    output_dir = Path(args.output or (Path("runs") / "label_audit" / output_stem)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, data_names = collect_dataset_pairs(args.data, args.split)
    if not pairs:
        print("没有找到图片/标签对。请检查 --data 和 --split。")
        return 2

    print("=" * 72)
    print("YOLO 标签审计")
    print("=" * 72)
    print(f"模型:   {', '.join(model_paths)}")
    print(f"数据:   {args.data}")
    print(f"图片数: {len(pairs)}")
    print(f"输出:   {output_dir}")
    if args.tta_flip or len(model_paths) > 1 or args.min_support > 1:
        print(f"一致性: 翻转TTA={args.tta_flip}, 最小支持数={args.min_support}, 合并IoU={args.consensus_iou}")
    print("-" * 72)

    nms = NMSConfig(conf_threshold=args.conf, iou_threshold=args.nms_iou, max_detections=args.max_det)
    engines = [YOLOInference(model_path, nms_config=nms, device=args.device, imgsz=args.imgsz) for model_path in model_paths]
    model_names = {int(k): str(v) for k, v in engines[0].classes.items()}
    if not data_names:
        data_names = model_names

    all_issues: list[Issue] = []
    extract_rows: dict[str, dict[str, Any]] = {}
    extract_issue_types = set(args.extract_issues) if args.extract_issues else None
    data_roots = unique_existing_paths([Path(args.data).resolve()] + [p.parent.resolve() for p, _, _ in pairs])
    issue_images = 0
    saved_vis = 0
    unreadable = 0
    started = time.time()

    for idx, (img_path, label_path, split) in enumerate(pairs, 1):
        img = imread_unicode(img_path)
        if img is None:
            unreadable += 1
            all_issues.append(
                Issue("unreadable_image", 1, split, str(img_path), str(label_path), note="OpenCV 无法读取图片")
            )
            continue

        h, w = img.shape[:2]
        labels, parse_issues = parse_label_file(label_path, w, h, args.min_area)
        predictions = run_consensus_inference(engines, img, args)
        issues = audit_image(img_path, label_path, split, img, labels, predictions, parse_issues, args, model_names, data_names)

        has_review_issue = bool(issues)
        if has_review_issue:
            issue_images += 1
        if (has_review_issue or args.keep_clean_vis) and not args.no_vis:
            if args.vis_limit <= 0 or saved_vis < args.vis_limit:
                issue_type = issues[0].issue_type if issues else "clean"
                rel_name = f"{split}_{img_path.stem}.jpg"
                vis_path = output_dir / "visualizations" / issue_type / rel_name
                vis = draw_audit_visualization(img, labels, predictions, issues, model_names, data_names)
                imwrite_unicode(vis_path, vis)
                for issue in issues:
                    issue.vis_path = str(vis_path)
                saved_vis += 1

        all_issues.extend(issues)

        if args.extract_dir:
            selected_issues = [
                issue
                for issue in issues
                if issue_selected_for_extract(issue, extract_issue_types, args.extract_max_priority)
            ]
            if selected_issues:
                key = str(img_path.resolve())
                confidences = [float(i.confidence) for i in selected_issues if i.confidence != ""]
                ious = [float(i.iou) for i in selected_issues if i.iou != ""]
                if key not in extract_rows:
                    extract_rows[key] = {
                        "image": str(img_path),
                        "label": str(label_path),
                        "vis_path": selected_issues[0].vis_path,
                        "issue_types": set(),
                        "max_confidence": "",
                        "min_iou": "",
                        "priority": min(i.priority for i in selected_issues),
                        "issue_count": 0,
                    }
                row = extract_rows[key]
                row["issue_types"].update(i.issue_type for i in selected_issues)
                row["issue_count"] += len(selected_issues)
                row["priority"] = min(row["priority"], min(i.priority for i in selected_issues))
                if not row["vis_path"]:
                    row["vis_path"] = selected_issues[0].vis_path
                if confidences:
                    current = float(row["max_confidence"]) if row["max_confidence"] != "" else 0.0
                    row["max_confidence"] = round(max(current, max(confidences)), 4)
                if ious:
                    current = float(row["min_iou"]) if row["min_iou"] != "" else 1.0
                    row["min_iou"] = round(min(current, min(ious)), 4)

        if idx % 50 == 0 or idx == len(pairs):
            elapsed = max(time.time() - started, 1e-6)
            print(f"进度: {idx}/{len(pairs)} 张图片，{len(all_issues)} 个问题，{idx / elapsed:.2f} 张/秒")

    all_issues.sort(key=lambda x: (x.priority, x.issue_type, -float(x.confidence or 0), str(x.image)))
    issue_counts = Counter(i.issue_type for i in all_issues)
    priority_counts = Counter(str(i.priority) for i in all_issues)

    write_csv(output_dir / "issues.csv", all_issues)
    write_review_lists(output_dir / "review_lists", all_issues)

    extract_summary = None
    if args.extract_dir:
        extract_rows_list = list(extract_rows.values())
        extract_rows_list.sort(key=lambda x: (x["priority"], -float(x["max_confidence"] or 0), str(x["image"])))
        extract_summary = extract_review_dataset(
            Path(args.extract_dir).resolve(),
            extract_rows_list,
            data_roots,
            output_dir,
        )

    summary = {
        "models": model_paths,
        "data": str(args.data),
        "splits": args.split or ["auto"],
        "output": str(output_dir),
        "total_images": len(pairs),
        "issue_images": issue_images,
        "total_issues": len(all_issues),
        "unreadable_images": unreadable,
        "saved_visualizations": saved_vis,
        "issue_counts": dict(issue_counts),
        "priority_counts": dict(priority_counts),
        "extract": extract_summary,
        "thresholds": {
            "conf": args.conf,
            "missing_conf": args.missing_conf,
            "missing_iou": args.missing_iou,
            "shift_conf": args.shift_conf,
            "shift_min_iou": args.shift_min_iou,
            "good_iou": args.good_iou,
            "unmatched_gt_iou": args.unmatched_gt_iou,
            "duplicate_iou": args.duplicate_iou,
            "min_area": args.min_area,
            "tta_flip": args.tta_flip,
            "consensus_iou": args.consensus_iou,
            "min_support": args.min_support,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 72)
    print(f"完成: 共 {len(pairs)} 张图片，{issue_images} 张有问题，共 {len(all_issues)} 个问题")
    for issue_type, count in issue_counts.most_common():
        print(f"  {issue_type}: {count}")
    print(f"总表:     {output_dir / 'issues.csv'}")
    print(f"分类表:   {output_dir / 'review_lists'}")
    if not args.no_vis:
        print(f"可视化:   {output_dir / 'visualizations'}")
    if extract_summary:
        print(f"提取目录: {extract_summary['extract_dir']}")
        print(f"提取清单: {extract_summary['manifest']}")
    print(f"摘要:     {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
