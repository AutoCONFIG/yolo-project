#!/usr/bin/env python3
"""
标签清洗工具 - 基于模型预测检测疑似误标注/漏标注
================================================
用训练好的模型对训练集推理，将预测结果与原始标注对比，找出差异大的样本。

检测维度:
1. 类别不一致: 预测类别与标注类别不匹配
2. IoU 过低:   预测框与标注框 IoU < 阈值
3. 漏检 (FN):  标注了但模型没检测到（疑似误标注）
4. 多检 (FP):  模型高置信度检测但无对应标注（疑似漏标注）

输出:
  - 文本报告: 每张图的疑似问题列表
  - 可视化图: GT(绿) + 预测(红/蓝) 叠加绘制，保存到 --save-dir
  - 统计摘要: 各类问题数量分布

Usage:
  # 基本用法: 检测标签问题并保存可视化
  python tools/check_labels.py --model runs/detect/train/weights/best.pt --data datasets/my_dataset

  # 调整阈值
  python tools/check_labels.py --model best.pt --data datasets/my --iou-thresh 0.3 --conf-thresh 0.25

  # 只生成报告不保存可视化图片
  python tools/check_labels.py --model best.pt --data datasets/my --no-vis

  # 指定任务类型
  python tools/check_labels.py --model best.pt --data datasets/my --task detect
  python tools/check_labels.py --model best.pt --data datasets/my --task pose --kpt-shape 4 3

  # 只检查特定 split
  python tools/check_labels.py --model best.pt --data datasets/my --split train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from utils.constants import IMG_EXTENSIONS, DEFAULT_KPT_SHAPE
from utils.config import setup_ultralytics_path
from utils.io import read_text_robust

setup_ultralytics_path()

from core.engine import YOLOInference
from core.types import NMSConfig


COLOR_GT = (0, 200, 0)
COLOR_PRED_MATCH = (0, 165, 255)
COLOR_PRED_FP = (0, 0, 255)
COLOR_FN = (128, 128, 128)


def parse_yolo_label(label_path: Path, kpt_shape: tuple[int, int] | None = None):
    """解析 YOLO 格式标签文件，返回标注列表。"""
    text = read_text_robust(label_path)
    annotations = []
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            kpts = []
            if kpt_shape is not None:
                nkpt, ndim = kpt_shape
                expected = 5 + nkpt * ndim
                if len(parts) >= expected:
                    for i in range(nkpt):
                        base = 5 + i * ndim
                        kx = float(parts[base])
                        ky = float(parts[base + 1])
                        kv = float(parts[base + 2]) if ndim == 3 else 1.0
                        kpts.append([kx, ky, kv])
            annotations.append({
                "cls_id": cls_id,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "kpts": kpts if kpts else None,
            })
        except (ValueError, IndexError):
            continue
    return annotations


def xywh_to_xyxy(bbox_xywh, img_w, img_h):
    cx, cy, w, h = bbox_xywh
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_predictions_to_gt(
    predictions: list[dict],
    gt_annotations: list[dict],
    img_w: int,
    img_h: int,
    iou_thresh: float,
    class_agnostic: bool = False,
):
    """将预测结果与 GT 标注进行匈牙利式贪心匹配。

    Returns:
        matches: [(pred_idx, gt_idx, iou, class_match), ...]
        unmatched_preds: [pred_idx, ...]  (FP: 疑似漏标注)
        unmatched_gts: [gt_idx, ...]      (FN: 疑似误标注)
    """
    if not predictions or not gt_annotations:
        return [], list(range(len(predictions))), list(range(len(gt_annotations)))

    gt_boxes = [xywh_to_xyxy((g["cx"], g["cy"], g["w"], g["h"]), img_w, img_h) for g in gt_annotations]
    pred_boxes = [p["bbox"] for p in predictions]

    iou_matrix = np.zeros((len(predictions), len(gt_annotations)))
    for i in range(len(predictions)):
        for j in range(len(gt_annotations)):
            if not class_agnostic and predictions[i]["class_id"] != gt_annotations[j]["cls_id"]:
                continue
            iou_matrix[i, j] = compute_iou(pred_boxes[i], gt_boxes[j])

    matched_pred = set()
    matched_gt = set()
    matches = []

    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        i = int(flat_idx // len(gt_annotations))
        j = int(flat_idx % len(gt_annotations))
        if i in matched_pred or j in matched_gt:
            continue
        if iou_matrix[i, j] < iou_thresh:
            break
        class_match = predictions[i]["class_id"] == gt_annotations[j]["cls_id"]
        matches.append((i, j, float(iou_matrix[i, j]), class_match))
        matched_pred.add(i)
        matched_gt.add(j)

    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred]
    unmatched_gts = [j for j in range(len(gt_annotations)) if j not in matched_gt]

    return matches, unmatched_preds, unmatched_gts


def find_dataset_structure(data_path: Path, split: str | None = None):
    """自动探测 YOLO 数据集目录结构，返回 (images_dir, labels_dir) 列表。"""
    results = []
    candidates = [split] if split else ["train", "val", "test"]

    for sp in candidates:
        img_dir = data_path / "images" / sp
        lbl_dir = data_path / "labels" / sp
        if img_dir.exists() and lbl_dir.exists():
            results.append((img_dir, lbl_dir, sp))
            continue
        img_dir = data_path / sp / "images"
        lbl_dir = data_path / sp / "labels"
        if img_dir.exists() and lbl_dir.exists():
            results.append((img_dir, lbl_dir, sp))

    if not results and not split:
        img_dir = data_path / "images"
        lbl_dir = data_path / "labels"
        if img_dir.exists() and lbl_dir.exists():
            results.append((img_dir, lbl_dir, ""))
        elif data_path.exists():
            has_images = any(data_path.glob(f"*{e}") for e in IMG_EXTENSIONS)
            has_labels = any(data_path.glob("*.txt"))
            if has_images and has_labels:
                results.append((data_path, data_path, ""))

    return results


def collect_image_label_pairs(img_dir: Path, lbl_dir: Path):
    """收集图片-标签对。"""
    pairs = []
    for ext in IMG_EXTENSIONS:
        for img_path in sorted(img_dir.rglob(f"*{ext}")):
            rel = img_path.relative_to(img_dir)
            lbl_path = lbl_dir / rel.with_suffix(".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


def draw_check_result(
    img: np.ndarray,
    gt_anns: list[dict],
    predictions: list[dict],
    matches: list[tuple],
    unmatched_preds: list[int],
    unmatched_gts: list[int],
    kpt_shape: tuple[int, int] | None = None,
):
    """在图片上同时绘制 GT 和预测结果，标注问题区域。"""
    vis = img.copy()
    h_img, w_img = vis.shape[:2]

    for j in unmatched_gts:
        ann = gt_anns[j]
        x1, y1, x2, y2 = [int(v) for v in xywh_to_xyxy((ann["cx"], ann["cy"], ann["w"], ann["h"]), w_img, h_img)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_FN, 2)
        cv2.putText(vis, f"GT:cls{ann['cls_id']}(FN)", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FN, 1)

    for i, j, iou, cls_match in matches:
        if not cls_match:
            ann = gt_anns[j]
            pred = predictions[i]
            gx1, gy1, gx2, gy2 = [int(v) for v in xywh_to_xyxy((ann["cx"], ann["cy"], ann["w"], ann["h"]), w_img, h_img)]
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), COLOR_GT, 2)
            cv2.putText(vis, f"GT:cls{ann['cls_id']}", (gx1, max(gy1 - 20, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT, 1)

            px1, py1, px2, py2 = [int(v) for v in pred["bbox"]]
            cv2.rectangle(vis, (px1, py1), (px2, py2), COLOR_PRED_FP, 2)
            cv2.putText(vis, f"P:cls{pred['class_id']}({pred['confidence']:.2f})", (px1, max(py1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRED_FP, 1)

    for i in unmatched_preds:
        pred = predictions[i]
        px1, py1, px2, py2 = [int(v) for v in pred["bbox"]]
        cv2.rectangle(vis, (px1, py1), (px2, py2), COLOR_PRED_FP, 2)
        label = f"FP:cls{pred['class_id']}({pred['confidence']:.2f})"
        cv2.putText(vis, label, (px1, max(py1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRED_FP, 1)

    return vis


def check_labels(
    model_path: str,
    data_path: str,
    iou_thresh: float = 0.3,
    conf_thresh: float = 0.25,
    iou_nms: float = 0.7,
    split: str | None = None,
    task: str = "detect",
    kpt_shape: tuple[int, int] | None = None,
    save_dir: str | None = None,
    no_vis: bool = False,
    report_only: bool = False,
    imgsz: int = 640,
    class_agnostic: bool = False,
    device: str = "auto",
    max_det: int = 300,
):
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"错误: 数据集路径不存在 {data_path}")
        return

    print("=" * 60)
    print("标签清洗工具 - 基于模型预测检测疑似误标注")
    print("=" * 60)
    print(f"模型:    {model_path}")
    print(f"数据集:  {data_path}")
    print(f"IoU阈值: {iou_thresh}")
    print(f"置信度:  {conf_thresh}")
    print(f"类别无关匹配: {'是' if class_agnostic else '否'}")
    print("-" * 60)

    nms_config = NMSConfig(
        conf_threshold=conf_thresh,
        iou_threshold=iou_nms,
        max_detections=max_det,
    )
    print("正在加载模型...")
    engine = YOLOInference(model_path, nms_config=nms_config, device=device, imgsz=imgsz)
    class_names = engine.classes
    print(f"模型类别数: {len(class_names)}, 类别: {class_names}")

    splits = find_dataset_structure(data_path, split)
    if not splits:
        print("错误: 未找到数据集 images/labels 目录结构")
        return

    all_issues = []
    total_images = 0
    total_gt = 0
    total_pred = 0
    issue_counter = Counter()
    class_fn_counter = Counter()
    class_fp_counter = Counter()
    class_mismatch_counter = Counter()

    if save_dir and not no_vis:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    for img_dir, lbl_dir, split_name in splits:
        pairs = collect_image_label_pairs(img_dir, lbl_dir)
        print(f"\nSplit '{split_name or 'root'}': {len(pairs)} 张图片")

        for idx, (img_path, lbl_path) in enumerate(pairs):
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            total_images += 1

            gt_anns = parse_yolo_label(lbl_path, kpt_shape)
            total_gt += len(gt_anns)

            result = engine(img_path)
            predictions = []
            for det in result.detections:
                total_pred += 1
                predictions.append({
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                })

            matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
                predictions, gt_anns, w_img, h_img, iou_thresh, class_agnostic
            )

            image_issues = []

            for i, j, iou_val, cls_match in matches:
                if not cls_match:
                    gt_cls = gt_anns[j]["cls_id"]
                    pred_cls = predictions[i]["class_id"]
                    gt_name = class_names.get(gt_cls, f"cls{gt_cls}")
                    pred_name = class_names.get(pred_cls, f"cls{pred_cls}")
                    image_issues.append({
                        "type": "class_mismatch",
                        "gt_cls": gt_cls, "gt_name": gt_name,
                        "pred_cls": pred_cls, "pred_name": pred_name,
                        "iou": round(iou_val, 3),
                        "confidence": round(predictions[i]["confidence"], 3),
                    })
                    issue_counter["class_mismatch"] += 1
                    class_mismatch_counter[(gt_cls, pred_cls)] += 1

            for j in unmatched_gts:
                gt_cls = gt_anns[j]["cls_id"]
                gt_name = class_names.get(gt_cls, f"cls{gt_cls}")
                image_issues.append({
                    "type": "fn_missed_gt",
                    "gt_cls": gt_cls, "gt_name": gt_name,
                    "gt_bbox": [round(v, 3) for v in (gt_anns[j]["cx"], gt_anns[j]["cy"], gt_anns[j]["w"], gt_anns[j]["h"])],
                })
                issue_counter["fn_missed_gt"] += 1
                class_fn_counter[gt_cls] += 1

            for i in unmatched_preds:
                pred_cls = predictions[i]["class_id"]
                pred_name = class_names.get(pred_cls, f"cls{pred_cls}")
                image_issues.append({
                    "type": "fp_extra_pred",
                    "pred_cls": pred_cls, "pred_name": pred_name,
                    "confidence": round(predictions[i]["confidence"], 3),
                    "pred_bbox": [round(v, 1) for v in predictions[i]["bbox"]],
                })
                issue_counter["fp_extra_pred"] += 1
                class_fp_counter[pred_cls] += 1

            if image_issues:
                all_issues.append({
                    "image": str(img_path),
                    "label": str(lbl_path),
                    "split": split_name,
                    "gt_count": len(gt_anns),
                    "pred_count": len(predictions),
                    "issues": image_issues,
                })

            if save_path and not no_vis and image_issues:
                vis = draw_check_result(
                    img, gt_anns, predictions, matches, unmatched_preds, unmatched_gts, kpt_shape
                )
                info_text = f"Issues: {len(image_issues)} | GT: {len(gt_anns)} Pred: {len(predictions)}"
                cv2.putText(vis, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                out_subdir = save_path / split_name if split_name else save_path
                out_subdir.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".jpg", vis)[1].tofile(str(out_subdir / f"{img_path.stem}.jpg"))

            if (idx + 1) % 100 == 0 or idx == len(pairs) - 1:
                print(f"  进度: {idx + 1}/{len(pairs)}")

    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    print(f"总图片数:     {total_images}")
    print(f"总 GT 标注:   {total_gt}")
    print(f"总预测框:     {total_pred}")
    print(f"有问题图片:   {len(all_issues)} ({len(all_issues) / max(total_images, 1) * 100:.1f}%)")

    print(f"\n问题类型分布:")
    for issue_type, count in issue_counter.most_common():
        type_labels = {
            "class_mismatch": "类别不一致",
            "fn_missed_gt":   "漏检(疑似误标注)",
            "fp_extra_pred":  "多检(疑似漏标注)",
        }
        print(f"  {type_labels.get(issue_type, issue_type)}: {count}")

    if class_fn_counter:
        print(f"\n漏检(FN)按类别:")
        for cls_id, count in class_fn_counter.most_common():
            name = class_names.get(cls_id, f"cls{cls_id}")
            print(f"  [{cls_id}] {name}: {count}")

    if class_fp_counter:
        print(f"\n多检(FP)按类别:")
        for cls_id, count in class_fp_counter.most_common():
            name = class_names.get(cls_id, f"cls{cls_id}")
            print(f"  [{cls_id}] {name}: {count}")

    if class_mismatch_counter:
        print(f"\n类别不一致详情 (GT → Pred):")
        for (gt_cls, pred_cls), count in class_mismatch_counter.most_common(20):
            gt_name = class_names.get(gt_cls, f"cls{gt_cls}")
            pred_name = class_names.get(pred_cls, f"cls{pred_cls}")
            print(f"  {gt_name} → {pred_name}: {count}")

    report_path = None
    if save_path:
        report_path = save_path / "label_check_report.json"
        report_data = {
            "summary": {
                "model": model_path,
                "data": str(data_path),
                "iou_thresh": iou_thresh,
                "conf_thresh": conf_thresh,
                "class_agnostic": class_agnostic,
                "total_images": total_images,
                "total_gt": total_gt,
                "total_pred": total_pred,
                "issue_images": len(all_issues),
                "issue_types": dict(issue_counter),
                "fn_by_class": {str(k): v for k, v in class_fn_counter.most_common()},
                "fp_by_class": {str(k): v for k, v in class_fp_counter.most_common()},
                "mismatch_pairs": {
                    f"{class_names.get(gt, f'cls{gt}')}->{class_names.get(pr, f'cls{pr}')}": cnt
                    for (gt, pr), cnt in class_mismatch_counter.most_common()
                },
            },
            "issues": all_issues,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {report_path}")

    if save_path and not no_vis:
        print(f"可视化图片: {save_path}")

    print("=" * 60)

    return all_issues


def parse_args():
    parser = argparse.ArgumentParser(
        description="标签清洗工具 - 基于模型预测检测疑似误标注/漏标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型路径 (如 runs/detect/train/weights/best.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="数据集根目录 (需包含 images/ 和 labels/ 子目录)")
    parser.add_argument("--iou-thresh", type=float, default=0.3,
                        help="匹配 IoU 阈值，低于此值视为不匹配 (默认: 0.3)")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="预测置信度过滤阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值 (默认: 0.7)")
    parser.add_argument("--split", type=str, default=None,
                        help="只检查指定 split (train/val/test)，默认检查全部")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["detect", "pose"],
                        help="任务类型 (默认: detect)")
    parser.add_argument("--kpt-shape", type=int, nargs=2, default=None,
                        metavar=("N_KPT", "N_DIM"),
                        help="关键点配置，仅 pose 任务 (如: 4 3)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="输出目录 (保存报告和可视化图片)")
    parser.add_argument("--no-vis", action="store_true", default=False,
                        help="不保存可视化图片，只生成报告")
    parser.add_argument("--class-agnostic", action="store_true", default=False,
                        help="类别无关匹配 (跨类别 IoU 匹配)")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备 (默认: auto)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--max-det", type=int, default=300,
                        help="每张图最大检测数 (默认: 300)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.save_dir:
        model_stem = Path(args.model).stem
        args.save_dir = str(Path("runs") / "label_check" / model_stem)

    kpt_shape = tuple(args.kpt_shape) if args.kpt_shape else None

    check_labels(
        model_path=args.model,
        data_path=args.data,
        iou_thresh=args.iou_thresh,
        conf_thresh=args.conf_thresh,
        iou_nms=args.iou,
        split=args.split,
        task=args.task,
        kpt_shape=kpt_shape,
        save_dir=args.save_dir,
        no_vis=args.no_vis,
        imgsz=args.imgsz,
        class_agnostic=args.class_agnostic,
        device=args.device,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
