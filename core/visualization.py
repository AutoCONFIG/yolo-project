"""Visualization utilities for drawing detection results on images."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.types import ImageResult
from utils.constants import COLOR_GRAY, COLOR_GREEN, COLOR_ORANGE, COLOR_WHITE


def draw_dashed_line(
    img,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """在两点之间绘制虚线。"""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist < 1:
        return
    dashes = int(dist / (dash_len + gap_len)) + 1
    for i in range(dashes):
        s = i * (dash_len + gap_len) / dist
        e = min((i * (dash_len + gap_len) + dash_len) / dist, 1.0)
        sx = int(x1 + (x2 - x1) * s)
        sy = int(y1 + (y2 - y1) * s)
        ex = int(x1 + (x2 - x1) * e)
        ey = int(y1 + (y2 - y1) * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def draw_detections(
    image: np.ndarray,
    result: ImageResult,
    classes: Dict[int, str],
    box_thickness: int = 2,
    font_scale: float = 0.5,
    show_labels: bool = True,
    show_conf: bool = True,
    mask_alpha: float = 0.4,
    kpt_radius: int = 5,
    kpt_line: bool = True,
    line_width: Optional[int] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    kpt_names: Optional[Dict[int, List[str]]] = None,
) -> np.ndarray:
    """在图像上绘制检测结果。

    支持: detect(边框), segment(掩码+边框), classify(概率文本),
    pose(关键点+边框), obb(旋转边框)。
    """
    output = image.copy()
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(len(classes) + 10, 3), dtype=np.uint8)

    # 分类任务
    if result.task_type == "classify" and result.probs is not None:
        y_offset = 30
        for class_name, prob in result.probs:
            text = f"{class_name}: {prob:.2f}"
            color = tuple(int(c) for c in colors[hash(class_name) % len(colors)])
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 2
            )
            cv2.rectangle(
                output,
                (10, y_offset - text_height - 5),
                (10 + text_width + 10, y_offset + 5),
                color,
                -1,
            )
            cv2.putText(
                output,
                text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_offset += text_height + 15
        return output

    # OBB 任务
    if result.task_type == "obb" and result.obb_boxes is not None:
        for obb in result.obb_boxes:
            points = np.array(obb["points"], dtype=np.int32).reshape((-1, 1, 2))
            class_id = obb["class_id"]
            color = tuple(int(c) for c in colors[class_id % len(colors)])
            cv2.polylines(output, [points], isClosed=True, color=color, thickness=box_thickness)
            if show_labels:
                label = obb["class_name"]
                if show_conf:
                    label += f" {obb['confidence']:.2f}"
                cx = int(np.mean(points[:, 0, 0]))
                cy = int(np.mean(points[:, 0, 1]))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(
                    output,
                    (cx - tw // 2 - 2, cy - th - 8),
                    (cx + tw // 2 + 2, cy - 3),
                    color,
                    -1,
                )
                cv2.putText(
                    output,
                    label,
                    (cx - tw // 2, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return output

    # detect / segment / pose
    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        class_id = det.class_id
        color = tuple(int(c) for c in colors[class_id % len(colors)])

        # 分割掩码
        if det.mask is not None:
            mask = det.mask.astype(bool)
            color_array = np.array(color, dtype=np.uint8)
            output[mask] = (output[mask].astype(np.float32) * (1 - mask_alpha) + color_array * mask_alpha).astype(np.uint8)
            contours, _ = cv2.findContours(
                det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, color, box_thickness)

        # 边框
        if det.mask is None:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        # 姿态关键点
        if det.keypoints is not None:
            kpts = np.array(det.keypoints)
            if kpts.ndim != 2:
                kpts = kpts.reshape(-1, kpts.shape[-1])

            def _get_vis(kpt_arr, idx):
                if len(kpt_arr) > 2:
                    v = kpt_arr[2]
                    if v >= 1.5:          # training label v=2 (visible)
                        return 2
                    elif v == 1.0:        # training label v=1 (occluded)
                        return 1
                    elif v >= 0.5:        # inference confidence >= 0.5 (visible)
                        return 2
                    else:
                        return 0
                return 2

            if kpt_line and skeleton is not None:
                for i, j in skeleton:
                    if i < len(kpts) and j < len(kpts):
                        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                        pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                        vis_i = _get_vis(kpts[i], i)
                        vis_j = _get_vis(kpts[j], j)
                        if vis_i == 2 and vis_j == 2:
                            cv2.line(output, pt1, pt2, COLOR_GREEN, max(2, box_thickness))
                        elif vis_i >= 1 and vis_j >= 1:
                            draw_dashed_line(output, pt1, pt2, COLOR_ORANGE, max(2, box_thickness))
                        elif vis_i >= 0 and vis_j >= 0:
                            draw_dashed_line(output, pt1, pt2, COLOR_GRAY, 1)

            for kidx, kpt in enumerate(kpts):
                x, y = int(kpt[0]), int(kpt[1])
                vis = _get_vis(kpt, kidx)
                if vis == 2:
                    cv2.circle(output, (x, y), kpt_radius, COLOR_GREEN, -1)
                    cv2.circle(output, (x, y), kpt_radius, COLOR_WHITE, 1)
                elif vis == 1:
                    cv2.circle(output, (x, y), kpt_radius, COLOR_ORANGE, -1)
                    cv2.circle(output, (x, y), kpt_radius, COLOR_WHITE, 1)
                    r = max(2, kpt_radius - 2)
                    cv2.line(output, (x - r, y - r), (x + r, y + r), COLOR_WHITE, 1)
                    cv2.line(output, (x - r, y + r), (x + r, y - r), COLOR_WHITE, 1)
                else:
                    cv2.circle(output, (x, y), kpt_radius, COLOR_GRAY, 1)

                if kpt_names and class_id in kpt_names and kidx < len(kpt_names[class_id]):
                    name = str(kpt_names[class_id][kidx])
                    txt = f"{kidx}:{name}"
                    (tw, th), _ = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, 1
                    )
                    tx = min(x + kpt_radius + 2, output.shape[1] - tw - 2)
                    ty = max(y - kpt_radius - 2, th + 2)
                    cv2.putText(
                        output,
                        txt,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.7,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # 标签
        if show_labels and det.keypoints is None:
            label = det.class_name
            if show_conf:
                label += f" {det.confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                output, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1
            )
            cv2.putText(
                output,
                label,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return output
