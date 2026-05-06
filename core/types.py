"""Core domain types for YOLO inference.

Data classes and configuration objects used across the inference pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.constants import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_KPT_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_TOPK,
)


# ─── Detection Result ───────────────────────────────────────────────────────


@dataclass
class DetectionResult:
    """单个检测结果。"""

    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None  # 分割掩码 (二值)
    keypoints: Optional[List[List[float]]] = None  # 姿态关键点 [(x, y, conf), ...]


# ─── Image Result ───────────────────────────────────────────────────────────


@dataclass
class ImageResult:
    """单张图片的检测结果。"""

    image_path: str
    image_shape: Tuple[int, int]  # (height, width)
    detections: List[DetectionResult] = field(default_factory=list)
    inference_time: float = 0.0
    task_type: str = "detect"  # detect, segment, classify, pose, obb
    probs: Optional[List[Tuple[str, float]]] = None  # 分类结果
    obb_boxes: Optional[List[Dict]] = None  # OBB 结果

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        result = {
            "image_path": self.image_path,
            "image_shape": list(self.image_shape),
            "inference_time_ms": round(self.inference_time * 1000, 2),
            "task_type": self.task_type,
        }

        if self.task_type == "classify" and self.probs:
            result["num_detections"] = len(self.probs)
            result["classifications"] = [
                {"class_name": name, "probability": round(prob, 4)}
                for name, prob in self.probs
            ]
        else:
            result["num_detections"] = len(self.detections)
            result["detections"] = []
            for d in self.detections:
                det_dict = {
                    "bbox": d.bbox,
                    "confidence": round(d.confidence, 4),
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                if d.mask is not None:
                    det_dict["mask"] = d.mask.tolist()
                if d.keypoints is not None:
                    det_dict["keypoints"] = d.keypoints
                result["detections"].append(det_dict)

        if self.task_type == "obb" and self.obb_boxes:
            result["obb"] = self.obb_boxes

        return result


# ─── NMS Config ─────────────────────────────────────────────────────────────


@dataclass
class NMSConfig:
    """NMS (非极大值抑制) 配置。"""

    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    max_detections: int = DEFAULT_MAX_DETECTIONS
    agnostic: bool = False
    kpt_thres: Optional[float] = None  # 仅 pose 任务使用, None=不传递给后端
    topk: Optional[int] = None  # 仅 classify 任务使用, None=不传递给后端
