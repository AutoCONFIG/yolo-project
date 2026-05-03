"""Core business layer for YOLO inference.

This package encapsulates all logic related to model inference,
result parsing, visualization, and video processing.

It sits between the CLI frontend (commands/) and the ultralytics backend.
"""

from core.engine import YOLOInference
from core.parser import parse_pytorch_result
from core.types import DetectionResult, ImageResult, NMSConfig
from core.video import get_image_files, get_video_files, inference_video, is_video_file
from core.visualization import draw_detections

__all__ = [
    "DetectionResult",
    "ImageResult",
    "NMSConfig",
    "YOLOInference",
    "draw_detections",
    "get_image_files",
    "get_video_files",
    "inference_video",
    "is_video_file",
    "parse_pytorch_result",
]
