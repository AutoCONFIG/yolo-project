"""Shared constants for the YOLO project.

Centralizes magic numbers, default values, color palettes, and file extensions
to avoid scattering hard-coded literals across the codebase.
"""

from pathlib import Path
from typing import Dict, Final, List, Set, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  Path defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PREDICT_OUTPUT: Final[str] = "runs/predict"

# ═══════════════════════════════════════════════════════════════════════════════
#  NMS / inference defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.7
DEFAULT_MAX_DETECTIONS: Final[int] = 300
DEFAULT_IMGSZ: Final[int] = 640
DEFAULT_BATCH_SIZE: Final[int] = 16
DEFAULT_KPT_THRESHOLD: Final[float] = 0.5
DEFAULT_TOPK: Final[int] = 5

# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_BOX_THICKNESS: Final[int] = 2
DEFAULT_FONT_SCALE: Final[float] = 0.5
DEFAULT_MASK_ALPHA: Final[float] = 0.4
DEFAULT_KPT_RADIUS: Final[int] = 5
DEFAULT_KPT_LINE: Final[bool] = True

# Pre-defined color palettes (BGR for OpenCV)
COLOR_GRAY: Final[Tuple[int, int, int]] = (160, 160, 160)
COLOR_ORANGE: Final[Tuple[int, int, int]] = (0, 165, 255)
COLOR_GREEN: Final[Tuple[int, int, int]] = (0, 220, 0)
COLOR_WHITE: Final[Tuple[int, int, int]] = (255, 255, 255)

# ═══════════════════════════════════════════════════════════════════════════════
#  Letterbox defaults
# ═══════════════════════════════════════════════════════════════════════════════

LETTERBOX_FILL_VALUE: Final[Tuple[int, int, int]] = (114, 114, 114)

# ═══════════════════════════════════════════════════════════════════════════════
#  Video defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_VIDEO_CODEC: Final[str] = "mp4v"
DEFAULT_VIDEO_FPS: Final[float] = 25.0

# ═══════════════════════════════════════════════════════════════════════════════
#  File extensions
# ═══════════════════════════════════════════════════════════════════════════════

IMG_EXTENSIONS: Final[Set[str]] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

VIDEO_EXTENSIONS: Final[Set[str]] = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Export formats
# ═══════════════════════════════════════════════════════════════════════════════

EXPORT_FORMATS: Final[Dict[str, Dict[str, str]]] = {
    "onnx": {"suffix": ".onnx", "desc": "ONNX"},
    "torchscript": {"suffix": ".torchscript", "desc": "TorchScript"},
    "openvino": {"suffix": "_openvino_model", "desc": "OpenVINO"},
    "engine": {"suffix": ".engine", "desc": "TensorRT"},
    "coreml": {"suffix": ".mlpackage", "desc": "CoreML"},
    "saved_model": {"suffix": "_saved_model", "desc": "TensorFlow SavedModel"},
    "pb": {"suffix": ".pb", "desc": "TensorFlow GraphDef"},
    "tflite": {"suffix": ".tflite", "desc": "TensorFlow Lite"},
    "edgetpu": {"suffix": "_edgetpu.tflite", "desc": "Edge TPU"},
    "tfjs": {"suffix": "_web_model", "desc": "TensorFlow.js"},
    "paddle": {"suffix": "_paddle_model", "desc": "PaddlePaddle"},
    "mnn": {"suffix": ".mnn", "desc": "MNN"},
    "ncnn": {"suffix": "_ncnn_model", "desc": "NCNN"},
    "rknn": {"suffix": "_rknn_model", "desc": "RKNN"},
    "executorch": {"suffix": "_executorch_model", "desc": "ExecuTorch"},
    "axelera": {"suffix": "_axelera_model", "desc": "Axelera AI"},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Color generators
# ═══════════════════════════════════════════════════════════════════════════════


def generate_class_colors(num_classes: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic pseudo-random colors for class visualization.

    Args:
        num_classes: Number of distinct classes.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (num_classes + 10, 3) with uint8 BGR colors.
        The +10 padding avoids index-overflow when hashing class names.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(num_classes + 10, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pose defaults (parking spot task)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_KPT_SHAPE: Final[Tuple[int, int]] = (4, 3)  # 4 keypoints, 3 dims (x, y, v)
DEFAULT_SKELETON: Final[List[Tuple[int, int]]] = [(0, 1), (1, 2), (2, 3), (3, 0)]
DEFAULT_KPT_NAMES: Final[List[str]] = [
    "front_left",
    "front_right",
    "rear_right",
    "rear_left",
]
DEFAULT_KPT_COLORS: Final[List[Tuple[int, int, int]]] = [
    (255, 0, 0),    # front_left  - blue
    (0, 255, 0),    # front_right - green
    (0, 0, 255),    # rear_right  - red
    (255, 255, 0),  # rear_left   - cyan
]
