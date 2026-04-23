"""
YOLO Inference Script
=====================
Batch inference script for YOLO models (PyTorch .pt and ONNX formats).

Supports:
- Single image or batch inference on directories
- Preserves subdirectory structure in output
- Configurable NMS post-processing parameters
- Both PyTorch and ONNX model formats
- Automatic task type detection (detect/segment/classify/pose/obb)

Usage:
    # PyTorch model inference
    python inference.py --model runs/detect/train/weights/best.pt --input images/ --output results/

    # ONNX model inference
    python inference.py --model model.onnx --input images/ --output results/ --format onnx

    # With custom NMS parameters
    python inference.py --model best.pt --input images/ --conf 0.5 --iou 0.45 --max-det 100
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Add ultralytics submodule to path
ULTRALYTICS_PATH = Path(__file__).parent / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

# Patch ultralytics downloads to use weights_dir (must be before ultralytics imports)
from utils.downloads import patch_ultralytics_downloads
patch_ultralytics_downloads()

import yaml


@dataclass
class DetectionResult:
    """Single detection result."""

    bbox: List[float]  # [x1, y1, x2, y2] or [x, y, w, h]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None  # Segmentation mask (binary)
    keypoints: Optional[List[List[float]]] = None  # Pose keypoints [(x, y, conf), ...]


@dataclass
class ImageResult:
    """Results for a single image."""

    image_path: str
    image_shape: Tuple[int, int]  # (height, width)
    detections: List[DetectionResult] = field(default_factory=list)
    inference_time: float = 0.0
    task_type: str = "detect"  # detect, segment, classify, pose, obb
    probs: Optional[List[Tuple[str, float]]] = None  # Classification results [(class_name, prob), ...]
    obb_boxes: Optional[List[Dict]] = None  # OBB results

    def to_dict(self) -> Dict[str, Any]:
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
            result["detections"] = [
                {
                    "bbox": d.bbox,
                    "confidence": round(d.confidence, 4),
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                for d in self.detections
            ]

        if self.task_type == "obb" and self.obb_boxes:
            result["obb"] = self.obb_boxes

        return result


class NMSConfig:
    """NMS (Non-Maximum Suppression) configuration."""

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        agnostic: bool = False,
        multi_label: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.agnostic = agnostic  # Class-agnostic NMS
        self.multi_label = multi_label  # Allow multiple labels per box


class YOLOInference:
    """YOLO inference engine supporting both PyTorch and ONNX models."""

    def __init__(
        self,
        model_path: str,
        nms_config: Optional[NMSConfig] = None,
        device: str = "auto",
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        batch_size: int = 1,
    ):
        self.model_path = Path(model_path)
        self.nms_config = nms_config or NMSConfig()
        self.device = device
        self.imgsz = imgsz
        self.classes_filter = classes
        self.batch_size = batch_size

        # Detect model format
        self.model_format = self._detect_format()

        # Load model
        self.model = None
        self.ort_session = None
        self.classes: Dict[int, str] = {}
        self._load_model()

    def _detect_format(self) -> str:
        """Detect model format from file extension."""
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        elif suffix == ".pt":
            return "pytorch"
        elif suffix in [".torchscript", ".engine", ".tflite"]:
            return suffix[1:]
        else:
            # Default to pytorch
            return "pytorch"

    def _load_model(self):
        """Load model based on format."""
        if self.model_format == "onnx":
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """Load PyTorch model using ultralytics."""
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

        # Get class names
        self.classes = self.model.names

        # Set device
        if self.device != "auto":
            self.model.to(self.device)

    def _load_onnx_model(self):
        """Load ONNX model using onnxruntime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference. Install with: pip install onnxruntime or onnxruntime-gpu")

        # Select providers based on device
        available = ort.get_available_providers()
        if self.device == "cuda" or (self.device == "auto" and "CUDAExecutionProvider" in available):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(str(self.model_path), providers=providers)

        # Get input shape
        model_inputs = self.ort_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # Try to load class names from metadata or use COCO classes
        self._load_onnx_classes()

    def _load_onnx_classes(self):
        """Load class names for ONNX model."""
        # Try to get from model metadata
        if self.ort_session:
            metadata = self.ort_session.get_modelmeta()
            if metadata and "names" in metadata.custom_metadata_map:
                try:
                    names = json.loads(metadata.custom_metadata_map["names"])
                    self.classes = {int(k): v for k, v in names.items()}
                    return
                except (json.JSONDecodeError, ValueError):
                    pass

        # Default to COCO classes
        self.classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Resize image with letterbox padding."""
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute new unpadded shape
        new_unpad = round(shape[1] * r), round(shape[0] * r)

        # Compute padding
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left), r

    def preprocess_onnx(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Preprocess image for ONNX inference."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_letterbox, pad, r = self.letterbox(img_rgb, (self.input_height, self.input_width))

        # Normalize and transpose
        img_data = img_letterbox.astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = img_data[None]  # Add batch dimension

        return img_data, pad, r

    def postprocess_onnx(
        self, outputs: np.ndarray, orig_shape: Tuple[int, int], pad: Tuple[int, int], r: float
    ) -> List[DetectionResult]:
        """Postprocess ONNX model outputs with NMS."""
        # outputs shape: (1, num_detections, 4 + num_classes) or (1, 4 + num_classes, num_detections)
        outputs = np.squeeze(outputs[0])

        # Handle transposed output
        if outputs.shape[0] > outputs.shape[1]:
            outputs = np.transpose(outputs)

        detections = []

        # Get boxes and scores
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        # Get max class scores
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= self.nms_config.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert boxes from center format to corner format
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Apply NMS
        if self.nms_config.agnostic:
            # Class-agnostic NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(),
                confidences.tolist(),
                self.nms_config.conf_threshold,
                self.nms_config.iou_threshold,
            )
            indices = np.array(indices).flatten()
        else:
            # Class-specific NMS
            indices = []
            for class_id in np.unique(class_ids):
                class_mask = class_ids == class_id
                class_boxes = boxes_xyxy[class_mask]
                class_confidences = confidences[class_mask]

                class_indices = cv2.dnn.NMSBoxes(
                    class_boxes.tolist(),
                    class_confidences.tolist(),
                    self.nms_config.conf_threshold,
                    self.nms_config.iou_threshold,
                )
                class_indices = np.array(class_indices).flatten()
                indices.extend(np.where(class_mask)[0][class_indices].tolist())
            indices = np.array(indices)

        # Limit detections
        if len(indices) > self.nms_config.max_detections:
            # Sort by confidence and take top detections
            sorted_indices = np.argsort(confidences[indices])[::-1]
            indices = indices[sorted_indices[: self.nms_config.max_detections]]

        # Scale boxes back to original image coordinates
        orig_h, orig_w = orig_shape

        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]

            # Remove padding and scale
            x1 = (x1 - pad[1]) / r
            y1 = (y1 - pad[0]) / r
            x2 = (x2 - pad[1]) / r
            y2 = (y2 - pad[0]) / r

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            class_id = int(class_ids[idx])
            class_name = self.classes.get(class_id, f"class_{class_id}")

            detections.append(
                DetectionResult(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(confidences[idx]),
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections

    def _detect_task_type(self) -> str:
        """Detect task type from model."""
        if self.model_format == "onnx":
            return "detect"  # Default for ONNX

        # Check model type from ultralytics
        if hasattr(self.model, "task"):
            return self.model.task

        # Fallback: check model name
        model_name = str(self.model_path).lower()
        if "-seg" in model_name:
            return "segment"
        elif "-cls" in model_name:
            return "classify"
        elif "-pose" in model_name:
            return "pose"
        elif "-obb" in model_name:
            return "obb"
        return "detect"

    def inference_pytorch(self, image: np.ndarray) -> ImageResult:
        """Run inference with PyTorch model."""
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.nms_config.conf_threshold,
            iou=self.nms_config.iou_threshold,
            max_det=self.nms_config.max_detections,
            agnostic_nms=self.nms_config.agnostic,
            classes=self.classes_filter,
            verbose=False,
        )

        image_shape = image.shape[:2]
        task_type = self._detect_task_type()

        detections = []
        probs = None
        obb_boxes = None

        if results and len(results) > 0:
            result = results[0]

            # Handle classification task
            if result.probs is not None:
                task_type = "classify"
                probs = []
                for idx in result.probs.top5:
                    class_name = self.classes.get(idx, f"class_{idx}")
                    prob = float(result.probs.data[idx])
                    probs.append((class_name, prob))

            # Handle OBB task
            elif result.obb is not None:
                task_type = "obb"
                obb_boxes = []
                for i in range(len(result.obb)):
                    box = result.obb[i]
                    xyxyxyxy = box.xyxyxyxy.squeeze().cpu().numpy()  # 8 points
                    conf = float(box.conf.cpu().numpy())
                    class_id = int(box.cls.cpu().numpy())
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    # Also add as detection for compatibility
                    detections.append(
                        DetectionResult(
                            bbox=box.xyxy.squeeze().cpu().numpy().tolist(),
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )
                    obb_boxes.append({
                        "points": xyxyxyxy.tolist(),
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name,
                    })

            # Handle detect/segment/pose tasks
            elif result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    # Get mask if available
                    mask = None
                    if result.masks is not None:
                        task_type = "segment"
                        # Get binary mask in original image coordinates
                        mask_data = result.masks.data[i].cpu().numpy()
                        # Scale mask to original image size
                        mask = cv2.resize(
                            mask_data.astype(np.uint8),
                            (image_shape[1], image_shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                    # Get keypoints if available
                    keypoints = None
                    if result.keypoints is not None:
                        task_type = "pose"
                        kpt_data = result.keypoints.data[i].cpu().numpy()
                        if kpt_data.shape[-1] == 3:
                            # (x, y, conf) format
                            keypoints = kpt_data.tolist()
                        else:
                            # (x, y) format
                            keypoints = kpt_data.tolist()

                    detections.append(
                        DetectionResult(
                            bbox=box.tolist(),
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                            mask=mask,
                            keypoints=keypoints,
                        )
                    )

        return ImageResult(
            image_path="array",
            image_shape=image_shape,
            detections=detections,
            task_type=task_type,
            probs=probs,
            obb_boxes=obb_boxes,
        )

    def inference_onnx(self, image: np.ndarray) -> ImageResult:
        """Run inference with ONNX model."""
        # Preprocess
        img_data, pad, r = self.preprocess_onnx(image)

        # Run inference
        model_inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.run(None, {model_inputs[0].name: img_data})

        # Postprocess
        orig_shape = image.shape[:2]
        detections = self.postprocess_onnx(outputs, orig_shape, pad, r)

        return ImageResult(
            image_path="array",
            image_shape=orig_shape,
            detections=detections,
            task_type="detect",  # ONNX doesn't easily support task detection
        )

    def __call__(self, image: Union[str, np.ndarray]) -> ImageResult:
        """Run inference on a single image."""
        # Load image if path provided
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            image_path = "array"

        image_shape = image.shape[:2]

        # Run inference
        start_time = time.perf_counter()

        if self.model_format == "onnx":
            result = self.inference_onnx(image)
        else:
            result = self.inference_pytorch(image)

        result.image_path = image_path
        result.inference_time = time.perf_counter() - start_time

        return result

    def inference_batch_pytorch(self, images: List[np.ndarray]) -> List[ImageResult]:
        """Run batch inference with PyTorch model."""
        start_time = time.perf_counter()

        results = self.model.predict(
            images,
            imgsz=self.imgsz,
            conf=self.nms_config.conf_threshold,
            iou=self.nms_config.iou_threshold,
            max_det=self.nms_config.max_detections,
            agnostic_nms=self.nms_config.agnostic,
            classes=self.classes_filter,
            verbose=False,
        )

        inference_time = time.perf_counter() - start_time
        per_image_time = inference_time / len(images)

        all_results = []
        task_type = self._detect_task_type()

        for idx, result in enumerate(results):
            image_shape = images[idx].shape[:2]
            detections = []
            probs = None
            obb_boxes = None

            # Handle classification task
            if result.probs is not None:
                task_type = "classify"
                probs = []
                for class_idx in result.probs.top5:
                    class_name = self.classes.get(class_idx, f"class_{class_idx}")
                    prob = float(result.probs.data[class_idx])
                    probs.append((class_name, prob))

            # Handle OBB task
            elif result.obb is not None:
                task_type = "obb"
                obb_boxes = []
                for i in range(len(result.obb)):
                    box = result.obb[i]
                    xyxyxyxy = box.xyxyxyxy.squeeze().cpu().numpy()
                    conf = float(box.conf.cpu().numpy())
                    class_id = int(box.cls.cpu().numpy())
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    detections.append(
                        DetectionResult(
                            bbox=box.xyxy.squeeze().cpu().numpy().tolist(),
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )
                    obb_boxes.append({
                        "points": xyxyxyxy.reshape(-1, 2).tolist(),
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name,
                    })

            # Handle segmentation task
            elif result.masks is not None:
                task_type = "segment"
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
                    conf = box.conf.cpu().numpy()
                    conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
                    class_id = int(box.cls.cpu().numpy().squeeze())
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    # Get mask
                    mask = None
                    if result.masks is not None and i < len(result.masks):
                        mask_tensor = result.masks[i].data
                        mask = mask_tensor.cpu().numpy().squeeze()
                        # Resize mask to original image size
                        mask_h, mask_w = mask.shape
                        orig_h, orig_w = image_shape
                        if mask_h != orig_h or mask_w != orig_w:
                            mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    detections.append(
                        DetectionResult(
                            bbox=[x1, y1, x2, y2],
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                            mask=mask,
                        )
                    )

            # Handle pose task
            elif result.keypoints is not None:
                task_type = "pose"
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
                    conf = box.conf.cpu().numpy()
                    conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
                    class_id = int(box.cls.cpu().numpy().squeeze())
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    # Get keypoints
                    keypoints = None
                    if result.keypoints is not None and i < len(result.keypoints):
                        kpts_data = result.keypoints[i].data.cpu().numpy()
                        keypoints = kpts_data.tolist()

                    detections.append(
                        DetectionResult(
                            bbox=[x1, y1, x2, y2],
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                            keypoints=keypoints,
                        )
                    )

            # Handle detection task
            elif result.boxes is not None:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
                    conf = box.conf.cpu().numpy()
                    conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
                    class_id = int(box.cls.cpu().numpy().squeeze())
                    class_name = self.classes.get(class_id, f"class_{class_id}")

                    detections.append(
                        DetectionResult(
                            bbox=[x1, y1, x2, y2],
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )

            all_results.append(ImageResult(
                image_path="array",
                image_shape=image_shape,
                detections=detections,
                inference_time=per_image_time,
                task_type=task_type,
                probs=probs,
                obb_boxes=obb_boxes,
            ))

        return all_results

    def inference_batch(self, images: List[np.ndarray]) -> List[ImageResult]:
        """Run batch inference on multiple images."""
        if self.model_format == "onnx":
            # ONNX doesn't support batch easily, fall back to sequential
            results = []
            for img in images:
                results.append(self.inference_onnx(img))
            return results
        else:
            return self.inference_batch_pytorch(images)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed line between two points."""
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
    skeleton: Optional[List[Tuple[int, int]]] = None,
    kpt_names: Optional[Dict[int, List[str]]] = None,
) -> np.ndarray:
    """Draw detection results on image based on task type.

    Supports:
    - detect: Bounding boxes
    - segment: Masks + bounding boxes
    - classify: Class probabilities text
    - pose: Keypoints + bounding boxes
    - obb: Oriented bounding boxes
    """
    output = image.copy()

    # Generate color palette
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes) + 10, 3), dtype=np.uint8)

    # Handle classification task
    if result.task_type == "classify" and result.probs is not None:
        # Draw classification results as text
        y_offset = 30
        for class_name, prob in result.probs:
            text = f"{class_name}: {prob:.2f}"
            color = tuple(int(c) for c in colors[hash(class_name) % len(colors)])

            # Draw background rectangle
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 2)
            cv2.rectangle(output, (10, y_offset - text_height - 5), (10 + text_width + 10, y_offset + 5), color, -1)

            # Draw text
            cv2.putText(output, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += text_height + 15

        return output

    # Handle OBB task
    if result.task_type == "obb" and result.obb_boxes is not None:
        for obb in result.obb_boxes:
            points = np.array(obb["points"], dtype=np.int32).reshape((-1, 1, 2))
            class_id = obb["class_id"]
            color = tuple(int(c) for c in colors[class_id % len(colors)])

            # Draw rotated bounding box
            cv2.polylines(output, [points], isClosed=True, color=color, thickness=box_thickness)

            # Draw label
            if show_labels:
                label = obb["class_name"]
                if show_conf:
                    label += f" {obb['confidence']:.2f}"

                # Get centroid for label position
                cx = int(np.mean(points[:, 0, 0]))
                cy = int(np.mean(points[:, 0, 1]))

                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(output, (cx - text_width // 2 - 2, cy - text_height - 8),
                             (cx + text_width // 2 + 2, cy - 3), color, -1)
                cv2.putText(output, label, (cx - text_width // 2, cy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return output

    # Create overlay for masks
    overlay = output.copy()

    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        class_id = det.class_id
        color = tuple(int(c) for c in colors[class_id % len(colors)])

        # Draw segmentation mask
        if det.mask is not None:
            mask = det.mask.astype(bool)
            # Fill mask with color
            output[mask] = output[mask] * (1 - mask_alpha) + np.array(color) * mask_alpha
            # Draw mask contour
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, color, box_thickness)

        # Draw bounding box (skip if mask already drew contour)
        if det.mask is None:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        # Draw pose keypoints
        if det.keypoints is not None:
            kpts = np.array(det.keypoints)
            if kpts.ndim != 2:
                kpts = kpts.reshape(-1, kpts.shape[-1])

            # Visibility colors: 0=not labeled(gray), 1=occluded(orange), 2=visible(green)
            VIS_COLOR_GRAY = (160, 160, 160)    # BGR gray for not-labeled
            VIS_COLOR_ORANGE = (0, 165, 255)    # BGR orange for occluded
            VIS_COLOR_GREEN = (0, 220, 0)       # BGR green for visible

            def _get_vis(kpt_arr, idx):
                """Get visibility value: 0=not labeled, 1=occluded, 2=visible."""
                if len(kpt_arr) > 2:
                    v = kpt_arr[2]
                    # YOLO convention: integer 0/1/2 for visibility
                    if v >= 1.5:
                        return 2  # visible
                    elif v >= 0.5:
                        return 1  # occluded
                    else:
                        return 0  # not labeled
                return 2  # default visible if no visibility field

            # Draw skeleton lines using config-driven skeleton
            if kpt_line and skeleton is not None:
                for i, j in skeleton:
                    if i < len(kpts) and j < len(kpts):
                        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                        pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                        vis_i = _get_vis(kpts[i], i)
                        vis_j = _get_vis(kpts[j], j)
                        # Both visible → solid green line
                        if vis_i == 2 and vis_j == 2:
                            cv2.line(output, pt1, pt2, VIS_COLOR_GREEN, max(2, box_thickness))
                        # At least one occluded → dashed orange line
                        elif vis_i >= 1 and vis_j >= 1:
                            _draw_dashed_line(output, pt1, pt2, VIS_COLOR_ORANGE, max(2, box_thickness))
                        # One not-labeled → skip line (or very faint dashed gray)
                        # We still draw a faint hint so the shape is visible
                        elif vis_i >= 0 and vis_j >= 0:
                            _draw_dashed_line(output, pt1, pt2, VIS_COLOR_GRAY, 1)

            # Draw keypoint circles with visibility differentiation
            for kidx, kpt in enumerate(kpts):
                x, y = int(kpt[0]), int(kpt[1])
                vis = _get_vis(kpt, kidx)

                if vis == 2:
                    # Visible: solid filled circle in green with white border
                    cv2.circle(output, (x, y), kpt_radius, VIS_COLOR_GREEN, -1)
                    cv2.circle(output, (x, y), kpt_radius, (255, 255, 255), 1)
                elif vis == 1:
                    # Occluded: filled circle in orange with X mark
                    cv2.circle(output, (x, y), kpt_radius, VIS_COLOR_ORANGE, -1)
                    cv2.circle(output, (x, y), kpt_radius, (255, 255, 255), 1)
                    # Draw X mark inside
                    r = max(2, kpt_radius - 2)
                    cv2.line(output, (x - r, y - r), (x + r, y + r), (255, 255, 255), 1)
                    cv2.line(output, (x - r, y + r), (x + r, y - r), (255, 255, 255), 1)
                else:
                    # Not labeled: hollow circle (ring only) in gray
                    cv2.circle(output, (x, y), kpt_radius, VIS_COLOR_GRAY, 1)

        # Draw label (skip for pose when keypoints are drawn)
        if show_labels and det.keypoints is None:
            label = det.class_name
            if show_conf:
                label += f" {det.confidence:.2f}"

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Draw label background
            cv2.rectangle(
                output,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw text
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


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def is_video_file(path: Path) -> bool:
    """Check if the given path is a video file."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def inference_video(
    engine: "YOLOInference",
    input_path: Path,
    output_path: Path,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    save_vis: bool = True,
    save_json: bool = False,
    verbose: bool = False,
    vis_cfg: Dict[str, Any] = None,
    args=None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    kpt_names: Optional[Dict[int, List[str]]] = None,
) -> List["ImageResult"]:
    """Run inference on a video file and save the annotated result as a video.

    Args:
        engine: YOLOInference engine instance.
        input_path: Path to input video file.
        output_path: Directory to save output video and optional JSON.
        fps: Output video FPS. None = use source FPS.
        codec: FourCC codec string (e.g. 'mp4v', 'xvid', 'h264').
        save_vis: Whether to save the annotated video.
        save_json: Whether to save per-frame detection results as JSON.
        verbose: Print per-frame progress.
        vis_cfg: Visualization config dict.
        args: CLI args for visualization overrides.

    Returns:
        List of ImageResult for each frame.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps if fps is not None else src_fps

    # Setup video writer
    writer = None
    if save_vis:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_path.mkdir(parents=True, exist_ok=True)
        video_out_path = output_path / f"{input_path.stem}_annotated{input_path.suffix if codec == 'mp4v' else '.mp4'}"
        writer = cv2.VideoWriter(str(video_out_path), fourcc, out_fps, (src_w, src_h))
        if not writer.isOpened():
            # Fallback to mp4v if requested codec fails
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_out_path = output_path / f"{input_path.stem}_annotated.mp4"
            writer = cv2.VideoWriter(str(video_out_path), fourcc, out_fps, (src_w, src_h))

    print(f"Video: {input_path.name} | {src_w}x{src_h} @ {src_fps:.1f}fps | {total_frames} frames")
    if writer:
        print(f"Output: {video_out_path}")

    all_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = engine(frame)
        result.image_path = f"{input_path.name}#frame{frame_idx}"
        all_results.append(result)

        if save_vis and writer is not None:
            vis = draw_detections(
                frame,
                result,
                engine.classes,
                box_thickness=vis_cfg.get("box_thickness", args.box_thickness) if vis_cfg else args.box_thickness,
                font_scale=vis_cfg.get("font_scale", args.font_scale) if vis_cfg else args.font_scale,
                show_labels=vis_cfg.get("show_labels", args.show_labels) if vis_cfg else args.show_labels,
                show_conf=vis_cfg.get("show_conf", args.show_conf) if vis_cfg else args.show_conf,
                skeleton=skeleton,
                kpt_names=kpt_names,
            )
            writer.write(vis)

        if verbose:
            print(f"  Frame {frame_idx+1}/{total_frames}: {len(result.detections)} detections, {result.inference_time*1000:.1f}ms")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    # Save JSON results
    if save_json:
        json_path = output_path / f"{input_path.stem}_results.json"
        json_data = {
            "video": str(input_path),
            "fps": out_fps,
            "total_frames": frame_idx,
            "results": [r.to_dict() for r in all_results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to: {json_path}")

    total_dets = sum(len(r.detections) for r in all_results)
    total_time = sum(r.inference_time for r in all_results)
    print(f"Video done: {frame_idx} frames, {total_dets} detections, avg {total_time/frame_idx*1000:.1f}ms/frame")

    return all_results


def get_video_files(input_path: Path) -> List[Path]:
    """Get all video files from input path (directory or single file)."""
    input_path = Path(input_path)

    if input_path.is_file():
        if is_video_file(input_path):
            return [input_path]
        return []
    elif input_path.is_dir():
        files = []
        for ext in VIDEO_EXTENSIONS:
            files.extend(input_path.rglob(f"*{ext}"))
            files.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def get_image_files(input_path: Path, extensions: List[str] = None) -> List[Path]:
    """Get all image files from input path."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in extensions:
            files.extend(input_path.rglob(f"*{ext}"))
            files.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def load_inference_config(config_path: str) -> Dict[str, Any]:
    """Load inference configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Batch Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # PyTorch model inference
    python inference.py --model best.pt --input images/ --output results/

    # ONNX model inference
    python inference.py --model model.onnx --input images/ --output results/

    # Single image inference
    python inference.py --model best.pt --input image.jpg --output results/

    # With custom NMS parameters
    python inference.py --model best.pt --input images/ --conf 0.5 --iou 0.45 --max-det 100

    # Save detection results as JSON
    python inference.py --model best.pt --input images/ --save-json
        """,
    )

    # Model settings
    parser.add_argument("--model", "-m", type=str, default=None, help="Model path (.pt or .onnx)")
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["auto", "pytorch", "onnx"],
        default="auto",
        help="Model format (auto-detect by default)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, 0, 0,1")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference")

    # Input/Output settings
    parser.add_argument("--input", "-i", type=str, default=None, help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="runs/inference", help="Output directory")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        default=True,
        help="Save visualization images",
    )
    parser.add_argument("--no-save-vis", action="store_false", dest="save_vis", help="Don't save visualization")
    parser.add_argument("--save-json", action="store_true", help="Save detection results as JSON")
    parser.add_argument("--save-txt", action="store_true", help="Save detection results as YOLO txt format")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detection images")

    # NMS settings
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS")
    parser.add_argument("--multi-label", action="store_true", default=True, help="Multi-label detection")
    parser.add_argument("--classes", type=int, nargs="+", help="只保留指定类别ID，如 --classes 0 1 2 (person=0, car=2)")

    # Visualization settings
    parser.add_argument("--box-thickness", type=int, default=2, help="Bounding box line thickness")
    parser.add_argument("--font-scale", type=float, default=0.5, help="Font scale for labels")
    parser.add_argument("--show-labels", action="store_true", default=True, help="Show class labels")
    parser.add_argument("--show-conf", action="store_true", default=True, help="Show confidence scores")
    parser.add_argument("--no-show-labels", action="store_false", dest="show_labels")
    parser.add_argument("--no-show-conf", action="store_false", dest="show_conf")

    # Video settings
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS (default: same as source)")
    parser.add_argument("--codec", type=str, default="mp4v", help="Output video codec (default: mp4v)")

    # Other settings
    parser.add_argument("--config", "-c", type=str, help="YAML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config file if specified
    config = {}
    if args.config:
        config = load_inference_config(args.config)

    # Override args with config values (config takes precedence)
    model_cfg = config.get("model", {})
    io_cfg = config.get("io", {})
    nms_cfg = config.get("nms", {})
    vis_cfg = config.get("visualization", {})

    model = model_cfg.get("path", args.model)
    imgsz = model_cfg.get("imgsz", args.imgsz)
    device = model_cfg.get("device", args.device)
    batch_size = model_cfg.get("batch", args.batch)
    classes_filter = model_cfg.get("classes", args.classes)

    input_path = io_cfg.get("input", args.input)
    output_path = io_cfg.get("output", args.output)
    save_vis = io_cfg.get("save_vis", args.save_vis)
    save_json = io_cfg.get("save_json", args.save_json)
    save_txt = io_cfg.get("save_txt", args.save_txt)
    save_crop = io_cfg.get("save_crop", args.save_crop)

    verbose = config.get("verbose", args.verbose)

    # Validate required parameters
    if not model:
        raise ValueError("--model or config model.path is required")
    if not input_path:
        raise ValueError("--input or config io.input is required")

    # Create NMS config
    nms_config = NMSConfig(
        conf_threshold=nms_cfg.get("conf", args.conf),
        iou_threshold=nms_cfg.get("iou", args.iou),
        max_detections=nms_cfg.get("max_det", args.max_det),
        agnostic=nms_cfg.get("agnostic", args.agnostic_nms),
        multi_label=nms_cfg.get("multi_label", args.multi_label),
    )

    # Initialize inference engine
    print(f"\n{'='*60}")
    print("YOLO Inference")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"NMS Config:")
    print(f"  - Confidence threshold: {nms_config.conf_threshold}")
    print(f"  - IoU threshold: {nms_config.iou_threshold}")
    print(f"  - Max detections: {nms_config.max_detections}")
    print(f"  - Class-agnostic: {nms_config.agnostic}")
    print(f"  - Classes filter: {classes_filter if classes_filter else 'All'}")
    print(f"{'='*60}\n")

    # Parse skeleton connections from config
    skeleton = None
    skeleton_cfg = vis_cfg.get("skeleton", None)
    if skeleton_cfg is not None:
        skeleton = [tuple(pair) for pair in skeleton_cfg]
    kpt_names = vis_cfg.get("kpt_names", None)

    engine = YOLOInference(
        model_path=model,
        nms_config=nms_config,
        device=device,
        imgsz=imgsz,
        classes=classes_filter,
        batch_size=batch_size,
    )

    # Get input files
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Video inference path — single video file
    if input_path.is_file() and is_video_file(input_path):
        video_cfg = config.get("video", {})
        out_fps = video_cfg.get("fps", args.fps)
        out_codec = video_cfg.get("codec", args.codec)

        inference_video(
            engine=engine,
            input_path=input_path,
            output_path=output_path / "video",
            fps=out_fps,
            codec=out_codec,
            save_vis=save_vis,
            save_json=save_json,
            verbose=verbose,
            vis_cfg=vis_cfg,
            args=args,
            skeleton=skeleton,
            kpt_names=kpt_names,
        )
        return

    # Directory input: smart scan for both videos and images
    video_files = []
    image_files = []
    if input_path.is_dir():
        video_files = get_video_files(input_path)
        image_files = get_image_files(input_path)
    elif input_path.is_file():
        image_files = [input_path]

    # Process videos
    if video_files:
        video_cfg = config.get("video", {})
        out_fps = video_cfg.get("fps", args.fps)
        out_codec = video_cfg.get("codec", args.codec)
        print(f"Found {len(video_files)} video(s) to process")

        for vf in video_files:
            if verbose:
                print(f"\nProcessing video: {vf}")
            inference_video(
                engine=engine,
                input_path=vf,
                output_path=output_path / "video",
                fps=out_fps,
                codec=out_codec,
                save_vis=save_vis,
                save_json=save_json,
                verbose=verbose,
                vis_cfg=vis_cfg,
                args=args,
                skeleton=skeleton,
                kpt_names=kpt_names,
            )

    # Process images
    if image_files:
        print(f"Found {len(image_files)} images to process")
    elif not video_files:
        print("No images or videos found!")
        return

    if not image_files:
        return

    # Process images in batches
    all_results = []
    total_detections = 0
    total_time = 0.0
    detected_task_type = None

    def save_single_result(result, image_file, image):
        """Save results for a single image."""
        nonlocal detected_task_type

        # Detect task type from first result
        if detected_task_type is None:
            detected_task_type = result.task_type

        # Calculate relative path for output
        if input_path.is_file():
            rel_path = image_file.name
        else:
            rel_path = image_file.relative_to(input_path)

        # Save visualization
        if save_vis:
            vis_output = draw_detections(
                image,
                result,
                engine.classes,
                box_thickness=vis_cfg.get("box_thickness", args.box_thickness),
                font_scale=vis_cfg.get("font_scale", args.font_scale),
                show_labels=vis_cfg.get("show_labels", args.show_labels),
                show_conf=vis_cfg.get("show_conf", args.show_conf),
                skeleton=skeleton,
                kpt_names=kpt_names,
            )

            vis_path = output_path / "vis" / rel_path
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis_output)

        # Save cropped detections
        if save_crop and result.detections:
            crop_dir = output_path / "crops" / rel_path.stem
            crop_dir.mkdir(parents=True, exist_ok=True)

            for j, det in enumerate(result.detections):
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                crop = image[y1:y2, x1:x2]
                crop_path = crop_dir / f"{det.class_name}_{j}.jpg"
                cv2.imwrite(str(crop_path), crop)

        # Save txt label (real-time)
        if save_txt:
            txt_dir = output_path / "labels"
            txt_path = txt_dir / str(rel_path.with_suffix(".txt"))
            txt_path.parent.mkdir(parents=True, exist_ok=True)

            img_h, img_w = result.image_shape

            with open(txt_path, "w") as f:
                # Handle classification results
                if result.task_type == "classify" and result.probs:
                    for class_name, prob in result.probs:
                        f.write(f"{class_name} {prob:.4f}\n")

                # Handle OBB results
                elif result.task_type == "obb" and result.obb_boxes:
                    for obb in result.obb_boxes:
                        # Write OBB in YOLO format: class x1 y1 x2 y2 x3 y3 x4 y4 conf
                        points = obb["points"]
                        line = f"{obb['class_id']}"
                        for pt in points:
                            line += f" {pt[0]/img_w:.6f} {pt[1]/img_h:.6f}"
                        line += f" {obb['confidence']:.4f}\n"
                        f.write(line)

                # Handle detect/segment/pose results
                else:
                    for det in result.detections:
                        x1, y1, x2, y2 = det.bbox

                        # Write segmentation mask if available
                        if det.mask is not None:
                            # Convert mask to polygon points
                            contours, _ = cv2.findContours(
                                det.mask.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            if contours:
                                # Use largest contour
                                contour = max(contours, key=cv2.contourArea)
                                # Normalize coordinates
                                points = contour.squeeze()
                                if len(points.shape) == 1:
                                    points = points.reshape(1, 2)
                                points_str = " ".join(f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in points)
                                f.write(f"{det.class_id} {points_str} {det.confidence:.4f}\n")
                            continue

                        # Write keypoints if available
                        if det.keypoints is not None:
                            kpts = np.array(det.keypoints, dtype=float)
                            if kpts.ndim == 3:
                                kpts = kpts.squeeze(0)
                            # YOLO pose format: class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... conf
                            cx = (x1 + x2) / 2 / img_w
                            cy = (y1 + y2) / 2 / img_h
                            w = (x2 - x1) / img_w
                            h = (y2 - y1) / img_h

                            kpts_str = ""
                            for kpt in kpts:
                                if len(kpt) >= 3:
                                    kpts_str += f" {kpt[0]/img_w:.6f} {kpt[1]/img_h:.6f} {kpt[2]:.4f}"
                                else:
                                    kpts_str += f" {kpt[0]/img_w:.6f} {kpt[1]/img_h:.6f} 1.0"

                            f.write(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}{kpts_str} {det.confidence:.4f}\n")
                            continue

                        # Standard detection format
                        cx = (x1 + x2) / 2 / img_w
                        cy = (y1 + y2) / 2 / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h

                        f.write(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {det.confidence:.4f}\n")

    # Batch processing
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

        # Read batch images
        batch_images = []
        valid_files = []
        for image_file in batch_files:
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Warning: Failed to load {image_file}")
                continue
            batch_images.append(image)
            valid_files.append(image_file)

        if not batch_images:
            continue

        # Run batch inference
        start_time = time.perf_counter()
        batch_results = engine.inference_batch(batch_images)
        batch_time = time.perf_counter() - start_time

        # Process and save results
        for result, image_file, image in zip(batch_results, valid_files, batch_images):
            result.image_path = str(image_file)
            all_results.append(result)
            total_detections += len(result.detections)
            total_time += result.inference_time

            if verbose:
                task_info = f"[{result.task_type}]" if result.task_type != "detect" else ""
                print(f"[{start_idx + len(valid_files)}/{len(image_files)}] {image_file.name}: {len(result.detections)} detections {task_info}, {result.inference_time*1000:.2f}ms")

            save_single_result(result, image_file, image)

    # Save JSON results
    if save_json:
        json_path = output_path / "results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = {
            "model": model,
            "nms_config": {
                "conf_threshold": nms_config.conf_threshold,
                "iou_threshold": nms_config.iou_threshold,
                "max_detections": nms_config.max_detections,
                "agnostic": nms_config.agnostic,
            },
            "total_images": len(image_files),
            "total_detections": total_detections,
            "results": [r.to_dict() for r in all_results],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Inference Summary")
    print(f"{'='*60}")
    print(f"Total images:    {len(image_files)}")
    print(f"Task type:       {detected_task_type or 'detect'}")
    print(f"Total detections: {total_detections}")
    print(f"Total time:      {total_time:.2f}s")
    print(f"Average time:    {total_time/len(image_files)*1000:.2f}ms per image")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
