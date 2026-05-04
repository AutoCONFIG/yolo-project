"""YOLO inference engine supporting PyTorch and ONNX backends."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from core.parser import parse_pytorch_result
from core.types import DetectionResult, ImageResult, NMSConfig
from utils.config import setup_ultralytics_path
from utils.constants import LETTERBOX_FILL_VALUE

setup_ultralytics_path()


def _clip_letterbox_coords(
    x1: float, y1: float, x2: float, y2: float,
    pad_left: int, pad_top: int,
    ratio: float,
    orig_w: int, orig_h: int,
) -> Tuple[float, float, float, float]:
    """将 letterbox 变换后的坐标逆变换回原图坐标。

    Args:
        x1, y1, x2, y2: letterbox 坐标系中的坐标
        pad_left, pad_top: letterbox 左/上边距
        ratio: 缩放比例 (原图/letterbox)
        orig_w, orig_h: 原图宽高

    Returns:
        原图坐标系中的 (x1, y1, x2, y2)
    """
    x1 = max(0.0, min((x1 - pad_left) / ratio, orig_w))
    y1 = max(0.0, min((y1 - pad_top) / ratio, orig_h))
    x2 = max(0.0, min((x2 - pad_left) / ratio, orig_w))
    y2 = max(0.0, min((y2 - pad_top) / ratio, orig_h))
    return x1, y1, x2, y2


class YOLOInference:
    """YOLO 推理引擎，支持 PyTorch 和 ONNX 模型。"""

    def __init__(
        self,
        model_path: str,
        nms_config: Optional[NMSConfig] = None,
        device: str = "auto",
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        batch_size: int = 1,
        stream: bool = False,
        half: bool = False,
        augment: bool = False,
        vid_stride: int = 1,
        retina_masks: bool = False,
        visualize: bool = False,
        embed: Optional[Union[List[int], int]] = None,
        int8: bool = False,
        line_width: Optional[int] = None,
        save_frames: bool = False,
        stream_buffer: bool = False,
        save_conf: bool = False,
        dnn: bool = False,
        end2end: Optional[bool] = None,
        show: bool = False,
    ):
        self.model_path = Path(model_path)
        self.nms_config = nms_config or NMSConfig()
        self.device = device
        self.imgsz = imgsz
        self.classes_filter = classes
        self.batch_size = batch_size
        self.stream = stream
        self.half = half
        self.augment = augment
        self.vid_stride = vid_stride
        self.retina_masks = retina_masks
        self.visualize = visualize
        self.embed = embed
        self.int8 = int8
        self.line_width = line_width
        self.save_frames = save_frames
        self.stream_buffer = stream_buffer
        self.save_conf = save_conf
        self.dnn = dnn
        self.end2end = end2end
        self.show = show

        self.model_format = self._detect_format()
        self.model = None
        self.ort_session = None
        self.classes: Dict[int, str] = {}
        self.task_type = "detect"  # 默认任务类型
        self._load_model()

    def _detect_format(self) -> str:
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        elif suffix == ".pt":
            return "pytorch"
        elif suffix in [".torchscript", ".engine", ".tflite"]:
            return suffix[1:]
        return "pytorch"

    def _load_model(self):
        if self.model_format == "onnx":
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.classes = self.model.names
        self.task_type = getattr(self.model, "task", "detect")
        if self.device != "auto":
            self.model.to(self.device)

    def _load_onnx_model(self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX 推理需要 onnxruntime，请安装: pip install onnxruntime 或 onnxruntime-gpu"
            )

        available = ort.get_available_providers()
        use_cuda = (
            self.device == "cuda"
            or (self.device == "auto" and "CUDAExecutionProvider" in available)
            or (self.device.replace(",", "").replace(" ", "").isdigit())
        )
        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(str(self.model_path), providers=providers)

        model_inputs = self.ort_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self._load_onnx_classes()

    def _load_onnx_classes(self):
        import ast

        self.task_type = "detect"
        if self.ort_session:
            metadata = self.ort_session.get_modelmeta()
            if metadata and hasattr(metadata, "custom_metadata_map"):
                custom_meta = metadata.custom_metadata_map
                if "task" in custom_meta:
                    task = custom_meta["task"]
                    if task in {"detect", "segment", "classify", "pose", "obb"}:
                        self.task_type = task
                if "names" in custom_meta:
                    names_str = custom_meta["names"]
                    for parse in (json.loads, ast.literal_eval):
                        try:
                            names = parse(names_str)
                            if isinstance(names, dict):
                                self.classes = {int(k): v for k, v in names.items()}
                                return
                        except Exception:
                            continue
        self.classes = {}

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int], float]:
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=LETTERBOX_FILL_VALUE
        )
        return img, (top, left), r

    def preprocess_onnx(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_letterbox, pad, r = self.letterbox(img_rgb, (self.input_height, self.input_width))
        img_data = img_letterbox.astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = img_data[None]
        return img_data, pad, r

    def postprocess_onnx(
        self, outputs: np.ndarray, orig_shape: Tuple[int, int], pad: Tuple[int, int], r: float
    ) -> List[DetectionResult]:
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        output = np.squeeze(output)

        if output.ndim == 1:
            return []

        # 自动判断输出布局: (C, N) 或 (N, C)。当 C < N 且 C 为较小的维度时，通常为 (C, N)
        if output.ndim == 2 and output.shape[0] < output.shape[1] and output.shape[0] <= 85:
            output = np.transpose(output)

        detections = []
        orig_h, orig_w = orig_shape

        # NMS 内嵌格式 (N, 6)
        if output.shape[1] == 6:
            for det in output:
                conf = float(det[4])
                if conf < self.nms_config.conf_threshold:
                    continue
                class_id = int(det[5])
                x1, y1, x2, y2 = det[:4]
                x1, y1, x2, y2 = _clip_letterbox_coords(
                    x1, y1, x2, y2, pad[1], pad[0], r, orig_w, orig_h
                )
                class_name = self.classes.get(class_id, f"class_{class_id}")
                detections.append(
                    DetectionResult(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
                if len(detections) >= self.nms_config.max_detections:
                    break
            return detections

        # 标准 YOLO 输出
        boxes = output[:, :4]
        scores = output[:, 4:]
        if scores.ndim > 1 and scores.shape[1] > 0:
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
        else:
            return []

        conf_mask = confidences >= self.nms_config.conf_threshold
        boxes = boxes[conf_mask]
        confidences = confidences[conf_mask]
        class_ids = class_ids[conf_mask]

        if len(boxes) == 0:
            return []

        boxes_xyxy = self._convert_boxes_to_xyxy(boxes)
        indices = self._apply_nms(boxes_xyxy, confidences, class_ids)

        if len(indices) > self.nms_config.max_detections:
            sorted_indices = np.argsort(confidences[indices])[::-1]
            indices = indices[sorted_indices[: self.nms_config.max_detections]]

        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            x1, y1, x2, y2 = _clip_letterbox_coords(
                x1, y1, x2, y2, pad[1], pad[0], r, orig_w, orig_h
            )
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

    def _convert_boxes_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.copy()
        # 判断是否为归一化坐标: 所有值在 [0, 1] 范围内且非全零
        if boxes.size > 0 and np.all((boxes >= 0) & (boxes <= 1.0)) and not np.allclose(boxes, 0):
            boxes[:, [0, 2]] *= self.input_width
            boxes[:, [1, 3]] *= self.input_height
            return boxes
        # xywh -> xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return xyxy

    def _apply_nms(
        self, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray
    ) -> np.ndarray:
        if self.nms_config.agnostic:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                confidences.tolist(),
                self.nms_config.conf_threshold,
                self.nms_config.iou_threshold,
            )
            return self._normalize_nms_indices(indices)

        all_indices = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_confidences = confidences[class_mask]
            indices = cv2.dnn.NMSBoxes(
                class_boxes.tolist(),
                class_confidences.tolist(),
                self.nms_config.conf_threshold,
                self.nms_config.iou_threshold,
            )
            normalized = self._normalize_nms_indices(indices)
            all_indices.extend(np.where(class_mask)[0][normalized].tolist())
        return np.array(all_indices, dtype=np.int64)

    @staticmethod
    def _normalize_nms_indices(indices) -> np.ndarray:
        if indices is None or len(indices) == 0:
            return np.array([], dtype=np.int64)
        if isinstance(indices, tuple):
            indices = indices[0] if len(indices) > 0 else np.array([])
        arr = np.array(indices)
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr.astype(np.int64)

    def _predict_kwargs(self) -> dict:
        kwargs = dict(
            imgsz=self.imgsz,
            conf=self.nms_config.conf_threshold,
            iou=self.nms_config.iou_threshold,
            max_det=self.nms_config.max_detections,
            agnostic_nms=self.nms_config.agnostic,
            classes=self.classes_filter,
            verbose=False,
            stream=self.stream,
            half=self.half,
            augment=self.augment,
            vid_stride=self.vid_stride,
            visualize=self.visualize,
            embed=self.embed,
            int8=self.int8,
            save_conf=self.save_conf,
            save_frames=self.save_frames,
            stream_buffer=self.stream_buffer,
            dnn=self.dnn,
            show=self.show,
        )
        # line_width: 仅在显式设置时传递
        if self.line_width is not None:
            kwargs["line_width"] = self.line_width
        # end2end: 仅在显式设置时传递
        if self.end2end is not None:
            kwargs["end2end"] = self.end2end
        # kpt_thres: 仅在显式设置时传递 (姿态任务关键点置信度过滤)
        if self.nms_config.kpt_thres is not None:
            kwargs["kpt_thres"] = self.nms_config.kpt_thres
        # 分割任务专用: 高分辨率分割掩码
        if self.task_type == "segment" and self.retina_masks:
            kwargs["retina_masks"] = self.retina_masks
        # topk: 仅在分类任务时传递给后端
        if self.task_type == "classify" and self.nms_config.topk is not None:
            kwargs["topk"] = self.nms_config.topk
        return kwargs

    def inference_pytorch(self, image: np.ndarray) -> ImageResult:
        start_time = time.perf_counter()
        results = self.model.predict(image, **self._predict_kwargs())
        inference_time = time.perf_counter() - start_time
        image_shape = image.shape[:2]
        if results and len(results) > 0:
            result = parse_pytorch_result(results[0], self.classes, image_shape)
        else:
            result = ImageResult(image_path="array", image_shape=image_shape)
        result.inference_time = inference_time
        return result

    def inference_onnx(self, image: np.ndarray) -> ImageResult:
        img_data, pad, r = self.preprocess_onnx(image)
        model_inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.run(None, {model_inputs[0].name: img_data})
        orig_shape = image.shape[:2]
        detections = self.postprocess_onnx(outputs, orig_shape, pad, r)
        return ImageResult(
            image_path="array",
            image_shape=orig_shape,
            detections=detections,
            task_type=self.task_type,
        )

    def __call__(self, image: Union[str, np.ndarray]) -> ImageResult:
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
        else:
            image_path = "array"

        start_time = time.perf_counter()
        if self.model_format == "onnx":
            result = self.inference_onnx(image)
        else:
            result = self.inference_pytorch(image)

        result.image_path = image_path
        result.inference_time = time.perf_counter() - start_time
        return result

    def inference_batch_pytorch(self, images: List[np.ndarray]) -> List[ImageResult]:
        start_time = time.perf_counter()
        results = self.model.predict(images, **self._predict_kwargs())
        inference_time = time.perf_counter() - start_time
        per_image_time = inference_time / max(len(images), 1)
        all_results = []
        for idx, result in enumerate(results):
            image_shape = images[idx].shape[:2]
            image_result = parse_pytorch_result(result, self.classes, image_shape)
            image_result.inference_time = per_image_time
            all_results.append(image_result)
        return all_results

    def inference_batch(self, images: List[np.ndarray]) -> List[ImageResult]:
        if self.model_format == "onnx":
            return [self.inference_onnx(img) for img in images]
        return self.inference_batch_pytorch(images)
