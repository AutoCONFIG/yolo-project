"""Result parsing utilities.

Convert ultralytics Result objects into structured ImageResult / DetectionResult.
"""

from typing import Dict, Tuple

from core.types import DetectionResult, ImageResult


def _parse_box(box, classes: Dict[int, str]):
    """从 ultralytics box 对象提取 bbox/conf/class_id/class_name。"""
    x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
    conf = box.conf.cpu().numpy()
    conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
    class_id = int(box.cls.cpu().numpy().squeeze())
    class_name = classes.get(class_id, f"class_{class_id}")
    return [x1, y1, x2, y2], conf, class_id, class_name


def _parse_mask(mask_data, image_shape: Tuple[int, int]):
    """将单张 mask tensor 转为与原图对齐的二值数组。"""
    mask = mask_data.cpu().numpy().squeeze()
    mask_h, mask_w = mask.shape
    orig_h, orig_w = image_shape
    if mask_h != orig_h or mask_w != orig_w:
        import cv2
        mask = cv2.resize(mask.astype("uint8"), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def _parse_detect_detections(boxes, classes: Dict[int, str], branch: str = None):
    """解析纯检测框分支。"""
    detections = []
    for i in range(len(boxes)):
        bbox, conf, class_id, class_name = _parse_box(boxes[i], classes)
        detections.append(
            DetectionResult(
                bbox=bbox,
                confidence=conf,
                class_id=class_id,
                class_name=class_name,
                branch=branch,
            )
        )
    return detections


def _parse_segment_detections(boxes, masks, classes: Dict[int, str], image_shape: Tuple[int, int], branch: str = None):
    """解析实例分割分支（boxes + masks）。"""
    detections = []
    for i in range(len(boxes)):
        bbox, conf, class_id, class_name = _parse_box(boxes[i], classes)
        mask = None
        if masks is not None and i < len(masks):
            mask = _parse_mask(masks[i].data, image_shape)
        detections.append(
            DetectionResult(
                bbox=bbox,
                confidence=conf,
                class_id=class_id,
                class_name=class_name,
                mask=mask,
                branch=branch,
            )
        )
    return detections


def _is_detect_segment_result(result) -> bool:
    """DetectSegmentResults 只含 detect/segment 两个嵌套分支，不具备单任务 Results 的 probs/boxes 等探测属性，
    且其 __getattr__ 对缺失属性直接抛 AttributeError，必须先于一切属性探测判断。"""
    return getattr(result, "detect", None) is not None and getattr(result, "segment", None) is not None


def parse_pytorch_result(result, classes: Dict[int, str], image_shape: Tuple[int, int]) -> ImageResult:
    """将 ultralytics Results 对象解析为 ImageResult。

    支持所有任务类型: detect-segment, classify, obb, segment, pose, detect。
    优先级: detect/segment 嵌套分支 -> probs -> obb -> masks -> keypoints -> boxes
    """
    detections = []
    probs = None
    obb_boxes = None
    task_type = "detect"

    if _is_detect_segment_result(result):
        task_type = "detect-segment"
        # 两个分支是独立类空间，各自携带自己的 names，不能用 engine 的全局 classes（仅 detect 类空间）
        detect_names = getattr(result.detect, "names", None) or classes
        segment_names = getattr(result.segment, "names", None) or classes
        if result.detect.boxes is not None:
            detections += _parse_detect_detections(result.detect.boxes, detect_names, branch="detect")
        if result.segment.boxes is not None:
            detections += _parse_segment_detections(
                result.segment.boxes, result.segment.masks, segment_names, image_shape, branch="segment"
            )

    elif result.probs is not None:
        task_type = "classify"
        probs = []
        for idx in result.probs.top5:
            class_name = classes.get(idx, f"class_{idx}")
            prob = float(result.probs.data[idx])
            probs.append((class_name, prob))

    elif result.obb is not None:
        task_type = "obb"
        obb_boxes = []
        for i in range(len(result.obb)):
            box = result.obb[i]
            xyxyxyxy = box.xyxyxyxy.cpu().numpy()
            if xyxyxyxy.ndim > 2:
                xyxyxyxy = xyxyxyxy.squeeze()
            if xyxyxyxy.ndim == 1:
                xyxyxyxy = xyxyxyxy.reshape(4, 2)
            conf = float(box.conf.cpu().numpy())
            class_id = int(box.cls.cpu().numpy())
            class_name = classes.get(class_id, f"class_{class_id}")

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

    elif result.masks is not None:
        task_type = "segment"
        detections = _parse_segment_detections(result.boxes, result.masks, classes, image_shape)

    elif result.keypoints is not None:
        task_type = "pose"
        for i in range(len(result.boxes)):
            bbox, conf, class_id, class_name = _parse_box(result.boxes[i], classes)

            keypoints = None
            if i < len(result.keypoints):
                kpts_data = result.keypoints[i].data.cpu().numpy()
                keypoints = kpts_data.tolist()

            detections.append(
                DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    keypoints=keypoints,
                )
            )

    elif result.boxes is not None:
        detections = _parse_detect_detections(result.boxes, classes)

    return ImageResult(
        image_path="array",
        image_shape=image_shape,
        detections=detections,
        task_type=task_type,
        probs=probs,
        obb_boxes=obb_boxes,
    )
