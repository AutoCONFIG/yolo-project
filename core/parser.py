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


def parse_pytorch_result(result, classes: Dict[int, str], image_shape: Tuple[int, int]) -> ImageResult:
    """将 ultralytics Results 对象解析为 ImageResult。

    支持所有任务类型: classify, obb, segment, pose, detect。
    优先级: probs -> obb -> masks -> keypoints -> boxes
    """
    detections = []
    probs = None
    obb_boxes = None
    task_type = "detect"

    if result.probs is not None:
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
        for i in range(len(result.boxes)):
            bbox, conf, class_id, class_name = _parse_box(result.boxes[i], classes)

            mask = None
            if i < len(result.masks):
                mask_tensor = result.masks[i].data
                mask = mask_tensor.cpu().numpy().squeeze()
                mask_h, mask_w = mask.shape
                orig_h, orig_w = image_shape
                if mask_h != orig_h or mask_w != orig_w:
                    import cv2
                    mask = cv2.resize(mask.astype("uint8"), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            detections.append(
                DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                )
            )

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
        for i in range(len(result.boxes)):
            bbox, conf, class_id, class_name = _parse_box(result.boxes[i], classes)

            detections.append(
                DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
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
