"""Result parsing utilities.

Convert ultralytics Result objects into structured ImageResult / DetectionResult.
"""

from typing import Dict, Tuple

from core.types import DetectionResult, ImageResult


def parse_pytorch_result(result, classes: Dict[int, str], image_shape: Tuple[int, int]) -> ImageResult:
    """将 ultralytics Results 对象解析为 ImageResult。

    支持所有任务类型: classify, obb, segment, pose, detect。
    优先级: probs -> obb -> masks -> keypoints -> boxes
    """
    detections = []
    probs = None
    obb_boxes = None
    task_type = "detect"

    # 分类任务
    if result.probs is not None:
        task_type = "classify"
        probs = []
        for idx in result.probs.top5:
            class_name = classes.get(idx, f"class_{idx}")
            prob = float(result.probs.data[idx])
            probs.append((class_name, prob))

    # OBB 任务
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

    # 分割任务
    elif result.masks is not None:
        task_type = "segment"
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
            conf = box.conf.cpu().numpy()
            conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
            class_id = int(box.cls.cpu().numpy().squeeze())
            class_name = classes.get(class_id, f"class_{class_id}")

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
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                )
            )

    # 姿态估计任务
    elif result.keypoints is not None:
        task_type = "pose"
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
            conf = box.conf.cpu().numpy()
            conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
            class_id = int(box.cls.cpu().numpy().squeeze())
            class_name = classes.get(class_id, f"class_{class_id}")

            keypoints = None
            if i < len(result.keypoints):
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

    # 检测任务
    elif result.boxes is not None:
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy.squeeze().cpu().numpy().tolist()
            conf = box.conf.cpu().numpy()
            conf = float(conf.item() if conf.ndim == 0 else conf.squeeze())
            class_id = int(box.cls.cpu().numpy().squeeze())
            class_name = classes.get(class_id, f"class_{class_id}")

            detections.append(
                DetectionResult(
                    bbox=[x1, y1, x2, y2],
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
