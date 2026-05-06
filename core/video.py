"""Video inference and file collection utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.engine import YOLOInference
from core.types import ImageResult
from core.visualization import draw_detections
from utils.constants import (
    DEFAULT_KPT_LINE,
    DEFAULT_KPT_RADIUS,
    DEFAULT_MASK_ALPHA,
    IMG_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


# ─── File utilities ─────────────────────────────────────────────────────────


def is_video_file(path: Path) -> bool:
    """检查路径是否为视频文件。"""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def get_video_files(input_path: Path) -> List[Path]:
    """从输入路径获取所有视频文件。"""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if is_video_file(input_path) else []
    elif input_path.is_dir():
        files = []
        for ext in VIDEO_EXTENSIONS:
            files.extend(input_path.rglob(f"*{ext}"))
            files.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    raise ValueError(f"输入路径不存在: {input_path}")


def get_image_files(input_path: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """从输入路径获取所有图像文件。"""
    if extensions is None:
        extensions = sorted(IMG_EXTENSIONS)
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in extensions:
            files.extend(input_path.rglob(f"*{ext}"))
            files.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    raise ValueError(f"输入路径不存在: {input_path}")


# ─── Video inference ────────────────────────────────────────────────────────


def inference_video(
    engine: YOLOInference,
    input_path: Path,
    output_path: Path,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    save_vis: bool = True,
    save_json: bool = False,
    verbose: bool = False,
    vis_cfg: Optional[Dict[str, Any]] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    kpt_names: Optional[Dict[int, List[str]]] = None,
    vid_stride: int = 1,
) -> List[ImageResult]:
    """对视频文件运行推理，保存标注结果。"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps if fps is not None else src_fps / vid_stride

    writer = None
    video_out_path = None
    if save_vis:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = input_path.suffix if codec == "mp4v" else ".mp4"
        video_out_path = output_path / f"{input_path.stem}_annotated{suffix}"
        writer = cv2.VideoWriter(str(video_out_path), fourcc, out_fps, (src_w, src_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_out_path = output_path / f"{input_path.stem}_annotated.mp4"
            writer = cv2.VideoWriter(str(video_out_path), fourcc, out_fps, (src_w, src_h))

    print(f"视频: {input_path.name} | {src_w}x{src_h} @ {src_fps:.1f}fps | {total_frames} 帧")
    if writer and video_out_path:
        print(f"输出: {video_out_path}")

    all_results = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % vid_stride != 0:
                frame_idx += 1
                continue

            result = engine(frame)
            result.image_path = f"{input_path.name}#frame{frame_idx}"
            all_results.append(result)

            if save_vis and writer is not None:
                vis = draw_detections(
                    frame,
                    result,
                    engine.classes,
                    box_thickness=vis_cfg.get("box_thickness", 2) if vis_cfg else 2,
                    font_scale=vis_cfg.get("font_scale", 0.5) if vis_cfg else 0.5,
                    show_labels=vis_cfg.get("show_labels", True) if vis_cfg else True,
                    show_conf=vis_cfg.get("show_conf", True) if vis_cfg else True,
                    line_width=vis_cfg.get("line_width") if vis_cfg else None,
                    mask_alpha=vis_cfg.get("mask_alpha", DEFAULT_MASK_ALPHA) if vis_cfg else DEFAULT_MASK_ALPHA,
                    kpt_radius=vis_cfg.get("kpt_radius", DEFAULT_KPT_RADIUS) if vis_cfg else DEFAULT_KPT_RADIUS,
                    kpt_line=vis_cfg.get("kpt_line", DEFAULT_KPT_LINE) if vis_cfg else DEFAULT_KPT_LINE,
                    skeleton=skeleton,
                    kpt_names=kpt_names,
                )
                writer.write(vis)

            if verbose:
                print(
                    f"  帧 {frame_idx+1}/{total_frames}: {len(result.detections)} 检测, "
                    f"{result.inference_time*1000:.1f}ms"
                )

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None and writer.isOpened():
            writer.release()

    if save_json:
        json_path = output_path / f"{input_path.stem}_results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = {
            "video": str(input_path),
            "fps": out_fps,
            "total_frames": frame_idx,
            "results": [r.to_dict() for r in all_results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON 结果保存至: {json_path}")

    total_dets = sum(len(r.detections) for r in all_results)
    total_time = sum(r.inference_time for r in all_results)
    if frame_idx > 0:
        print(
            f"视频处理完成: {frame_idx} 帧, {total_dets} 检测, "
            f"平均 {total_time/frame_idx*1000:.1f}ms/帧"
        )
    else:
        print("视频处理完成: 0 帧")

    return all_results
