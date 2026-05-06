"""
YOLO Tracking Module
====================
CLI frontend for running YOLO object tracking.

Tracking is built on top of detection/segmentation/pose inference,
adding multi-object tracking (ByteTrack / BoT-SORT) with persistent IDs.

Usage::

    python yolo.py track --config configs/predict/example/track_example.yaml
    python yolo.py track --model best.pt --input video.mp4 --tracker botsort.yaml
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from core import (
    get_video_files,
    is_video_file,
)
from core.engine import YOLOInference
from core.parser import parse_pytorch_result
from core.types import NMSConfig
from core.visualization import draw_detections
from utils.config import (
    config_from_args,
    get_nested_value,
    load_yaml_config,
    merge_configs,
    resolve_config_value,
    set_boolean_argument,
    setup_ultralytics_path,
    to_bool,
)
from utils.constants import DEFAULT_IMGSZ

setup_ultralytics_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 目标跟踪",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python yolo.py track --model best.pt --input video.mp4
    python yolo.py track --config configs/predict/example/track_example.yaml
    python yolo.py track --model best.pt --input video.mp4 --tracker bytetrack.yaml
        """,
    )

    parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--model", "-m", type=str, default=None, help="模型路径 (.pt 或 .onnx)")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None, help="设备: auto, cpu, cuda, 0")
    parser.add_argument("--batch", type=int, default=None, help="推理批大小")
    parser.add_argument("--tracker", type=str, default=None, help="跟踪器配置 (botsort.yaml / bytetrack.yaml)")

    set_boolean_argument(parser, "stream", "stream", help_true="流式推理", help_false="非流式")
    set_boolean_argument(parser, "half", "half", help_true="FP16 半精度", help_false="全精度")
    set_boolean_argument(parser, "augment", "augment", help_true="TTA 增强", help_false="无 TTA")
    set_boolean_argument(parser, "retina_masks", "retina-masks", help_true="高分辨率掩码", help_false="标准掩码")
    parser.add_argument("--vid-stride", type=int, default=None, help="视频帧步长")
    set_boolean_argument(parser, "visualize", "visualize", help_true="可视化特征", help_false="不可视化")
    set_boolean_argument(parser, "int8", "int8", help_true="INT8 量化", help_false="无 INT8")
    set_boolean_argument(parser, "dnn", "dnn", help_true="OpenCV DNN ONNX 推理", help_false="不使用 DNN")
    set_boolean_argument(parser, "end2end", "end2end", help_true="端到端检测头 (YOLO26/YOLOv10)", help_false="标准检测头")
    set_boolean_argument(parser, "save_conf", "save-conf", help_true="保存置信度到结果", help_false="不保存置信度")
    set_boolean_argument(parser, "stream_buffer", "stream-buffer", help_true="流式缓冲所有帧", help_false="只保留最新帧")
    parser.add_argument("--embed", type=int, nargs="+", default=None, help="特征嵌入层索引")
    parser.add_argument("--kpt-thres", type=float, default=None, help="关键点阈值 (仅姿态估计)")
    parser.add_argument("--topk", type=int, default=None, help="分类 Top-K (仅分类任务)")

    parser.add_argument("--input", "-i", type=str, default=None, help="输入视频或目录")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出目录")
    set_boolean_argument(parser, "save_vis", "save-vis", help_true="保存可视化", help_false="不保存可视化", neg_prefix="no-")
    set_boolean_argument(parser, "save_json", "save-json", help_true="保存 JSON 结果", help_false="不保存 JSON")
    set_boolean_argument(parser, "save_txt", "save-txt", help_true="保存 YOLO txt 标签", help_false="不保存 txt")
    set_boolean_argument(parser, "save_crop", "save-crop", help_true="保存裁剪目标", help_false="不保存裁剪")

    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=None, help="最大检测数")
    set_boolean_argument(parser, "agnostic_nms", "agnostic-nms", help_true="类别无关 NMS", help_false="类别特定 NMS")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="类别过滤")

    parser.add_argument("--box-thickness", type=int, default=None, help="边框线宽")
    parser.add_argument("--font-scale", type=float, default=None, help="字体比例")
    set_boolean_argument(parser, "show_labels", "show-labels", help_true="显示标签", help_false="不显示标签", neg_prefix="no-")
    set_boolean_argument(parser, "show_conf", "show-conf", help_true="显示置信度", help_false="不显示置信度", neg_prefix="no-")

    parser.add_argument("--fps", type=float, default=None, help="输出视频 FPS")
    parser.add_argument("--codec", type=str, default=None, help="视频编码器")

    parser.add_argument("--line-width", type=int, default=None, help="后端渲染边框线宽 (自动缩放)")
    set_boolean_argument(parser, "show_boxes", "show-boxes", help_true="显示检测框", help_false="不显示检测框", neg_prefix="no-")
    set_boolean_argument(parser, "save_frames", "save-frames", help_true="保存视频帧", help_false="不保存帧", neg_prefix="no-")
    set_boolean_argument(parser, "show", "show", help_true="弹出窗口显示结果", help_false="不弹出窗口")
    set_boolean_argument(parser, "verbose", "verbose", help_true="详细输出", help_false="简洁输出", neg_prefix="no-")
    set_boolean_argument(parser, "persist", "persist", help_true="跨调用持久化跟踪器", help_false="不持久化")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    model_cfg = config_from_args(
        args,
        plain=("model", "imgsz", "device", "batch", "classes",
               "vid_stride", "line_width", "tracker", "embed", "topk", "kpt_thres"),
        boolean=("stream", "half", "augment", "retina_masks", "visualize",
                 "int8", "save_frames", "stream_buffer", "save_conf", "dnn", "end2end", "show",
                 "show_boxes", "persist"),
        rename={"model": "path"},
    )
    if model_cfg:
        config["model"] = model_cfg

    io_cfg = config_from_args(
        args, plain=("input", "output"), boolean=("save_vis", "save_json", "save_txt", "save_crop")
    )
    if io_cfg:
        config["io"] = io_cfg

    nms_cfg = config_from_args(
        args, plain=("conf", "iou", "max_det"), boolean=("agnostic_nms",)
    )
    if nms_cfg:
        config["nms"] = {**config.get("nms", {}), **nms_cfg}

    vis_cfg = config_from_args(
        args, plain=("box_thickness", "font_scale"), boolean=("show_labels", "show_conf")
    )
    if vis_cfg:
        config["visualization"] = {**config.get("visualization", {}), **vis_cfg}

    video_cfg = config_from_args(args, plain=("fps", "codec"))
    if video_cfg:
        config["video"] = video_cfg

    v = to_bool(getattr(args, "verbose", None))
    if v is not None:
        config.setdefault("output", {})["verbose"] = v

    return config


def track(config: Dict) -> None:
    """运行 YOLO 目标跟踪。"""
    model_path = get_nested_value(config, "model", "path")
    imgsz = get_nested_value(config, "model", "imgsz", default=DEFAULT_IMGSZ)
    device = get_nested_value(config, "model", "device", default="auto")
    batch_size = get_nested_value(config, "model", "batch", default=1)
    classes_filter = get_nested_value(config, "model", "classes")
    tracker_cfg = (
        get_nested_value(config, "model", "tracker")
        or config.get("tracker", "botsort.yaml")
    )
    persist = get_nested_value(config, "model", "persist", default=False)

    stream = get_nested_value(config, "model", "stream", default=False)
    half = get_nested_value(config, "model", "half", default=False)
    augment = get_nested_value(config, "model", "augment", default=False)
    vid_stride = get_nested_value(config, "model", "vid_stride", default=1)
    visualize = get_nested_value(config, "model", "visualize", default=False)
    int8 = get_nested_value(config, "model", "int8", default=False)
    line_width = get_nested_value(config, "model", "line_width")
    save_frames = get_nested_value(config, "model", "save_frames", default=False)
    stream_buffer = get_nested_value(config, "model", "stream_buffer", default=False)
    save_conf = get_nested_value(config, "model", "save_conf", default=False)
    dnn = get_nested_value(config, "model", "dnn", default=False)
    end2end = get_nested_value(config, "model", "end2end")
    show = get_nested_value(config, "model", "show", default=False)
    show_boxes = get_nested_value(config, "model", "show_boxes")
    retina_masks = get_nested_value(config, "model", "retina_masks", default=False)
    embed = get_nested_value(config, "model", "embed")

    input_path = get_nested_value(config, "io", "input")
    output_path = get_nested_value(config, "io", "output", default="runs/track")
    save_vis = get_nested_value(config, "io", "save_vis", default=True)
    save_json = get_nested_value(config, "io", "save_json", default=False)
    save_txt = get_nested_value(config, "io", "save_txt", default=False)
    save_crop = get_nested_value(config, "io", "save_crop", default=False)
    verbose = get_nested_value(config, "output", "verbose", default=False)

    if not model_path:
        raise ValueError("--model 或配置 model.path 是必需的")
    if not input_path:
        raise ValueError("--input 或配置 io.input 是必需的")

    nms_config = NMSConfig(
        conf_threshold=get_nested_value(config, "nms", "conf", default=0.1),
        iou_threshold=get_nested_value(config, "nms", "iou", default=0.7),
        max_detections=get_nested_value(config, "nms", "max_det", default=300),
        agnostic=get_nested_value(config, "nms", "agnostic_nms", default=False),
        kpt_thres=resolve_config_value(config, ("model", "kpt_thres"), ("nms", "kpt_thres")),
        topk=resolve_config_value(config, ("model", "topk"), ("nms", "topk")),
    )

    skeleton_cfg = get_nested_value(config, "visualization", "skeleton")
    skeleton = [tuple(pair) for pair in skeleton_cfg] if skeleton_cfg is not None else None
    kpt_names = get_nested_value(config, "visualization", "kpt_names")

    print(f"\n{'='*60}")
    print("YOLO 目标跟踪")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"设备: {device}")
    print(f"图像尺寸: {imgsz}")
    print(f"跟踪器: {tracker_cfg}")
    print(f"NMS: conf={nms_config.conf_threshold}, iou={nms_config.iou_threshold}")
    print(f"{'='*60}\n")

    engine = YOLOInference(
        model_path=model_path,
        nms_config=nms_config,
        device=device,
        imgsz=imgsz,
        classes=classes_filter,
        batch_size=batch_size,
        stream=stream,
        half=half,
        augment=augment,
        vid_stride=vid_stride,
        retina_masks=retina_masks,
        visualize=visualize,
        embed=embed,
        int8=int8,
        line_width=line_width,
        save_frames=save_frames,
        stream_buffer=stream_buffer,
        save_conf=save_conf,
        dnn=dnn,
        end2end=end2end,
        show=show,
        show_boxes=show_boxes,
    )

    if engine.model_format != "pytorch":
        raise ValueError("跟踪模式仅支持 PyTorch (.pt) 模型")

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    vis_cfg = config.get("visualization", {})
    video_cfg = config.get("video", {})

    video_files = []
    if input_path_obj.is_file() and is_video_file(input_path_obj):
        video_files = [input_path_obj]
    elif input_path_obj.is_dir():
        video_files = get_video_files(input_path_obj)

    if not video_files:
        raise ValueError(f"未找到视频文件: {input_path} (跟踪模式仅支持视频输入)")

    for vf in video_files:
        if verbose:
            print(f"\n跟踪视频: {vf}")
        _track_video(
            engine=engine,
            input_path=vf,
            output_path=output_path_obj / "video",
            tracker_cfg=tracker_cfg,
            persist=persist,
            fps=video_cfg.get("fps"),
            codec=video_cfg.get("codec", "mp4v"),
            save_vis=save_vis,
            save_json=save_json,
            verbose=verbose,
            vis_cfg=vis_cfg,
            skeleton=skeleton,
            kpt_names=kpt_names,
        )

    print(f"\n{'='*60}")
    print("跟踪完成")
    print(f"处理视频数: {len(video_files)}")
    print(f"输出保存至: {output_path_obj}")
    print(f"{'='*60}\n")


def _track_video(
    engine: YOLOInference,
    input_path: Path,
    output_path: Path,
    tracker_cfg: str = "botsort.yaml",
    persist: bool = False,
    fps=None,
    codec: str = "mp4v",
    save_vis: bool = True,
    save_json: bool = False,
    verbose: bool = False,
    vis_cfg: dict = None,
    skeleton=None,
    kpt_names=None,
):
    """使用 ultralytics model.track() 进行视频跟踪。"""

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.mkdir(parents=True, exist_ok=True)
    out_fps = fps or orig_fps

    writer = None
    if save_vis:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path / f"{input_path.stem}_tracked.mp4"), fourcc, out_fps, (w, h))

    frame_idx = 0
    total_detections = 0
    all_results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            predict_kwargs = engine.get_predict_kwargs()
            results = engine.model.track(
                source=frame,
                persist=persist,
                tracker=tracker_cfg,
                stream=False,
                **predict_kwargs,
            )

            if not results or len(results) == 0:
                if writer:
                    writer.write(frame)
                continue

            r = results[0]
            image_result = parse_pytorch_result(r, engine.classes, frame.shape[:2])

            det_count = len(image_result.detections)
            total_detections += det_count

            if verbose and frame_idx % 100 == 0:
                print(f"  帧 {frame_idx}/{total_frames}: {det_count} 检测")

            if save_vis:
                vis = draw_detections(
                    frame, image_result, engine.classes,
                    box_thickness=vis_cfg.get("box_thickness", 2) if vis_cfg else 2,
                    font_scale=vis_cfg.get("font_scale", 0.5) if vis_cfg else 0.5,
                    show_labels=vis_cfg.get("show_labels", True) if vis_cfg else True,
                    show_conf=vis_cfg.get("show_conf", True) if vis_cfg else True,
                    skeleton=skeleton, kpt_names=kpt_names,
                )
                if writer:
                    writer.write(vis)

            all_results.append(image_result)

    finally:
        cap.release()
        if writer:
            writer.release()

    if verbose:
        print(f"  完成: {frame_idx} 帧, {total_detections} 总检测")

    return all_results


def main():
    args = parse_args()
    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)
        track(config)
    except KeyboardInterrupt:
        print("\n跟踪被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}\n错误: {e}\n{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
