"""
YOLO Prediction / Inference Module
=====================================
CLI frontend for running YOLO inference.

All business logic lives in ``core/``; this module only handles
argument parsing and delegates to ``core.engine``.

Usage::

    python yolo.py predict --config configs/predict/chaoyuan.yaml
    python yolo.py predict --model best.pt --input images/ --output results/
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from core import (
    YOLOInference,
    draw_detections,
    get_image_files,
    get_video_files,
    inference_video,
    is_video_file,
)
from core.types import NMSConfig
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
from utils.constants import (
    DEFAULT_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_KPT_LINE,
    DEFAULT_KPT_RADIUS,
    DEFAULT_MASK_ALPHA,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_PREDICT_OUTPUT,
)

setup_ultralytics_path()


def _save_result(
    result,
    image_file: Path,
    image: np.ndarray,
    *,
    output_path: Path,
    input_path: Path,
    classes: Dict[int, str],
    vis_cfg: Dict[str, Any],
    save_vis: bool,
    save_crop: bool,
    save_txt: bool,
    skeleton,
    kpt_names,
) -> str:
    """保存单张图像的推理结果（可视化、裁剪、标签文件）。

    Returns:
        检测到的任务类型字符串。
    """
    rel_path = Path(image_file.name) if input_path.is_file() else image_file.relative_to(input_path)

    if save_vis:
        vis_output = draw_detections(
            image, result, classes,
            box_thickness=vis_cfg.get("box_thickness", 2),
            font_scale=vis_cfg.get("font_scale", 0.5),
            show_labels=vis_cfg.get("show_labels", True),
            show_conf=vis_cfg.get("show_conf", True),
            mask_alpha=vis_cfg.get("mask_alpha", DEFAULT_MASK_ALPHA),
            kpt_radius=vis_cfg.get("kpt_radius", DEFAULT_KPT_RADIUS),
            kpt_line=vis_cfg.get("kpt_line", DEFAULT_KPT_LINE),
            line_width=vis_cfg.get("line_width"),
            skeleton=skeleton, kpt_names=kpt_names,
        )
        vis_path = output_path / "vis" / rel_path
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_path), vis_output)

    if save_crop and result.detections:
        crop_dir = output_path / "crops" / rel_path.stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        for j, det in enumerate(result.detections):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            crop_path = crop_dir / f"{det.class_name}_{j}.jpg"
            cv2.imwrite(str(crop_path), crop)

    if save_txt:
        txt_dir = output_path / "labels"
        txt_path = txt_dir / str(rel_path.with_suffix(".txt"))
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        img_h, img_w = result.image_shape
        with open(txt_path, "w") as f:
            if result.task_type == "classify" and result.probs:
                for class_name, prob in result.probs:
                    f.write(f"{class_name} {prob:.4f}\n")
            elif result.task_type == "obb" and result.obb_boxes:
                for obb in result.obb_boxes:
                    points = obb["points"]
                    line = f"{obb['class_id']}"
                    for pt in points:
                        line += f" {pt[0]/img_w:.6f} {pt[1]/img_h:.6f}"
                    line += f" {obb['confidence']:.4f}\n"
                    f.write(line)
            else:
                for det in result.detections:
                    x1, y1, x2, y2 = det.bbox
                    if det.mask is not None:
                        contours, _ = cv2.findContours(
                            det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            contour = max(contours, key=cv2.contourArea)
                            pts = contour.squeeze()
                            if len(pts.shape) == 1:
                                pts = pts.reshape(1, 2)
                            pts_str = " ".join(f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in pts)
                            f.write(f"{det.class_id} {pts_str} {det.confidence:.4f}\n")
                        continue
                    if det.keypoints is not None:
                        kpts = np.array(det.keypoints, dtype=float)
                        if kpts.ndim == 3:
                            kpts = kpts.squeeze(0)
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
                    cx = (x1 + x2) / 2 / img_w
                    cy = (y1 + y2) / 2 / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    f.write(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {det.confidence:.4f}\n")

    return result.task_type


# ─── Argument parser ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m commands.predict --model best.pt --input images/ --output results/
    python -m commands.predict --model model.onnx --input images/
    python yolo.py predict --config configs/predict/chaoyuan.yaml
        """,
    )

    parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--model", "-m", type=str, default=None, help="模型路径 (.pt 或 .onnx)")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default=None, help="设备: auto, cpu, cuda, 0")
    parser.add_argument("--batch", type=int, default=None, help="推理批大小")

    set_boolean_argument(parser, "stream", "stream", help_true="流式推理", help_false="非流式")
    set_boolean_argument(parser, "half", "half", help_true="FP16 半精度", help_false="全精度")
    set_boolean_argument(parser, "augment", "augment", help_true="TTA 增强", help_false="无 TTA")
    parser.add_argument("--vid-stride", type=int, default=None, help="视频帧步长")
    set_boolean_argument(parser, "retina_masks", "retina-masks", help_true="高分辨率掩码", help_false="标准掩码")
    set_boolean_argument(parser, "visualize", "visualize", help_true="可视化特征", help_false="不可视化")
    parser.add_argument("--embed", type=int, nargs="+", default=None, help="特征嵌入层索引")
    set_boolean_argument(parser, "int8", "int8", help_true="INT8 量化", help_false="无 INT8")
    set_boolean_argument(parser, "dnn", "dnn", help_true="OpenCV DNN ONNX 推理", help_false="不使用 DNN")
    set_boolean_argument(parser, "end2end", "end2end", help_true="端到端检测头 (YOLO26/YOLOv10)", help_false="标准检测头")
    parser.add_argument("--kpt-thres", type=float, default=None, help="关键点阈值 (仅姿态估计)")
    parser.add_argument("--topk", type=int, default=None, help="分类 Top-K (仅分类任务)")
    set_boolean_argument(parser, "save_conf", "save-conf", help_true="保存置信度到结果", help_false="不保存置信度")
    set_boolean_argument(parser, "stream_buffer", "stream-buffer", help_true="流式缓冲所有帧", help_false="只保留最新帧")

    parser.add_argument("--input", "-i", type=str, default=None, help="输入图像或目录")
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

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config: Dict[str, Any] = {}

    # Model
    model_cfg = config_from_args(
        args,
        plain=("model", "imgsz", "device", "batch", "classes",
               "vid_stride", "embed", "line_width", "topk", "kpt_thres"),
        boolean=("stream", "half", "augment", "retina_masks", "visualize",
                 "int8", "save_frames", "stream_buffer", "save_conf", "dnn", "end2end", "show",
                 "show_boxes"),
        rename={"model": "path"},
    )
    if model_cfg:
        config["model"] = model_cfg

    # IO
    io_cfg = config_from_args(
        args, plain=("input", "output"), boolean=("save_vis", "save_json", "save_txt", "save_crop")
    )
    if io_cfg:
        config["io"] = io_cfg

    # NMS
    nms_cfg = config_from_args(
        args, plain=("conf", "iou", "max_det"), boolean=("agnostic_nms",)
    )
    if nms_cfg:
        config["nms"] = {**config.get("nms", {}), **nms_cfg}

    # Visualization
    vis_cfg = config_from_args(
        args, plain=("box_thickness", "font_scale"), boolean=("show_labels", "show_conf")
    )
    if vis_cfg:
        config["visualization"] = {**config.get("visualization", {}), **vis_cfg}

    # Video
    video_cfg = config_from_args(args, plain=("fps", "codec"))
    if video_cfg:
        config["video"] = video_cfg

    # Verbose: 同时支持 output.verbose 和根级 verbose（向后兼容）
    v = to_bool(getattr(args, "verbose", None))
    if v is not None:
        config.setdefault("output", {})["verbose"] = v

    return config


# ─── Main prediction orchestration ──────────────────────────────────────────


def predict(config: Dict) -> None:
    """运行 YOLO 推理。"""
    model_path = get_nested_value(config, "model", "path")
    imgsz = get_nested_value(config, "model", "imgsz", default=DEFAULT_IMGSZ)
    device = get_nested_value(config, "model", "device", default="auto")
    batch_size = get_nested_value(config, "model", "batch", default=1)
    classes_filter = get_nested_value(config, "model", "classes")

    stream = get_nested_value(config, "model", "stream", default=False)
    half = get_nested_value(config, "model", "half", default=False)
    augment = get_nested_value(config, "model", "augment", default=False)
    vid_stride = get_nested_value(config, "model", "vid_stride", default=1)
    retina_masks = get_nested_value(config, "model", "retina_masks", default=False)
    visualize = get_nested_value(config, "model", "visualize", default=False)
    embed = get_nested_value(config, "model", "embed")
    int8 = get_nested_value(config, "model", "int8", default=False)
    line_width = get_nested_value(config, "model", "line_width")
    save_frames = get_nested_value(config, "model", "save_frames", default=False)
    stream_buffer = get_nested_value(config, "model", "stream_buffer", default=False)
    save_conf = get_nested_value(config, "model", "save_conf", default=False)
    dnn = get_nested_value(config, "model", "dnn", default=False)
    end2end = resolve_config_value(config, ("model", "end2end"), ("nms", "end2end"), default=None)
    show = get_nested_value(config, "model", "show", default=False)
    show_boxes = get_nested_value(config, "model", "show_boxes")

    input_path = get_nested_value(config, "io", "input")
    output_path = get_nested_value(config, "io", "output", default=DEFAULT_PREDICT_OUTPUT)
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
        conf_threshold=get_nested_value(config, "nms", "conf", default=0.25),
        iou_threshold=get_nested_value(config, "nms", "iou", default=DEFAULT_IOU_THRESHOLD),
        max_detections=get_nested_value(config, "nms", "max_det", default=DEFAULT_MAX_DETECTIONS),
        agnostic=get_nested_value(config, "nms", "agnostic_nms", default=False),
        kpt_thres=resolve_config_value(config, ("model", "kpt_thres"), ("nms", "kpt_thres")),
        topk=resolve_config_value(config, ("model", "topk"), ("nms", "topk")),
    )

    skeleton_cfg = get_nested_value(config, "visualization", "skeleton")
    skeleton = [tuple(pair) for pair in skeleton_cfg] if skeleton_cfg is not None else None
    kpt_names = get_nested_value(config, "visualization", "kpt_names")

    print(f"\n{'='*60}")
    print("YOLO 推理")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"设备: {device}")
    print(f"图像尺寸: {imgsz}")
    print(f"批大小: {batch_size}")
    print(f"NMS: conf={nms_config.conf_threshold}, iou={nms_config.iou_threshold}, max_det={nms_config.max_detections}")
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

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    # 视频推理 — 单个视频文件
    if input_path_obj.is_file() and is_video_file(input_path_obj):
        video_cfg = config.get("video", {})
        vis_cfg = config.get("visualization", {})
        inference_video(
            engine=engine,
            input_path=input_path_obj,
            output_path=output_path_obj / "video",
            fps=video_cfg.get("fps"),
            codec=video_cfg.get("codec", "mp4v"),
            save_vis=save_vis,
            save_json=save_json,
            verbose=verbose,
            vis_cfg=vis_cfg,
            skeleton=skeleton,
            kpt_names=kpt_names,
            vid_stride=vid_stride,
        )
        return

    # 目录输入: 扫描视频和图像
    video_files = []
    image_files = []
    if input_path_obj.is_dir():
        video_files = get_video_files(input_path_obj)
        image_files = get_image_files(input_path_obj)
    elif input_path_obj.is_file():
        image_files = [input_path_obj]

    # 处理视频
    if video_files:
        vis_cfg = config.get("visualization", {})
        video_cfg = config.get("video", {})
        for vf in video_files:
            if verbose:
                print(f"\n处理视频: {vf}")
            inference_video(
                engine=engine,
                input_path=vf,
                output_path=output_path_obj / "video",
                fps=video_cfg.get("fps"),
                codec=video_cfg.get("codec", "mp4v"),
                save_vis=save_vis,
                save_json=save_json,
                verbose=verbose,
                vis_cfg=vis_cfg,
                skeleton=skeleton,
                kpt_names=kpt_names,
                vid_stride=vid_stride,
            )

    # 处理图像
    if image_files:
        print(f"找到 {len(image_files)} 张图像待处理")
    elif not video_files:
        print("未找到图像或视频！")
        return

    if not image_files:
        return

    all_results = []
    total_detections = 0
    total_time = 0.0
    detected_task_type = None
    vis_cfg = config.get("visualization", {})

    num_batches = (len(image_files) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

        batch_images = []
        valid_files = []
        for image_file in batch_files:
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"警告: 无法加载 {image_file}")
                continue
            batch_images.append(image)
            valid_files.append(image_file)

        if not batch_images:
            continue

        batch_results = engine.inference_batch(batch_images)

        for local_idx, (result, image_file, image) in enumerate(zip(batch_results, valid_files, batch_images)):
            result.image_path = str(image_file)
            all_results.append(result)
            total_detections += len(result.detections)
            total_time += result.inference_time

            if verbose:
                task_info = f"[{result.task_type}]" if result.task_type != "detect" else ""
                print(f"[{start_idx + local_idx + 1}/{len(image_files)}] {image_file.name}: {len(result.detections)} 检测 {task_info}, {result.inference_time*1000:.2f}ms")

            _save_result(
                result, image_file, image,
                output_path=output_path_obj,
                input_path=input_path_obj,
                classes=engine.classes,
                vis_cfg=vis_cfg,
                save_vis=save_vis,
                save_crop=save_crop,
                save_txt=save_txt,
                skeleton=skeleton,
                kpt_names=kpt_names,
            )
            
            # Track task type from first result
            if detected_task_type is None:
                detected_task_type = result.task_type

    if save_json:
        json_path = output_path_obj / "results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = {
            "model": model_path,
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
        print(f"结果保存至: {json_path}")

    print(f"\n{'='*60}")
    print("推理摘要")
    print(f"{'='*60}")
    print(f"总图像数:    {len(image_files)}")
    print(f"任务类型:    {detected_task_type or 'detect'}")
    print(f"总检测数:    {total_detections}")
    print(f"总耗时:      {total_time:.2f}s")
    if image_files:
        print(f"平均耗时:    {total_time/len(image_files)*1000:.2f}ms/张")
    print(f"输出保存至: {output_path_obj}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)
        cli_config = args_to_config(args)
        # CLI overrides YAML for explicitly-specified args
        config = merge_configs(config, cli_config)  # YAML in base, CLI in override = CLI wins
        predict(config)
    except KeyboardInterrupt:
        print("\n推理被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}\n错误: {e}\n{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
