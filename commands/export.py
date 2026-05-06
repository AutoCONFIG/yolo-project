"""
YOLO Export Module
===================
Standalone model export module invoked via ``python yolo.py export`` or
``python -m commands.export``.

Export trained YOLO models to various deployment formats:
ONNX, TensorRT, TorchScript, OpenVINO, etc.

Uses YAML config files and/or CLI arguments.  CLI takes precedence over YAML.

Typical usage::

    # Via unified entry point
    python yolo.py export --model best.pt
    python yolo.py export --model best.pt --format engine --half true

    # Direct invocation
    python -m commands.export --model best.pt --format onnx

    # Using config file
    python -m commands.export --config configs/export/onnx.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from utils.config import (
    config_from_args,
    get_nested_value,
    load_yaml_config,
    merge_configs,
    set_boolean_argument,
    setup_ultralytics_path,
)
from utils.constants import DEFAULT_IMGSZ, EXPORT_FORMATS

setup_ultralytics_path()
from ultralytics import YOLO


# ─── Argument parser ─────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 模型导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 导出为 ONNX (默认)
    python -m commands.export --model best.pt

    # 导出为 ONNX 并启用 FP16 和动态输入
    python -m commands.export --model best.pt --half true --dynamic true

    # 导出为 TensorRT (FP16)
    python -m commands.export --model best.pt --format engine --half true

    # 导出为 OpenVINO (INT8 量化)
    python -m commands.export --model best.pt --format openvino --int8 true --data coco8.yaml

    # 使用配置文件
    python -m commands.export --config configs/export/onnx.yaml

    # 导出并验证
    python -m commands.export --model best.pt --verify true --source test_image.jpg
        """,
    )

    # ── Config ────────────────────────────────────────────────────────
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--list-formats", action="store_true", help="列出所有支持的导出格式")

    # ── Model ─────────────────────────────────────────────────────────
    parser.add_argument("--model", "-m", type=str, default=None, help="模型路径 (如 best.pt)")
    parser.add_argument(
        "--format", "-f", type=str, default=None,
        choices=list(EXPORT_FORMATS.keys()),
        help="导出格式 (默认: onnx)",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸 (默认 640)")
    parser.add_argument("--batch", type=int, default=None, help="导出批大小 (默认 1)")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu, 0, 0,1")

    # ── ONNX options ──────────────────────────────────────────────────
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset 版本 (自动检测)")
    set_boolean_argument(parser, "simplify", "simplify", help_true="简化 ONNX 图 (默认开启)", help_false="不简化")
    set_boolean_argument(parser, "dynamic", "dynamic", help_true="动态输入形状", help_false="固定输入形状")
    set_boolean_argument(parser, "half", "half", help_true="FP16 半精度导出", help_false="全精度导出")
    set_boolean_argument(parser, "nms", "nms", help_true="在导出模型中嵌入 NMS", help_false="不嵌入 NMS")

    # ── NMS options (when nms=True) ──────────────────────────────────────
    parser.add_argument("--conf", type=float, default=None, help="NMS 置信度阈值 (nms=True 时生效, 默认 0.25)")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值 (nms=True 时生效, 默认 0.7)")
    parser.add_argument("--max-det", type=int, default=None, help="最大检测数 (nms=True 时生效, 默认 300)")
    set_boolean_argument(parser, "agnostic_nms", "agnostic-nms", help_true="类别无关 NMS", help_false="类别特定 NMS")
    set_boolean_argument(parser, "end2end", "end2end", help_true="端到端检测头 (YOLO26/YOLOv10)", help_false="标准检测头")

    # ── TorchScript options ────────────────────────────────────────────
    set_boolean_argument(parser, "optimize", "optimize", help_true="TorchScript 移动端优化", help_false="不优化")

    # ── Quantization ──────────────────────────────────────────────────
    set_boolean_argument(parser, "int8", "int8", help_true="INT8 量化 (TensorRT/OpenVINO)", help_false="不使用 INT8")
    parser.add_argument("--data", type=str, default=None, help="INT8 校准数据集配置")
    parser.add_argument("--fraction", type=float, default=None, help="INT8 校准数据集比例 (默认 1.0)")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT 工作区大小 (GB, 默认 4)")

    # ── TensorFlow options ─────────────────────────────────────────────
    set_boolean_argument(parser, "keras", "keras", help_true="导出 TF SavedModel 时使用 Keras", help_false="不使用 Keras")

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument("--output", "-o", type=str, default=None, help="自定义输出路径")
    set_boolean_argument(parser, "verbose", "verbose", help_true="详细输出", help_false="简洁输出")

    # ── Verification ──────────────────────────────────────────────────
    set_boolean_argument(parser, "verify", "verify", help_true="导出后验证模型", help_false="不验证")
    parser.add_argument("--source", type=str, default=None, help="验证用测试图像路径 (不设置则使用虚拟图像)")

    return parser.parse_args()


# ─── CLI → nested config ──────────────────────────────────────────────


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config = {}

    # ── Model ──
    model_cfg = config_from_args(
        args,
        plain=("model", "format", "imgsz", "batch", "device"),
        rename={"model": "path"},
    )
    if model_cfg:
        config["model"] = model_cfg

    # ── Export options ──
    export_plain = (
        "opset", "data", "fraction", "workspace",
        "conf", "iou", "max_det",
    )
    export_bool = (
        "simplify", "dynamic", "half", "nms",
        "optimize", "int8", "keras", "agnostic_nms", "end2end",
    )
    export_cfg = config_from_args(args, plain=export_plain, boolean=export_bool)
    if export_cfg:
        config["export"] = {**config.get("export", {}), **export_cfg}

    # ── Output ──
    out_cfg = config_from_args(
        args,
        plain=("output",),
        boolean=("verbose",),
        rename={"output": "path"},
    )
    if out_cfg:
        config["output"] = {**config.get("output", {}), **out_cfg}

    # ── Verification ──
    verify_cfg = config_from_args(
        args,
        plain=("source",),
        boolean=("verify",),
        rename={"verify": "enabled"},
    )
    if verify_cfg:
        config["verify"] = {**config.get("verify", {}), **verify_cfg}

    return config


# ─── Verify exported model ────────────────────────────────────────────


def verify_export(exported_path: str, imgsz: int = 640, source: str | None = None):
    """通过运行快速推理测试来验证导出的模型。"""
    import cv2
    import numpy as np

    print(f"\n{'='*60}")
    print("验证导出模型")
    print(f"{'='*60}")

    ext = Path(exported_path).suffix.lower()
    exported_path = str(exported_path)

    loadable_formats = {".pt", ".onnx", ".torchscript", ".engine", ".tflite"}
    is_dir = Path(exported_path).is_dir()

    if ext in loadable_formats or is_dir:
        try:
            model = YOLO(exported_path)

            if source and Path(source).exists():
                test_img = source
            else:
                dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                dummy_path = Path(exported_path).parent / "_verify_test.jpg"
                cv2.imwrite(str(dummy_path), dummy)
                test_img = str(dummy_path)

            results = model.predict(test_img, imgsz=imgsz, verbose=False)

            if results and len(results) > 0:
                n_det = len(results[0].boxes) if hasattr(results[0], "boxes") and results[0].boxes is not None else 0
                print(f"验证通过 - 测试推理检测到 {n_det} 个目标")

                if results[0].speed:
                    speed = results[0].speed
                    print(f"推理速度: 预处理={speed.get('preprocess', 0):.1f}ms, "
                          f"推理={speed.get('inference', 0):.1f}ms, "
                          f"后处理={speed.get('postprocess', 0):.1f}ms")
            else:
                print("验证通过 - 模型运行成功 (无返回结果)")

            # 清理虚拟图像
            if source is None:
                dummy_path = Path(exported_path).parent / "_verify_test.jpg"
                if dummy_path.exists():
                    dummy_path.unlink()

        except Exception as e:
            print(f"验证警告 - 无法验证导出模型: {e}")
    else:
        print(f"验证跳过 - 格式 '{ext}' 需要外部运行时进行验证")

    print(f"{'='*60}\n")


# ─── Export ────────────────────────────────────────────────────────────


def export(config: Dict):
    """将 YOLO 模型导出为指定格式。"""
    model_path = get_nested_value(config, "model", "path")
    fmt = get_nested_value(config, "model", "format", default="onnx")
    imgsz = get_nested_value(config, "model", "imgsz", default=DEFAULT_IMGSZ)
    batch = get_nested_value(config, "model", "batch", default=1)
    device = get_nested_value(config, "model", "device")

    # 导出选项
    opset = get_nested_value(config, "export", "opset")
    simplify = get_nested_value(config, "export", "simplify", default=True)
    dynamic = get_nested_value(config, "export", "dynamic", default=False)
    half = get_nested_value(config, "export", "half", default=False)
    nms = get_nested_value(config, "export", "nms", default=False)
    optimize = get_nested_value(config, "export", "optimize", default=False)
    int8 = get_nested_value(config, "export", "int8", default=False)
    data = get_nested_value(config, "export", "data")
    fraction = get_nested_value(config, "export", "fraction")
    workspace = get_nested_value(config, "export", "workspace")
    keras = get_nested_value(config, "export", "keras")

    # NMS 选项 (nms=True 时生效)
    conf = get_nested_value(config, "export", "conf")
    iou_export = get_nested_value(config, "export", "iou")
    max_det = get_nested_value(config, "export", "max_det")
    agnostic_nms = get_nested_value(config, "export", "agnostic_nms")
    end2end = get_nested_value(config, "export", "end2end")

    # 输出选项
    output_path = get_nested_value(config, "output", "path")
    verbose = get_nested_value(config, "output", "verbose", default=True)

    # 验证选项
    verify = get_nested_value(config, "verify", "enabled", default=False)
    source = get_nested_value(config, "verify", "source")

    # 验证
    if not model_path:
        raise ValueError("--model 或配置 model.path 是必需的")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    if fmt not in EXPORT_FORMATS:
        supported = ", ".join(sorted(EXPORT_FORMATS.keys()))
        raise ValueError(f"不支持的格式: '{fmt}'. 支持的格式: {supported}")

    # 打印配置
    format_info = EXPORT_FORMATS[fmt]
    print(f"\n{'='*60}")
    print("YOLO 模型导出配置")
    print(f"{'='*60}")
    print(f"模型:        {model_path}")
    print(f"格式:        {fmt} ({format_info['desc']})")
    print(f"图像尺寸:    {imgsz}")
    print(f"批大小:      {batch}")
    print(f"设备:        {device or 'auto'}")
    print(f"FP16 (半精度): {half}")
    print(f"动态输入:    {dynamic}")
    print(f"简化:        {simplify}")
    print(f"NMS:         {nms}")
    print(f"优化:        {optimize}")
    print(f"INT8:        {int8}")
    if opset:
        print(f"Opset:       {opset}")
    if int8 and data:
        print(f"校准数据:    {data}")
    if int8 and fraction != 1.0:
        print(f"数据比例:    {fraction}")
    if fmt == "engine" and workspace:
        print(f"工作区:      {workspace} GB")
    if fmt == "saved_model" and keras:
        print(f"Keras:       {keras}")
    if output_path:
        print(f"输出:        {output_path}")
    print(f"{'='*60}\n")

    # 加载模型
    model = YOLO(str(model_path))

    # 构建导出参数
    export_args = {
        "format": fmt,
        "imgsz": imgsz,
        "batch": batch,
        "simplify": simplify,
        "dynamic": dynamic,
        "half": half,
        "nms": nms,
        "int8": int8,
        "optimize": optimize,
        "verbose": verbose,
    }

    # 可选参数
    if device is not None:
        export_args["device"] = device
    if opset is not None:
        export_args["opset"] = opset
    if data is not None:
        export_args["data"] = data
    if fraction is not None:
        export_args["fraction"] = fraction
    if workspace is not None and fmt == "engine":
        export_args["workspace"] = workspace
    if keras is not None and fmt == "saved_model":
        export_args["keras"] = keras
    # NMS params
    if conf is not None:
        export_args["conf"] = conf
    if iou_export is not None:
        export_args["iou"] = iou_export
    if max_det is not None:
        export_args["max_det"] = max_det
    if agnostic_nms is not None:
        export_args["agnostic_nms"] = agnostic_nms
    if end2end is not None:
        export_args["end2end"] = end2end

    # 运行导出
    start_time = time.perf_counter()
    exported_path = model.export(**export_args)
    elapsed = time.perf_counter() - start_time

    # 确定实际输出路径
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        src = Path(exported_path)
        if src.is_dir():
            if output_path.exists():
                if output_path.is_file():
                    output_path.unlink()
                else:
                    shutil.rmtree(output_path)
            shutil.copytree(src, output_path)
            final_path = str(output_path)
        elif output_path.is_dir():
            dst = output_path / src.name
            shutil.copy2(src, dst)
            final_path = str(dst)
        else:
            shutil.copy2(src, output_path)
            final_path = str(output_path)
    else:
        final_path = str(exported_path)

    # 显示结果
    print(f"\n{'='*60}")
    print("导出完成！")
    print(f"{'='*60}")
    print(f"格式:       {format_info['desc']}")
    print(f"输出:       {final_path}")

    # 文件大小
    final = Path(final_path)
    if final.is_file():
        size_bytes = final.stat().st_size
        if size_bytes > 1024 * 1024:
            print(f"文件大小:   {size_bytes / (1024 * 1024):.1f} MB")
        elif size_bytes > 1024:
            print(f"文件大小:   {size_bytes / 1024:.1f} KB")
        else:
            print(f"文件大小:   {size_bytes} B")
    elif final.is_dir():
        total_size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file())
        if total_size > 1024 * 1024:
            print(f"目录大小:   {total_size / (1024 * 1024):.1f} MB")
        else:
            print(f"目录大小:   {total_size / 1024:.1f} KB")

    print(f"导出耗时:    {elapsed:.2f}s")
    print(f"{'='*60}\n")

    # 验证
    if verify:
        verify_export(final_path, imgsz, source)

    return final_path


# ─── Main ─────────────────────────────────────────────────────────────


def print_formats():
    """打印所有支持的导出格式。"""
    print("\n支持的导出格式:")
    print(f"{'格式':<15} {'后缀':<25} {'描述'}")
    print("-" * 60)
    for key, info in EXPORT_FORMATS.items():
        print(f"{key:<15} {info['suffix']:<25} {info['desc']}")
    print()


def main():
    args = parse_args()

    try:
        if args.list_formats:
            print_formats()
            return

        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)  # YAML in base, CLI in override = CLI wins

        export(config)
    except KeyboardInterrupt:
        print("\n导出被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"错误: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()