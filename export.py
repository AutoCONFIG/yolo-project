"""
YOLO Model Export Script
========================
Export trained YOLO models to various deployment formats (ONNX, TensorRT, TorchScript, OpenVINO, etc.).

Supports configuration via:
1. YAML config files (recommended)
2. Command line arguments (override config file)

Usage:
    # Export to ONNX (default)
    python export.py --model runs/detect/train/weights/best.pt

    # Export to ONNX with FP16 and dynamic shapes
    python export.py --model best.pt --format onnx --half true --dynamic true

    # Export to TensorRT
    python export.py --model best.pt --format engine --half true

    # Export to OpenVINO with INT8 quantization
    python export.py --model best.pt --format openvino --int8 true --data coco8.yaml

    # Using config file
    python export.py --config configs/export/onnx.yaml

    # Export and verify with a test image
    python export.py --model best.pt --verify true --source test_image.jpg
"""

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

# Add ultralytics submodule to path
ULTRALYTICS_PATH = Path(__file__).parent / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO  # noqa: E402

from utils.config import get_nested_value, load_yaml_config, merge_configs, set_boolean_argument, to_bool


# ─── Supported export formats ─────────────────────────────────────────

EXPORT_FORMATS = {
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


# ─── Argument parser ─────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Model Export Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export to ONNX (default)
    python export.py --model best.pt

    # Export to ONNX with FP16 and dynamic shapes
    python export.py --model best.pt --half true --dynamic true

    # Export to TensorRT with FP16
    python export.py --model best.pt --format engine --half true

    # Export to OpenVINO with INT8 quantization
    python export.py --model best.pt --format openvino --int8 true --data coco8.yaml

    # Using config file
    python export.py --config configs/export/onnx.yaml

    # Export and verify
    python export.py --model best.pt --verify true --source test_image.jpg
        """,
    )

    # Config file
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List all supported export formats",
    )

    # ── Model ──────────────────────────────────────────────────────────
    parser.add_argument("--model", "-m", type=str, default=None, help="Model path (e.g. best.pt)")
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default=None,
        choices=list(EXPORT_FORMATS.keys()),
        help="Export format (default: onnx)",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size (default 640)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size for export (default 1)")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, 0, 0,1")

    # ── ONNX options ───────────────────────────────────────────────────
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version (auto-detect if not set)")
    parser.add_argument(
        "--simplify",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Simplify ONNX graph (default true)",
    )
    parser.add_argument(
        "--dynamic",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Dynamic input shapes (default false)",
    )
    parser.add_argument(
        "--half",
        type=str,
        default=None,
        choices=["true", "false"],
        help="FP16 half precision export",
    )
    parser.add_argument(
        "--nms",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Embed NMS in exported model",
    )

    # ── TorchScript options ──────────────────────────────────────────────
    parser.add_argument(
        "--optimize",
        type=str,
        default=None,
        choices=["true", "false"],
        help="TorchScript mobile optimization (default false)",
    )

    # ── Quantization ───────────────────────────────────────────────────
    parser.add_argument(
        "--int8",
        type=str,
        default=None,
        choices=["true", "false"],
        help="INT8 quantization (TensorRT/OpenVINO)",
    )
    parser.add_argument("--data", type=str, default=None, help="Dataset config for INT8 calibration")
    parser.add_argument(
        "--workspace",
        type=int,
        default=None,
        help="TensorRT workspace size in GB (default 4)",
    )

    # ── Output ─────────────────────────────────────────────────────────
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output path")
    set_boolean_argument(
        parser, "verbose", "verbose", help_true="Verbose output", help_false="Quiet output"
    )

    # ── Verification ───────────────────────────────────────────────────
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Verify exported model with a test inference",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Test image path for verification (uses a dummy image if not set)",
    )

    return parser.parse_args()


# ─── CLI → nested config ──────────────────────────────────────────────


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command line args to nested config dict."""
    config = {}

    # ── Model ──
    model_config = {}
    if args.model:
        model_config["path"] = args.model
    if args.format:
        model_config["format"] = args.format
    if args.imgsz is not None:
        model_config["imgsz"] = args.imgsz
    if args.batch is not None:
        model_config["batch"] = args.batch
    if args.device is not None:
        model_config["device"] = args.device
    if model_config:
        config["model"] = model_config

    # ── Export options ──
    export_config = {}
    if args.opset is not None:
        export_config["opset"] = args.opset
    v = to_bool(args.simplify)
    if v is not None:
        export_config["simplify"] = v
    v = to_bool(args.dynamic)
    if v is not None:
        export_config["dynamic"] = v
    v = to_bool(args.half)
    if v is not None:
        export_config["half"] = v
    v = to_bool(args.nms)
    if v is not None:
        export_config["nms"] = v
    v = to_bool(args.optimize)
    if v is not None:
        export_config["optimize"] = v
    v = to_bool(args.int8)
    if v is not None:
        export_config["int8"] = v
    if args.data:
        export_config["data"] = args.data
    if args.workspace is not None:
        export_config["workspace"] = args.workspace
    if export_config:
        config["export"] = {**config.get("export", {}), **export_config}

    # ── Output ──
    output_config = {}
    if args.output:
        output_config["path"] = args.output
    v = args.verbose
    if v is not None:
        output_config["verbose"] = v
    if output_config:
        config["output"] = {**config.get("output", {}), **output_config}

    # ── Verification ──
    verify_config = {}
    v = to_bool(args.verify)
    if v is not None:
        verify_config["enabled"] = v
    if args.source:
        verify_config["source"] = args.source
    if verify_config:
        config["verify"] = {**config.get("verify", {}), **verify_config}

    return config


# ─── Verify exported model ────────────────────────────────────────────


def verify_export(exported_path: str, model_path: str, imgsz: int = 640, source: str | None = None):
    """Verify the exported model by running a quick inference test.

    Args:
        exported_path: Path to the exported model file.
        model_path: Path to the original PyTorch model.
        imgsz: Image size for inference.
        source: Optional test image path. Uses a dummy image if None.
    """
    import cv2
    import numpy as np

    print(f"\n{'='*60}")
    print("Verifying Exported Model")
    print(f"{'='*60}")

    ext = Path(exported_path).suffix.lower()
    exported_path = str(exported_path)

    # Determine if the exported model can be loaded by YOLO directly
    loadable_formats = {".pt", ".onnx", ".torchscript", ".engine", ".tflite"}
    is_dir = Path(exported_path).is_dir()

    if ext in loadable_formats or is_dir:
        try:
            model = YOLO(exported_path)

            # Prepare test image
            if source and Path(source).exists():
                test_img = source
            else:
                # Create a dummy test image
                dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                dummy_path = Path(exported_path).parent / "_verify_test.jpg"
                cv2.imwrite(str(dummy_path), dummy)
                test_img = str(dummy_path)

            results = model.predict(test_img, imgsz=imgsz, verbose=False)

            if results and len(results) > 0:
                n_det = len(results[0].boxes) if hasattr(results[0], "boxes") and results[0].boxes is not None else 0
                print(f"Verification PASSED - {n_det} detections from test inference")

                if results[0].speed:
                    speed = results[0].speed
                    print(f"Inference speed: preprocess={speed.get('preprocess', 0):.1f}ms, "
                          f"inference={speed.get('inference', 0):.1f}ms, "
                          f"postprocess={speed.get('postprocess', 0):.1f}ms")
            else:
                print("Verification PASSED - model runs successfully (no results returned)")

            # Clean up dummy image
            if source is None:
                dummy_path = Path(exported_path).parent / "_verify_test.jpg"
                if dummy_path.exists():
                    dummy_path.unlink()

        except Exception as e:
            print(f"Verification WARNING - Could not verify exported model: {e}")
    else:
        print(f"Verification SKIPPED - Format '{ext}' requires external runtime for verification")

    print(f"{'='*60}\n")


# ─── Export ────────────────────────────────────────────────────────────


def export(config: Dict):
    """Export a YOLO model to the specified format."""
    model_path = get_nested_value(config, "model", "path")
    fmt = get_nested_value(config, "model", "format", default="onnx")
    imgsz = get_nested_value(config, "model", "imgsz", default=640)
    batch = get_nested_value(config, "model", "batch", default=1)
    device = get_nested_value(config, "model", "device")

    # Export options
    opset = get_nested_value(config, "export", "opset")
    simplify = get_nested_value(config, "export", "simplify", default=True)
    dynamic = get_nested_value(config, "export", "dynamic", default=False)
    half = get_nested_value(config, "export", "half", default=False)
    nms = get_nested_value(config, "export", "nms", default=False)
    optimize = get_nested_value(config, "export", "optimize", default=False)
    int8 = get_nested_value(config, "export", "int8", default=False)
    data = get_nested_value(config, "export", "data")
    workspace = get_nested_value(config, "export", "workspace", default=4)

    # Output options
    output_path = get_nested_value(config, "output", "path")
    verbose = get_nested_value(config, "output", "verbose", default=True)

    # Verification options
    verify = get_nested_value(config, "verify", "enabled", default=False)
    source = get_nested_value(config, "verify", "source")

    # Validate
    if not model_path:
        raise ValueError("--model or config model.path is required")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if fmt not in EXPORT_FORMATS:
        supported = ", ".join(sorted(EXPORT_FORMATS.keys()))
        raise ValueError(f"Unsupported format: '{fmt}'. Supported formats: {supported}")

    # Print config
    format_info = EXPORT_FORMATS[fmt]
    print(f"\n{'='*60}")
    print("YOLO Model Export Configuration")
    print(f"{'='*60}")
    print(f"Model:        {model_path}")
    print(f"Format:       {fmt} ({format_info['desc']})")
    print(f"Image size:   {imgsz}")
    print(f"Batch size:   {batch}")
    print(f"Device:       {device or 'auto'}")
    print(f"FP16 (half):  {half}")
    print(f"Dynamic:      {dynamic}")
    print(f"Simplify:     {simplify}")
    print(f"NMS:          {nms}")
    print(f"Optimize:     {optimize}")
    print(f"INT8:         {int8}")
    if opset:
        print(f"Opset:        {opset}")
    if int8 and data:
        print(f"Calibration:  {data}")
    if fmt == "engine" and workspace:
        print(f"Workspace:    {workspace} GB")
    if output_path:
        print(f"Output:       {output_path}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(str(model_path))

    # Build export arguments
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

    # Optional args
    if device is not None:
        export_args["device"] = device
    if opset is not None:
        export_args["opset"] = opset
    if data is not None:
        export_args["data"] = data
    if workspace is not None and fmt == "engine":
        export_args["workspace"] = workspace

    # Run export
    start_time = time.perf_counter()

    exported_path = model.export(**export_args)

    elapsed = time.perf_counter() - start_time

    # Determine actual output path
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        src = Path(exported_path)
        if src.is_dir():
            # Destination must be a directory when source is a directory
            if output_path.exists():
                if output_path.is_file():
                    output_path.unlink()
                else:
                    shutil.rmtree(output_path)
            shutil.copytree(src, output_path)
        else:
            # Source is a file: destination can be file or directory
            if output_path.is_dir():
                shutil.copy2(src, output_path / src.name)
                final_path = str(output_path / src.name)
            else:
                shutil.copy2(src, output_path)
            final_path = str(output_path) if not output_path.is_dir() else str(output_path / src.name)
    else:
        final_path = str(exported_path)

    # Show results
    print(f"\n{'='*60}")
    print("Export Completed!")
    print(f"{'='*60}")
    print(f"Format:       {format_info['desc']}")
    print(f"Output:       {final_path}")

    # File size
    final = Path(final_path)
    if final.is_file():
        size_bytes = final.stat().st_size
        if size_bytes > 1024 * 1024:
            print(f"File size:    {size_bytes / (1024 * 1024):.1f} MB")
        elif size_bytes > 1024:
            print(f"File size:    {size_bytes / 1024:.1f} KB")
        else:
            print(f"File size:    {size_bytes} B")
    elif final.is_dir():
        total_size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file())
        if total_size > 1024 * 1024:
            print(f"Dir size:     {total_size / (1024 * 1024):.1f} MB")
        else:
            print(f"Dir size:     {total_size / 1024:.1f} KB")

    print(f"Export time:  {elapsed:.2f}s")
    print(f"{'='*60}\n")

    # Verify
    if verify:
        verify_export(final_path, str(model_path), imgsz, source)

    return final_path


# ─── Main ─────────────────────────────────────────────────────────────


def print_formats():
    """Print all supported export formats."""
    print("\nSupported export formats:")
    print(f"{'Format':<15} {'Suffix':<25} {'Description'}")
    print("-" * 60)
    for key, info in EXPORT_FORMATS.items():
        print(f"{key:<15} {info['suffix']:<25} {info['desc']}")
    print()


def main():
    args = parse_args()

    try:
        # Handle --list-formats before anything else
        if args.list_formats:
            print_formats()
            return

        # Load config from file if specified
        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        # Merge CLI args into config (CLI takes precedence)
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)

        # Run export
        export(config)
    except KeyboardInterrupt:
        print("\nExport interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
