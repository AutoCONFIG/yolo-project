"""
YOLO Training and Validation Script
====================================
A flexible frontend for training and validating YOLO models using the ultralytics submodule.

Supports configuration via:
1. YAML config files (recommended)
2. Command line arguments (override config file)

Usage:
    # Using config file
    python train.py --config configs/default.yaml

    # Override specific settings via CLI
    python train.py --config configs/default.yaml --epochs 50 --batch 32 --no-amp --no-plots

    # Use CLI only (with defaults)
    python train.py --mode train --model yolo26n.pt --data coco8.yaml --epochs 100
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

# Add ultralytics submodule to path
ULTRALYTICS_PATH = Path(__file__).parent / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO

from utils.config import get_nested_value, load_yaml_config, merge_configs, set_boolean_argument, to_bool


# ─── Argument parser ─────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Training and Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with config file
    python train.py --config configs/default.yaml

    # Train with CLI overrides
    python train.py --config configs/default.yaml --epochs 50 --batch 32 --no-plots

    # Quick training without config file
    python train.py --mode train --model yolo26n.pt --data coco8.yaml --epochs 100

    # Validate a trained model
    python train.py --mode val --model runs/detect/train/weights/best.pt --data coco8.yaml

    # Resume training
    python train.py --mode train --model runs/detect/train/weights/last.pt --resume

    # Boolean overrides: use --flag to enable, --no-flag to disable
    python train.py --config configs/default.yaml --amp --no-cos-lr --no-plots
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # ── Mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "predict"],
        default=None,
        help="Mode: train, val, or predict",
    )

    # ── Model ─────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default=None, help="Model path or name (e.g. yolo26n.pt)")
    parser.add_argument("--model-yaml", type=str, default=None, help="YAML config for building model from scratch")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained weights path (string). In YAML use True/False; in CLI use a path string.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "segment", "classify", "pose", "obb"],
        default=None,
        help="Task type",
    )

    # ── Data ──────────────────────────────────────────────────────────
    parser.add_argument("--data", type=str, default=None, help="Dataset config YAML file")
    parser.add_argument("--split", type=str, default=None, help="Dataset split: val / test / train")

    # ── Core training ─────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default 100)")
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Max training hours — overrides --epochs if set",
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default 16)")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default 640)")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, 0, 0,1, mps")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers per rank (default 8)")
    set_boolean_argument(parser, "resume", "resume", help_true="Resume training from last checkpoint")
    parser.add_argument("--patience", type=int, default=None, help="Early stop patience (default 100)")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=["ram", "disk"],
        help="Cache images in RAM or disk for speed",
    )
    parser.add_argument(
        "--pretrained-bool",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Use pretrained model (boolean via YAML-compatible string). Supersedes --pretrained when set.",
        dest="pretrained_bool",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save checkpoints (boolean via string)",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Automatic mixed precision training (boolean via string)",
    )
    parser.add_argument(
        "--rect",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Rectangular batch training (boolean via string)",
    )
    parser.add_argument(
        "--single-cls",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Treat all classes as one (boolean via string)",
    )
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of dataset to use (0.0-1.0)")
    parser.add_argument("--freeze", type=int, nargs="+", default=None, help="Freeze first N layers (int or list)")
    parser.add_argument(
        "--multi-scale",
        type=float,
        default=None,
        help="Multi-scale training range as fraction of imgsz (0.0 = off)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default=None,
        choices=["true", "false", "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
        help="torch.compile mode for training",
    )
    parser.add_argument(
        "--end2end",
        type=str,
        default=None,
        choices=["true", "false"],
        help="End-to-end head for YOLO26/YOLOv10 (boolean via string)",
    )
    parser.add_argument("--nbs", type=int, default=None, help="Nominal batch size for loss normalization (default 64)")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Profile ONNX/TensorRT speeds (boolean via string)",
    )

    # ── Optimizer ─────────────────────────────────────────────────────
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer: SGD/Adam/AdamW/auto (default auto)")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (default 0.01)")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR fraction (default 0.01)")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum / Adam beta1 (default 0.937)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay L2 (default 0.0005)")
    set_boolean_argument(
        parser, "cos_lr", "cos-lr",
        help_true="Use cosine learning rate scheduler",
        help_false="Disable cosine LR (use step decay)",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="Disable mosaic for last N epochs (default 10, 0 = keep enabled)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default 0)")
    parser.add_argument(
        "--deterministic",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Deterministic ops for reproducibility (boolean via string)",
    )

    # ── Warmup ────────────────────────────────────────────────────────
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs (default 3.0)")
    parser.add_argument("--warmup-momentum", type=float, default=None, help="Initial warmup momentum (default 0.8)")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, help="Warmup bias LR (default 0.1)")

    # ── Loss gains ────────────────────────────────────────────────────
    parser.add_argument("--box", type=float, default=None, help="Box loss gain (default 7.5)")
    parser.add_argument("--cls", type=float, default=None, help="Class loss gain (default 0.5)")
    parser.add_argument("--cls-pw", type=float, default=None, help="Class weight power for imbalance (default 0.0)")
    parser.add_argument("--dfl", type=float, default=None, help="Distribution focal loss gain (default 1.5)")
    parser.add_argument("--pose", type=float, default=None, help="Pose loss gain (default 12.0)")
    parser.add_argument("--kobj", type=float, default=None, help="Keypoint objectness loss gain (default 1.0)")
    parser.add_argument("--rle", type=float, default=None, help="RLE loss gain (default 1.0)")
    parser.add_argument("--angle", type=float, default=None, help="OBB angle loss gain (default 1.0)")

    # ── Task-specific ─────────────────────────────────────────────────
    parser.add_argument(
        "--overlap-mask",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Merge instance masks during training (segment only, boolean via string)",
    )
    parser.add_argument("--mask-ratio", type=int, default=None, help="Mask downsample ratio (segment only, default 4)")
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout for classification head (classify only, 0.0-1.0)",
    )

    # ── Augmentation ──────────────────────────────────────────────────
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation (default 0.015)")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation (default 0.7)")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation (default 0.4)")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation degrees ± (default 0.0)")
    parser.add_argument("--translate", type=float, default=None, help="Translation fraction ± (default 0.1)")
    parser.add_argument("--scale", type=float, default=None, help="Scale gain ± (default 0.5)")
    parser.add_argument("--shear", type=float, default=None, help="Shear degrees ± (default 0.0)")
    parser.add_argument("--perspective", type=float, default=None, help="Perspective fraction (default 0.0)")
    parser.add_argument("--flipud", type=float, default=None, help="Vertical flip probability (default 0.0)")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability (default 0.5)")
    parser.add_argument("--bgr", type=float, default=None, help="RGB↔BGR channel swap probability (default 0.0)")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability (default 1.0)")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability (default 0.0)")
    parser.add_argument("--cutmix", type=float, default=None, help="CutMix augmentation probability (default 0.0)")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste probability (segment only, default 0.0)")
    parser.add_argument(
        "--copy-paste-mode",
        type=str,
        default=None,
        choices=["flip", "mixup"],
        help="Copy-paste strategy for segmentation (default flip)",
    )
    parser.add_argument(
        "--auto-augment",
        type=str,
        default=None,
        choices=["randaugment", "autoaugment", "augmix"],
        help="Auto augmentation policy (classify, default randaugment)",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=None,
        help="Random erasing probability (classify, default 0.4)",
    )

    # ── Validation / NMS (shared between val & train-time val) ────────
    parser.add_argument("--val", type=str, default=None, choices=["true", "false"], help="Run validation during training (boolean via string)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold for NMS (default 0.7)")
    parser.add_argument("--max-det", type=int, default=None, help="Max detections per image (default 300)")
    parser.add_argument(
        "--half",
        type=str,
        default=None,
        choices=["true", "false"],
        help="FP16 half-precision inference (boolean via string)",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save plots/images during train/val (boolean via string)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save COCO JSON results (boolean via string)",
    )
    parser.add_argument(
        "--dnn",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Use OpenCV DNN for ONNX inference (boolean via string, val/predict only)",
    )
    parser.add_argument(
        "--agnostic-nms",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Class-agnostic NMS (boolean via string)",
    )
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Filter by class IDs (e.g. 0 or 0 1 2)")
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Test-time augmentation (boolean via string)",
    )

    # ── Prediction ────────────────────────────────────────────────────
    parser.add_argument("--source", type=str, default=None, help="Input source: path/URL/stream/camera index")
    parser.add_argument(
        "--stream-buffer",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Buffer all stream frames (boolean via string, predict only)",
    )
    parser.add_argument(
        "--embed",
        nargs="+",
        type=int,
        default=None,
        help="Return feature embeddings from given layer indices (predict only)",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Show results in window (boolean via string)",
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save results as .txt files (boolean via string)",
    )
    parser.add_argument(
        "--save-conf",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save confidence with results (boolean via string)",
    )
    parser.add_argument(
        "--save-crop",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save cropped prediction regions (boolean via string)",
    )
    parser.add_argument(
        "--save-frames",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Save individual video frames (boolean via string)",
    )
    parser.add_argument(
        "--show-labels",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Draw class labels on images (boolean via string)",
    )
    parser.add_argument(
        "--show-boxes",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Draw bounding boxes on images (boolean via string)",
    )
    parser.add_argument(
        "--show-conf",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Draw confidence values on images (boolean via string)",
    )
    parser.add_argument("--line-width", type=int, default=None, help="Box line width (auto-scale if unset)")
    parser.add_argument("--vid-stride", type=int, default=None, help="Video frame stride (default 1)")
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Visualize model features (boolean via string)",
    )
    parser.add_argument(
        "--retina-masks",
        type=str,
        default=None,
        choices=["true", "false"],
        help="High-res segmentation masks (boolean via string)",
    )

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=None, help="Project name for results root")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--save-period", type=int, default=None, help="Save checkpoint every N epochs (-1 = off)")
    parser.add_argument(
        "--exist-ok",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Overwrite existing project/name (boolean via string)",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Verbose output during training (boolean via string)",
    )

    # ── Misc ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Custom config YAML that overrides defaults (ultralytics --cfg)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Tracker config: botsort.yaml or bytetrack.yaml",
    )

    return parser.parse_args()


_AUG_KEYS = [
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic",
    "mixup", "cutmix", "copy_paste", "copy_paste_mode",
    "auto_augment", "erasing",
]


# ─── CLI → nested config ──────────────────────────────────────────────


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command line args to nested config dict."""
    config = {}
    if args.mode:
        config["mode"] = args.mode

    model_cfg = {}
    if args.model:
        model_cfg["name"] = args.model
    if args.model_yaml:
        model_cfg["yaml"] = args.model_yaml
    if args.pretrained:
        model_cfg["pretrained"] = args.pretrained
    elif args.pretrained_bool is not None:
        model_cfg["pretrained"] = to_bool(args.pretrained_bool)
    if args.task:
        model_cfg["task"] = args.task
    if model_cfg:
        config["model"] = model_cfg

    data_cfg = {}
    if args.data:
        data_cfg["config"] = args.data
    if args.split:
        data_cfg["split"] = args.split
    if data_cfg:
        config["data"] = {**config.get("data", {}), **data_cfg}

    train_cfg = {}
    for field in ("epochs", "time", "batch", "imgsz", "device", "workers", "patience",
                  "cache", "fraction", "freeze", "multi_scale", "nbs", "optimizer",
                  "lr0", "lrf", "momentum", "weight_decay", "close_mosaic", "seed",
                  "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
                  "box", "cls", "cls_pw", "dfl", "pose", "kobj", "rle", "angle",
                  "mask_ratio", "dropout"):
        v = getattr(args, field)
        if v is not None:
            train_cfg[field] = v
    for field in ("save", "amp", "rect", "single_cls", "end2end", "profile",
                  "deterministic", "overlap_mask"):
        v = to_bool(getattr(args, field))
        if v is not None:
            train_cfg[field] = v
    if args.resume is not None:
        train_cfg["resume"] = args.resume
    if args.cos_lr is not None:
        train_cfg["cos_lr"] = args.cos_lr
    if args.compile is not None:
        b = to_bool(args.compile)
        train_cfg["compile"] = b if b is not None else args.compile
    if train_cfg:
        config["train"] = {**config.get("train", {}), **train_cfg}

    aug_cfg = {k: getattr(args, k) for k in _AUG_KEYS if getattr(args, k) is not None}
    if aug_cfg:
        config["augmentation"] = {**config.get("augmentation", {}), **aug_cfg}

    val_cfg = {}
    for field in ("val", "half", "plots", "save_json", "dnn", "agnostic_nms", "augment"):
        v = to_bool(getattr(args, field))
        if v is not None:
            val_cfg[field] = v
    for field in ("conf", "iou", "max_det", "classes"):
        v = getattr(args, field)
        if v is not None:
            val_cfg[field] = v
    if val_cfg:
        config["validation"] = {**config.get("validation", {}), **val_cfg}

    pred_cfg = {}
    for field in ("show", "save_txt", "save_conf", "save_crop", "save_frames",
                  "show_labels", "show_boxes", "show_conf", "visualize",
                  "retina_masks", "stream_buffer"):
        v = to_bool(getattr(args, field))
        if v is not None:
            pred_cfg[field] = v
    for field in ("source", "line_width", "vid_stride", "embed"):
        v = getattr(args, field)
        if v is not None:
            pred_cfg[field] = v
    if pred_cfg:
        config["predict"] = {**config.get("predict", {}), **pred_cfg}

    out_cfg = {}
    for field in ("exist_ok", "verbose"):
        v = to_bool(getattr(args, field))
        if v is not None:
            out_cfg[field] = v
    for field in ("project", "name", "save_period"):
        v = getattr(args, field)
        if v is not None:
            out_cfg[field] = v
    if out_cfg:
        config["output"] = {**config.get("output", {}), **out_cfg}

    if args.cfg:
        config["cfg"] = args.cfg
    if args.tracker:
        config["tracker"] = args.tracker

    return config


# ─── Train ────────────────────────────────────────────────────────────


def train(config: Dict):
    """Train a YOLO model."""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")
    model_yaml = get_nested_value(config, "model", "yaml")
    pretrained = get_nested_value(config, "model", "pretrained")
    data_config = get_nested_value(config, "data", "config", default="coco8.yaml")

    train_args = {
        "data": data_config,
        "epochs": get_nested_value(config, "train", "epochs", default=100),
        "batch": get_nested_value(config, "train", "batch", default=16),
        "imgsz": get_nested_value(config, "train", "imgsz", default=640),
        "workers": get_nested_value(config, "train", "workers", default=8),
        "optimizer": get_nested_value(config, "train", "optimizer", default="auto"),
        "lr0": get_nested_value(config, "train", "lr0", default=0.01),
        "lrf": get_nested_value(config, "train", "lrf", default=0.01),
        "momentum": get_nested_value(config, "train", "momentum", default=0.937),
        "weight_decay": get_nested_value(config, "train", "weight_decay", default=0.0005),
        "patience": get_nested_value(config, "train", "patience", default=100),
        "amp": get_nested_value(config, "train", "amp", default=True),
        "save": get_nested_value(config, "train", "save", default=True),
        "cos_lr": get_nested_value(config, "train", "cos_lr", default=False),
        "close_mosaic": get_nested_value(config, "train", "close_mosaic", default=10),
        "seed": get_nested_value(config, "train", "seed", default=0),
        "deterministic": get_nested_value(config, "train", "deterministic", default=True),
        "warmup_epochs": get_nested_value(config, "train", "warmup_epochs", default=3.0),
        "warmup_momentum": get_nested_value(config, "train", "warmup_momentum", default=0.8),
        "warmup_bias_lr": get_nested_value(config, "train", "warmup_bias_lr", default=0.1),
        "box": get_nested_value(config, "train", "box", default=7.5),
        "cls": get_nested_value(config, "train", "cls", default=0.5),
        "dfl": get_nested_value(config, "train", "dfl", default=1.5),
        "nbs": get_nested_value(config, "train", "nbs", default=64),
        "save_period": get_nested_value(config, "train", "save_period", default=-1),
        "exist_ok": get_nested_value(config, "output", "exist_ok", default=False),
        "verbose": get_nested_value(config, "output", "verbose", default=True),
        "resume": get_nested_value(config, "train", "resume", default=False),
        "fraction": get_nested_value(config, "train", "fraction", default=1.0),
        "multi_scale": get_nested_value(config, "train", "multi_scale", default=0.0),
    }

    for key in (
        "device", "cache", "time", "rect", "single_cls", "freeze",
        "compile", "end2end", "profile", "classes", "augmentations",
        "cls_pw", "pose", "kobj", "rle", "angle", "overlap_mask",
        "mask_ratio", "dropout",
    ):
        v = get_nested_value(config, "train", key)
        if v is not None:
            train_args[key] = v

    if pretrained is not None:
        train_args["pretrained"] = pretrained
    task = get_nested_value(config, "model", "task")
    if task is not None:
        train_args["task"] = task

    for key in ("val", "conf", "iou", "max_det", "half", "plots", "save_json"):
        v = get_nested_value(config, "validation", key)
        if v is not None:
            train_args[key] = v
    train_args["split"] = get_nested_value(config, "data", "split", default="val")

    for key in ("project", "name"):
        v = get_nested_value(config, "output", key)
        if v is not None:
            train_args[key] = v

    for key in _AUG_KEYS:
        v = get_nested_value(config, "augmentation", key)
        if v is not None:
            train_args[key] = v

    if config.get("cfg") is not None:
        train_args["cfg"] = config["cfg"]
    if config.get("tracker") is not None:
        train_args["tracker"] = config["tracker"]

    resume = train_args["resume"]
    project = train_args.get("project")
    name = train_args.get("name")

    project_root = Path(__file__).parent.resolve()
    save_dir = (project_root / str(project) / str(name)).resolve() if project or name else None

    print(f"\n{'='*60}")
    print("YOLO Training Configuration")
    print(f"{'='*60}")
    print(f"Model:      {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Data:       {data_config}")
    print(f"Epochs:     {train_args['epochs']}")
    print(f"Batch:      {train_args['batch']}")
    print(f"Image size: {train_args['imgsz']}")
    print(f"Device:     {train_args.get('device', 'auto')}")
    print(f"Cache:      {train_args.get('cache', 'False')}")
    print(f"Optimizer:  {train_args['optimizer']}")
    print(f"Learning rate: {train_args['lr0']}")
    if resume and save_dir:
        print(f"Resume:     {save_dir / 'weights' / 'last.pt'}")
    print(f"{'='*60}\n")

    last_pt = (save_dir / "weights" / "last.pt") if save_dir else None
    if resume and last_pt and last_pt.exists():
        print(f"Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        if resume:
            print(f"Checkpoint not found: {last_pt}, starting fresh training")
            resume = False
        if model_yaml:
            model = YOLO(model_yaml)
            if isinstance(pretrained, str):
                model.load(pretrained)
        else:
            model = YOLO(model_name)

    results = model.train(**train_args)

    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"Last weights: {results.save_dir}/weights/last.pt")
    print(f"{'='*60}\n")

    return results


# ─── Validate ─────────────────────────────────────────────────────────


def validate(config: Dict):
    """Validate a YOLO model."""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")

    val_args = {
        "data": get_nested_value(config, "data", "config", default="coco8.yaml"),
        "split": get_nested_value(config, "data", "split", default="val"),
        "imgsz": get_nested_value(config, "train", "imgsz", default=640),
        "batch": get_nested_value(config, "train", "batch", default=16),
        "conf": get_nested_value(config, "validation", "conf", default=0.25),
        "iou": get_nested_value(config, "validation", "iou", default=0.7),
        "plots": get_nested_value(config, "validation", "plots", default=False),
        "save_json": get_nested_value(config, "validation", "save_json", default=False),
        "dnn": get_nested_value(config, "validation", "dnn", default=False),
        "agnostic_nms": get_nested_value(config, "validation", "agnostic_nms", default=False),
        "augment": get_nested_value(config, "validation", "augment", default=False),
        "half": get_nested_value(config, "validation", "half", default=False),
        "max_det": get_nested_value(config, "validation", "max_det", default=300),
        "verbose": get_nested_value(config, "output", "verbose", default=True),
    }

    device = get_nested_value(config, "train", "device")
    if device is not None:
        val_args["device"] = device

    classes = get_nested_value(config, "validation", "classes")
    if classes is not None:
        val_args["classes"] = classes

    for key in ("project", "name"):
        v = get_nested_value(config, "output", key)
        if v is not None:
            val_args[key] = v

    print(f"\n{'='*60}")
    print("YOLO Validation Configuration")
    print(f"{'='*60}")
    print(f"Model:      {model_name}")
    print(f"Data:       {val_args['data']}")
    print(f"Split:      {val_args['split']}")
    print(f"Image size: {val_args['imgsz']}")
    print(f"Batch:      {val_args['batch']}")
    print(f"Device:     {val_args.get('device', 'auto')}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)
    metrics = model.val(**val_args)

    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    print(f"{'='*60}\n")

    return metrics


# ─── Predict ──────────────────────────────────────────────────────────


def predict(config: Dict):
    """Run prediction with a YOLO model."""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")
    source = get_nested_value(config, "predict", "source")

    if not source:
        print("Error: --source is required for prediction mode")
        sys.exit(1)

    predict_args = {
        "source": source,
        "imgsz": get_nested_value(config, "train", "imgsz", default=640),
        "conf": get_nested_value(config, "validation", "conf", default=0.25),
        "iou": get_nested_value(config, "validation", "iou", default=0.7),
        "show": get_nested_value(config, "predict", "show", default=False),
        "save_txt": get_nested_value(config, "predict", "save_txt", default=False),
        "save_conf": get_nested_value(config, "predict", "save_conf", default=False),
        "save_crop": get_nested_value(config, "predict", "save_crop", default=False),
        "save_frames": get_nested_value(config, "predict", "save_frames", default=False),
        "show_labels": get_nested_value(config, "predict", "show_labels", default=True),
        "show_boxes": get_nested_value(config, "predict", "show_boxes", default=True),
        "show_conf": get_nested_value(config, "predict", "show_conf", default=True),
        "agnostic_nms": get_nested_value(config, "validation", "agnostic_nms", default=False),
        "augment": get_nested_value(config, "validation", "augment", default=False),
        "half": get_nested_value(config, "validation", "half", default=False),
        "max_det": get_nested_value(config, "validation", "max_det", default=300),
        "visualize": get_nested_value(config, "predict", "visualize", default=False),
        "retina_masks": get_nested_value(config, "predict", "retina_masks", default=False),
        "stream_buffer": get_nested_value(config, "predict", "stream_buffer", default=False),
        "verbose": get_nested_value(config, "output", "verbose", default=True),
    }

    device = get_nested_value(config, "train", "device")
    if device is not None:
        predict_args["device"] = device

    classes = get_nested_value(config, "validation", "classes")
    if classes is not None:
        predict_args["classes"] = classes

    for key in ("project", "name"):
        v = get_nested_value(config, "output", key)
        if v is not None:
            predict_args[key] = v

    for key in ("line_width", "vid_stride", "embed"):
        v = get_nested_value(config, "predict", key)
        if v is not None:
            predict_args[key] = v

    print(f"\n{'='*60}")
    print("YOLO Prediction Configuration")
    print(f"{'='*60}")
    print(f"Model:  {model_name}")
    print(f"Source: {source}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)
    results = model.predict(**predict_args)

    print(f"\n{'='*60}")
    print("Prediction Completed!")
    print(f"{'='*60}\n")

    return results


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    try:
        # Load config from file if specified
        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        # Merge CLI args into config (CLI takes precedence)
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)

        # Determine mode
        mode = config.get("mode", "train")

        # Run appropriate mode
        if mode == "train":
            train(config)
        elif mode == "val":
            validate(config)
        elif mode == "predict":
            predict(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
