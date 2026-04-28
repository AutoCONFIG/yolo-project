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
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add ultralytics submodule to path
ULTRALYTICS_PATH = Path(__file__).parent / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

# Patch ultralytics downloads to use weights_dir (must be before ultralytics imports)
from utils.downloads import patch_ultralytics_downloads
patch_ultralytics_downloads()

from ultralytics import YOLO


# ─── Boolean CLI helper ───────────────────────────────────────────────
# Creates paired flags: --flag (sets True) and --no-flag (sets False).
# Omitting both yields None (YAML default preserved).


def set_boolean_argument(
    parser: argparse.ArgumentParser,
    dest: str,
    flag_name: str | None = None,
    *,
    neg_prefix: str = "no-",
    help_true: str = "",
    help_false: str = "",
) -> None:
    """Add a paired boolean argument (e.g. --amp / --no-amp) to a parser.

    Parameters
    ----------
    parser :
        The argparse.ArgumentParser to add arguments to.
    dest :
        Destination attribute name in the parsed Namespace.
    flag_name :
        The positive flag text (default: *dest* with underscores replaced by hyphens).
    neg_prefix :
        Prefix for the negative flag (default ``"no-"``).
    help_true / help_false :
        Help text for the positive / negative flags.
    """
    flag = flag_name or dest.replace("_", "-")

    positive = f"--{flag}"
    negative = f"--{neg_prefix}{flag}"

    group = parser.add_mutually_exclusive_group()

    # Positive flag
    group.add_argument(
        positive,
        dest=dest,
        action="store_const",
        const=True,
        default=None,
        help=help_true or f"Enable {flag}",
    )
    # Negative flag
    group.add_argument(
        negative,
        dest=dest,
        action="store_const",
        const=False,
        default=None,
        help=help_false or f"Disable {flag}",
    )


# ─── Config loaders ───────────────────────────────────────────────────


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def merge_configs(base_config: Dict, override_args: Dict) -> Dict:
    """Merge override args into base config."""
    result = base_config.copy()
    for key, value in override_args.items():
        if value is not None:
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = {**result[key], **value}
            else:
                result[key] = value
    return result


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


# ─── CLI → nested config ──────────────────────────────────────────────


def to_bool(value: str | None) -> bool | None:
    """Convert 'true'/'false' string to bool. Returns None for None or unknown."""
    if value is None:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return None


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command line args to nested config dict."""
    config = {}

    # Direct mappings
    if args.mode:
        config["mode"] = args.mode

    # ── Model ──
    model_config = {}
    if args.model:
        model_config["name"] = args.model
    if args.model_yaml:
        model_config["yaml"] = args.model_yaml
    # pretrained: CLI can pass a path (--pretrained) or a bool string (--pretrained-bool true/false)
    if args.pretrained:
        model_config["pretrained"] = args.pretrained
    elif args.pretrained_bool is not None:
        model_config["pretrained"] = to_bool(args.pretrained_bool)
    if args.task:
        model_config["task"] = args.task
    if model_config:
        config["model"] = model_config

    # ── Data ──
    data_config = {}
    if args.data:
        data_config["config"] = args.data
    if args.split:
        data_config["split"] = args.split
    if data_config:
        config["data"] = {**config.get("data", {}), **data_config}

    # ── Train ──
    train_config = {}
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.time is not None:
        train_config["time"] = args.time
    if args.batch is not None:
        train_config["batch"] = args.batch
    if args.imgsz is not None:
        train_config["imgsz"] = args.imgsz
    if args.device is not None:
        train_config["device"] = args.device
    if args.workers is not None:
        train_config["workers"] = args.workers
    if args.resume is not None:
        train_config["resume"] = args.resume
    if args.patience is not None:
        train_config["patience"] = args.patience
    if args.cache is not None:
        train_config["cache"] = args.cache
    v = to_bool(args.save)
    if v is not None:
        train_config["save"] = v
    v = to_bool(args.amp)
    if v is not None:
        train_config["amp"] = v
    v = to_bool(args.rect)
    if v is not None:
        train_config["rect"] = v
    v = to_bool(args.single_cls)
    if v is not None:
        train_config["single_cls"] = v
    if args.fraction is not None:
        train_config["fraction"] = args.fraction
    if args.freeze is not None:
        train_config["freeze"] = args.freeze
    if args.multi_scale is not None:
        train_config["multi_scale"] = args.multi_scale
    v = args.compile
    if v is not None:
        b = to_bool(v)
        train_config["compile"] = b if b is not None else v
    v = to_bool(args.end2end)
    if v is not None:
        train_config["end2end"] = v
    if args.nbs is not None:
        train_config["nbs"] = args.nbs
    v = to_bool(args.profile)
    if v is not None:
        train_config["profile"] = v
    # Optimizer
    if args.optimizer:
        train_config["optimizer"] = args.optimizer
    if args.lr0 is not None:
        train_config["lr0"] = args.lr0
    if args.lrf is not None:
        train_config["lrf"] = args.lrf
    if args.momentum is not None:
        train_config["momentum"] = args.momentum
    if args.weight_decay is not None:
        train_config["weight_decay"] = args.weight_decay
    v = args.cos_lr
    if v is not None:
        train_config["cos_lr"] = v
    if args.close_mosaic is not None:
        train_config["close_mosaic"] = args.close_mosaic
    if args.seed is not None:
        train_config["seed"] = args.seed
    v = to_bool(args.deterministic)
    if v is not None:
        train_config["deterministic"] = v
    # Warmup
    if args.warmup_epochs is not None:
        train_config["warmup_epochs"] = args.warmup_epochs
    if args.warmup_momentum is not None:
        train_config["warmup_momentum"] = args.warmup_momentum
    if args.warmup_bias_lr is not None:
        train_config["warmup_bias_lr"] = args.warmup_bias_lr
    # Loss gains
    if args.box is not None:
        train_config["box"] = args.box
    if args.cls is not None:
        train_config["cls"] = args.cls
    if args.cls_pw is not None:
        train_config["cls_pw"] = args.cls_pw
    if args.dfl is not None:
        train_config["dfl"] = args.dfl
    if args.pose is not None:
        train_config["pose"] = args.pose
    if args.kobj is not None:
        train_config["kobj"] = args.kobj
    if args.rle is not None:
        train_config["rle"] = args.rle
    if args.angle is not None:
        train_config["angle"] = args.angle
    # Task-specific
    v = to_bool(args.overlap_mask)
    if v is not None:
        train_config["overlap_mask"] = v
    if args.mask_ratio is not None:
        train_config["mask_ratio"] = args.mask_ratio
    if args.dropout is not None:
        train_config["dropout"] = args.dropout
    if train_config:
        config["train"] = {**config.get("train", {}), **train_config}

    # ── Augmentation ──
    aug_config = {}
    if args.hsv_h is not None:
        aug_config["hsv_h"] = args.hsv_h
    if args.hsv_s is not None:
        aug_config["hsv_s"] = args.hsv_s
    if args.hsv_v is not None:
        aug_config["hsv_v"] = args.hsv_v
    if args.degrees is not None:
        aug_config["degrees"] = args.degrees
    if args.translate is not None:
        aug_config["translate"] = args.translate
    if args.scale is not None:
        aug_config["scale"] = args.scale
    if args.shear is not None:
        aug_config["shear"] = args.shear
    if args.perspective is not None:
        aug_config["perspective"] = args.perspective
    if args.flipud is not None:
        aug_config["flipud"] = args.flipud
    if args.fliplr is not None:
        aug_config["fliplr"] = args.fliplr
    if args.bgr is not None:
        aug_config["bgr"] = args.bgr
    if args.mosaic is not None:
        aug_config["mosaic"] = args.mosaic
    if args.mixup is not None:
        aug_config["mixup"] = args.mixup
    if args.cutmix is not None:
        aug_config["cutmix"] = args.cutmix
    if args.copy_paste is not None:
        aug_config["copy_paste"] = args.copy_paste
    if args.copy_paste_mode is not None:
        aug_config["copy_paste_mode"] = args.copy_paste_mode
    if args.auto_augment is not None:
        aug_config["auto_augment"] = args.auto_augment
    if args.erasing is not None:
        aug_config["erasing"] = args.erasing
    if aug_config:
        config["augmentation"] = {**config.get("augmentation", {}), **aug_config}

    # ── Validation ──
    val_config = {}
    v = to_bool(args.val)
    if v is not None:
        val_config["val"] = v
    if args.conf is not None:
        val_config["conf"] = args.conf
    if args.iou is not None:
        val_config["iou"] = args.iou
    if args.max_det is not None:
        val_config["max_det"] = args.max_det
    v = to_bool(args.half)
    if v is not None:
        val_config["half"] = v
    v = to_bool(args.plots)
    if v is not None:
        val_config["plots"] = v
    v = to_bool(args.save_json)
    if v is not None:
        val_config["save_json"] = v
    v = to_bool(args.dnn)
    if v is not None:
        val_config["dnn"] = v
    v = to_bool(args.agnostic_nms)
    if v is not None:
        val_config["agnostic_nms"] = v
    if args.classes is not None:
        val_config["classes"] = args.classes
    v = to_bool(args.augment)
    if v is not None:
        val_config["augment"] = v
    if val_config:
        config["validation"] = {**config.get("validation", {}), **val_config}

    # ── Predict ──
    predict_config = {}
    if args.source:
        predict_config["source"] = args.source
    v = to_bool(args.show)
    if v is not None:
        predict_config["show"] = v
    v = to_bool(args.save_txt)
    if v is not None:
        predict_config["save_txt"] = v
    v = to_bool(args.save_conf)
    if v is not None:
        predict_config["save_conf"] = v
    v = to_bool(args.save_crop)
    if v is not None:
        predict_config["save_crop"] = v
    v = to_bool(args.save_frames)
    if v is not None:
        predict_config["save_frames"] = v
    v = to_bool(args.show_labels)
    if v is not None:
        predict_config["show_labels"] = v
    v = to_bool(args.show_boxes)
    if v is not None:
        predict_config["show_boxes"] = v
    v = to_bool(args.show_conf)
    if v is not None:
        predict_config["show_conf"] = v
    if args.line_width is not None:
        predict_config["line_width"] = args.line_width
    if args.vid_stride is not None:
        predict_config["vid_stride"] = args.vid_stride
    v = to_bool(args.visualize)
    if v is not None:
        predict_config["visualize"] = v
    v = to_bool(args.retina_masks)
    if v is not None:
        predict_config["retina_masks"] = v
    v = to_bool(args.stream_buffer)
    if v is not None:
        predict_config["stream_buffer"] = v
    if args.embed is not None:
        predict_config["embed"] = args.embed
    if predict_config:
        config["predict"] = {**config.get("predict", {}), **predict_config}

    # ── Output ──
    output_config = {}
    if args.project:
        output_config["project"] = args.project
    if args.name:
        output_config["name"] = args.name
    if args.save_period is not None:
        output_config["save_period"] = args.save_period
    v = to_bool(args.exist_ok)
    if v is not None:
        output_config["exist_ok"] = v
    v = to_bool(args.verbose)
    if v is not None:
        output_config["verbose"] = v
    if output_config:
        config["output"] = {**config.get("output", {}), **output_config}

    # ── Misc ──
    if args.cfg:
        config["cfg"] = args.cfg
    if args.tracker:
        config["tracker"] = args.tracker

    return config


# ─── Helpers ──────────────────────────────────────────────────────────


def get_nested_value(config: Dict, *keys, default=None):
    """Get nested value from config dict."""
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ─── Train ────────────────────────────────────────────────────────────


def train(config: Dict):
    """Train a YOLO model."""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")
    model_yaml = get_nested_value(config, "model", "yaml")
    pretrained = get_nested_value(config, "model", "pretrained")
    data_config = get_nested_value(config, "data", "config", default="coco8.yaml")

    # ── Core training settings ──
    epochs = get_nested_value(config, "train", "epochs", default=100)
    time = get_nested_value(config, "train", "time")
    batch = get_nested_value(config, "train", "batch", default=16)
    imgsz = get_nested_value(config, "train", "imgsz", default=640)
    device = get_nested_value(config, "train", "device")
    workers = get_nested_value(config, "train", "workers", default=8)
    resume = get_nested_value(config, "train", "resume", default=False)
    patience = get_nested_value(config, "train", "patience", default=100)
    cache = get_nested_value(config, "train", "cache")
    amp = get_nested_value(config, "train", "amp", default=True)
    save = get_nested_value(config, "train", "save", default=True)
    rect = get_nested_value(config, "train", "rect", default=False)
    single_cls = get_nested_value(config, "train", "single_cls", default=False)
    fraction = get_nested_value(config, "train", "fraction", default=1.0)
    freeze = get_nested_value(config, "train", "freeze")
    multi_scale = get_nested_value(config, "train", "multi_scale", default=0.0)
    compile_flag = get_nested_value(config, "train", "compile")
    end2end = get_nested_value(config, "train", "end2end")
    nbs = get_nested_value(config, "train", "nbs", default=64)
    profile = get_nested_value(config, "train", "profile", default=False)

    # ── Optimizer settings ──
    optimizer = get_nested_value(config, "train", "optimizer", default="auto")
    lr0 = get_nested_value(config, "train", "lr0", default=0.01)
    lrf = get_nested_value(config, "train", "lrf", default=0.01)
    momentum = get_nested_value(config, "train", "momentum", default=0.937)
    weight_decay = get_nested_value(config, "train", "weight_decay", default=0.0005)
    cos_lr = get_nested_value(config, "train", "cos_lr", default=False)
    close_mosaic = get_nested_value(config, "train", "close_mosaic", default=10)
    seed = get_nested_value(config, "train", "seed", default=0)
    deterministic = get_nested_value(config, "train", "deterministic", default=True)

    # ── Warmup ──
    warmup_epochs = get_nested_value(config, "train", "warmup_epochs", default=3.0)
    warmup_momentum = get_nested_value(config, "train", "warmup_momentum", default=0.8)
    warmup_bias_lr = get_nested_value(config, "train", "warmup_bias_lr", default=0.1)

    # ── Loss gains ──
    box = get_nested_value(config, "train", "box", default=7.5)
    cls = get_nested_value(config, "train", "cls", default=0.5)
    cls_pw = get_nested_value(config, "train", "cls_pw", default=0.0)
    dfl = get_nested_value(config, "train", "dfl", default=1.5)
    pose = get_nested_value(config, "train", "pose", default=12.0)
    kobj = get_nested_value(config, "train", "kobj", default=1.0)
    rle = get_nested_value(config, "train", "rle", default=1.0)
    angle = get_nested_value(config, "train", "angle", default=1.0)

    # ── Task-specific settings ──
    overlap_mask = get_nested_value(config, "train", "overlap_mask", default=True)
    mask_ratio = get_nested_value(config, "train", "mask_ratio", default=4)
    dropout = get_nested_value(config, "train", "dropout", default=0.0)
    task = get_nested_value(config, "model", "task")

    # ── Validation during training ──
    val = get_nested_value(config, "validation", "val", default=True)
    val_split = get_nested_value(config, "data", "split", default="val")
    conf = get_nested_value(config, "validation", "conf")
    iou = get_nested_value(config, "validation", "iou", default=0.7)
    max_det = get_nested_value(config, "validation", "max_det", default=300)
    half = get_nested_value(config, "validation", "half", default=False)
    plots = get_nested_value(config, "validation", "plots", default=True)
    save_json = get_nested_value(config, "validation", "save_json", default=False)

    # ── Output settings ──
    project = get_nested_value(config, "output", "project")
    name = get_nested_value(config, "output", "name")
    save_period = get_nested_value(config, "output", "save_period", default=-1)
    exist_ok = get_nested_value(config, "output", "exist_ok", default=False)
    verbose = get_nested_value(config, "output", "verbose", default=True)

    # ── Misc ──
    cfg_override = config.get("cfg")
    tracker = config.get("tracker")

    # Print config
    print(f"\n{'='*60}")
    print("YOLO Training Configuration")
    print(f"{'='*60}")
    print(f"Model:      {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Data:       {data_config}")
    print(f"Epochs:     {epochs}")
    print(f"Batch:      {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device:     {device or 'auto'}")
    print(f"Cache:      {cache or 'False'}")
    print(f"Optimizer:  {optimizer}")
    print(f"Learning rate: {lr0}")
    print(f"{'='*60}\n")

    # Load model
    if model_yaml:
        model = YOLO(model_yaml)
        if isinstance(pretrained, str):
            model.load(pretrained)
    else:
        model = YOLO(model_name)

    # Build training arguments
    train_args = {
        "data": data_config,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "patience": patience,
        "amp": amp,
        "save": save,
        "cos_lr": cos_lr,
        "close_mosaic": close_mosaic,
        "seed": seed,
        "deterministic": deterministic,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
        "box": box,
        "cls": cls,
        "dfl": dfl,
        "nbs": nbs,
        "save_period": save_period,
        "exist_ok": exist_ok,
        "verbose": verbose,
        "resume": resume,
    }

    # Add optional args
    if device is not None:
        train_args["device"] = device
    if pretrained is not None:
        train_args["pretrained"] = pretrained
    if cache is not None:
        train_args["cache"] = cache
    if time is not None:
        train_args["time"] = time
    if rect is not None:
        train_args["rect"] = rect
    if single_cls is not None:
        train_args["single_cls"] = single_cls
    if fraction is not None:
        train_args["fraction"] = fraction
    if freeze is not None:
        train_args["freeze"] = freeze
    if multi_scale is not None:
        train_args["multi_scale"] = multi_scale
    if compile_flag is not None:
        train_args["compile"] = compile_flag
    if end2end is not None:
        train_args["end2end"] = end2end
    if profile is not None:
        train_args["profile"] = profile
    if cls_pw is not None:
        train_args["cls_pw"] = cls_pw
    if pose is not None:
        train_args["pose"] = pose
    if kobj is not None:
        train_args["kobj"] = kobj
    if rle is not None:
        train_args["rle"] = rle
    if angle is not None:
        train_args["angle"] = angle
    if overlap_mask is not None:
        train_args["overlap_mask"] = overlap_mask
    if mask_ratio is not None:
        train_args["mask_ratio"] = mask_ratio
    if dropout is not None:
        train_args["dropout"] = dropout
    if task is not None:
        train_args["task"] = task
    if val is not None:
        train_args["val"] = val
    if val_split is not None:
        train_args["split"] = val_split
    if conf is not None:
        train_args["conf"] = conf
    if iou is not None:
        train_args["iou"] = iou
    if max_det is not None:
        train_args["max_det"] = max_det
    if half is not None:
        train_args["half"] = half
    if plots is not None:
        train_args["plots"] = plots
    if save_json is not None:
        train_args["save_json"] = save_json
    if project is not None:
        train_args["project"] = project
    if name is not None:
        train_args["name"] = name
    if cfg_override is not None:
        train_args["cfg"] = cfg_override
    if tracker is not None:
        train_args["tracker"] = tracker

    aug_keys = [
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
        "scale", "shear", "perspective", "flipud", "fliplr",
        "bgr", "mosaic", "mixup", "cutmix", "copy_paste",
        "copy_paste_mode", "auto_augment", "erasing",
    ]
    for key in aug_keys:
        value = get_nested_value(config, "augmentation", key)
        if value is not None:
            train_args[key] = value

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
    data_config = get_nested_value(config, "data", "config", default="coco8.yaml")
    split = get_nested_value(config, "data", "split", default="val")

    imgsz = get_nested_value(config, "train", "imgsz", default=640)
    batch = get_nested_value(config, "train", "batch", default=16)
    device = get_nested_value(config, "train", "device")

    # Validation settings
    conf = get_nested_value(config, "validation", "conf", default=0.25)
    iou = get_nested_value(config, "validation", "iou", default=0.7)
    plots = get_nested_value(config, "validation", "plots", default=False)
    save_json = get_nested_value(config, "validation", "save_json", default=False)
    dnn = get_nested_value(config, "validation", "dnn", default=False)
    agnostic_nms = get_nested_value(config, "validation", "agnostic_nms", default=False)
    classes = get_nested_value(config, "validation", "classes")
    augment = get_nested_value(config, "validation", "augment", default=False)
    half = get_nested_value(config, "validation", "half", default=False)
    max_det = get_nested_value(config, "validation", "max_det", default=300)

    project = get_nested_value(config, "output", "project")
    name = get_nested_value(config, "output", "name")
    verbose = get_nested_value(config, "output", "verbose", default=True)

    print(f"\n{'='*60}")
    print("YOLO Validation Configuration")
    print(f"{'='*60}")
    print(f"Model:      {model_name}")
    print(f"Data:       {data_config}")
    print(f"Split:      {split}")
    print(f"Image size: {imgsz}")
    print(f"Batch:      {batch}")
    print(f"Device:     {device or 'auto'}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(model_name)

    # Build validation arguments
    val_args = {
        "data": data_config,
        "imgsz": imgsz,
        "batch": batch,
        "split": split,
        "conf": conf,
        "iou": iou,
        "plots": plots,
        "save_json": save_json,
        "verbose": verbose,
        "dnn": dnn,
        "agnostic_nms": agnostic_nms,
        "augment": augment,
        "half": half,
        "max_det": max_det,
    }

    if device is not None:
        val_args["device"] = device
    if project:
        val_args["project"] = project
    if name:
        val_args["name"] = name
    if classes is not None:
        val_args["classes"] = classes

    # Validate
    metrics = model.val(**val_args)

    # Print metrics
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

    imgsz = get_nested_value(config, "train", "imgsz", default=640)
    device = get_nested_value(config, "train", "device")

    conf = get_nested_value(config, "validation", "conf", default=0.25)
    iou = get_nested_value(config, "validation", "iou", default=0.7)
    classes = get_nested_value(config, "validation", "classes")
    agnostic_nms = get_nested_value(config, "validation", "agnostic_nms", default=False)
    augment = get_nested_value(config, "validation", "augment", default=False)
    half = get_nested_value(config, "validation", "half", default=False)
    max_det = get_nested_value(config, "validation", "max_det", default=300)

    show = get_nested_value(config, "predict", "show", default=False)
    save_txt = get_nested_value(config, "predict", "save_txt", default=False)
    save_conf = get_nested_value(config, "predict", "save_conf", default=False)
    save_crop = get_nested_value(config, "predict", "save_crop", default=False)
    save_frames = get_nested_value(config, "predict", "save_frames", default=False)
    show_labels = get_nested_value(config, "predict", "show_labels", default=True)
    show_boxes = get_nested_value(config, "predict", "show_boxes", default=True)
    show_conf = get_nested_value(config, "predict", "show_conf", default=True)
    line_width = get_nested_value(config, "predict", "line_width")
    vid_stride = get_nested_value(config, "predict", "vid_stride")
    visualize = get_nested_value(config, "predict", "visualize", default=False)
    retina_masks = get_nested_value(config, "predict", "retina_masks", default=False)
    stream_buffer = get_nested_value(config, "predict", "stream_buffer", default=False)
    embed = get_nested_value(config, "predict", "embed")

    project = get_nested_value(config, "output", "project")
    name = get_nested_value(config, "output", "name")
    verbose = get_nested_value(config, "output", "verbose", default=True)

    print(f"\n{'='*60}")
    print("YOLO Prediction Configuration")
    print(f"{'='*60}")
    print(f"Model:  {model_name}")
    print(f"Source: {source}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(model_name)

    # Build prediction arguments
    predict_args = {
        "source": source,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "show": show,
        "save_txt": save_txt,
        "save_conf": save_conf,
        "save_crop": save_crop,
        "save_frames": save_frames,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "show_boxes": show_boxes,
        "agnostic_nms": agnostic_nms,
        "augment": augment,
        "half": half,
        "max_det": max_det,
        "visualize": visualize,
        "retina_masks": retina_masks,
        "stream_buffer": stream_buffer,
        "verbose": verbose,
    }

    if device is not None:
        predict_args["device"] = device
    if project:
        predict_args["project"] = project
    if name:
        predict_args["name"] = name
    if classes is not None:
        predict_args["classes"] = classes
    if line_width is not None:
        predict_args["line_width"] = line_width
    if vid_stride is not None:
        predict_args["vid_stride"] = vid_stride
    if embed is not None:
        predict_args["embed"] = embed

    # Predict
    results = model.predict(**predict_args)

    print(f"\n{'='*60}")
    print("Prediction Completed!")
    print(f"{'='*60}\n")

    return results


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

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
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
