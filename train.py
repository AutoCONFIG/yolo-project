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
    python train.py --config configs/default.yaml --epochs 50 --batch 32

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

from ultralytics import YOLO


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Training and Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with config file
    python train.py --config configs/default.yaml

    # Train with CLI overrides
    python train.py --config configs/default.yaml --train.epochs 50 --train.batch 32

    # Quick training without config file
    python train.py --mode train --model yolo26n.pt --data coco8.yaml --epochs 100

    # Validate a trained model
    python train.py --mode val --model runs/detect/train/weights/best.pt --data coco8.yaml

    # Resume training
    python train.py --mode train --model runs/detect/train/weights/last.pt --resume
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

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "predict"],
        default=None,
        help="Mode: train, val, or predict",
    )

    # Model settings
    parser.add_argument("--model", type=str, default=None, help="Model path or name")
    parser.add_argument("--model-yaml", type=str, default=None, help="YAML config for building model")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weights path")

    # Data settings
    parser.add_argument("--data", type=str, default=None, help="Dataset config file")

    # Training settings (dot notation for nested config)
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, 0, 0,1, mps")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--resume", action="store_true", help="Resume training")

    # Optimizer settings
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR factor")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")

    # Augmentation settings
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV-Hue")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV-Saturation")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV-Value")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation degrees")
    parser.add_argument("--translate", type=float, default=None, help="Translation")
    parser.add_argument("--scale", type=float, default=None, help="Scaling")
    parser.add_argument("--shear", type=float, default=None, help="Shear")
    parser.add_argument("--perspective", type=float, default=None, help="Perspective")
    parser.add_argument("--flipud", type=float, default=None, help="Flip up-down")
    parser.add_argument("--fliplr", type=float, default=None, help="Flip left-right")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic")
    parser.add_argument("--mixup", type=float, default=None, help="Mixup")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste")

    # Validation settings
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold")
    parser.add_argument("--split", type=str, default=None, help="Dataset split")
    parser.add_argument("--plots", action="store_true", help="Save plots")
    parser.add_argument("--save-json", action="store_true", help="Save JSON")

    # Prediction settings
    parser.add_argument("--source", type=str, default=None, help="Source for prediction")

    # Output settings
    parser.add_argument("--project", type=str, default=None, help="Project name")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--save-period", type=int, default=None, help="Save period")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command line args to nested config dict."""
    config = {}

    # Direct mappings
    if args.mode:
        config["mode"] = args.mode

    # Model config
    model_config = {}
    if args.model:
        model_config["name"] = args.model
    if args.model_yaml:
        model_config["yaml"] = args.model_yaml
    if args.pretrained:
        model_config["pretrained"] = args.pretrained
    if model_config:
        config["model"] = model_config

    # Data config
    data_config = {}
    if args.data:
        data_config["config"] = args.data
    if args.split:
        data_config["split"] = args.split
    if data_config:
        config["data"] = {**config.get("data", {}), **data_config}

    # Train config
    train_config = {}
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch is not None:
        train_config["batch"] = args.batch
    if args.imgsz is not None:
        train_config["imgsz"] = args.imgsz
    if args.device is not None:
        train_config["device"] = args.device
    if args.workers is not None:
        train_config["workers"] = args.workers
    if args.resume:
        train_config["resume"] = True
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
    if train_config:
        config["train"] = {**config.get("train", {}), **train_config}

    # Augmentation config
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
    if args.mosaic is not None:
        aug_config["mosaic"] = args.mosaic
    if args.mixup is not None:
        aug_config["mixup"] = args.mixup
    if args.copy_paste is not None:
        aug_config["copy_paste"] = args.copy_paste
    if aug_config:
        config["augmentation"] = {**config.get("augmentation", {}), **aug_config}

    # Validation config
    val_config = {}
    if args.conf is not None:
        val_config["conf"] = args.conf
    if args.iou is not None:
        val_config["iou"] = args.iou
    if args.plots:
        val_config["plots"] = True
    if args.save_json:
        val_config["save_json"] = True
    if val_config:
        config["validation"] = {**config.get("validation", {}), **val_config}

    # Predict config
    predict_config = {}
    if args.source:
        predict_config["source"] = args.source
    if predict_config:
        config["predict"] = {**config.get("predict", {}), **predict_config}

    # Output config
    output_config = {}
    if args.project:
        output_config["project"] = args.project
    if args.name:
        output_config["name"] = args.name
    if args.save_period is not None:
        output_config["save_period"] = args.save_period
    if args.exist_ok:
        output_config["exist_ok"] = True
    if args.verbose:
        output_config["verbose"] = True
    if output_config:
        config["output"] = {**config.get("output", {}), **output_config}

    return config


def get_nested_value(config: Dict, *keys, default=None):
    """Get nested value from config dict."""
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def train(config: Dict):
    """Train a YOLO model."""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")
    model_yaml = get_nested_value(config, "model", "yaml")
    pretrained = get_nested_value(config, "model", "pretrained")
    data_config = get_nested_value(config, "data", "config", default="coco8.yaml")

    epochs = get_nested_value(config, "train", "epochs", default=100)
    batch = get_nested_value(config, "train", "batch", default=16)
    imgsz = get_nested_value(config, "train", "imgsz", default=640)
    device = get_nested_value(config, "train", "device")
    workers = get_nested_value(config, "train", "workers", default=8)
    resume = get_nested_value(config, "train", "resume", default=False)

    optimizer = get_nested_value(config, "train", "optimizer", default="auto")
    lr0 = get_nested_value(config, "train", "lr0", default=0.01)
    lrf = get_nested_value(config, "train", "lrf", default=0.01)
    momentum = get_nested_value(config, "train", "momentum", default=0.937)
    weight_decay = get_nested_value(config, "train", "weight_decay", default=0.0005)

    project = get_nested_value(config, "output", "project")
    name = get_nested_value(config, "output", "name")
    save_period = get_nested_value(config, "output", "save_period", default=-1)
    exist_ok = get_nested_value(config, "output", "exist_ok", default=False)
    verbose = get_nested_value(config, "output", "verbose", default=True)

    # Print config
    print(f"\n{'='*60}")
    print("YOLO Training Configuration")
    print(f"{'='*60}")
    print(f"Model:      {model_name}")
    print(f"Data:       {data_config}")
    print(f"Epochs:     {epochs}")
    print(f"Batch:      {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device:     {device or 'auto'}")
    print(f"Optimizer:  {optimizer}")
    print(f"Learning rate: {lr0}")
    print(f"{'='*60}\n")

    # Load model
    if model_yaml:
        model = YOLO(model_yaml)
        if pretrained:
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
        "save_period": save_period,
        "exist_ok": exist_ok,
        "verbose": verbose,
        "resume": resume,
    }

    # Add optional args
    if device:
        train_args["device"] = device
    if project:
        train_args["project"] = project
    if name:
        train_args["name"] = name

    # Add augmentation settings
    aug_keys = [
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
        "scale", "shear", "perspective", "flipud", "fliplr",
        "mosaic", "mixup", "copy_paste", "erasing", "cutmix"
    ]
    for key in aug_keys:
        value = get_nested_value(config, "augmentation", key)
        if value is not None:
            train_args[key] = value

    # Train
    results = model.train(**train_args)

    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"Last weights: {results.save_dir}/weights/last.pt")
    print(f"{'='*60}\n")

    return results


def validate(config: Dict):
    """Validate a YOLO model."""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")
    data_config = get_nested_value(config, "data", "config", default="coco8.yaml")
    split = get_nested_value(config, "data", "split", default="val")

    imgsz = get_nested_value(config, "train", "imgsz", default=640)
    batch = get_nested_value(config, "train", "batch", default=16)
    device = get_nested_value(config, "train", "device")

    conf = get_nested_value(config, "validation", "conf", default=0.25)
    iou = get_nested_value(config, "validation", "iou", default=0.7)
    plots = get_nested_value(config, "validation", "plots", default=False)
    save_json = get_nested_value(config, "validation", "save_json", default=False)

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
    }

    if device:
        val_args["device"] = device
    if project:
        val_args["project"] = project
    if name:
        val_args["name"] = name

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

    project = get_nested_value(config, "output", "project")
    name = get_nested_value(config, "output", "name")

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
    }

    if device:
        predict_args["device"] = device
    if project:
        predict_args["project"] = project
    if name:
        predict_args["name"] = name

    # Predict
    results = model.predict(**predict_args)

    print(f"\n{'='*60}")
    print("Prediction Completed!")
    print(f"{'='*60}\n")

    return results


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
