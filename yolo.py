"""
YOLO Unified CLI Entry Point
=============================

Usage:

    python yolo.py --config configs/train/chaoyuan.yaml
    python yolo.py --config configs/validate/val.yaml
    python yolo.py --config configs/predict/chaoyuan.yaml
    python yolo.py --config configs/export/example/onnx/detect_example.yaml

Mode is auto-detected from the 'mode' field in the YAML config file.
"""

import sys


MODES = {
    "train": "commands.train",
    "val": "commands.val",
    "predict": "commands.predict",
    "track": "commands.track",
    "export": "commands.export",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    import yaml

    config_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--config", "-c") and i < len(sys.argv) - 1:
            config_path = sys.argv[i + 1]
            break

    if config_path is None:
        print("Error: --config is required")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mode = cfg.get("mode")
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)

    if mode not in MODES:
        print(f"Error: invalid or missing 'mode' in config (got: {mode})")
        print(f"Valid modes: {', '.join(MODES.keys())}")
        sys.exit(1)

    sys.argv = ["yolo.py", mode] + sys.argv[1:]
    import importlib
    importlib.import_module(MODES[mode]).main()


if __name__ == "__main__":
    main()
