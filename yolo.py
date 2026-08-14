"""
YOLO Unified CLI Entry Point
=============================

Usage:

    python yolo.py configs/train/chaoyuan.yaml
    python yolo.py configs/validate/val.yaml
    python yolo.py configs/predict/chaoyuan.yaml
    python yolo.py configs/export/example/onnx/detect_example.yaml

Mode is auto-detected from the 'mode' field in the YAML config file.
"""

import sys
from pathlib import Path


MODES = {
    "train": "commands.train",
    "val": "commands.val",
    "predict": "commands.predict",
    "track": "commands.track",
    "export": "commands.export",
}


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return
    if len(sys.argv) != 2:
        print("Error: expected exactly one YAML config path")
        sys.exit(1)

    import yaml

    config_path = Path(sys.argv[1])
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        print(f"Error: config must be a YAML file: {config_path}")
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mode = cfg.get("mode") if isinstance(cfg, dict) else None
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)

    if mode not in MODES:
        print(f"Error: invalid or missing 'mode' in config (got: {mode})")
        print(f"Valid modes: {', '.join(MODES.keys())}")
        sys.exit(1)

    sys.argv = ["yolo.py", "--config", str(config_path)]
    import importlib
    importlib.import_module(MODES[mode]).main()


if __name__ == "__main__":
    main()
