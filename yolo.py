"""
YOLO Unified CLI Entry Point
=============================
Route commands to the appropriate module:

    python yolo.py train   --config configs/train/chaoyuan.yaml
    python yolo.py val     --config configs/validate/val.yaml
    python yolo.py predict --model best.pt --source images/
    python yolo.py track   --model best.pt --input video.mp4
    python yolo.py export  --model best.pt --format onnx

Each command module can also be invoked directly:

    python -m commands.train   --config ...
    python -m commands.val     --config ...
    python -m commands.predict --model ...
    python -m commands.track   --model ...
    python -m commands.export  --model ...
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
        print("\nAvailable modes:")
        for mode in MODES:
            print(f"  {mode}")
        print(f"\nUse: python yolo.py <mode> --help  for mode-specific options")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    mode = sys.argv[1]

    if mode not in MODES:
        print(f"Unknown mode: {mode}")
        print(f"Available modes: {', '.join(MODES.keys())}")
        sys.exit(1)

    # Shift argv so the target module's argparse works normally.
    # sys.argv[0] becomes a virtual program name; sys.argv[1:] = remaining args.
    sys.argv = [f"yolo.py {mode}"] + sys.argv[2:]

    module_name = MODES[mode]
    import importlib
    mod = importlib.import_module(module_name)
    mod.main()


if __name__ == "__main__":
    main()