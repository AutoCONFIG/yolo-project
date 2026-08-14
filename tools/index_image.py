#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

# ─── How to run ───
#   python tools/index_image.py
#   uv run tools/index_image.py
# ──────────────────

from __future__ import annotations

import random
from pathlib import Path
from typing import Final


ROOT_DIRECTORY: Final = Path("/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/datasets/road_seg")
IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
LABEL_SUFFIXES: Final = ("_det.txt", "_seg.txt")
TRAIN_RATIO: Final = 0.9
RANDOM_SEED: Final = 36


def main() -> None:
    dataset = ROOT_DIRECTORY.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    images = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and any(path.with_name(f"{path.stem}{suffix}").is_file() for suffix in LABEL_SUFFIXES)
    )
    random.Random(RANDOM_SEED).shuffle(images)
    train_count = int(len(images) * TRAIN_RATIO)
    splits = (("train_seg_det.txt", images[:train_count]), ("val_seg_det.txt", images[train_count:]))
    for name, paths in splits:
        (dataset / name).write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    print(f"完成: 训练集 {train_count} 张，验证集 {len(images) - train_count} 张")
    print(f"输出: {dataset / 'train_seg_det.txt'}")
    print(f"输出: {dataset / 'val_seg_det.txt'}")


if __name__ == "__main__":
    main()
