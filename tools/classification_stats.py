#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML>=6.0"]
# ///
# ─── How to run ───
# 1. 修改下面的 INPUT_PATH
# 2. 在项目根目录运行: python tools/classification_stats.py
"""统计 Ultralytics YOLO 本地数据集的类别分布。"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

# 可填写分类数据集目录、标签型数据集目录、data.yaml 或图片清单 TXT。
INPUT_PATH: Final[Path] = Path(r"datasets/classify")

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
)
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test", "minival")
CLASSIFICATION_VAL_NAMES: Final[tuple[str, ...]] = ("val", "validation", "valid")


@dataclass(frozen=True, slots=True)
class SplitStats:
    """一个数据集划分的不可变统计结果。"""

    name: str
    counts: tuple[tuple[int, int], ...]
    images: int
    label_files: int = 0
    missing_labels: int = 0
    empty_labels: int = 0
    invalid_lines: int = 0

    @property
    def instances(self) -> int:
        """返回该划分的类别实例总数。"""
        return sum(count for _, count in self.counts)


class DatasetInputError(Exception):
    """表示无法解析本地 YOLO 数据集输入。"""


def image_files(directory: Path) -> tuple[Path, ...]:
    """递归返回 Ultralytics 支持的本地图片文件。"""
    return tuple(
        sorted(
            (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: str(path).lower(),
        )
    )


def classification_splits(root: Path) -> tuple[tuple[str, Path], ...]:
    """识别 Ultralytics 分类数据集的 train/val/test 目录。"""
    train = root / "train"
    if not train.is_dir():
        return ()
    val = next((root / name for name in CLASSIFICATION_VAL_NAMES if (root / name).is_dir()), None)
    return tuple(
        (name, path)
        for name, path in (("train", train), ("val", val), ("test", root / "test"))
        if path is not None and path.is_dir()
    )


def scan_classification(split: str, directory: Path, class_ids: dict[str, int]) -> SplitStats:
    """按图片的一级类别目录统计分类数据集。"""
    counts: Counter[int] = Counter({class_id: 0 for class_id in class_ids.values()})
    total_images = 0
    for class_dir in sorted((path for path in directory.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        files = image_files(class_dir)
        counts[class_ids.setdefault(class_dir.name, len(class_ids))] += len(files)
        total_images += len(files)
    return SplitStats(split, tuple(sorted(counts.items())), total_images)


def load_yaml(path: Path) -> tuple[Path, dict[int, str], tuple[tuple[str, tuple[Path, ...]], ...]]:
    """按 Ultralytics YAML 路径规则读取类别名称和数据划分。"""
    with path.open(encoding="utf-8", errors="ignore") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise DatasetInputError(f"YAML 顶层必须是映射: {path}")

    root_value = raw.get("path")
    root = Path(str(root_value)) if root_value else path.parent
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    names_value = raw.get("names")
    names: dict[int, str] = {}
    if isinstance(names_value, list):
        names = {index: str(name) for index, name in enumerate(names_value)}
    elif isinstance(names_value, dict):
        names = {int(index): str(name) for index, name in names_value.items()}
    elif isinstance(raw.get("nc"), int):
        names = {index: f"class_{index}" for index in range(raw["nc"])}

    splits: list[tuple[str, tuple[Path, ...]]] = []
    for split in SPLITS:
        value = raw.get(split)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        paths = tuple(Path(str(item)) if Path(str(item)).is_absolute() else (root / str(item)).resolve() for item in values)
        splits.append((split, paths))
    return root, names, tuple(splits)


def images_from_source(source: Path) -> tuple[Path, ...]:
    """像 Ultralytics BaseDataset 一样读取图片目录或 TXT 图片清单。"""
    if source.is_dir():
        return image_files(source)
    if not source.is_file():
        raise DatasetInputError(f"数据源不存在: {source}")
    if source.suffix.lower() != ".txt":
        raise DatasetInputError(f"数据源必须是图片目录或 TXT 清单: {source}")

    parent = str(source.parent) + os.sep
    with source.open(encoding="utf-8") as file:
        lines = (line.strip() for line in file)
        paths = (line.replace("./", parent, 1) if line.startswith("./") else line for line in lines if line)
        return tuple(Path(path) for path in paths if Path(path).suffix.lower() in IMAGE_EXTENSIONS)


def label_path(image: Path) -> Path:
    """按 Ultralytics img2label_paths 规则把图片路径映射为标签路径。"""
    marker = f"{os.sep}images{os.sep}"
    replacement = f"{os.sep}labels{os.sep}"
    return Path(replacement.join(str(image).rsplit(marker, 1))).with_suffix(".txt")


def scan_labels(split: str, sources: tuple[Path, ...], names: dict[int, str]) -> SplitStats:
    """统计检测、分割、关键点和 OBB 标签每行首列的类别实例。"""
    counts: Counter[int] = Counter({class_id: 0 for class_id in names})
    images = tuple(image for source in sources for image in images_from_source(source))
    label_files = missing = empty = invalid = 0
    for image in images:
        label = label_path(image)
        if not label.is_file():
            missing += 1
            continue
        label_files += 1
        with label.open(encoding="utf-8", errors="ignore") as file:
            lines = tuple(line.strip() for line in file if line.strip())
        if not lines:
            empty += 1
            continue
        for line in lines:
            token = line.split(maxsplit=1)[0]
            try:
                value = float(token)
                class_id = int(value)
                if value != class_id or class_id < 0 or (names and class_id not in names):
                    raise ValueError
            except ValueError:
                invalid += 1
                continue
            counts[class_id] += 1
    return SplitStats(split, tuple(sorted(counts.items())), len(images), label_files, missing, empty, invalid)


def print_distribution(stats: SplitStats, names: dict[int, str]) -> None:
    """打印一个数据集划分的类别分布和数据质量摘要。"""
    rows = tuple((names.get(class_id, f"class_{class_id}"), count) for class_id, count in stats.counts)
    width = max((len(name) for name, _ in rows), default=4)
    print(f"\n{'=' * 60}\n数据集划分: {stats.name}\n{'=' * 60}")
    print(f"{'类别':<{width}}  {'数量':>10}  {'占比':>10}\n{'-' * 60}")
    for name, count in rows:
        percentage = count / stats.instances * 100 if stats.instances else 0.0
        print(f"{name:<{width}}  {count:>10}  {percentage:>9.2f}%")
    print(f"{'-' * 60}\n{'合计':<{width}}  {stats.instances:>10}  {100.0 if stats.instances else 0.0:>9.2f}%")
    print(f"图片: {stats.images}")
    if stats.label_files or stats.missing_labels or stats.empty_labels or stats.invalid_lines:
        print(
            f"标签文件: {stats.label_files} | 缺失标签: {stats.missing_labels} | "
            f"空标签: {stats.empty_labels} | 无效标签行: {stats.invalid_lines}"
        )


def combined_stats(stats: tuple[SplitStats, ...]) -> SplitStats:
    """合并所有数据集划分。"""
    counts: Counter[int] = Counter()
    for item in stats:
        counts.update(dict(item.counts))
    return SplitStats(
        "总体（所有划分合并）",
        tuple(sorted(counts.items())),
        sum(item.images for item in stats),
        sum(item.label_files for item in stats),
        sum(item.missing_labels for item in stats),
        sum(item.empty_labels for item in stats),
        sum(item.invalid_lines for item in stats),
    )


def run(input_path: Path) -> int:
    """解析输入、执行统计并打印结果。"""
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise DatasetInputError(f"输入路径不存在: {path}")

    class_dirs = classification_splits(path) if path.is_dir() else ()
    if class_dirs:
        class_ids: dict[str, int] = {}
        results = tuple(scan_classification(split, directory, class_ids) for split, directory in class_dirs)
        names = {class_id: name for name, class_id in class_ids.items()}
        kind = "classify"
    else:
        yaml_path = path if path.suffix.lower() in {".yaml", ".yml"} else None
        if path.is_dir() and yaml_path is None:
            yaml_files = tuple(path.glob("*.yaml")) + tuple(path.glob("*.yml"))
            yaml_path = yaml_files[0] if len(yaml_files) == 1 else None
        if yaml_path:
            root, names, sources = load_yaml(yaml_path)
        elif path.suffix.lower() == ".txt":
            root, names, sources = path.parent, {}, ((path.stem, (path,)),)
        else:
            root, names = path, {}
            sources = tuple(
                (split, (candidates[0],))
                for split in SPLITS
                if (candidates := tuple(candidate for candidate in (root / f"{split}.txt", root / "images" / split) if candidate.exists()))
            )
        if not sources:
            raise DatasetInputError(f"未找到 YOLO 数据划分: {path}")
        results = tuple(scan_labels(split, split_sources, names) for split, split_sources in sources)
        kind = "detect/segment/pose/obb"

    print(f"数据集: {path}\n识别类型: {kind}")
    for stats in results:
        print_distribution(stats, names)
    if len(results) > 1:
        print_distribution(combined_stats(results), names)
    return 0 if any(stats.images for stats in results) else 1


def main() -> int:
    """命令行入口。"""
    try:
        return run(INPUT_PATH)
    except (DatasetInputError, OSError, yaml.YAMLError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
