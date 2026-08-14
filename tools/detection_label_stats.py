#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML>=6.0"]
# ///
# ─── How to run ───
# 1. 修改下面的 CONFIG_PATH
# 2. 在项目根目录运行: python tools/detection_label_stats.py
"""统计 YOLO 检测数据集的图片数量、目标数量和标签类别分布。"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

CONFIG_PATH: Final[Path] = Path(r"configs/datasets/example/detect_example.yaml")
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
)
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test", "minival")


@dataclass(frozen=True, slots=True)
class SplitStats:
    """一个检测数据集划分的统计结果。"""

    name: str
    counts: tuple[tuple[int, int], ...]
    images: int
    label_files: int
    missing_labels: int
    empty_labels: int
    invalid_lines: int

    @property
    def instances(self) -> int:
        """返回有效目标总数。"""
        return sum(count for _, count in self.counts)


def load_config(path: Path) -> tuple[dict[int, str], tuple[tuple[str, tuple[Path, ...]], ...]]:
    """读取 YOLO 数据 YAML 并解析各划分路径。"""
    with path.open(encoding="utf-8", errors="ignore") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射: {path}")

    root = Path(str(data.get("path") or path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    names_value = data.get("names")
    if isinstance(names_value, list):
        names = {index: str(name) for index, name in enumerate(names_value)}
    elif isinstance(names_value, dict):
        names = {int(index): str(name) for index, name in names_value.items()}
    elif isinstance(data.get("nc"), int):
        names = {index: f"class_{index}" for index in range(data["nc"])}
    else:
        names = {}

    splits: list[tuple[str, tuple[Path, ...]]] = []
    for split in SPLITS:
        value = data.get(split)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        sources = tuple(Path(str(item)) if Path(str(item)).is_absolute() else (root / str(item)).resolve() for item in values)
        splits.append((split, sources))
    if not splits:
        raise ValueError("YAML 中未配置 train、val、test 或 minival")
    return names, tuple(splits)


def image_files(source: Path) -> tuple[Path, ...]:
    """从图片目录或 Ultralytics TXT 清单读取图片路径。"""
    if source.is_dir():
        return tuple(
            sorted(
                (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda path: str(path).lower(),
            )
        )
    if not source.is_file():
        raise FileNotFoundError(f"数据源不存在: {source}")
    if source.suffix.lower() != ".txt":
        raise ValueError(f"数据源必须是图片目录或 TXT 清单: {source}")

    parent = str(source.parent) + os.sep
    with source.open(encoding="utf-8") as file:
        lines = (line.strip() for line in file)
        paths = (line.replace("./", parent, 1) if line.startswith("./") else line for line in lines if line)
        return tuple(Path(item) for item in paths if Path(item).suffix.lower() in IMAGE_EXTENSIONS)


def label_path(image: Path) -> Path:
    """按 Ultralytics 规则把图片路径映射为标签路径。"""
    marker = f"{os.sep}images{os.sep}"
    replacement = f"{os.sep}labels{os.sep}"
    return Path(replacement.join(str(image).rsplit(marker, 1))).with_suffix(".txt")


def scan_split(name: str, sources: tuple[Path, ...], names: dict[int, str]) -> SplitStats:
    """统计一个划分中的检测目标和标签质量。"""
    counts: Counter[int] = Counter({class_id: 0 for class_id in names})
    images = tuple(image for source in sources for image in image_files(source))
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
    return SplitStats(name, tuple(sorted(counts.items())), len(images), label_files, missing, empty, invalid)


def merge_stats(stats: tuple[SplitStats, ...]) -> SplitStats:
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


def print_stats(stats: SplitStats, names: dict[int, str]) -> None:
    """将检测标签统计结果打印到控制台。"""
    rows = tuple((names.get(class_id, f"class_{class_id}"), count) for class_id, count in stats.counts)
    width = max((len(name) for name, _ in rows), default=4)
    print(f"\n{'=' * 60}\n数据集划分: {stats.name}\n{'=' * 60}")
    print(f"{'类别':<{width}}  {'数量':>10}  {'占比':>10}\n{'-' * 60}")
    for name, count in rows:
        percentage = count / stats.instances * 100 if stats.instances else 0.0
        print(f"{name:<{width}}  {count:>10}  {percentage:>9.2f}%")
    print(f"{'-' * 60}\n{'目标合计':<{width}}  {stats.instances:>10}  {100.0 if stats.instances else 0.0:>9.2f}%")
    print(
        f"图片: {stats.images} | 标签文件: {stats.label_files} | 缺失标签: {stats.missing_labels} | "
        f"空标签: {stats.empty_labels} | 无效标签行: {stats.invalid_lines}"
    )


def main() -> int:
    """读取 CONFIG_PATH 并输出各划分及总体统计。"""
    try:
        config = CONFIG_PATH.expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(f"数据集 YAML 不存在: {config}")
        names, sources = load_config(config)
        results = tuple(scan_split(split, split_sources, names) for split, split_sources in sources)
        print(f"数据集配置: {config}")
        for stats in results:
            print_stats(stats, names)
        if len(results) > 1:
            print_stats(merge_stats(results), names)
        return 0 if any(stats.images for stats in results) else 1
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
