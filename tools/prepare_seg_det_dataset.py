#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

# ─── How to run ───
#   Edit the paths below, then run: python tools/prepare_seg_det_dataset.py
# ──────────────────

from __future__ import annotations

import filecmp
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final


DETECT_IMAGES_DIRECTORY: Final[Path | None] = Path(
    "/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/datasets/detect/20260727_11分类任务"
)
DETECT_LABELS_DIRECTORY: Final[Path | None] = DETECT_IMAGES_DIRECTORY
SEGMENT_IMAGES_DIRECTORY: Final[Path | None] = Path(
    "/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/datasets/seg/highway_road"
)
SEGMENT_LABELS_DIRECTORY: Final[Path | None] = SEGMENT_IMAGES_DIRECTORY
OUTPUT_DIRECTORY: Final = Path("/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/datasets/road_seg")
IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


class Task(StrEnum):
    DETECT = "det"
    SEGMENT = "seg"


class MergeError(RuntimeError):
    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}")


@dataclass(frozen=True, slots=True)
class TaskSource:
    images: Path
    labels: Path


def normalize_detection_label(path: Path, content: bytes) -> bytes:
    lines = []
    for line_number, line in enumerate(content.splitlines(), 1):
        fields = line.split()
        if fields and len(fields) not in {5, 6}:
            raise MergeError(path, f"第 {line_number} 行检测标签应为 5 列，或带置信度的 6 列")
        lines.append(b" ".join(fields[:5]))
    return b"\n".join(lines) + (b"\n" if content.endswith((b"\n", b"\r")) else b"")


def prepare_task(task: Task, source: TaskSource, output: Path) -> None:
    image_count = 0
    missing_count = 0
    images = sorted(
        path for path in source.images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image in images:
        relative = image.relative_to(source.images)
        plain_label = source.labels / relative.with_suffix(".txt")
        normalized_label = plain_label.with_name(f"{plain_label.stem}_{task.value}.txt")
        source_label = plain_label if plain_label.is_file() else normalized_label
        if not source_label.is_file():
            missing_count += 1
            continue
        destination_image = output / relative
        if destination_image.exists() or destination_image.is_symlink():
            if not destination_image.is_file() or not filecmp.cmp(image, destination_image, shallow=False):
                raise MergeError(destination_image, "输出位置已有另一张不同的图片")
        else:
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_image.symlink_to(image.resolve())

        destination_label = destination_image.with_name(f"{destination_image.stem}_{task.value}.txt")
        source_content = source_label.read_bytes()
        content = normalize_detection_label(source_label, source_content) if task is Task.DETECT else source_content
        if destination_label.exists():
            destination_content = destination_label.read_bytes()
            if destination_content == source_content and destination_content != content:
                destination_label.write_bytes(content)
            elif destination_content != content:
                raise MergeError(destination_label, "同任务标签已存在且内容不同")
        else:
            destination_label.write_bytes(content)
        image_count += 1
    if image_count == 0:
        raise MergeError(source.labels, "没有找到与图片相对路径对应的 .txt 标签")
    print(f"完成 {task.value}: 合并 {image_count} 张，跳过 {missing_count} 张无标签图片")


def main() -> None:
    configured_count = 0
    try:
        configurations = (
            (Task.DETECT, DETECT_IMAGES_DIRECTORY, DETECT_LABELS_DIRECTORY),
            (Task.SEGMENT, SEGMENT_IMAGES_DIRECTORY, SEGMENT_LABELS_DIRECTORY),
        )
        for task, images, labels in configurations:
            if images is None and labels is None:
                continue
            if images is None or labels is None:
                raise MergeError(Path(__file__), f"{task.value} 的图片目录和标签目录必须同时配置")
            if not images.is_dir():
                raise MergeError(images, "图片目录不存在")
            if not labels.is_dir():
                raise MergeError(labels, "标签目录不存在")
            prepare_task(task, TaskSource(images.resolve(), labels.resolve()), OUTPUT_DIRECTORY.resolve())
            configured_count += 1
        if configured_count == 0:
            raise MergeError(Path(__file__), "请先在文件开头配置检测或分割数据路径")
    except MergeError as error:
        print(f"错误: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"输出: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
