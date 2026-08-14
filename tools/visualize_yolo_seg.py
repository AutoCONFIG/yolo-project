#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=2.0",
#     "opencv-python>=4.10",
#     "typer>=0.16",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Open a converted YOLO label or launch the native file chooser:
#      uv run tools/visualize_yolo_seg.py [LABEL_seg.txt]
# 3. Save one rendered preview without opening a window:
#      uv run tools/visualize_yolo_seg.py LABEL_seg.txt --save-preview preview.jpg
# ──────────────────

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, cast

import cv2
import numpy as np
import typer
from numpy.typing import NDArray

WINDOW_NAME: Final = "YOLO Segmentation Viewer"
HEADER_HEIGHT: Final = 42
IMAGE_SUFFIXES: Final = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PALETTE: Final = (
    (0, 159, 230),
    (233, 180, 86),
    (115, 158, 0),
    (66, 228, 240),
    (178, 114, 0),
    (0, 94, 213),
    (167, 121, 204),
    (128, 128, 128),
)


class ViewerError(RuntimeError):
    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f"{path}: {detail}")


def find_image(label_path: Path) -> Path:
    if not label_path.stem.endswith("_seg"):
        raise ViewerError(label_path, "标签文件名必须以 _seg.txt 结尾")
    image_stem = label_path.stem.removesuffix("_seg")
    matches = [
        path
        for path in label_path.parent.iterdir()
        if path.is_file() and path.stem == image_stem and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise ViewerError(label_path, f"同目录应有且仅有一张名为 {image_stem} 的图片，实际找到 {len(matches)} 张")
    return matches[0]


def load_image(path: Path) -> NDArray[np.uint8]:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ViewerError(path, "图片无法读取")
    return cast(NDArray[np.uint8], image)


def read_segments(path: Path) -> list[tuple[int, NDArray[np.float64]]]:
    segments: list[tuple[int, NDArray[np.float64]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        try:
            class_id = int(values[0])
            coordinates = np.asarray([float(value) for value in values[1:]], dtype=np.float64)
        except ValueError as error:
            raise ViewerError(path, f"第 {line_number} 行含有非数值字段") from error
        if class_id < 0 or len(coordinates) < 6 or len(coordinates) % 2:
            raise ViewerError(path, f"第 {line_number} 行不是有效的 YOLO segmentation 标签")
        if np.any((coordinates < 0) | (coordinates > 1)):
            raise ViewerError(path, f"第 {line_number} 行存在 [0, 1] 之外的坐标")
        segments.append((class_id, coordinates.reshape(-1, 2)))
    return segments


def dataset_root(label_path: Path) -> Path:
    return next((parent for parent in label_path.parents if (parent / "classes.txt").is_file()), label_path.parent)


def class_names(label_path: Path) -> tuple[str, ...]:
    classes_file = dataset_root(label_path) / "classes.txt"
    return tuple(classes_file.read_text(encoding="utf-8").splitlines()) if classes_file.is_file() else ()


def render(label_path: Path, position: int = 1, total: int = 1) -> NDArray[np.uint8]:
    image = load_image(find_image(label_path))
    segments = read_segments(label_path)
    names = class_names(label_path)
    overlay = image.copy()
    height, width = image.shape[:2]
    outlines: list[tuple[NDArray[np.int32], tuple[int, int, int], str]] = []

    for class_id, normalized in segments:
        points = np.rint(normalized * np.array([width, height])).astype(np.int32)
        color = PALETTE[class_id % len(PALETTE)]
        name = names[class_id] if class_id < len(names) else f"class_{class_id}"
        cv2.fillPoly(overlay, [points], color)
        outlines.append((points, color, name))

    rendered = cv2.addWeighted(overlay, 0.42, image, 0.58, 0)
    for points, color, name in outlines:
        cv2.polylines(rendered, [points], True, color, 2, cv2.LINE_AA)
        anchor = tuple(np.mean(points, axis=0).astype(np.int32).tolist())
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        left = min(max(anchor[0] - text_size[0] // 2, 0), width - text_size[0] - 8)
        top = min(max(anchor[1] - text_size[1] // 2 - 4, 0), height - text_size[1] - 8)
        text_color = (20, 20, 20) if sum(color) > 380 else (245, 245, 245)
        cv2.rectangle(rendered, (left, top), (left + text_size[0] + 8, top + text_size[1] + 8), color, -1)
        cv2.putText(rendered, name, (left + 4, top + text_size[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)

    header = np.full((HEADER_HEIGHT, width, 3), (24, 26, 29), dtype=np.uint8)
    status = f"{position}/{total}  polygons: {len(outlines)}  [A/Left] previous  [D/Right] next  [O] open  [Q] quit"
    cv2.putText(header, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 238, 242), 1, cv2.LINE_AA)
    return cast(NDArray[np.uint8], np.vstack((header, rendered)))


def choose_label() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(title="Choose a YOLO segmentation label", filetypes=[("YOLO", "*_seg.txt")])
    root.destroy()
    return Path(selected) if selected else None


def browse(selected: Path) -> None:
    while True:
        files = sorted(dataset_root(selected).rglob("*_seg.txt"))
        index = files.index(selected)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1400, 820)
        while True:
            cv2.imshow(WINDOW_NAME, render(files[index], index + 1, len(files)))
            key = cv2.waitKeyEx(0)
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return
            if key in (81, 2424832, 65361, ord("a"), ord("A")):
                index = (index - 1) % len(files)
            if key in (32, 83, 2555904, 65363, ord("d"), ord("D")):
                index = (index + 1) % len(files)
            if key in (ord("o"), ord("O")):
                replacement = choose_label()
                if replacement is not None:
                    selected = replacement.resolve()
                    cv2.destroyAllWindows()
                    break


def save_image(path: Path, image: NDArray[np.uint8]) -> None:
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ViewerError(path, "预览文件扩展名必须为 jpg、jpeg 或 png")
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ViewerError(path, "预览图片编码失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def main(
    label_path: Annotated[Path | None, typer.Argument(exists=True, dir_okay=False, resolve_path=True)] = None,
    save_preview: Annotated[Path | None, typer.Option("--save-preview", dir_okay=False, resolve_path=True)] = None,
) -> None:
    """Browse converted ``*_seg.txt`` labels or save one rendered preview."""
    selected = label_path or choose_label()
    if selected is None:
        raise typer.Exit()
    try:
        if save_preview is not None:
            save_image(save_preview, render(selected))
            typer.echo(f"已保存: {save_preview}")
            return
        browse(selected)
    except (OSError, ViewerError) as error:
        typer.echo(f"错误: {error}", err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    typer.run(main)
