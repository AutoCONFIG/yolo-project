#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic>=2.12",
#     "typer>=0.16",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run tools/convert_coco_seg_to_yolo.py SOURCE OUTPUT
# 3. Or make executable and run:
#      chmod +x tools/convert_coco_seg_to_yolo.py && ./tools/convert_coco_seg_to_yolo.py SOURCE OUTPUT
# ──────────────────

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class CocoImage(FrozenModel):
    id: int
    file_name: str
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class CocoAnnotation(FrozenModel):
    image_id: int
    category_id: int
    segmentation: list[list[float]]
    iscrowd: bool = False


class CocoCategory(FrozenModel):
    id: int
    name: str


class CocoDocument(FrozenModel):
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]


class DatasetFormatError(RuntimeError):
    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}")


def read_document(path: Path) -> CocoDocument:
    try:
        document = CocoDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise DatasetFormatError(path, f"COCO JSON 格式错误: {error}") from error
    if len(document.images) != 1:
        raise DatasetFormatError(path, f"每个 JSON 必须只描述一张图片，实际为 {len(document.images)} 张")
    return document


def find_label_root(path: Path) -> Path:
    label_root = next((parent for parent in path.parents if parent.name == "labels"), None)
    if label_root is None:
        raise DatasetFormatError(path, "JSON 必须位于名为 labels 的目录下")
    return label_root


def image_relative_path(path: Path, image: CocoImage) -> Path:
    relative = Path(image.file_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetFormatError(path, f"非法图片路径: {image.file_name}")
    return relative


def annotation_lines(
    path: Path,
    document: CocoDocument,
    class_ids: dict[int, int],
) -> list[str]:
    image = document.images[0]
    lines: list[str] = []
    for annotation in document.annotations:
        if annotation.image_id != image.id:
            raise DatasetFormatError(path, f"annotation.image_id={annotation.image_id} 与图片 ID 不一致")
        if annotation.iscrowd:
            continue
        if annotation.category_id not in class_ids:
            raise DatasetFormatError(path, f"未知 category_id: {annotation.category_id}")
        for polygon in annotation.segmentation:
            if len(polygon) < 6 or len(polygon) % 2:
                raise DatasetFormatError(path, "segmentation 多边形至少需要 3 个点且坐标数必须为偶数")
            points = [
                (
                    min(max(polygon[index], 0.0), image.width) / image.width,
                    min(max(polygon[index + 1], 0.0), image.height) / image.height,
                )
                for index in range(0, len(polygon), 2)
            ]
            if len(set(points)) < 3:
                raise DatasetFormatError(path, "多边形裁剪到图像边界后不足 3 个不同的点")
            coordinates = " ".join(f"{value:.6f}" for point in points for value in point)
            lines.append(f"{class_ids[annotation.category_id]} {coordinates}")
    return lines


def write_once_or_same(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise DatasetFormatError(path, "输出文件已存在且内容不同")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def link_once_or_same(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise DatasetFormatError(destination, "输出图片已存在且不是指向源图片的符号链接")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def convert_dataset(source: Path, output: Path) -> tuple[int, int, tuple[str, ...]]:
    json_files = sorted(source.rglob("*.json"))
    if not json_files:
        raise DatasetFormatError(source, "未找到 JSON 标签")

    category_names: dict[int, str] = {}
    for json_path in json_files:
        for category in read_document(json_path).categories:
            known_name = category_names.get(category.id)
            if known_name is not None and known_name != category.name:
                raise DatasetFormatError(json_path, f"category_id={category.id} 对应了多个名称")
            category_names[category.id] = category.name
    if not category_names:
        raise DatasetFormatError(source, "所有 JSON 都缺少 categories")
    categories = tuple(sorted(category_names.items()))
    class_ids = {category_id: index for index, (category_id, _) in enumerate(categories)}
    polygon_count = 0

    for json_path in json_files:
        document = read_document(json_path)
        label_root = find_label_root(json_path)
        image_relative = image_relative_path(json_path, document.images[0])
        source_image = label_root.parent / "images" / image_relative
        if not source_image.is_file():
            raise DatasetFormatError(json_path, f"找不到对应图片: {source_image}")
        prefix = Path() if label_root == source else label_root.parent.relative_to(source)
        destination_image = output / prefix / image_relative
        destination_label = destination_image.with_name(f"{destination_image.stem}_seg.txt")
        lines = annotation_lines(json_path, document, class_ids)
        link_once_or_same(source_image, destination_image)
        write_once_or_same(destination_label, "".join(f"{line}\n" for line in lines))
        polygon_count += len(lines)

    names = tuple(name for _, name in categories)
    write_once_or_same(output / "classes.txt", "".join(f"{name}\n" for name in names))
    return len(json_files), polygon_count, names


def main(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
) -> None:
    """Convert SOURCE into same-directory image and ``*_seg.txt`` pairs at OUTPUT."""
    if source == output:
        raise typer.BadParameter("SOURCE 与 OUTPUT 不能相同")
    try:
        image_count, polygon_count, names = convert_dataset(source, output)
    except DatasetFormatError as error:
        typer.echo(f"错误: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"完成: {image_count} 张图片，{polygon_count} 个多边形，{len(names)} 个类别")
    typer.echo(f"输出: {output}")


if __name__ == "__main__":
    typer.run(main)
