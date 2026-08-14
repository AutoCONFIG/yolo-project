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
#      uv run tools/convert_coco_set_seg_to_yolo.py SOURCE OUTPUT
# 3. Or make executable and run:
#      chmod +x tools/convert_coco_set_seg_to_yolo.py && ./tools/convert_coco_set_seg_to_yolo.py SOURCE OUTPUT
# ──────────────────
#
# 与 convert_coco_seg_to_yolo.py 的区别：
#   - 本脚本处理"set 集 JSON"布局：一个 JSON 包含多张图的 annotations，
#     与同级 images/ 目录并列（如 certain/ 下 trainset.json / valset.json）。
#   - 读取规则与远端算法一致：源图片 = JSON所在目录 / "images" / basename(file_name)，
#     同时兼容裸文件名（"000.jpg"）和带 images/ 前缀（"images/foo.png"）两种写法。
#   - convert_coco_seg_to_yolo.py 处理的是"每 JSON 一图 + labels/ 目录"布局，保持不变。

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import json

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
    if not document.images:
        raise DatasetFormatError(path, "JSON 不包含任何图片")
    return document


def looks_like_coco(path: Path) -> bool:
    """判断 JSON 是否为 COCO 标注文档（同时含 images/annotations/categories 三个字段）。

    数据集目录里可能混入非标注 JSON（如打包流程生成的 processing_summary.json），
    用结构判断而非文件名黑名单，对这类文件鲁棒。缺字段的不是标注，跳过即可。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(raw, dict) and {"images", "annotations", "categories"} <= raw.keys()


def resolve_source_image(
    json_path: Path,
    image: CocoImage,
    index_cache: dict[Path, dict[str, Path]],
) -> tuple[Path, Path]:
    """定位源图片，返回 (源图片路径, images根目录)。

    读取规则与远端算法 ``img_path / file_name`` 语义一致，但远端各数据集的
    ``img_path`` 配置不同，且部分数据集（如 snap_picture）把图片按摄像头位置分了
    子目录。这里用回退链兼容所有已知写法，无需外部配置：

    1. ``json_dir / file_name``        —— file_name 含相对路径（"images/foo.png"、
                                          "images/<摄像头>/foo.jpg"）时直接命中。
    2. ``json_dir / "images" / file_name`` —— file_name 为裸文件名且图片直接在
                                          同级 images/ 下时命中。
    3. 在 ``json_dir / "images"`` 下按 basename 递归查找 —— file_name 为裸文件名
                                          但图片在 images/ 的子目录中（snap_picture）时命中。

    images 根目录统一为 ``json_dir / "images"``；输出时保留图片相对该根的子路径，
    避免不同子目录下同名文件冲突。
    """
    relative = Path(image.file_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise DatasetFormatError(json_path, f"非法图片路径: {image.file_name}")
    json_dir = json_path.parent
    images_root = json_dir / "images"

    candidates = [json_dir / relative, images_root / relative]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, images_root

    # 裸文件名但图片嵌套在 images/ 子目录下：按 basename 在 images_root 下查一次。
    index = index_cache.get(images_root)
    if index is None:
        index = {}
        if images_root.is_dir():
            for path in images_root.rglob("*"):
                if path.is_file():
                    index.setdefault(path.name, path)
        index_cache[images_root] = index
    hit = index.get(relative.name)
    if hit is not None:
        return hit, images_root

    raise DatasetFormatError(json_path, f"找不到对应图片: {image.file_name}")



def image_lines(
    path: Path,
    image: CocoImage,
    annotations: list[CocoAnnotation],
    class_ids: dict[int, int],
) -> list[str]:
    lines: list[str] = []
    for annotation in annotations:
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
    all_json_files = sorted(source.rglob("*.json"))
    if not all_json_files:
        raise DatasetFormatError(source, "未找到 JSON 标签")
    # 跳过非 COCO 标注 JSON（如打包流程生成的 processing_summary.json）。
    json_files: list[Path] = []
    skipped: list[Path] = []
    for json_path in all_json_files:
        if looks_like_coco(json_path):
            json_files.append(json_path)
        else:
            skipped.append(json_path)
    if skipped:
        typer.echo(f"跳过非标注 JSON（{len(skipped)} 个）: {', '.join(str(p) for p in skipped)}", err=True)
    if not json_files:
        raise DatasetFormatError(source, "未找到 COCO 标注 JSON")

    # 第一遍：聚合所有 JSON 的 categories，做跨文件一致性校验。
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

    image_count = 0
    polygon_count = 0
    index_cache: dict[Path, dict[str, Path]] = {}
    # 第二遍：按图生成 *_seg.txt 并符号链接图片。
    for json_path in json_files:
        document = read_document(json_path)
        annotations_by_image: dict[int, list[CocoAnnotation]] = {}
        for annotation in document.annotations:
            annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

        seen_ids: set[int] = set()
        for image in document.images:
            if image.id in seen_ids:
                raise DatasetFormatError(json_path, f"重复的 image id: {image.id}")
            seen_ids.add(image.id)

            source_image, images_root = resolve_source_image(json_path, image, index_cache)
            # 保留图片相对 images 根的子路径（snap_picture 这类按摄像头分目录的不会丢层级）。
            try:
                within_images = source_image.relative_to(images_root)
            except ValueError:
                within_images = Path(source_image.name)

            prefix = Path() if json_path.parent == source else json_path.parent.relative_to(source)
            destination_image = output / prefix / "images" / within_images
            destination_label = destination_image.with_name(f"{destination_image.stem}_seg.txt")

            lines = image_lines(json_path, image, annotations_by_image.get(image.id, []), class_ids)
            link_once_or_same(source_image, destination_image)
            write_once_or_same(destination_label, "".join(f"{line}\n" for line in lines))
            image_count += 1
            polygon_count += len(lines)

    names = tuple(name for _, name in categories)
    write_once_or_same(output / "classes.txt", "".join(f"{name}\n" for name in names))
    return image_count, polygon_count, names


def main(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
) -> None:
    """Convert set-JSON COCO segmentation SOURCE into image and ``*_seg.txt`` pairs at OUTPUT."""
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
