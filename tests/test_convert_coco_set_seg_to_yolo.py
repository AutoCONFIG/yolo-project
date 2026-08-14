from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "convert_coco_set_seg_to_yolo.py"


def test_converts_set_coco_polygon_to_yolo(tmp_path: Path) -> None:
    source = tmp_path / "source"
    images_dir = source / "batch" / "images"
    images_dir.mkdir(parents=True)

    # 两张图：一张用裸文件名，一张用 "images/" 前缀，覆盖两种 file_name 写法。
    (images_dir / "frame.jpg").write_bytes(b"image-a")
    (images_dir / "scene.png").write_bytes(b"image-b")

    json_path = source / "batch" / "trainset.json"
    json_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "frame.jpg", "width": 100, "height": 50},
                    {"id": 2, "file_name": "images/scene.png", "width": 200, "height": 100},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 5,
                        "segmentation": [[-10, -5, 100, 0, 120, 50]],
                        "bbox": [-10, -5, 130, 55],
                        "area": 3000,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 2,
                        "segmentation": [[0, 0, 200, 0, 200, 100]],
                        "bbox": [0, 0, 200, 100],
                        "area": 20000,
                    },
                    # 第二张图上同一类别的第二个多边形。
                    {
                        "id": 3,
                        "image_id": 2,
                        "category_id": 5,
                        "segmentation": [[0, 0, 50, 0, 50, 25]],
                        "bbox": [0, 0, 50, 25],
                        "area": 1250,
                    },
                ],
                "categories": [
                    {"id": 2, "name": "road", "supercategory": "object"},
                    {"id": 5, "name": "guardrail", "supercategory": "object"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # 类别按 id 升序：road(2)->0, guardrail(5)->1
    assert (output / "batch" / "images" / "frame_seg.txt").read_text(encoding="utf-8") == (
        "1 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000\n"
    )
    scene_lines = (output / "batch" / "images" / "scene_seg.txt").read_text(encoding="utf-8").splitlines()
    assert scene_lines == [
        "0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000",
        "1 0.000000 0.000000 0.250000 0.000000 0.250000 0.250000",
    ]
    assert (output / "batch" / "images" / "frame.jpg").resolve() == (images_dir / "frame.jpg").resolve()
    assert (output / "batch" / "images" / "scene.png").resolve() == (images_dir / "scene.png").resolve()
    assert (output / "classes.txt").read_text(encoding="utf-8") == "road\nguardrail\n"
    assert "2 张图片，3 个多边形，2 个类别" in result.stdout


def test_empty_label_image_writes_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    images_dir = source / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "neg.jpg").write_bytes(b"image")

    (source / "valset.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "neg.jpg", "width": 10, "height": 10}],
                # iscrowd 标注会被跳过，导致该图无有效多边形 → 空标签文件。
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[0, 0, 1, 0, 1, 1]],
                        "iscrowd": 1,
                    }
                ],
                "categories": [{"id": 1, "name": "road"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "images" / "neg_seg.txt").read_text(encoding="utf-8") == ""
    assert "1 张图片，0 个多边形，1 个类别" in result.stdout


def test_skips_non_coco_json_like_processing_summary(tmp_path: Path) -> None:
    source = tmp_path / "source"
    images_dir = source / "batch" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame.jpg").write_bytes(b"image")

    (source / "batch" / "trainset.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "frame.jpg", "width": 100, "height": 50}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[0, 0, 100, 0, 100, 50]],
                    }
                ],
                "categories": [{"id": 1, "name": "road"}],
            }
        ),
        encoding="utf-8",
    )
    # 远端数据集打包流程生成的统计文件，非 COCO 标注，应被跳过而非报错。
    (source / "processing_summary.json").write_text(
        json.dumps(
            {
                "total_files": 32,
                "certain": {"copied_count": 83},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "跳过非标注 JSON" in result.stderr
    assert "processing_summary.json" in result.stderr
    assert (output / "batch" / "images" / "frame_seg.txt").read_text(encoding="utf-8").startswith("0 ")
    assert "1 张图片" in result.stdout


def test_resolves_bare_filename_nested_in_images_subdir(tmp_path: Path) -> None:
    """snap_picture 这类布局：file_name 是裸文件名，但图片按摄像头分到 images/<摄像头>/ 下。"""
    source = tmp_path / "source"
    # 图片在 images/ 的子目录里，file_name 却只写裸文件名。
    cam_dir = source / "snap" / "images" / "cam01"
    cam_dir.mkdir(parents=True)
    (cam_dir / "frame.jpg").write_bytes(b"image")

    (source / "snap" / "trainset.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "frame.jpg", "width": 100, "height": 50}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[0, 0, 100, 0, 100, 50]],
                    }
                ],
                "categories": [{"id": 1, "name": "road"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # 保留摄像头子目录层级，避免不同子目录下同名文件冲突。
    assert (output / "snap" / "images" / "cam01" / "frame.jpg").resolve() == (cam_dir / "frame.jpg").resolve()
    assert (output / "snap" / "images" / "cam01" / "frame_seg.txt").read_text(encoding="utf-8").startswith("0 ")
    assert "1 张图片" in result.stdout


