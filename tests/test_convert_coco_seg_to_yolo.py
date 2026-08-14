from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "convert_coco_seg_to_yolo.py"


def test_converts_per_image_coco_polygon_to_yolo(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = source / "batch" / "images" / "camera" / "frame.jpg"
    label = source / "batch" / "labels" / "camera" / "frame.json"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "camera\\frame.jpg", "width": 100, "height": 50}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 5,
                        "segmentation": [[-10, -5, 100, 0, 120, 50]],
                        "bbox": [-10, -5, 130, 55],
                        "area": 3000,
                        "iscrowd": 0,
                    }
                ],
                "categories": [
                    {"id": 2, "name": "road", "supercategory": "object"},
                    {"id": 5, "name": "guardrail", "supercategory": "object"},
                ],
            }
        ),
        encoding="utf-8",
    )
    empty_categories_image = source / "batch" / "images" / "negative.jpg"
    empty_categories_label = source / "batch" / "labels" / "negative.json"
    empty_categories_image.write_bytes(b"image")
    empty_categories_label.write_text(
        json.dumps(
            {
                "images": [{"id": 2, "file_name": "negative.jpg", "width": 100, "height": 50}],
                "annotations": [
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 2,
                        "segmentation": [[0, 0, 50, 0, 50, 25]],
                        "bbox": [0, 0, 50, 25],
                        "area": 1250,
                        "iscrowd": 0,
                    }
                ],
                "categories": [],
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
    assert (output / "batch" / "camera" / "frame_seg.txt").read_text(encoding="utf-8") == (
        "1 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000\n"
    )
    assert (output / "batch" / "camera" / "frame.jpg").resolve() == image.resolve()
    assert (output / "batch" / "negative_seg.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (output / "classes.txt").read_text(encoding="utf-8") == "road\nguardrail\n"
