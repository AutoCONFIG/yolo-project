from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT = Path(__file__).parents[1] / "tools" / "visualize_yolo_seg.py"
COMPATIBLE_SCRIPT = Path(__file__).parents[1] / "tools" / "visualize_labels.py"


def test_saves_polygon_overlay_preview(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    label_path = tmp_path / "frame_seg.txt"
    cv2.imwrite(str(image_path), np.full((50, 100, 3), 255, dtype=np.uint8))
    label_path.write_text("0 0.1 0.2 0.9 0.2 0.5 0.9\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("road\n", encoding="utf-8")
    preview_path = tmp_path / "preview.jpg"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(label_path), "--save-preview", str(preview_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    preview = cv2.imread(str(preview_path))
    assert preview is not None
    assert preview.shape == (92, 100, 3)
    assert np.any(preview[20:50, :, 1] < 240)


def test_visualize_labels_preserves_subdirectories(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    nested = dataset / "scene" / "camera"
    nested.mkdir(parents=True)
    source_image = tmp_path / "source.jpg"
    image_path = nested / "frame.jpg"
    label_path = nested / "frame_seg.txt"
    output = tmp_path / "rendered"
    cv2.imwrite(str(source_image), np.full((50, 100, 3), 255, dtype=np.uint8))
    image_path.symlink_to(source_image)
    label_path.write_text("0 0.1 0.2 0.9 0.2 0.5 0.9\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(COMPATIBLE_SCRIPT), str(dataset), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "scene" / "camera" / "frame.jpg").is_file()
