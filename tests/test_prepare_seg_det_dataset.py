from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "prepare_seg_det_dataset.py"


def test_merges_separate_yolo_tasks_and_normalizes_label_names(tmp_path: Path) -> None:
    det_images = tmp_path / "det" / "images"
    det_labels = tmp_path / "det" / "labels"
    seg_images = tmp_path / "seg" / "images"
    seg_labels = tmp_path / "seg" / "labels"
    relative_image = Path("batch/camera/frame.jpg")
    relative_label = relative_image.with_suffix(".txt")
    relative_seg_label = relative_label.with_name(f"{relative_label.stem}_seg.txt")
    for root in (det_images, seg_images):
        (root / relative_image).parent.mkdir(parents=True)
        (root / relative_image).write_bytes(b"same image")
    for root in (det_labels, seg_labels):
        (root / relative_label).parent.mkdir(parents=True)
    (det_labels / relative_label).write_text("0 0.5 0.5 0.2 0.2 0.93\n", encoding="utf-8")
    (seg_labels / relative_seg_label).write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
    output = tmp_path / "merged"

    module = runpy.run_path(str(SCRIPT))
    main = module["main"]
    main.__globals__["DETECT_IMAGES_DIRECTORY"] = det_images
    main.__globals__["DETECT_LABELS_DIRECTORY"] = det_labels
    main.__globals__["SEGMENT_IMAGES_DIRECTORY"] = seg_images
    main.__globals__["SEGMENT_LABELS_DIRECTORY"] = seg_labels
    main.__globals__["OUTPUT_DIRECTORY"] = output
    main()

    detect_output = output / "batch/camera/frame_det.txt"
    assert (output / relative_image).read_bytes() == b"same image"
    assert detect_output.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    assert (output / "batch/camera/frame_seg.txt").read_text(encoding="utf-8") == "0 0 0 1 0 1 1\n"

    detect_output.write_text("0 0.5 0.5 0.2 0.2 0.93\n", encoding="utf-8")
    main()
    assert detect_output.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
