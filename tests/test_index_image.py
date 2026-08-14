from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "index_image.py"


def test_indexes_only_images_with_det_or_seg_labels(tmp_path: Path) -> None:
    labeled = {
        tmp_path / "a" / "det.jpg",
        tmp_path / "a" / "seg.png",
        tmp_path / "b" / "both.jpeg",
    }
    for image in (*labeled, tmp_path / "unlabeled.jpg"):
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")

    (tmp_path / "a" / "det_det.txt").write_text("bad content is still an existing label\n", encoding="utf-8")
    (tmp_path / "a" / "seg_seg.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "both_det.txt").write_text("", encoding="utf-8")
    (tmp_path / "b" / "both_seg.txt").write_text("", encoding="utf-8")

    module = runpy.run_path(str(SCRIPT))
    main = module["main"]
    main.__globals__["ROOT_DIRECTORY"] = tmp_path
    main()

    train = (tmp_path / "train_seg_det.txt").read_text(encoding="utf-8").splitlines()
    val = (tmp_path / "val_seg_det.txt").read_text(encoding="utf-8").splitlines()
    assert set(train + val) == {str(path.resolve()) for path in labeled}
    assert set(train).isdisjoint(val)


def test_preserves_symlink_paths_for_colocated_labels(tmp_path: Path) -> None:
    source = tmp_path / "source" / "sample.jpg"
    source.parent.mkdir()
    source.write_bytes(b"image")
    dataset = tmp_path / "merged"
    dataset.mkdir()
    image = dataset / "sample.jpg"
    image.symlink_to(source)
    (dataset / "sample_det.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    module = runpy.run_path(str(SCRIPT))
    main = module["main"]
    main.__globals__["ROOT_DIRECTORY"] = dataset
    main()

    indexed = (dataset / "train_seg_det.txt").read_text(encoding="utf-8").splitlines()
    indexed += (dataset / "val_seg_det.txt").read_text(encoding="utf-8").splitlines()
    assert indexed == [str(image)]
