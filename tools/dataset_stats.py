#!/usr/bin/env python3
"""Dataset statistics analyzer for YOLO format datasets."""
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Dataset root path")
    p.add_argument("--names", nargs="+", default=None, help="Class names (optional if dataset YAML exists)")
    p.add_argument("--out", default="dataset_stats.txt", help="Output text report")
    return p.parse_args()


def load_names_from_yaml(root: Path):
    """尝试从数据集目录下的 YAML 文件读取类别名称。"""
    import yaml
    for yaml_path in root.rglob("*.yaml"):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "names" in data:
                names = data["names"]
                if isinstance(names, dict):
                    return [names[i] for i in sorted(names.keys())]
                if isinstance(names, list):
                    return names
        except Exception:
            continue
    return None


def analyze_split(root: Path, split: str, names: list):
    lbl_dir = root / "labels" / split
    if not lbl_dir.exists():
        return None

    label_files = sorted(lbl_dir.glob("*.txt"))
    total_images = len(label_files)
    total_objects = 0
    class_counts = Counter()
    objs_per_image = []
    all_areas = defaultdict(list)
    all_ratios = defaultdict(list)
    all_centers = defaultdict(list)
    quality_issues = []
    cooccurrence = Counter()

    for lf in label_files:
        lines = lf.read_text().strip().splitlines()
        n = len(lines)
        objs_per_image.append(n)
        total_objects += n
        classes_in_image = set()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                quality_issues.append(f"{lf.name}: bad format: {line}")
                continue
            try:
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                quality_issues.append(f"{lf.name}: non-numeric: {line}")
                continue

            if cid < 0 or cid >= len(names):
                quality_issues.append(f"{lf.name}: invalid class {cid}: {line}")
                continue

            if w <= 0 or h <= 0:
                quality_issues.append(f"{lf.name}: non-positive size: {line}")
                continue
            if x - w/2 < -0.01 or x + w/2 > 1.01 or y - h/2 < -0.01 or y + h/2 > 1.01:
                quality_issues.append(f"{lf.name}: out of bounds: {line}")
                continue

            class_counts[cid] += 1
            classes_in_image.add(cid)
            area = w * h
            all_areas[cid].append(area)
            ratio = w / h if h > 0 else 0
            all_ratios[cid].append(ratio)
            all_centers[cid].append((x, y))

        for c in classes_in_image:
            for d in classes_in_image:
                if c <= d:
                    cooccurrence[(c, d)] += 1

    stats = {
        "split": split,
        "images": total_images,
        "objects": total_objects,
        "avg_objs": total_objects / max(total_images, 1),
        "class_counts": class_counts,
        "objs_per_image": np.array(objs_per_image) if objs_per_image else np.array([0]),
        "areas": dict(all_areas),
        "ratios": dict(all_ratios),
        "centers": dict(all_centers),
        "issues": quality_issues,
        "cooccurrence": cooccurrence,
    }
    return stats

def print_report(stats, names, file=None):
    def p(*args):
        print(*args, file=file)

    p("=" * 70)
    p(f"Dataset Analysis Report - {stats['split'].upper()}")
    p("=" * 70)
    p(f"Images:        {stats['images']}")
    p(f"Objects:       {stats['objects']}")
    p(f"Avg obj/img:   {stats['avg_objs']:.2f}")

    opi = stats["objs_per_image"]
    p(f"\nObjects per image:")
    p(f"  Min: {opi.min()}, Max: {opi.max()}, Mean: {opi.mean():.2f}, Median: {np.median(opi):.0f}")
    for k, v in sorted(Counter(opi).items())[:10]:
        p(f"    {int(k)} obj: {v} images ({v/len(opi)*100:.1f}%)")

    p(f"\nClass Distribution:")
    total = stats["objects"]
    for cid, name in enumerate(names):
        c = stats["class_counts"].get(cid, 0)
        p(f"  [{cid}] {name}: {c} ({c/max(total,1)*100:.1f}%)")

    p(f"\nObject Size (normalized area):")
    for cid, name in enumerate(names):
        areas = np.array(stats["areas"].get(cid, []))
        if len(areas) == 0:
            continue
        p(f"  [{cid}] {name}: n={len(areas)}")
        p(f"    area: min={areas.min():.5f}, max={areas.max():.4f}, mean={areas.mean():.5f}, median={np.median(areas):.5f}")
        tiny = np.sum(areas < 0.0001)
        small = np.sum((areas >= 0.0001) & (areas < 0.001))
        medium = np.sum((areas >= 0.001) & (areas < 0.01))
        large = np.sum(areas >= 0.01)
        p(f"    tiny(<0.01%):{tiny} small:{small} medium:{medium} large(>1%):{large}")

    p(f"\nObject Position (center x,y):")
    for cid, name in enumerate(names):
        centers = np.array(stats["centers"].get(cid, []))
        if len(centers) == 0:
            continue
        left = np.sum(centers[:, 0] < 0.33)
        center = np.sum((centers[:, 0] >= 0.33) & (centers[:, 0] < 0.67))
        right = np.sum(centers[:, 0] >= 0.67)
        p(f"  [{cid}] {name}: left={left} center={center} right={right}")

    p(f"\nCo-occurrence (images containing both classes):")
    for (c1, c2), cnt in sorted(stats["cooccurrence"].items()):
        if c1 == c2:
            p(f"  [{c1}] {names[c1]} alone: {cnt} images")
        else:
            p(f"  [{c1}] {names[c1]} + [{c2}] {names[c2]}: {cnt} images")

    p(f"\nQuality Issues: {len(stats['issues'])}")
    for issue in stats["issues"][:20]:
        p(f"  {issue}")
    if len(stats["issues"]) > 20:
        p(f"  ... and {len(stats['issues']) - 20} more")
    p("=" * 70)

if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root)
    names = args.names
    if names is None:
        names = load_names_from_yaml(root)
        if names is None:
            print("错误: 无法从数据集目录自动读取类别名称，请使用 --names 参数指定")
            sys.exit(1)
        print(f"自动读取类别: {names}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for split in ["train", "val"]:
            stats = analyze_split(root, split, names)
            if stats:
                print_report(stats, names, file=f)
                print_report(stats, names, file=sys.stdout)

    print(f"\nReport saved to: {out_path}")
