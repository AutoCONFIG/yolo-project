"""
Dataset Preparation Tool
========================
将原始标注数据转换为 YOLO pose 训练所需的目录结构。

输入: source 目录下的子文件夹，每个子文件夹含 .jpg + .txt
  - 有 .txt 的 .jpg = 有标注，保留
  - 无 .txt 的 .jpg = 无标注，跳过

输出: output 目录
  images/
    train/
    val/
  labels/
    train/
    val/

Usage:
    python tools/prepare_dataset.py
    python tools/prepare_dataset.py --source ../汇总 --output datasets/parking_pose
    python tools/prepare_dataset.py --val-ratio 0.2 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

# ========== 可直接修改的默认参数 ==========
DEFAULT_SOURCE = "../汇总"                # 原始数据目录
DEFAULT_OUTPUT = "../datasets/parking_pose"   # 输出目录
DEFAULT_VAL_RATIO = 0.2                   # 验证集比例
DEFAULT_SEED = 42                         # 随机种子
# ==========================================


def find_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """找出所有有标注的 (图片, 标签) 对。"""
    pairs = []
    for txt_file in sorted(source_dir.glob("*.txt")):
        jpg_file = txt_file.with_suffix(".jpg")
        if jpg_file.exists():
            pairs.append((jpg_file, txt_file))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="准备停车位关键点数据集")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="原始数据目录")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="验证集比例")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    val_ratio = args.val_ratio

    if not source.exists():
        print(f"错误: 源目录不存在 {source}")
        return

    # 收集所有有标注的图片-标签对
    all_pairs = []
    subdirs = sorted([d for d in source.iterdir() if d.is_dir()])
    for subdir in subdirs:
        pairs = find_pairs(subdir)
        print(f"  {subdir.name}: {len(pairs)} 有标注样本")
        all_pairs.extend(pairs)

    print(f"\n总计: {len(all_pairs)} 有标注样本")

    if not all_pairs:
        print("没有找到有标注的样本，退出")
        return

    # 打乱并划分
    random.seed(args.seed)
    random.shuffle(all_pairs)
    val_count = int(len(all_pairs) * val_ratio)
    val_pairs = all_pairs[:val_count]
    train_pairs = all_pairs[val_count:]

    print(f"训练集: {len(train_pairs)}")
    print(f"验证集: {len(val_pairs)}")

    # 创建输出目录
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 复制文件，用子目录名做前缀避免重名冲突
    def copy_pairs(pairs, split):
        for jpg, txt in pairs:
            prefix = jpg.parent.name
            stem = jpg.stem
            dst_jpg = output / "images" / split / f"{prefix}_{stem}.jpg"
            dst_txt = output / "labels" / split / f"{prefix}_{stem}.txt"
            shutil.copy2(jpg, dst_jpg)
            shutil.copy2(txt, dst_txt)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    print(f"\n数据集已生成: {output.resolve()}")
    print(f"  images/train/: {len(train_pairs)} 张")
    print(f"  images/val/:   {len(val_pairs)} 张")


if __name__ == "__main__":
    main()
