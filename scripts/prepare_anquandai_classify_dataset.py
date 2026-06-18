"""
将 36 类组合标注拆分为独立分类目录，并按 YOLO classify/ImageFolder 格式输出软链接。

原始标注格式：
每张图片同名 .txt 中存储 class_id（0~35）

class_id = 安全带值 * 12 + 抽烟值 * 4 + 电话值

输出结构：
  belt/train/<类别名>/xxx.jpg
  belt/val/<类别名>/xxx.jpg
  smoke/train/<类别名>/xxx.jpg
  smoke/val/<类别名>/xxx.jpg
  phone/train/<类别名>/xxx.jpg
  phone/val/<类别名>/xxx.jpg

YOLO classify 原生读取 ImageFolder 目录，不读取 dataset YAML。
"""

import argparse
import os
import random
from collections import defaultdict

SRC_DIR = r"/data2/kaiyun/datasets_archive/anquandai"
DST_DIR = r"/data2/kaiyun/datasets_anquandai"
VAL_RATIO = 0.2
SEED = 0

BELT_OPTIONS = ["已系安全带", "无安全带", "其他"]
SMOKE_OPTIONS = ["抽烟", "无抽烟", "其他"]
PHONE_OPTIONS = ["打电话", "玩手机", "无电话", "其他"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


CATEGORIES = {
    "belt": BELT_OPTIONS,
    "smoke": SMOKE_OPTIONS,
    "phone": PHONE_OPTIONS,
}


def decode_class_id(class_id):
    """从 36 类 class_id 解码出三个维度的索引。"""
    belt_idx = class_id // 12
    smoke_idx = (class_id % 12) // 4
    phone_idx = class_id % 4
    return belt_idx, smoke_idx, phone_idx


def collect_images(root_dir):
    """递归收集所有图片文件，返回 [(相对路径, 完整路径), ...]。"""
    results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                results.append((rel_path, full_path))

    results.sort()
    return results


def safe_filename_from_relpath(rel_path):
    """将相对路径转成安全文件名，避免不同子目录下同名图片互相覆盖。"""
    return rel_path.replace("\\", "__").replace("/", "__")


def symlink_image(src_file, dst_file):
    """创建图片软链接；目标文件或软链接已存在时会删除后重建。"""
    src_file = os.path.abspath(src_file)
    dst_file = os.path.abspath(dst_file)

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    if os.path.lexists(dst_file):
        if os.path.isdir(dst_file) and not os.path.islink(dst_file):
            raise IsADirectoryError(f"目标路径已存在且是目录，无法覆盖: {dst_file}")
        os.remove(dst_file)

    os.symlink(src_file, dst_file)


def read_class_id(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("空标签")

    class_id = int(content.split()[0])

    if not (0 <= class_id <= 35):
        raise ValueError(f"class_id 超出范围 0~35: {class_id}")

    return class_id


def build_records(src_dir):
    """读取图片及标签，返回可拆分样本记录。"""
    images = collect_images(src_dir)
    records = []
    skipped = 0
    error_count = 0

    for rel_path, full_path in images:
        base_name = os.path.splitext(rel_path)[0]
        label_path = os.path.join(src_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            skipped += 1
            print(f"[跳过] 未找到标签: {label_path}")
            continue

        try:
            class_id = read_class_id(label_path)
        except Exception as e:
            error_count += 1
            skipped += 1
            print(f"[错误] 读取标签失败: {label_path}, error={e}")
            continue

        records.append({
            "rel_path": rel_path,
            "full_path": full_path,
            "class_id": class_id,
            "indexes": decode_class_id(class_id),
        })

    return images, records, skipped, error_count


def assign_splits(records, val_ratio, seed):
    """按 36 类组合标签分组拆分，尽量保持 train/val 类别组合分布一致。"""
    grouped = defaultdict(list)

    for record in records:
        grouped[record["class_id"]].append(record)

    split_records = {"train": [], "val": []}

    for class_id, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item["rel_path"])
        rng = random.Random(seed + class_id)
        rng.shuffle(items)

        if len(items) <= 1:
            val_count = 0
        else:
            val_count = round(len(items) * val_ratio)
            val_count = max(1, min(val_count, len(items) - 1))

        split_records["val"].extend(items[:val_count])
        split_records["train"].extend(items[val_count:])

    split_records["train"].sort(key=lambda item: item["rel_path"])
    split_records["val"].sort(key=lambda item: item["rel_path"])
    return split_records


def create_output_dirs(dst_dir):
    for dataset_name, class_list in CATEGORIES.items():
        for split in ("train", "val"):
            for class_name in class_list:
                os.makedirs(os.path.join(dst_dir, dataset_name, split, class_name), exist_ok=True)


def link_records(dst_dir, split_records):
    count = {
        "belt": {"train": 0, "val": 0},
        "smoke": {"train": 0, "val": 0},
        "phone": {"train": 0, "val": 0},
    }
    symlink_error_count = 0

    for split, records in split_records.items():
        for record in records:
            belt_idx, smoke_idx, phone_idx = record["indexes"]
            img_filename = safe_filename_from_relpath(record["rel_path"])

            targets = (
                ("belt", BELT_OPTIONS[belt_idx]),
                ("smoke", SMOKE_OPTIONS[smoke_idx]),
                ("phone", PHONE_OPTIONS[phone_idx]),
            )

            for dataset_name, class_name in targets:
                dst_file = os.path.join(dst_dir, dataset_name, split, class_name, img_filename)

                try:
                    symlink_image(record["full_path"], dst_file)
                    count[dataset_name][split] += 1
                except Exception as e:
                    symlink_error_count += 1
                    print(f"[错误] 创建软连接失败: {record['full_path']} -> {dst_file}, error={e}")

    return count, symlink_error_count


def print_class_stats(dst_dir):
    for dataset_name, class_list in CATEGORIES.items():
        print(f"\n  [{dataset_name}]")

        for split in ("train", "val"):
            split_total = 0
            print(f"    {split}:")

            for class_name in class_list:
                class_dir = os.path.join(dst_dir, dataset_name, split, class_name)

                if os.path.exists(class_dir):
                    num = len([
                        f for f in os.listdir(class_dir)
                        if f.lower().endswith(IMAGE_EXTENSIONS)
                    ])
                else:
                    num = 0

                split_total += num
                print(f"      {class_name}: {num} 张")

            print(f"      小计: {split_total} 张")


def split_labels(src_dir, dst_dir, val_ratio=VAL_RATIO, seed=SEED):
    """拆分标注，不修改源数据，只在输出目录中创建软链接。"""
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    if src_dir == dst_dir:
        print("错误：源目录和目标目录不能相同！")
        return 1

    if not (0.0 < val_ratio < 1.0):
        print(f"错误：val_ratio 必须在 0 和 1 之间，当前为 {val_ratio}")
        return 1

    images, records, skipped, error_count = build_records(src_dir)

    if not images:
        print(f"在 {src_dir} 中未找到图片文件")
        return 1

    if not records:
        print("没有可用样本，未生成数据集")
        return 1

    split_records = assign_splits(records, val_ratio, seed)
    create_output_dirs(dst_dir)
    count, symlink_error_count = link_records(dst_dir, split_records)

    print("\n完成！")
    print(f"  源目录: {src_dir}")
    print(f"  目标目录: {dst_dir}")
    print(f"  共发现 {len(images)} 张图片")
    print(f"  有效样本 {len(records)} 张")
    print(f"  train 样本 {len(split_records['train'])} 张")
    print(f"  val 样本 {len(split_records['val'])} 张")
    print(f"  跳过 {skipped} 张（无标签或标签格式错误）")
    print(f"  标签错误 {error_count} 个")
    print(f"  软连接错误 {symlink_error_count} 个")
    print(f"  安全带: train={count['belt']['train']} 张, val={count['belt']['val']} 张")
    print(f"  抽烟:   train={count['smoke']['train']} 张, val={count['smoke']['val']} 张")
    print(f"  电话:   train={count['phone']['train']} 张, val={count['phone']['val']} 张")

    print_class_stats(dst_dir)
    return 0 if symlink_error_count == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser(description="将 36 类安全带组合标注拆分为 YOLO classify 软链接数据集")
    parser.add_argument("--src-dir", default=SRC_DIR, help="源目录，包含图片和同名 .txt 标签")
    parser.add_argument("--dst-dir", default=DST_DIR, help="输出根目录")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子，默认 0")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("YOLO 分类标注拆分工具 - train/val 软连接版")
    print("=" * 50)
    print("\n配置:")
    print(f"  源目录: {args.src_dir}")
    print(f"  输出目录: {args.dst_dir}")
    print(f"  验证集比例: {args.val_ratio}")
    print(f"  随机种子: {args.seed}")
    print()

    if not os.path.isdir(args.src_dir):
        print(f"错误：源目录不存在: {args.src_dir}")
        return 1

    return split_labels(args.src_dir, args.dst_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
