#!/usr/bin/env python3
"""
数据集准备工具 - 最终版
======================
将原始标注数据转换为 YOLO 训练所需的目录结构，支持目标检测和关键点任务。

主要功能:
1. 支持 .jpg/.jpeg/.png 混合格式
2. 递归遍历所有子文件夹
3. 生成 train.txt/val.txt 路径清单
4. 自动检测任务类型 (detect / pose)，也可通过 --task 手动指定
5. 目标检测模式: 标签 5 列 (cls cx cy w h)，bbox 裁剪到 [0,1]
6. 关键点模式: 超画幅关键点贴边处理 (坐标裁剪到 [0,1], visibility=0)。根据画幅内关键点重新计算并裁剪边界框
7. 支持负样本提取 (空标签图片)

输入结构:
  source/
    任务A/
      人员1/
        批次1/
          0001.jpg
          0001.txt
        批次2/
          0002.png
          0002.txt

输出结构:
  output/
    images/train/
    images/val/
    labels/train/
    labels/val/
    train.txt   (绝对路径列表)
    val.txt     (绝对路径列表)
    dataset.yaml

标签处理规则:
  detect 模式:
    - 列数 = 5 (cls cx cy w h)
    - 类别ID为负数 -> 跳过该行
    - bbox 裁剪到 [0,1] 范围

  pose 模式:
    - 列数 = 5 + nkpt * ndim
    - 类别ID为负数 -> 跳过该行
    - 关键点超出 [0,1] 范围 -> 坐标贴边, v=0
    - 仅用 v>0 的可见关键点重新计算bbox，并将bbox裁剪到 [0,1] 范围
    - 格式错误或列数不足的行 -> 跳过并警告

Usage:
  python prepare_dataset.py
  python prepare_dataset.py --source /path/to/data --output ../datasets/parking_pose
  python prepare_dataset.py --val-ratio 0.2 --empty-ratio 0.1 --seed 42
  python prepare_dataset.py --task detect --source /path/to/detect_data
  python prepare_dataset.py --task pose --kpt-shape 4 3
  python prepare_dataset.py --task auto  (默认，自动检测)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中, 以便导入 utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.constants import IMG_EXTENSIONS
from utils.io import read_text_robust

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ========== 可修改的默认参数 ==========
DEFAULT_SOURCE = "/home/yun/Desktop/20260430任务_1/ok"
DEFAULT_OUTPUT = "/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/datasets/ok"
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_EMPTY_RATIO = 0  # 负样本比例 (0=不提取, 0.1=10%)
# ======================================

# YOLO pose 标签格式 (num_keypoints, ndim)
KPT_SHAPE = (4, 3)  # 4个关键点，每个点有 (x, y, visibility) 3个值


def _ensure_positive_extent(
    min_val: float, max_val: float, eps: float = 1e-4, bounds: tuple = (0.0, 1.0)
) -> tuple[float, float]:
    """确保 max_val > min_val，同时保持在 bounds 范围内。

    当 bbox 宽高为0或负时，向 bounds 上限方向扩展 eps。
    若 min_val 已接近上限，则同时向 bounds 下限方向收缩。
    """
    if max_val <= min_val:
        lo, hi = bounds
        if min_val >= hi - eps:
            min_val = max(lo, min_val - eps)
        max_val = min(hi, min_val + eps)
    return min_val, max_val


def detect_task_type(pairs: list[tuple[Path, Path]]) -> tuple[str, tuple[int, int] | None]:
    """
    自动检测标签的任务类型和关键点配置。

    通过采样标签文件的列数来判断:
    - 5列 (cls cx cy w h) → "detect" (目标检测)
    - 偶数列 (>5, 每2列一对) → "segment" (实例分割)
    - 5 + nkpt*ndim 列  → "pose"   (关键点/姿态估计)

    Returns:
        (task_type, kpt_shape):
            task_type: "detect", "segment" 或 "pose"
            kpt_shape: detect/segment时为 None, pose时为 (nkpt, ndim)
    """
    col_counts: dict[int, int] = {}
    sample_limit = min(len(pairs), 50)

    for _, txt_path in pairs[:sample_limit]:
        try:
            text = read_text_robust(txt_path)
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                n = len(parts)
                col_counts[n] = col_counts.get(n, 0) + 1
        except (IOError, OSError) as e:
            logger.warning(f"跳过无法读取的文件: {txt_path}: {e}")
            continue

    if not col_counts:
        return "detect", None

    dominant_cols = max(col_counts, key=col_counts.get)

    if dominant_cols == 5:
        return "detect", None

    extra = dominant_cols - 5
    if extra <= 0:
        return "detect", None

    for ndim in (3, 2):
        if extra % ndim == 0:
            nkpt = extra // ndim
            if nkpt > 0:
                return "pose", (nkpt, ndim)

    if extra % 2 == 0 and extra >= 6:
        return "segment", None

    return "detect", None


def find_pairs_recursive(source_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """
    递归查找所有图片和对应的标签文件（使用 os.walk 替代 rglob，速度更快）。
    """
    labeled_pairs = []
    empty_images = []

    for root, dirs, files in os.walk(source_dir):
        file_set = set(files)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMG_EXTENSIONS:
                continue

            img_path = Path(root) / f
            txt_name = os.path.splitext(f)[0] + ".txt"

            if txt_name not in file_set:
                empty_images.append(img_path)
                continue

            txt_path = Path(root) / txt_name
            if os.path.getsize(txt_path) == 0:
                empty_images.append(img_path)
                continue

            try:
                text = read_text_robust(txt_path)
                if not any(line.strip() for line in text.splitlines()):
                    empty_images.append(img_path)
                    continue
            except (IOError, OSError):
                empty_images.append(img_path)
                continue

            labeled_pairs.append((img_path, txt_path))

    return labeled_pairs, empty_images


def fix_label_line(line: str, task_type: str = "pose", kpt_shape: tuple[int, int] | None = None) -> tuple[str, list[str]]:
    """
    修正单个标签行。

    detect 模式:
    1. 过滤类别ID为负数的行
    2. 将 bbox 裁剪到 [0,1] 范围

    pose 模式:
    1. 过滤类别ID为负数的行
    2. 修正超画幅关键点(贴边，v=0)
    3. 根据修正后的关键点重新计算bbox
    4. 将新的bbox裁剪到[0,1]并处理宽高为0的情况
    """
    parts = line.strip().split()
    if not parts:
        return "", []

    bbox_end = 5  # cls + cx + cy + w + h

    # 检查类别ID
    try:
        cls_id = int(parts[0])
    except ValueError:
        return "", [f"类别ID非整数: {parts[0]}"]
    
    if cls_id < 0:
        return "", [f"类别ID为负数: {cls_id}"]

    if task_type in ("detect", "segment"):
        if len(parts) < bbox_end:
            return "", [f"列数不足: {len(parts)} < {bbox_end}"]

        try:
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            return "", ["bbox 格式错误"]

        x1 = max(0.0, min(1.0, cx - w / 2.0))
        y1 = max(0.0, min(1.0, cy - h / 2.0))
        x2 = max(0.0, min(1.0, cx + w / 2.0))
        y2 = max(0.0, min(1.0, cy + h / 2.0))

        x1, x2 = _ensure_positive_extent(x1, x2)
        y1, y2 = _ensure_positive_extent(y1, y2)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        out = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

        if task_type == "segment" and len(parts) > bbox_end:
            seg_coords = []
            for i in range(bbox_end, len(parts)):
                try:
                    v = float(parts[i])
                    v = max(0.0, min(1.0, v))
                    seg_coords.append(f"{v:.6f}")
                except ValueError:
                    continue
            out.extend(seg_coords)

        return " ".join(out), []

    # ===== pose 模式: 处理关键点 + bbox =====
    nkpt, ndim = kpt_shape or KPT_SHAPE
    kpt_cols = nkpt * ndim
    expected = bbox_end + kpt_cols

    if len(parts) < expected:
        return "", [f"列数不足: {len(parts)} < {expected}"]

    # 1. 解析并修正关键点
    kpts = []
    warnings = []
    for i in range(nkpt):
        base = bbox_end + i * ndim
        try:
            x = float(parts[base])
            y = float(parts[base + 1])
            v = float(parts[base + 2]) if ndim == 3 else 1.0
        except (ValueError, IndexError):
            return "", [f"第 {i} 个关键点格式错误"]

        oob = x < 0 or x > 1 or y < 0 or y > 1
        if oob:
            warnings.append(f"kpt{i}({x:.3f},{y:.3f},v={int(v)}) 超画幅")

            # 关键点贴边
            if x < 0: x = 0.0
            elif x > 1: x = 1.0

            if y < 0: y = 0.0
            elif y > 1: y = 1.0

            # 超画幅设为不可见
            v = 0.0

        kpts.append((x, y, int(v)))

    try:
        orig_cx = float(parts[1])
        orig_cy = float(parts[2])
        orig_w = float(parts[3])
        orig_h = float(parts[4])
        orig_x1 = orig_cx - orig_w / 2.0
        orig_y1 = orig_cy - orig_h / 2.0
        orig_x2 = orig_cx + orig_w / 2.0
        orig_y2 = orig_cy + orig_h / 2.0
    except (ValueError, IndexError):
        return "", ["原始 bbox 格式错误"]

    visible_kpts = [(x, y) for x, y, v in kpts if v > 0]
    if visible_kpts:
        x_coords = [k[0] for k in visible_kpts]
        y_coords = [k[1] for k in visible_kpts]
    else:
        x_coords = [k[0] for k in kpts]
        y_coords = [k[1] for k in kpts]

    x_min = min(min(x_coords), orig_x1)
    x_max = max(max(x_coords), orig_x2)
    y_min = min(min(y_coords), orig_y1)
    y_max = max(max(y_coords), orig_y2)

    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)

    x_min_orig, x_max_orig = x_min, x_max
    y_min_orig, y_max_orig = y_min, y_max
    x_min, x_max = _ensure_positive_extent(x_min, x_max)
    y_min, y_max = _ensure_positive_extent(y_min, y_max)
    if (x_min, x_max) != (x_min_orig, x_max_orig):
        warnings.append("框宽度为0，已调整为极小值")
    if (y_min, y_max) != (y_min_orig, y_max_orig):
        warnings.append("框高度为0，已调整为极小值")

    # 重新计算 cx, cy, w, h
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # 6. 组装输出行
    out = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    
    for x, y, v in kpts:
        out.append(f"{x:.6f}")
        out.append(f"{y:.6f}")
        if ndim == 3:
            out.append(f"{v}")

    return " ".join(out), warnings


def copy_or_link(src: Path, dst: Path, mode: str = "soft") -> None:
    """
    根据 mode 选择复制或链接方式：
      hard = 硬链接（同文件系统，不占用额外空间，最快）
      soft = 软链接/符号链接
      copy = 完整复制（跨文件系统兼容）
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # 跨文件系统，回退到 copy
    elif mode == "soft":
        os.symlink(src.resolve(), dst)
        return
    shutil.copy2(src, dst)


def process_label(src_txt: Path, dst_txt: Path, task_type: str = "pose", kpt_shape: tuple[int, int] | None = None) -> tuple[int, int, int]:
    """
    处理整个标签文件。
    """
    written = 0
    skipped = 0
    warns = 0

    text = read_text_robust(src_txt)
    with open(dst_txt, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue

            fixed, warnings = fix_label_line(line, task_type=task_type, kpt_shape=kpt_shape)
            if not fixed:
                skipped += 1
                warns += len(warnings)
                if warnings:
                    print(f"    [警告] {src_txt.name} 第 {line_no} 行: {warnings[0]}")
                continue

            fout.write(fixed + "\n")
            written += 1
            if warnings:
                warns += len(warnings)

    return written, skipped, warns


def make_safe_name(img_path: Path, source: Path) -> tuple[str, str]:
    """
    根据相对路径生成安全文件名。
    """
    rel = img_path.relative_to(source).with_suffix("")
    safe_base = "_".join(rel.parts)
    suffix = img_path.suffix
    return safe_base, suffix


def main():
    parser = argparse.ArgumentParser(
        description="准备 YOLO 训练数据集，支持目标检测(detect)和关键点(pose)任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="原始数据目录（默认: %(default)s）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="输出目录（默认: %(default)s）")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help="验证集比例（默认: %(default).2f）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子（默认: %(default)d）")
    parser.add_argument("--empty-ratio", type=float, default=DEFAULT_EMPTY_RATIO,
                        help="负样本比例（0=不提取，默认: %(default).2f）")
    parser.add_argument("--task", type=str, choices=["auto", "detect", "segment", "pose"],
                        default="auto",
                        help="任务类型: auto=自动检测, detect=目标检测, segment=实例分割, pose=关键点（默认: auto）")
    parser.add_argument("--kpt-shape", type=int, nargs=2, default=None,
                        metavar=("N_KPT", "N_DIM"),
                        help="手动指定关键点配置 (如: 4 3)，仅pose模式有效，auto模式覆盖自动检测")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="类别名称列表 (如: --classes person car dog)，默认使用 class_0 class_1 ...")
    parser.add_argument("--link-mode", type=str, choices=["hard", "soft", "copy"], default="soft",
                        help="文件复制方式: hard=硬链接(最快,同文件系统), soft=软链接(默认,支持跨文件系统), copy=完整复制")

    args = parser.parse_args()

    if not args.source:
        parser.error("--source 是必需的，请指定原始数据目录")
    if not args.output:
        parser.error("--output 是必需的，请指定输出目录")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if not source.exists():
        print(f"错误: 源目录不存在 {source}")
        return

    random.seed(args.seed)

    print("=" * 60)
    print("数据集准备工具")
    print("=" * 60)
    print(f"源目录: {source}")
    print(f"输出目录: {output}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"负样本比例: {args.empty_ratio}")
    print(f"随机种子: {args.seed}")
    print("-" * 60)

    print("正在扫描源目录...")
    all_pairs, all_empty_images = find_pairs_recursive(source)
    print(f"找到 {len(all_pairs)} 个有标注样本，{len(all_empty_images)} 个无标注图片")

    if not all_pairs and args.empty_ratio == 0:
        print("错误: 没有找到有标注的样本，且未启用负样本提取")
        return

    # ===== 任务类型检测 =====
    task_type = args.task
    kpt_shape = tuple(args.kpt_shape) if args.kpt_shape else None

    if task_type == "auto":
        print("\n自动检测任务类型...")
        detected_type, detected_kpt = detect_task_type(all_pairs)
        task_type = detected_type
        # 手动 --kpt-shape 可覆盖自动检测结果
        if kpt_shape is None:
            kpt_shape = detected_kpt
        print(f"  检测结果: {task_type}" + (f" (kpt_shape={kpt_shape})" if kpt_shape else ""))
    elif task_type == "pose":
        # pose 模式下，kpt_shape 必须有值
        if kpt_shape is None:
            # 尝试自动检测来获取 kpt_shape
            _, detected_kpt = detect_task_type(all_pairs)
            kpt_shape = detected_kpt if detected_kpt else KPT_SHAPE
        print(f"\n任务类型: pose (kpt_shape={kpt_shape})")
    elif task_type == "segment":
        kpt_shape = None
        print(f"\n任务类型: segment (实例分割)")
    else:
        # detect 模式
        kpt_shape = None
        print(f"\n任务类型: detect (目标检测)")

    # 打印关键点配置
    if task_type == "pose":
        print(f"关键点配置: {kpt_shape[0]}个点, {kpt_shape[1]}维")
    print("-" * 60)

    if all_pairs:
        random.shuffle(all_pairs)
        val_count = int(len(all_pairs) * args.val_ratio)
        val_pairs = all_pairs[:val_count]
        train_pairs = all_pairs[val_count:]
    else:
        train_pairs, val_pairs = [], []

    empty_train, empty_val = [], []
    if args.empty_ratio > 0 and all_empty_images:
        empty_count = min(int(len(all_pairs) * args.empty_ratio), len(all_empty_images))
        if empty_count > 0:
            random.shuffle(all_empty_images)
            selected_empty = all_empty_images[:empty_count]
            empty_val_count = int(empty_count * args.val_ratio)
            empty_val = selected_empty[:empty_val_count]
            empty_train = selected_empty[empty_val_count:]

    print("\n数据集划分:")
    print(f"  训练集: {len(train_pairs)} 有标注 + {len(empty_train)} 无标注 = {len(train_pairs) + len(empty_train)}")
    print(f"  验证集: {len(val_pairs)} 有标注 + {len(empty_val)} 无标注 = {len(val_pairs) + len(empty_val)}")

    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0
    total_warns = 0

    def process_split(pairs, empty_imgs, split_name, list_file):
        nonlocal total_written, total_skipped, total_warns
        print(f"\n处理 {split_name} 集...")

        for img_path, txt_path in pairs:
            safe_base, suffix = make_safe_name(img_path, source)
            dst_img = output / "images" / split_name / f"{safe_base}{suffix}"
            dst_txt = output / "labels" / split_name / f"{safe_base}.txt"

            copy_or_link(img_path, dst_img, mode=args.link_mode)
            list_file.write(f"{dst_img.resolve()}\n")

            written, skipped, warns = process_label(txt_path, dst_txt, task_type=task_type, kpt_shape=kpt_shape)
            total_written += written
            total_skipped += skipped
            total_warns += warns

        for img_path in empty_imgs:
            safe_base, suffix = make_safe_name(img_path, source)
            dst_img = output / "images" / split_name / f"{safe_base}{suffix}"
            dst_txt = output / "labels" / split_name / f"{safe_base}.txt"

            copy_or_link(img_path, dst_img, mode=args.link_mode)
            list_file.write(f"{dst_img.resolve()}\n")
            dst_txt.touch()

    with open(output / "train.txt", "w", encoding="utf-8") as train_list_file, \
         open(output / "val.txt", "w", encoding="utf-8") as val_list_file:
        process_split(train_pairs, empty_train, "train", train_list_file)
        process_split(val_pairs, empty_val, "val", val_list_file)

    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)
    print(f"输出目录: {output}")
    print("\n目录结构:")
    print(f"  {output}/images/train/")
    print(f"  {output}/images/val/")
    print(f"  {output}/labels/train/")
    print(f"  {output}/labels/val/")
    print(f"  {output}/train.txt")
    print(f"  {output}/val.txt")

    print("\n标签处理统计:")
    print(f"  成功写入标签行: {total_written}")
    if total_skipped > 0:
        print(f"  跳过的标签行: {total_skipped} (类别负数/格式错误)")
    if total_warns > 0:
        print(f"  修正警告(关键点超画幅/框为0): {total_warns}")

    yaml_content = f"""# dataset.yaml
path: {output}  # dataset root dir
train: images/train  # train images
val: images/val    # val images

"""

    if task_type == "pose":
        yaml_content += f"# Keypoints\nkpt_shape: {kpt_shape}  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visibility)\n\n"

    if args.classes:
        names_lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(args.classes))
    else:
        max_cls = 0
        for _, txt_path in all_pairs[:100]:
            try:
                text = read_text_robust(txt_path)
                for line in text.splitlines():
                    parts = line.strip().split()
                    if parts:
                        max_cls = max(max_cls, int(parts[0]))
            except (IOError, ValueError) as e:
                logger.warning(f"解析类别时跳过文件: {txt_path}: {e}")
                continue
        names_lines = "\n".join(f"  {i}: class_{i}" for i in range(max_cls + 1))

    yaml_content += f"""# Classes
names:
{names_lines}
"""

    yaml_path = output / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n已生成配置文件: {yaml_path}")


if __name__ == "__main__":
    main()