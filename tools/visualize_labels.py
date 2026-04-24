#!/usr/bin/env python3
"""
YOLO Pose 标签可视化工具
========================
将 YOLO pose 格式的 txt 标签绘制到对应图片上，用于快速检查标注质量。

支持功能:
1. 单文件或批量目录可视化
2. 自动匹配图片（jpg/jpeg/png）
3. 绘制边界框、关键点、骨架连线
4. 关键点可见性区分（可见/遮挡/画幅外）
5. 交互式逐张浏览或批量保存

标签格式 (每行):
  class_id cx cy w h kpt1_x kpt1_y kpt1_v ... kptN_x kptN_y kptN_v

Usage:
  python visualize_labels.py --labels runs/inference/labels --images runs/inference/vis
  python visualize_labels.py --labels path/to/labels --images path/to/images --save-dir output/vis
  python visualize_labels.py --labels path/to/labels --images path/to/images --browse
  python visualize_labels.py --labels path/to/labels --images path/to/images --kpt-shape 4 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# ========== 硬编码配置（修改这里即可，命令行参数可覆盖） ==========
CONFIG = {
    "labels": r"D:\parking_pose\labels",       # 标签目录或单个txt文件路径
    "images": [r"D:\parking_pose\images"],         # 图片目录列表（可多个）
    "save_dir": r"D:\vis_labeled", # 保存目录（设为 None 则必须用 --browse）
    "browse": False,                          # 交互式浏览
    "filter_empty": False,                    # 跳过无标注图片
}
# ====================================================================

# ========== 默认参数 ==========
DEFAULT_KPT_SHAPE = (4, 3)  # 4个关键点，每个3维 (x, y, visibility)
DEFAULT_BOX_COLOR = (0, 255, 0)  # BGR: 绿色
DEFAULT_KPT_COLORS = [
    (255, 0, 0),    # 前左 - 蓝
    (0, 255, 0),    # 前右 - 绿
    (0, 0, 255),    # 后右 - 红
    (255, 255, 0),  # 后左 - 青
]
DEFAULT_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 顺时针连线
KPT_NAMES = ["front_left", "front_right", "rear_right", "rear_left"]

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ================================


def parse_label_file(label_path: Path, kpt_shape: tuple[int, int]):
    """解析 YOLO pose 标签文件，返回标注列表。"""
    nkpt, ndim = kpt_shape
    bbox_end = 5  # cls + cx + cy + w + h
    expected_cols = bbox_end + nkpt * ndim

    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) < expected_cols:
                print(f"  [警告] {label_path.name} 第{line_no}行: 列数{len(parts)} < 期望{expected_cols}")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                keypoints = []
                for i in range(nkpt):
                    base = bbox_end + i * ndim
                    x = float(parts[base])
                    y = float(parts[base + 1])
                    v = float(parts[base + 2]) if ndim == 3 else 1.0
                    keypoints.append((x, y, v))

                annotations.append({
                    "cls_id": cls_id,
                    "bbox": (cx, cy, w, h),
                    "keypoints": keypoints,
                })
            except (ValueError, IndexError) as e:
                print(f"  [警告] {label_path.name} 第{line_no}行: 解析失败 ({e})")
                continue

    return annotations


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=8, gap_len=4):
    """绘制虚线。"""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = max(abs(x2 - x1), abs(y2 - y1))
    if dist == 0:
        return
    for i in range(0, dist, dash_len + gap_len):
        s = i / dist
        e = min(i + dash_len, dist) / dist
        sx = int(x1 + (x2 - x1) * s)
        sy = int(y1 + (y2 - y1) * s)
        ex = int(x1 + (x2 - x1) * e)
        ey = int(y1 + (y2 - y1) * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def draw_annotations(
    img: np.ndarray,
    annotations: list[dict],
    kpt_shape: tuple[int, int],
    box_color: tuple[int, int, int],
    kpt_colors: list[tuple[int, int, int]],
    skeleton: list[tuple[int, int]],
    box_thickness: int = 2,
    kpt_radius: int = 5,
    skeleton_thickness: int = 2,
    font_scale: float = 0.5,
    show_labels: bool = True,
    show_conf: bool = False,
):
    """在图片上绘制边界框、关键点和骨架。"""
    h_img, w_img = img.shape[:2]
    nkpt, ndim = kpt_shape

    for ann in annotations:
        cx, cy, w, h = ann["bbox"]
        # 转换为像素坐标
        x1 = int((cx - w / 2) * w_img)
        y1 = int((cy - h / 2) * h_img)
        x2 = int((cx + w / 2) * w_img)
        y2 = int((cy + h / 2) * h_img)

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)

        # 绘制标签
        if show_labels:
            label = f"cls{ann['cls_id']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # 绘制骨架连线
        kpts = ann["keypoints"]
        for i, j in skeleton:
            if i < nkpt and j < nkpt:
                xi, yi, vi = kpts[i]
                xj, yj, vj = kpts[j]
                pt1 = (int(xi * w_img), int(yi * h_img))
                pt2 = (int(xj * w_img), int(yj * h_img))
                # 不可见的关键点坐标可能无效(0,0)，跳过
                if vi == 0 and (xi == 0 and yi == 0):
                    continue
                if vj == 0 and (xj == 0 and yj == 0):
                    continue
                if vi > 0 and vj > 0:
                    # 两个都可见/遮挡: 实线
                    cv2.line(img, pt1, pt2, (200, 200, 200), skeleton_thickness)
                else:
                    # 至少一个不可见: 虚线
                    _draw_dashed_line(img, pt1, pt2, (100, 100, 100), skeleton_thickness)

        # 绘制关键点
        for idx, (kx, ky, v) in enumerate(kpts):
            px, py = int(kx * w_img), int(ky * h_img)
            color = kpt_colors[idx % len(kpt_colors)]

            if v == 0:
                # 不可见: 红色空心圆 + 小叉 + v=0 标注
                red = (0, 0, 255)
                cv2.circle(img, (px, py), kpt_radius, red, 1)
                cv2.line(img, (px - 3, py - 3), (px + 3, py + 3), red, 1)
                cv2.line(img, (px - 3, py + 3), (px + 3, py - 3), red, 1)
            elif v == 1:
                # 遮挡: 半填充 + 黄色边框
                cv2.circle(img, (px, py), kpt_radius, color, -1)
                cv2.circle(img, (px, py), kpt_radius, (0, 255, 255), 2)
            else:
                # 可见(v=2): 实心圆
                cv2.circle(img, (px, py), kpt_radius, color, -1)

            # 关键点名称 + 可见性标注
            if idx < len(KPT_NAMES):
                v_tag = {0: "v0", 1: "v1", 2: "v2"}.get(int(v), f"v{int(v)}")
                cv2.putText(img, f"{KPT_NAMES[idx]}({v_tag})", (px + kpt_radius + 2, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, 1)

    return img


def find_image_for_label(label_path: Path, image_dirs: list[Path], labels_dir: Path | None = None) -> Path | None:
    """根据标签文件名在图片目录中查找对应图片。
    如果提供了 labels_dir，会保留标签的相对子目录结构在图片目录中查找。
    """
    stem = label_path.stem
    for img_dir in image_dirs:
        # 先尝试保留相对路径查找（如 20260411任务/徐靖981/frame_000000.jpg）
        if labels_dir is not None:
            rel = label_path.relative_to(labels_dir).parent
            for ext in IMG_EXTENSIONS:
                candidate = img_dir / rel / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
        # 兜底：仅按文件名在图片目录下递归查找
        for ext in IMG_EXTENSIONS:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        # 递归查找同名图片
        for ext in IMG_EXTENSIONS:
            matches = list(img_dir.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
    return None


def collect_label_files(labels_dir: Path) -> list[Path]:
    """收集目录下所有txt标签文件。"""
    files = sorted(labels_dir.rglob("*.txt"))
    # 过滤掉空文件
    return [f for f in files if f.stat().st_size > 0]


def browse_images(entries: list[tuple[Path, Path]], args):
    """交互式逐张浏览：按 q 退出，其他键下一张。"""
    print(f"\n交互式浏览: 共 {len(entries)} 张")
    print("  按 q/Esc 退出，其他键下一张\n")

    for i, (label_path, img_path) in enumerate(entries):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [跳过] 无法读取: {img_path}")
            continue

        annotations = parse_label_file(label_path, args.kpt_shape)
        vis = draw_annotations(
            img, annotations, args.kpt_shape,
            args.box_color, args.kpt_colors, args.skeleton,
            box_thickness=args.box_thickness,
            kpt_radius=args.kpt_radius,
            skeleton_thickness=args.skeleton_thickness,
            font_scale=args.font_scale,
            show_labels=args.show_labels,
        )

        # 添加信息栏
        info = f"[{i + 1}/{len(entries)}] {img_path.name} | {len(annotations)} objects | q=quit"
        bar = np.zeros((30, vis.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        vis = np.vstack([bar, vis])

        cv2.imshow("YOLO Pose Label Viewer", vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):  # q 或 Esc
            break

    cv2.destroyAllWindows()


def batch_save(entries: list[tuple[Path, Path]], args):
    """批量保存可视化结果。"""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n批量保存: 共 {len(entries)} 张 -> {save_dir}")
    for i, (label_path, img_path) in enumerate(entries):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [跳过] 无法读取: {img_path}")
            continue

        annotations = parse_label_file(label_path, args.kpt_shape)
        vis = draw_annotations(
            img, annotations, args.kpt_shape,
            args.box_color, args.kpt_colors, args.skeleton,
            box_thickness=args.box_thickness,
            kpt_radius=args.kpt_radius,
            skeleton_thickness=args.skeleton_thickness,
            font_scale=args.font_scale,
            show_labels=args.show_labels,
        )

        out_path = save_dir / f"{img_path.stem}.jpg"
        cv2.imencode(".jpg", vis)[1].tofile(str(out_path))

        if (i + 1) % 50 == 0 or i == len(entries) - 1:
            print(f"  进度: {i + 1}/{len(entries)}")

    print(f"完成! 保存到 {save_dir}")


def parse_color(s: str) -> tuple[int, int, int]:
    """解析颜色字符串 (BGR)，支持 'R,G,B' 或 '#RRGGBB'。"""
    if s.startswith("#"):
        hex_s = s[1:]
        r, g, b = int(hex_s[0:2], 16), int(hex_s[2:4], 16), int(hex_s[4:6], 16)
        return (b, g, r)  # BGR
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"颜色格式错误: {s} (应为 R,G,B 或 #RRGGBB)")
    return tuple(int(p.strip()) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Pose 标签可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--labels", type=str, default=CONFIG["labels"],
                        help="标签目录或单个txt文件路径")
    parser.add_argument("--images", type=str, nargs="+", default=CONFIG["images"],
                        help="图片目录（可指定多个，按优先级搜索）")
    parser.add_argument("--save-dir", type=str, default=CONFIG["save_dir"],
                        help="保存可视化结果的目录（不指定则必须用 --browse）")
    parser.add_argument("--browse", action="store_true", default=CONFIG["browse"],
                        help="交互式逐张浏览")
    parser.add_argument("--kpt-shape", type=int, nargs=2, default=DEFAULT_KPT_SHAPE,
                        metavar=("N_KPT", "N_DIM"),
                        help=f"关键点配置 (默认: {DEFAULT_KPT_SHAPE[0]} {DEFAULT_KPT_SHAPE[1]})")
    parser.add_argument("--box-color", type=parse_color, default=DEFAULT_BOX_COLOR,
                        help=f"边界框颜色 R,G,B (默认: 0,255,0)")
    parser.add_argument("--box-thickness", type=int, default=2,
                        help="边界框线宽 (默认: 2)")
    parser.add_argument("--kpt-radius", type=int, default=5,
                        help="关键点圆半径 (默认: 5)")
    parser.add_argument("--skeleton-thickness", type=int, default=2,
                        help="骨架线宽 (默认: 2)")
    parser.add_argument("--font-scale", type=float, default=0.5,
                        help="字体大小 (默认: 0.5)")
    parser.add_argument("--show-labels", action="store_true", default=True,
                        help="显示类别标签 (默认开启)")
    parser.add_argument("--no-labels", action="store_false", dest="show_labels",
                        help="不显示类别标签")
    parser.add_argument("--filter-empty", action="store_true", default=CONFIG["filter_empty"],
                        help="跳过没有标注的图片")

    args = parser.parse_args()
    args.kpt_shape = tuple(args.kpt_shape)
    args.kpt_colors = DEFAULT_KPT_COLORS
    args.skeleton = DEFAULT_SKELETON

    if not args.browse and not args.save_dir:
        parser.error("请指定 --save-dir 或使用 --browse")

    labels_path = Path(args.labels)
    image_dirs = [Path(p).resolve() for p in args.images]

    # 收集标签文件
    if labels_path.is_file():
        label_files = [labels_path]
    elif labels_path.is_dir():
        label_files = collect_label_files(labels_path)
    else:
        print(f"错误: 路径不存在 {labels_path}")
        return

    if not label_files:
        print("未找到标签文件")
        return

    print(f"找到 {len(label_files)} 个标签文件")

    # 匹配图片
    entries = []
    missing = 0
    for lbl in label_files:
        img = find_image_for_label(lbl, image_dirs, labels_dir=labels_path)
        if img:
            entries.append((lbl, img))
        else:
            missing += 1

    if missing > 0:
        print(f"  {missing} 个标签未找到对应图片")

    if args.filter_empty:
        entries = [(l, i) for l, i in entries if parse_label_file(l, args.kpt_shape)]

    if not entries:
        print("没有可可视化的图片")
        return

    print(f"匹配到 {len(entries)} 张图片")

    if args.browse:
        browse_images(entries, args)
    else:
        batch_save(entries, args)


if __name__ == "__main__":
    main()
