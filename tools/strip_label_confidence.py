"""
clean_yolo_labels.py
====================
清洗 YOLO label 文件:
  1. 移除每行多余的置信度列(第 6 列及以后),只保留前 5 列  class cx cy w h
  2. 修正归一化坐标越界: cx/cy/w/h 裁剪到 [0,1],处理负值与 >1 的情况

YOLO 标准格式每行 5 列:  class cx cy w h  (均归一化到 [0,1])

用法
----
直接运行(无需命令行参数):
    python clean_yolo_labels.py

所有可调参数都在下方 ===== 配置区 ===== 修改即可。
"""

import sys
from pathlib import Path

# ============================== 配置区 ==============================
# 数据集根目录,递归遍历所有子目录下的 *.txt
ROOT = r"/data2/kaiyun/datasets_11_classes"

# False = dry-run(只统计不修改,默认); True = 实际写入修改
APPLY = True

# 清洗后是否删除目录下的 *.cache (YOLOv5 label cache)
# 建议 True:让下次训练重新扫描 label,否则还会读到旧的缓存断言
CLEAR_CACHE = False

# 坐标裁剪下限/上限(归一化坐标应落在 [0,1])
CLAMP_MIN = 0.0
CLAMP_MAX = 1.0

# 打印示例的最大数量
MAX_SAMPLES = 10
# ===================================================================


def is_int_token(tok):
    """token 是否为整数(允许前导符号,不允许小数点)。"""
    if not tok:
        return False
    s = tok.lstrip("+-")
    return s.isdigit()


def parse_lines(path):
    """
    读取文件,返回 (lines, max_cols, is_label)。
    - lines: list[str],保留原始行(含换行)。
    - max_cols: 非空行的最大列数。
    - is_label: 所有非空行首 token 都是整数 => 视为 label 文件。
    空文件返回 is_label=False。
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        print(f"  [skip] 读取失败 {path}: {e}")
        return None, 0, False

    max_cols = 0
    is_label = True
    has_nonempty = False
    for line in raw:
        s = line.strip()
        if not s:
            continue
        has_nonempty = True
        toks = s.split()
        max_cols = max(max_cols, len(toks))
        if not is_int_token(toks[0]):
            is_label = False
            break

    if not has_nonempty:
        is_label = False
    return raw, max_cols, is_label


def clamp01(v):
    return max(CLAMP_MIN, min(CLAMP_MAX, v))


def clean_file(path):
    """
    清洗单个 label 文件:
      - 截断到前 5 列
      - 越界坐标一律裁剪到 [CLAMP_MIN, CLAMP_MAX]
    返回 (changed, kept_lines, had_outlier)
      changed     : 是否需要写回(有改动)
      kept_lines  : 清洗后的行列表(含换行)
      had_outlier : 是否存在越界坐标
    """
    raw, max_cols, is_label = parse_lines(path)
    if not is_label or raw is None:
        return False, None, False

    changed = False
    had_outlier = False
    kept = []

    for line in raw:
        s = line.rstrip("\n")
        if not s.strip():
            kept.append(line)  # 保留空行原样
            continue

        toks = s.split()
        # 1) 多余列 -> 截断到前 5 列
        if len(toks) > 5:
            toks = toks[:5]
            changed = True

        # label 行至少要有 5 列;不足的跳过(脏数据)
        if len(toks) < 5:
            changed = True
            continue

        cls = toks[0]
        try:
            cx, cy, w, h = (float(toks[1]), float(toks[2]),
                            float(toks[3]), float(toks[4]))
        except ValueError:
            changed = True
            continue

        # 2) 越界检测与裁剪
        vals = [cx, cy, w, h]
        if any(v < CLAMP_MIN or v > CLAMP_MAX for v in vals):
            had_outlier = True
            changed = True
            cx, cy, w, h = clamp01(cx), clamp01(cy), clamp01(w), clamp01(h)

        kept.append(f"{cls} {cx:.6g} {cy:.6g} {w:.6g} {h:.6g}\n")

    return changed, kept, had_outlier


def main():
    root = Path(ROOT)
    if not root.exists():
        print(f"错误:路径不存在: {root}", file=sys.stderr)
        sys.exit(1)

    # 统计
    total_txt = 0
    label_files = 0
    need_fix_cols = 0      # 多余置信度列
    need_fix_outlier = 0   # 越界坐标
    skipped_index = 0
    skipped_empty = 0
    written = 0
    samples = []

    print(f"扫描根目录: {root}")
    print(f"模式: {'APPLY (写入)' if APPLY else 'DRY-RUN (只统计,不改)'}")
    print(f"越界策略: clamp (裁剪到 [{CLAMP_MIN}, {CLAMP_MAX}])")
    if CLEAR_CACHE:
        print("清理 .cache: 是")
    print("-" * 70)

    for path in sorted(root.rglob("*.txt")):
        total_txt += 1
        raw, max_cols, is_label = parse_lines(path)
        if raw is None:
            continue
        if not is_label:
            if max_cols == 0:
                skipped_empty += 1
            else:
                skipped_index += 1
            continue
        label_files += 1

        if max_cols > 5:
            need_fix_cols += 1

        changed, kept, had_outlier = clean_file(path)
        if had_outlier:
            need_fix_outlier += 1

        if changed and len(samples) < MAX_SAMPLES:
            reasons = []
            if max_cols > 5:
                reasons.append(f"{max_cols}列")
            if had_outlier:
                reasons.append("越界")
            samples.append((path, "+".join(reasons) if reasons else "?"))

        if APPLY and changed and kept is not None:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.writelines(kept)
            written += 1

    print(f"扫描 txt 文件总数            : {total_txt}")
    print(f"  - label 文件              : {label_files}")
    print(f"  - 索引/非 label 文件      : {skipped_index}")
    print(f"  - 空文件                  : {skipped_empty}")
    print(f"  - 含多余置信度列(>5列)    : {need_fix_cols}")
    print(f"  - 含越界坐标              : {need_fix_outlier}")
    if APPLY:
        print(f"  - 已写入修改的文件数      : {written}")

    if samples:
        print(f"\n示例(最多 {MAX_SAMPLES} 个):")
        for p, reason in samples:
            print(f"  [{reason}] {p}")

    if need_fix_cols == 0 and need_fix_outlier == 0:
        print("\n数据集已是标准格式,无需清洗。")
    elif not APPLY:
        print("\n以上为 DRY-RUN 结果。确认无误后把配置区 APPLY 改为 True 再运行一次。")

    # 清理 cache
    if CLEAR_CACHE:
        cache_files = list(root.rglob("*.cache"))
        if cache_files:
            for c in cache_files:
                try:
                    c.unlink()
                except OSError as e:
                    print(f"  [warn] 删除失败 {c}: {e}")
            print(f"\n已删除 {len(cache_files)} 个 .cache 文件。")
        else:
            print("\n未找到 .cache 文件,无需清理。")
    elif APPLY and (need_fix_cols > 0 or need_fix_outlier > 0):
        print("\n提示:建议同时清理 YOLOv5 label cache,"
              "把配置区 CLEAR_CACHE 改为 True。")


if __name__ == "__main__":
    main()
