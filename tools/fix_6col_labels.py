#!/usr/bin/env python3
"""
修复 YOLO 标签文件：删除多余的第六列（置信度分数等）
支持进度条显示
用法: python fix_6col_labels.py <根目录> [--dry-run]
"""

import os
import sys
from tqdm import tqdm

def fix_label_file(file_path, dry_run=False):
    """修复单个标签文件，删除六行中的最后一列"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except:
            # 无法读取的文件跳过
            return False

    new_lines = []
    modified = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue

        parts = stripped.split()
        if len(parts) == 6:
            new_line = ' '.join(parts[:5])
            new_lines.append(new_line + '\n')
            modified = True
        else:
            new_lines.append(line)

    if modified and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return modified

def main():
    if len(sys.argv) < 2:
        print("用法: python fix_6col_labels.py <根目录> [--dry-run]")
        sys.exit(1)

    root_dir = sys.argv[1]
    dry_run = '--dry-run' in sys.argv

    if not os.path.isdir(root_dir):
        print(f"错误: '{root_dir}' 不是一个有效的目录")
        sys.exit(1)

    print(f"扫描目录: {root_dir}")
    print(f"模拟运行: {'是' if dry_run else '否'}")
    print("正在收集所有 .txt 文件...")

    # 收集所有txt文件路径
    txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.txt'):
                txt_files.append(os.path.join(dirpath, fname))

    total_files = len(txt_files)
    print(f"找到 {total_files} 个 .txt 文件\n")

    # 带进度条处理
    fixed_count = 0
    with tqdm(total=total_files, desc="处理进度", unit="文件", ncols=80) as pbar:
        for file_path in txt_files:
            if fix_label_file(file_path, dry_run=dry_run):
                fixed_count += 1
                if not dry_run:
                    pbar.set_postfix_str(f"已修复: {fixed_count}")
            pbar.update(1)

    print(f"\n完成: 共扫描 {total_files} 个文件，需要修复 {fixed_count} 个")
    if dry_run:
        print("提示: 这是模拟运行，实际修复请去掉 --dry-run 参数")

if __name__ == "__main__":
    main()