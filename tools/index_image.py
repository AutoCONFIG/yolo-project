"""
图片索引工具 - 极速优化版
用于构建YOLO数据集的图片索引文件

优化特性:
- 支持跳过图片损坏检查，只校验标签匹配性
- 多进程批量预处理元数据
- 快速标签解析
"""

import os
import random
import logging
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

# ===================================================================
# 配置参数设置区域 - 在这里统一修改所有参数
# ===================================================================

# 路径配置
ROOT_DIRECTORY = "/data2/kaiyun/datasets_11_classes_remap"  # 图片根目录路径
OUTPUT_DIRECTORY = None  # 输出目录，None表示使用图片根目录路径

# 按子文件夹模式配置
SCAN_BY_SUBFOLDERS = True  # True: 按一级子文件夹分别生成txt
                           # False: 扫描整个目录，输出到根目录
SKIP_EXISTING_FOLDERS = True  # True: 跳过已有txt的子文件夹

# 分割比例配置
TRAIN_RATIO = 0.9    # 训练集比例
VAL_RATIO = 0.1      # 验证集比例
ALLOW_OVERLAP = False  # 是否允许训练集和验证集重复

# 其他配置
RANDOM_SEED = 36  # 随机种子
FILE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图片格式
LABEL_SUFFIX = ".txt"  # 标签文件后缀
EXPECTED_LABEL_IDS = "0-11"  # 期望的标签ID范围配置

# ============ 核心功能配置 =============
SKIP_IMAGE_CHECK = True       # 【新增】True: 不验证图片是否损坏，只要有txt就索引
                              # False: 会读取图片并尝试解码，剔除损坏图片

CHECK_JPEG_INTEGRITY = False   # 是否检查JPEG结构（当SKIP_IMAGE_CHECK为False时有效）
ENABLE_BATCH_PREPROCESS = True # 启用多进程元数据预处理
PREPROCESS_WORKERS = 12        # 预处理进程数

# 输出文件名配置
TRAIN_FILE_NAME = "train_20260807.txt"
VAL_FILE_NAME = "test_20260807.txt"
LOG_FILE_NAME = "index_image_errors.log"

# ===================================================================
# 以下代码无需修改
# ===================================================================

def parse_expected_label_ids(expected_str):
    if not expected_str or not isinstance(expected_str, str):
        return set()
    allowed_ids = set()
    parts = expected_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            try:
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    start, end = int(range_parts[0]), int(range_parts[1])
                    for i in range(start, end + 1): allowed_ids.add(i)
            except: continue
        else:
            try: allowed_ids.add(int(part))
            except: continue
    return allowed_ids

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, LOG_FILE_NAME)
    logger = logging.getLogger('index_image')
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def _check_file_metadata(args):
    """多进程检查文件元数据及标签内容"""
    img_path, label_suffix, allowed_ids = args
    result = {
        'path': img_path,
        'label_exists': False,
        'label_empty': True,
        'label_valid': True, 
        'error': None
    }
    
    try:
        dir_name = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(dir_name, base_name + label_suffix)
        
        if os.path.exists(label_path):
            result['label_exists'] = True
            content = ""
            # 尝试不同编码读取标签
            for enc in ['utf-8', 'gbk']:
                try:
                    with open(label_path, 'r', encoding=enc) as f:
                        lines = f.readlines()
                        content = "".join(lines).strip()
                        if content:
                            result['label_empty'] = False
                            # 校验ID是否越界
                            for line in lines:
                                parts = line.strip().split()
                                if parts and int(parts[0]) not in allowed_ids:
                                    result['label_valid'] = False
                                    break
                    break 
                except: continue
    except Exception as e:
        result['error'] = str(e)
    return result

def batch_preprocess_metadata(image_paths, num_workers, label_suffix, allowed_ids):
    print(f"  [元数据预处理] 并行检查 {len(image_paths)} 个文件的标签状态...")
    args_list = [(path, label_suffix, allowed_ids) for path in image_paths]
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_check_file_metadata, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  预处理进度", ncols=80):
            results.append(future.result())
    return results

def process_single_folder_optimized(folder_path, train_ratio, val_ratio, allow_overlap, 
                                   seed, file_extensions, logger):
    if seed is not None: random.seed(seed)
    allowed_ids = parse_expected_label_ids(EXPECTED_LABEL_IDS)
    
    # 扫描所有图片
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in file_extensions):
                image_paths.append(os.path.abspath(os.path.join(root, file)))
    
    if not image_paths:
        print(f"  [跳过] 文件夹中无图片: {folder_path}")
        return None

    total_images = len(image_paths)
    valid_image_paths = []
    missing_count = 0
    empty_count = 0
    invalid_label_count = 0
    corrupt_img_count = 0

    # 1. 批量预处理元数据
    metadata_results = batch_preprocess_metadata(image_paths, PREPROCESS_WORKERS, LABEL_SUFFIX, allowed_ids)
    metadata_map = {r['path']: r for r in metadata_results}

    # 2. 筛选与校验
    print(f"  [筛选] 正在根据配置校验数据...")
    for img_path in image_paths:
        meta = metadata_map.get(img_path)
        if not meta: continue
        
        # 标签校验
        if not meta['label_exists']:
            missing_count += 1
            continue
        if meta['label_empty']:
            empty_count += 1
            continue
        if not meta['label_valid']:
            invalid_label_count += 1
            continue

        # 图片校验 (根据配置决定是否执行)
        if not SKIP_IMAGE_CHECK:
            try:
                # 尝试解码图片
                img = cv2.imread(img_path)
                if img is None or img.shape[0] <= 0:
                    corrupt_img_count += 1
                    continue
            except:
                corrupt_img_count += 1
                continue
        
        # 通过校验
        valid_image_paths.append(img_path)

    if not valid_image_paths:
        print(f"  [警告] 经过筛选后无有效数据: {folder_path}")
        return None

    # 3. 分割数据
    random.shuffle(valid_image_paths)
    total_valid = len(valid_image_paths)
    train_count = int(total_valid * train_ratio)
    
    train_paths = valid_image_paths[:train_count]
    val_paths = valid_image_paths[train_count:]

    # 4. 写入文件
    for file_name, paths in [(TRAIN_FILE_NAME, train_paths), (VAL_FILE_NAME, val_paths)]:
        with open(os.path.join(folder_path, file_name), 'w', encoding='utf-8') as f:
            for p in paths: f.write(p + '\n')

    return {
        'total': total_images,
        'valid': total_valid,
        'train': len(train_paths),
        'val': len(val_paths),
        'missing': missing_count,
        'empty': empty_count,
        'invalid_label': invalid_label_count,
        'corrupt_img': corrupt_img_count
    }

def process_subfolders(root_directory, train_ratio, val_ratio, allow_overlap, seed, 
                      file_extensions, skip_existing):
    logger = setup_logger(root_directory)
    subfolders = [os.path.join(root_directory, d) for d in os.listdir(root_directory) 
                  if os.path.isdir(os.path.join(root_directory, d))]
    subfolders.sort()

    print(f"模式: 按子文件夹扫描 | 损坏检查: {'关闭' if SKIP_IMAGE_CHECK else '开启'}")
    
    total_stats = {'valid': 0, 'train': 0, 'val': 0}

    for i, folder in enumerate(subfolders, 1):
        folder_name = os.path.basename(folder)
        if skip_existing and os.path.exists(os.path.join(folder, TRAIN_FILE_NAME)):
            print(f"[{i}/{len(subfolders)}] 跳过: {folder_name}")
            continue
            
        print(f"[{i}/{len(subfolders)}] 处理: {folder_name}")
        res = process_single_folder_optimized(folder, train_ratio, val_ratio, allow_overlap, seed, file_extensions, logger)
        
        if res:
            total_stats['valid'] += res['valid']
            total_stats['train'] += res['train']
            total_stats['val'] += res['val']
            print(f"    成功: {res['valid']} | 缺失标签: {res['missing']} | 空标签: {res['empty']}")
            if res['corrupt_img'] > 0: print(f"    损坏图片(剔除): {res['corrupt_img']}")

    # 合并根目录
    print(f"\n[合并] 正在汇总到根目录...")
    for out_name in [TRAIN_FILE_NAME, VAL_FILE_NAME]:
        with open(os.path.join(root_directory, out_name), 'w', encoding='utf-8') as outfile:
            for folder in subfolders:
                part_path = os.path.join(folder, out_name)
                if os.path.exists(part_path):
                    with open(part_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())

    print(f"\n=== 任务完成 ===")
    print(f"总有效索引: {total_stats['valid']} (训练: {total_stats['train']}, 验证: {total_stats['val']})")

def split_image_files():
    if SCAN_BY_SUBFOLDERS:
        process_subfolders(ROOT_DIRECTORY, TRAIN_RATIO, VAL_RATIO, ALLOW_OVERLAP, 
                          RANDOM_SEED, FILE_EXTENSIONS, SKIP_EXISTING_FOLDERS)
    else:
        logger = setup_logger(ROOT_DIRECTORY)
        process_single_folder_optimized(ROOT_DIRECTORY, TRAIN_RATIO, VAL_RATIO, ALLOW_OVERLAP, 
                                       RANDOM_SEED, FILE_EXTENSIONS, logger)

if __name__ == "__main__":
    split_image_files()
