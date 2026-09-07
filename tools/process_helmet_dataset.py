"""
process_helmet_dataset.py
完整的头盔数据集处理流程（流式处理，无中间文件）：
1. 转换类别（大框判断头盔）- 支持2分类/5分类/保持原始分类
2. 立即裁剪小图（严格以单一大框+内部小框外接为基准，多大框重叠不合并，截断部分保留标签）
3. 引入IoS机制，精准过滤远处无关目标
4. 支持自适应扩展像素与最小图片像素面积过滤
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import math


# ==================== 配置参数（统一配置区域）====================
class Config:
    """配置类 - 所有参数在这里统一修改"""
    
    # 源数据集路径
    SOURCE_ROOT = r"/data2/kaiyun/datasets_feijidongche"
    
    # 最终输出路径（裁剪后的小图数据集）
    FINAL_OUTPUT = r"/data2/kaiyun/datasets_feijidongche_xiaotu"
    
    # 分类模式选择
    # "binary": 二分类（0-未戴头盔, 1-戴头盔）
    # "multi": 五分类（0-自行车, 1-电动车戴头盔, 2-电动车未戴, 3-三轮车戴头盔, 4-三轮车未戴）
    # "original": 保持原始分类，不进行头盔判断，只按大框裁剪
    CLASSIFICATION_MODE = "original"  # 可选: "binary", "multi" 或 "original"
    
    # 扩展像素策略
    EXPAND_MODE = "adaptive"  # 可选: "fixed" 或 "adaptive"
    FIXED_EXPAND_PIXELS = 40
    
    # 自适应扩展像素配置
    ADAPTIVE_EXPAND_CONFIG = {
        'small': {'max_size': 100, 'expand': 10},
        'medium': {'max_size': 300, 'expand': 30},
        'large': {'max_size': float('inf'), 'expand': 60}
    }
    
    # IoS阈值 (Intersection over Small box)
    IOS_THRESHOLD = 0.5
    
    # 标签可见占比阈值（当框被裁剪边界截断时，只有可见面积占原始面积的比例>=此阈值才保留该标签）
    VISIBLE_RATIO_THRESHOLD = 0.3  
    
    # 最小裁剪尺寸（像素，单边）
    MIN_SIZE = 10
    
    # ★ 新增：最小图片像素面积（宽×高乘积小于此值的裁剪图直接丢弃，例如65536即为256x256）
    MIN_IMAGE_PIXELS = 65536
    
    # 标签框最小尺寸（归一化坐标，相对于裁剪后的图片，宽或高小于此值则过滤）
    MIN_BOX_SIZE = 0.02

# ================================================================

def imread_chinese(image_path):
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        return None

def imwrite_chinese(image_path, image):
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        ext = os.path.splitext(image_path)[1]
        is_success, buffer = cv2.imencode(ext, image)
        if is_success:
            with open(image_path, 'wb') as f:
                f.write(buffer)
            return True
        return False
    except Exception as e:
        return False


class HelmetDatasetProcessor:
    """头盔数据集处理器（流式处理）"""
    
    def __init__(self, source_root: str, output_root: str, mode: str = "binary",
                 expand_mode: str = "fixed", fixed_expand: int = 40,
                 adaptive_config: Dict = None, ios_threshold: float = 0.5,
                 visible_ratio: float = 0.3, min_size: int = 10, 
                 min_image_pixels: int = 65536, min_box_size: float = 0.02):
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.mode = mode
        self.expand_mode = expand_mode
        self.fixed_expand = fixed_expand
        self.adaptive_config = adaptive_config or {}
        self.ios_threshold = ios_threshold
        self.visible_ratio = visible_ratio
        self.min_size = min_size
        self.min_image_pixels = min_image_pixels
        self.min_box_size = min_box_size
        
        self.original_class_names = ['person', 'head', 'helmet', 'bicycle_person', 
                                     'motorperson', 'tricar_person', 'hand', 'motorperson_canopy']
        
        if mode == "binary":
            self.new_class_names = ['no_helmet', 'with_helmet']
        elif mode == "multi":
            self.new_class_names = [
                'bicycle_person', 'motorperson_with_helmet', 'motorperson_no_helmet',
                'tricar_person_with_helmet', 'tricar_person_no_helmet'
            ]
        elif mode == "original":
            self.new_class_names = self.original_class_names.copy()
        else:
            raise ValueError(f"不支持的分类模式: {mode}")
            
        self.bicycle_person_id = 3
        self.motorperson_id = 4
        self.tricar_person_id = 5
        self.motorperson_canopy_id = 7
        self.head_id = 1
        self.helmet_id = 2
        
        # 只有这些大框才会触发裁剪
        self.large_box_classes = [3, 4, 5, 7]
        
        self.stats = {
            'total_images': 0, 'processed_images': 0, 'total_boxes': 0,
            'converted_boxes': 0, 'cropped_boxes': 0, 'inner_boxes': 0,
            'skipped_small': 0, 'skipped_small_area': 0, 'skipped_ios': 0, 
            'skipped_visible': 0, 'failed_read': 0, 'union_box_used': 0,
        }
        for name in self.new_class_names:
            self.stats[name] = 0
    
    def calculate_expand_pixels(self, box_yolo: List[float], img_w: int, img_h: int) -> int:
        if self.expand_mode == "fixed":
            return self.fixed_expand
        _, _, width, height = box_yolo
        max_side = max(width * img_w, height * img_h)
        for category in ['small', 'medium', 'large']:
            if category in self.adaptive_config:
                config = self.adaptive_config[category]
                if max_side <= config['max_size']:
                    return config['expand']
        return 40
    
    def calculate_box_area(self, box: List[float]) -> float:
        return box[2] * box[3]
    
    def clip_box(self, box: List[float]) -> List[float]:
        x_center, y_center, width, height = box
        x1, y1 = x_center - width / 2, y_center - height / 2
        x2, y2 = x_center + width / 2, y_center + height / 2
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
    
    def calculate_intersection_area(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
        ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
        ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
        if ix1 >= ix2 or iy1 >= iy2: return 0.0
        return (ix2 - ix1) * (iy2 - iy1)
    
    def calculate_ios(self, small_box: List[float], large_box: List[float]) -> float:
        intersection = self.calculate_intersection_area(small_box, large_box)
        small_area = self.calculate_box_area(small_box)
        if small_area == 0: return 0.0
        return intersection / small_area
    
    def calculate_reference_point(self, large_box: List[float]) -> Tuple[float, float]:
        x_center, y_center, width, height = large_box
        return x_center, (y_center - height / 2) + height / 5
    
    def distance_to_point(self, box: List[float], point: Tuple[float, float]) -> float:
        return math.sqrt((box[0] - point[0]) ** 2 + (box[1] - point[1]) ** 2)
    
    def determine_helmet_status(self, large_box: List[float], heads: List[List[float]], helmets: List[List[float]]) -> int:
        has_head, has_helmet = len(heads) > 0, len(helmets) > 0
        if has_head and not has_helmet: return 0
        if has_helmet and not has_head: return 1
        if not has_head and not has_helmet: return 0
        ref_point = self.calculate_reference_point(large_box)
        min_head_dist = min(self.distance_to_point(h, ref_point) for h in heads)
        min_helmet_dist = min(self.distance_to_point(h, ref_point) for h in helmets)
        return 0 if min_head_dist < min_helmet_dist else 1
    
    def get_new_class_id(self, original_class_id: int, helmet_status: int) -> int:
        if self.mode == "binary":
            return helmet_status
        if original_class_id == self.bicycle_person_id: return 0
        elif original_class_id in [self.motorperson_id, self.motorperson_canopy_id]: return 1 if helmet_status == 1 else 2
        elif original_class_id == self.tricar_person_id: return 3 if helmet_status == 1 else 4
        return -1

    def calculate_union_box(self, boxes: List[List[float]]) -> List[float]:
        if not boxes: return [0, 0, 0, 0]
        min_x1, min_y1, max_x2, max_y2 = 1.0, 1.0, 0.0, 0.0
        for box in boxes:
            min_x1 = min(min_x1, box[0] - box[2] / 2)
            min_y1 = min(min_y1, box[1] - box[3] / 2)
            max_x2 = max(max_x2, box[0] + box[2] / 2)
            max_y2 = max(max_y2, box[1] + box[3] / 2)
        return [(min_x1+max_x2)/2, (min_y1+max_y2)/2, max_x2-min_x1, max_y2-min_y1]

    def convert_boxes(self, label_path: Path) -> Tuple[List[Dict], List[Dict]]:
        if not label_path.exists(): return [], []
        
        with open(label_path, 'r') as f: lines = f.readlines()
        all_boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            class_id = int(parts[0])
            box = self.clip_box([float(x) for x in parts[1:5]])
            if box[2] <= 0 or box[3] <= 0: continue
            all_boxes.append({'class_id': class_id, 'box': box})
            self.stats['total_boxes'] += 1
        
        large_boxes = [b for b in all_boxes if b['class_id'] in self.large_box_classes]
        
        converted_boxes = []
        for lb in large_boxes:
            lb_box = lb['box']
            original_class = lb['class_id']
            
            inner_boxes_for_union = [lb_box]
            heads_in = []
            helmets_in = []
            
            for b in all_boxes:
                if b is lb: continue
                ios = self.calculate_ios(b['box'], lb_box)
                
                if ios >= self.ios_threshold:
                    # ★ 核心修改：只有非大框类别（小框）才参与外接框计算，撑大裁剪区域
                    # 其他大框即使在内部或有重叠，也不撑大当前大框的裁剪区域，保证严格以单一大框为基准
                    if b['class_id'] not in self.large_box_classes:
                        inner_boxes_for_union.append(b['box'])
                    
                    # 如果是binary/multi模式，记录内部头/头盔用于判断主框类别
                    if self.mode != "original":
                        if b['class_id'] == self.head_id: heads_in.append(b['box'])
                        elif b['class_id'] == self.helmet_id: helmets_in.append(b['box'])
                elif ios > 0:
                    self.stats['skipped_ios'] += 1
            
            union_box = self.calculate_union_box(inner_boxes_for_union)
            
            if self.mode == "original":
                new_class_id = original_class
            else:
                helmet_status = self.determine_helmet_status(lb_box, heads_in, helmets_in)
                new_class_id = 0 if (self.mode == "multi" and original_class == self.bicycle_person_id) else self.get_new_class_id(original_class, helmet_status)
            
            if new_class_id >= 0:
                if union_box != lb_box:
                    self.stats['union_box_used'] += 1
                
                converted_boxes.append({
                    'class_id': new_class_id,
                    'original_class_id': original_class,
                    'box': lb_box,
                    'union_box': union_box,
                    'original_box_ref': lb
                })
                
                self.stats['converted_boxes'] += 1
                if new_class_id < len(self.new_class_names):
                    self.stats[self.new_class_names[new_class_id]] += 1
        
        return converted_boxes, all_boxes
    
    def crop_box_to_region(self, box_yolo: List[float], crop_region_normalized: List[float]) -> Tuple[List[float], float]:
        bx1, by1 = box_yolo[0] - box_yolo[2] / 2, box_yolo[1] - box_yolo[3] / 2
        bx2, by2 = box_yolo[0] + box_yolo[2] / 2, box_yolo[1] + box_yolo[3] / 2
        cx1, cy1, cx2, cy2 = crop_region_normalized
        
        ix1, iy1 = max(bx1, cx1), max(by1, cy1)
        ix2, iy2 = min(bx2, cx2), min(by2, cy2)
        
        if ix1 >= ix2 or iy1 >= iy2: return None, 0.0
        
        original_area = self.calculate_box_area(box_yolo)
        visible_area = (ix2 - ix1) * (iy2 - iy1)
        visible_ratio = visible_area / original_area if original_area > 0 else 0.0
        
        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        new_x1, new_y1 = (ix1 - cx1) / crop_w, (iy1 - cy1) / crop_h
        new_x2, new_y2 = (ix2 - cx1) / crop_w, (iy2 - cy1) / crop_h
        
        new_x_center, new_y_center = (new_x1 + new_x2) / 2, (new_y1 + new_y2) / 2
        new_width, new_height = new_x2 - new_x1, new_y2 - new_y1
        
        if new_width < self.min_box_size or new_height < self.min_box_size: return None, visible_ratio
        return [new_x_center, new_y_center, new_width, new_height], visible_ratio

    def crop_and_save(self, image: np.ndarray, converted_boxes: List[Dict], all_boxes: List[Dict], img_path: Path, rel_path: Path):
        img_h, img_w = image.shape[:2]
        
        for idx, box_info in enumerate(converted_boxes):
            class_id = box_info['class_id']
            union_box_yolo = box_info['union_box']
            class_name = self.new_class_names[class_id] if class_id < len(self.new_class_names) else f"class_{class_id}"
            
            # 1. 基于外接框计算裁剪区域
            expand_pixels = self.calculate_expand_pixels(union_box_yolo, img_w, img_h)
            crop_x1 = int(union_box_yolo[0] * img_w - union_box_yolo[2] * img_w / 2 - expand_pixels)
            crop_y1 = int(union_box_yolo[1] * img_h - union_box_yolo[3] * img_h / 2 - expand_pixels)
            crop_x2 = int(union_box_yolo[0] * img_w + union_box_yolo[2] * img_w / 2 + expand_pixels)
            crop_y2 = int(union_box_yolo[1] * img_h + union_box_yolo[3] * img_h / 2 + expand_pixels)
            
            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = min(img_w, crop_x2), min(img_h, crop_y2)
            
            crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
            
            # 尺寸检查1：单边过小
            if crop_w < self.min_size or crop_h < self.min_size:
                self.stats['skipped_small'] += 1
                continue
                
            # ★ 尺寸检查2：像素总面积过小（如目标太远）
            if crop_w * crop_h < self.min_image_pixels:
                self.stats['skipped_small_area'] += 1
                continue
            
            cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_region_normalized = [crop_x1 / img_w, crop_y1 / img_h, crop_x2 / img_w, crop_y2 / img_h]
            
            # 2. 生成标签：遍历所有原始框，保留在裁剪区域内的标签（包含截断的其他大框）
            label_lines = []
            inner_count = 0
            
            for orig_box in all_boxes:
                new_box, vis_ratio = self.crop_box_to_region(orig_box['box'], crop_region_normalized)
                
                if new_box is not None:
                    if vis_ratio < self.visible_ratio:
                        self.stats['skipped_visible'] += 1
                        continue
                    
                    is_main_box = (orig_box is box_info['original_box_ref'])
                    
                    # 主框使用转换后ID，其余所有框（包括其他大框和小框）均保留原始ID
                    label_class_id = class_id if is_main_box else orig_box['class_id']
                    
                    label_lines.append(f"{label_class_id} {new_box[0]:.6f} {new_box[1]:.6f} {new_box[2]:.6f} {new_box[3]:.6f}\n")
                    if not is_main_box:
                        inner_count += 1
                        self.stats['inner_boxes'] += 1
            
            # 3. 保存文件
            output_subdir = self.output_root / rel_path.parent
            base_name, ext = img_path.stem, img_path.suffix
            
            output_img_name = f"{base_name}_class{class_id}_{class_name}_box{idx}_exp{expand_pixels}_lbl{len(label_lines)}{ext}"
            output_label_name = f"{base_name}_class{class_id}_{class_name}_box{idx}_exp{expand_pixels}_lbl{len(label_lines)}.txt"
            
            if imwrite_chinese(str(output_subdir / output_img_name), cropped_img):
                (output_subdir / output_label_name).parent.mkdir(parents=True, exist_ok=True)
                with open(output_subdir / output_label_name, 'w') as f:
                    f.writelines(label_lines)
                self.stats['cropped_boxes'] += 1
    
    def process_single_image(self, img_path: Path):
        label_path = img_path.with_suffix('.txt')
        converted_boxes, all_boxes = self.convert_boxes(label_path)
        if not converted_boxes: return
        
        image = imread_chinese(str(img_path))
        if image is None:
            self.stats['failed_read'] += 1
            return
        
        rel_path = img_path.relative_to(self.source_root)
        self.crop_and_save(image, converted_boxes, all_boxes, img_path, rel_path)
        self.stats['processed_images'] += 1
    
    def process_dataset(self):
        mode_name = {"binary": "二分类", "multi": "五分类"}.get(self.mode, "原始分类(不判断头盔)")
        expand_name = "自适应扩展" if self.expand_mode == "adaptive" else f"固定扩展({self.fixed_expand}px)"
        
        print("=" * 70)
        print(f"数据集流式裁剪处理 (单一大框基准，IoS过滤，小图丢弃)")
        print("=" * 70)
        print(f"源路径: {self.source_root}\n输出路径: {self.output_root}")
        print(f"分类模式: {mode_name}\n扩展模式: {expand_name}")
        print(f"IoS阈值: {self.ios_threshold} (小框至少有{self.ios_threshold*100}%在大框内才算内部框)")
        print(f"标签可见度阈值: {self.visible_ratio:.1%}")
        print(f"最小图片像素面积: {self.min_image_pixels} (低于此值的裁剪图丢弃)")
        print(f"输出类别定义: {dict(zip(range(len(self.new_class_names)), self.new_class_names))}")
        print("-" * 70)
        
        for img_path in self.source_root.rglob('*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']: continue
            self.stats['total_images'] += 1
            self.process_single_image(img_path)
            if self.stats['total_images'] % 100 == 0:
                print(f"已处理 {self.stats['total_images']} 张图片, 裁剪 {self.stats['cropped_boxes']} 个大框...")
        
        print("\n" + "=" * 70)
        print("处理完成！统计信息：")
        print(f"原始总框数: {self.stats['total_boxes']}")
        print(f"触发裁剪的大框数: {self.stats['converted_boxes']} (其中内部小框超出大框自动外扩次数: {self.stats['union_box_used']})")
        print(f"实际裁剪保存数: {self.stats['cropped_boxes']}")
        print(f"保留的内部/重叠标签: {self.stats['inner_boxes']}")
        print(f"过滤的远处擦边框(IoS不足): {self.stats['skipped_ios']}")
        print(f"过滤的严重截断框(可见度不足): {self.stats['skipped_visible']}")
        print(f"丢弃的过小裁剪图(单边过小): {self.stats['skipped_small']}")
        print(f"丢弃的过小裁剪图(像素面积不足): {self.stats['skipped_small_area']}")
        print(f"\n各类别统计(仅主大框):")
        for i, name in enumerate(self.new_class_names):
            print(f"  {i}: {name:30s} - {self.stats[name]:6d} 个")
        print("=" * 70)

def main():
    config = Config()
    processor = HelmetDatasetProcessor(
        config.SOURCE_ROOT, config.FINAL_OUTPUT, config.CLASSIFICATION_MODE,
        config.EXPAND_MODE, config.FIXED_EXPAND_PIXELS, config.ADAPTIVE_EXPAND_CONFIG,
        config.IOS_THRESHOLD, config.VISIBLE_RATIO_THRESHOLD, config.MIN_SIZE, 
        config.MIN_IMAGE_PIXELS, config.MIN_BOX_SIZE
    )
    processor.process_dataset()

if __name__ == "__main__":
    main()