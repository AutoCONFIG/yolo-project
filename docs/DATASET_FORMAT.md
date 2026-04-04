# YOLO 数据集格式说明

本文档说明 YOLO 训练所需的数据集目录结构和配置文件格式。

## 目录结构

```
yolo-project/
├── datasets/
│   └── your_dataset/           # 数据集根目录（自定义名称）
│       ├── images/
│       │   ├── train/          # 训练图像目录
│       │   │   ├── img001.jpg
│       │   │   ├── img002.jpg
│       │   │   └── ...
│       │   ├── val/            # 验证图像目录
│       │   │   ├── img101.jpg
│       │   │   └── ...
│       │   └── test/           # 测试图像目录（可选）
│       │       └── ...
│       └── labels/
│           ├── train/          # 训练标签目录
│           │   ├── img001.txt  # 与图像同名，后缀为 .txt
│           │   ├── img002.txt
│           │   └── ...
│           ├── val/            # 验证标签目录
│           │   ├── img101.txt
│           │   └── ...
│           └── test/           # 测试标签目录（可选）
│               └── ...
└── configs/
    └── your_dataset.yaml       # 数据集配置文件
```

## 标签文件格式（YOLO 格式）

每个图像对应一个同名的 `.txt` 标签文件，每行表示一个目标：

```
class_id center_x center_y width height
```

### 参数说明

| 参数 | 说明 | 取值范围 |
|------|------|----------|
| `class_id` | 类别ID，从0开始 | 0, 1, 2, ... |
| `center_x` | 边界框中心X坐标（归一化） | 0.0 ~ 1.0 |
| `center_y` | 边界框中心Y坐标（归一化） | 0.0 ~ 1.0 |
| `width` | 边界框宽度（归一化） | 0.0 ~ 1.0 |
| `height` | 边界框高度（归一化） | 0.0 ~ 1.0 |

### 示例

假设图像尺寸为 640×480，有一个边界框：
- 左上角坐标：(160, 120)
- 右下角坐标：(480, 360)
- 类别：person（类别ID=0）

计算过程：
```
中心点X = (160 + 480) / 2 / 640 = 0.5
中心点Y = (120 + 360) / 2 / 480 = 0.5
宽度 = (480 - 160) / 640 = 0.5
高度 = (360 - 120) / 480 = 0.5
```

标签文件 `img001.txt` 内容：
```
0 0.5 0.5 0.5 0.5
1 0.2 0.3 0.1 0.15
```

### 转换公式

从像素坐标转换为归一化坐标：

```python
# 假设
img_width = 640
img_height = 480
x1, y1 = 160, 120  # 左上角
x2, y2 = 480, 360  # 右下角

# 计算 YOLO 格式
center_x = ((x1 + x2) / 2) / img_width
center_y = ((y1 + y2) / 2) / img_height
width = (x2 - x1) / img_width
height = (y2 - y1) / img_height
```

从归一化坐标转换为像素坐标：

```python
# 假设
img_width = 640
img_height = 480
center_x, center_y, width, height = 0.5, 0.5, 0.5, 0.5

# 计算像素坐标
x1 = (center_x - width / 2) * img_width
y1 = (center_y - height / 2) * img_height
x2 = (center_x + width / 2) * img_width
y2 = (center_y + height / 2) * img_height
```

## 数据集配置文件（YAML）

在 `configs/` 目录下创建数据集配置文件：

```yaml
# configs/my_dataset.yaml

# 数据集根目录路径
# 支持：相对路径、绝对路径、数据集名称（自动下载）
path: ../datasets/my_dataset

# 图像路径（相对于 path）
train: images/train   # 训练集图像目录
val: images/val       # 验证集图像目录
test: images/test     # 测试集图像目录（可选）

# 类别定义
# class_id 必须从 0 开始，连续递增
names:
  0: person
  1: car
  2: dog
  3: cat
  4: bicycle

# 可选：自动下载链接
# download: https://example.com/my_dataset.zip
```

### 配置文件字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `path` | 是 | 数据集根目录路径 |
| `train` | 是 | 训练图像目录（相对路径） |
| `val` | 是 | 验证图像目录（相对路径） |
| `test` | 否 | 测试图像目录（相对路径） |
| `names` | 是 | 类别名称字典，key 为类别ID |
| `download` | 否 | 自动下载 URL |

## 使用方法

### 1. 准备数据集

将数据按照上述目录结构组织好。

### 2. 创建配置文件

在 `configs/` 目录下创建 `your_dataset.yaml`。

### 3. 开始训练

```bash
# 使用配置文件训练
python train.py --config configs/default.yaml --data your_dataset.yaml

# 或者命令行直接指定
python train.py --mode train --model yolo26n.pt --data your_dataset.yaml --epochs 100
```

## 常见问题

### Q: 图像格式支持哪些？

支持常见图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### Q: 标签文件中没有目标怎么办？

创建一个空的 `.txt` 文件即可（文件必须存在，但内容为空）。

### Q: 如何验证数据集是否正确？

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
# 训练前会自动检查数据集
results = model.train(data="your_dataset.yaml", epochs=1)
```

### Q: 如何划分训练集/验证集/测试集？

常见划分比例：
- 训练集：70% ~ 80%
- 验证集：10% ~ 20%
- 测试集：10% ~ 20%

可以使用脚本自动划分：

```python
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_images, source_labels, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """划分数据集"""
    images = list(Path(source_images).glob("*.jpg"))
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }
    
    for split, split_images in splits.items():
        for img_path in split_images:
            # 复制图像
            dst_img = Path(output_dir) / "images" / split / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst_img)
            
            # 复制标签
            label_path = Path(source_labels) / (img_path.stem + ".txt")
            if label_path.exists():
                dst_label = Path(output_dir) / "labels" / split / (img_path.stem + ".txt")
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(label_path, dst_label)
```

## 示例项目结构

完整的训练项目结构示例：

```
yolo-project/
├── train.py                    # 训练脚本
├── inference.py                # 推理脚本
├── run_train.sh               # 训练启动脚本
├── run_inference.sh           # 推理启动脚本
├── configs/
│   ├── default.yaml           # 默认训练配置
│   ├── val.yaml               # 验证配置
│   ├── inference.yaml         # 推理配置
│   ├── dataset_example.yaml   # 示例数据集配置
│   └── my_dataset.yaml        # 你的数据集配置
├── datasets/
│   └── my_dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── runs/                       # 训练输出目录（自动生成）
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.csv
└── ultralytics/               # 子模块
```
