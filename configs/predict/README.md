# YOLO预测配置

本目录包含YOLO预测的配置文件，按任务类型分类。

## 目录结构

```
configs/predict/
├── chaoyuan.yaml        # 潮源模型预测配置
├── parking_pose.yaml    # 停车位姿态预测配置
├── example/             # 示例配置目录
│   ├── detect_example.yaml
│   ├── segment_example.yaml
│   ├── classify_example.yaml
│   ├── pose_example.yaml
│   └── obb_example.yaml
```

## 使用方法

### 命令行方式

```bash
# 使用配置文件
python yolo.py predict --config configs/predict/example/detect_example.yaml

# 覆盖配置文件中的参数
python yolo.py predict --config configs/predict/example/detect_example.yaml --conf 0.5 --half

# 不使用配置文件，纯命令行
python yolo.py predict --model best.pt --input images/ --output results/ --conf 0.25
```

### Python代码方式

```python
from commands.predict import YOLOInference, NMSConfig

# 创建推理引擎
engine = YOLOInference(
    model_path="runs/detect/train/weights/best.pt",
    nms_config=NMSConfig(conf_threshold=0.25, iou_threshold=0.7),
    device="cuda",
    imgsz=640,
    half=True,        # FP16半精度
    stream=False,     # 流式推理
    augment=False,    # TTA增强
)

# 单张图片推理
result = engine("image.jpg")

# 批量推理
import cv2
images = [cv2.imread(f"img{i}.jpg") for i in range(10)]
results = engine.inference_batch(images)
```

## 配置文件参数说明

### model 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| path | str | 必填 | 模型路径 (.pt 或 .onnx) |
| format | str | "auto" | 模型格式: auto, pytorch, onnx |
| imgsz | int | 640 | 输入图像尺寸 |
| device | str | "auto" | 运行设备: auto, cpu, cuda, 0, 0,1 |
| batch | int | 1 | 推理批次大小 |
| classes | list/null | null | 类别过滤 |
| stream | bool | false | 流式推理模式 |
| half | bool | false | FP16半精度推理 |
| augment | bool | false | TTA测试时增强 |
| vid_stride | int | 1 | 视频帧间隔 |
| int8 | bool | false | INT8量化推理 |
| visualize | bool | false | 可视化特征 |
| embed | list/null | null | 提取特征嵌入 |

### nms 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| conf | float | 0.25 | 置信度阈值 |
| iou | float | 0.7 | NMS IoU阈值 |
| max_det | int | 300 | 每图最大检测数 |
| agnostic | bool | false | 类别无关NMS |
| kpt_thres | float | 0.5 | 姿态关键点置信度阈值 |
| topk | int | 5 | 分类 Top-K 类别数 |

### io 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| input | str | 必填 | 输入路径 |
| output | str | "runs/predict" | 输出目录 |
| save_vis | bool | true | 保存可视化结果 |
| save_json | bool | false | 保存JSON结果 |
| save_txt | bool | true | 保存YOLO格式标签 |
| save_crop | bool | false | 保存裁剪目标 |
| save_frames | bool | false | 保存视频帧 |

### visualization 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| box_thickness | int | 2 | 框线粗细 |
| font_scale | float | 0.5 | 字体大小 |
| show_labels | bool | true | 显示标签 |
| show_conf | bool | true | 显示置信度 |
| line_width | int/null | null | 框线宽度 |
| skeleton | list | null | 骨架连接定义 (姿态估计) |
| kpt_names | dict | null | 关键点名称 (姿态估计) |

### video 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| fps | float/null | null | 输出FPS |
| codec | str | "mp4v" | 编码器 |
| stream_buffer | bool | false | 流缓冲 |

## 任务特定参数

### segment_example.yaml 特有

```yaml
segmentation:
  retina_masks: false      # 高分辨率掩码 (质量更好但较慢)
  mask_threshold: 0.5      # 掩码二值化阈值

visualization:
  mask_alpha: 0.4          # 掩码透明度 (0-1, 建议0.3-0.5)
```

### classify_example.yaml 特有

```yaml
classification:
  topk: 5                  # 返回前k个预测结果

model:
  imgsz: 224               # 分类通常224x224 (检测通常640)
```

### pose_example.yaml 特有

```yaml
pose:
  kpt_thres: 0.5           # 关键点置信度阈值
  skeleton: [...]          # 骨架连接定义
  kpt_names: [...]         # 关键点名称
  kpt_radius: 5            # 关键点半径
  kpt_line: true           # 绘制骨架连线
```

## 快速开始

### 1. 检测任务

```bash
# 使用配置文件
python yolo.py predict --config configs/predict/example/detect_example.yaml

# 或命令行
python yolo.py predict --model yolov8n.pt --input images/ --output results/
```

### 2. 分割任务

```bash
python yolo.py predict --config configs/predict/example/segment_example.yaml
```

### 3. 分类任务

```bash
python yolo.py predict --config configs/predict/example/classify_example.yaml
```

### 4. 姿态估计

```bash
python yolo.py predict --config configs/predict/example/pose_example.yaml
```

### 5. 旋转框检测

```bash
python yolo.py predict --config configs/predict/example/obb_example.yaml
```

## 注意事项

1. **配置文件优先级**: 命令行参数 > 配置文件 > 默认值
2. **任务匹配**: 确保配置文件与模型任务类型匹配
3. **GPU内存**: OBB和分割任务内存占用大，适当减小batch
4. **精度选择**: half=true 提速明显且精度损失小，推荐开启
5. **视频处理**: 大视频务必开启 stream=true 避免OOM
