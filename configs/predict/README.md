# YOLO 推理配置
# =============

本目录包含 YOLO 推理（predict）的配置文件，按任务类型分类。

## 目录结构

```
configs/predict/
├── chaoyuan.yaml        # 潮源模型推理配置
├── parking_pose.yaml    # 停车位姿态推理配置
├── example/             # 示例配置目录
│   ├── detect_example.yaml
│   ├── segment_example.yaml
│   ├── classify_example.yaml
│   ├── pose_example.yaml
│   ├── obb_example.yaml
│   └── track_example.yaml
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

### Python 代码方式

```python
from core.engine import YOLOInference
from core.types import NMSConfig

# 创建推理引擎
engine = YOLOInference(
    model_path="runs/detect/train/weights/best.pt",
    nms_config=NMSConfig(conf_threshold=0.25, iou_threshold=0.7),
    device="cuda",
    imgsz=640,
    half=True,
    stream=False,
    augment=False,
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
| imgsz | int | 640 | 输入图像尺寸 (分类建议 224) |
| device | str/null | null | 计算设备: null=自动, cpu, cuda, 0, 0,1 等 |
| batch | int | 1 | 推理批次大小 |
| classes | list/null | null | 类别过滤 (null=全部, [0,1]=只保留指定类别) |
| stream | bool | false | 流式推理模式 (视频/摄像头推荐 true) |
| half | bool | false | FP16 半精度推理 (需 GPU 支持) |
| augment | bool | false | 测试时增强 TTA (慢但可能提升精度) |
| vid_stride | int | 1 | 视频跳帧间隔 |
| visualize | bool | false | 可视化模型特征图 (调试用) |
| embed | list/null | null | 特征嵌入层索引 (调试用) |
| int8 | bool | false | INT8 量化推理 (ONNX/TensorRT 等支持) |
| dnn | bool | false | 使用 OpenCV DNN 进行 ONNX 推理 |
| show | bool | false | 弹窗显示结果 (需图形界面支持) |
| save_frames | bool | false | 保存视频推理的每一帧为图片 |
| stream_buffer | bool | false | 流式缓冲所有帧 (true=缓冲, false=只保留最新帧) |
| line_width | int/null | null | 后端渲染线宽 (null=自动缩放) |
| show_boxes | bool/null | null | 后端是否绘制检测框 |
| save_conf | bool | false | 保存置信度到结果 (后端保存) |
| retina_masks | bool | false | 高分辨率分割掩码 (segment 任务专用) |
| end2end | bool/null | null | 端到端检测头 (YOLO26/YOLOv10, 免 NMS) |
| kpt_thres | float/null | null | 关键点置信度阈值 (pose 任务专用) |
| topk | int/null | null | 分类 Top-K 结果数 (classify 任务专用) |

### nms 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| conf | float | 0.25 | 置信度阈值 |
| iou | float | 0.7 | NMS IoU 阈值 (越大保留越多框) |
| max_det | int | 300 | 每图最大检测数 |
| agnostic_nms | bool | false | 类别无关 NMS (不同类别框也做 NMS) |

### io 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| input | str | 必填 | 输入路径 (图片/目录/视频/URL/摄像头编号) |
| output | str | "runs/predict" | 输出目录路径 |
| save_vis | bool | true | 保存可视化结果图片/视频 |
| save_json | bool | false | 保存推理结果为 JSON 格式 |
| save_txt | bool | false | 保存 YOLO 格式 txt 标签 |
| save_crop | bool | false | 保存裁剪后的检测目标图片 |

### visualization 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| box_thickness | int | 2 | 前端可视化边框线宽 |
| font_scale | float | 0.5 | 前端可视化标签字体比例 |
| show_labels | bool | true | 显示类别标签 (前端可视化) |
| show_conf | bool | true | 显示置信度分数 (前端可视化) |
| mask_alpha | float | 0.4 | 分割掩码叠加透明度 (segment 任务专用) |
| kpt_radius | int | 5 | 关键点圆点半径 (pose 任务专用) |
| kpt_line | bool | true | 绘制关键点骨架连线 (pose 任务专用) |
| skeleton | list/null | null | 骨架连接定义 (pose 任务专用) |
| kpt_names | dict/null | null | 关键点名称 (pose 任务专用) |

### video 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| fps | float/null | null | 输出视频 FPS (null=保持原始 FPS) |
| codec | str | "mp4v" | 视频编码器 fourcc 码 |

### output 部分

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| verbose | bool | true | 详细日志输出 |

### tracker (track 任务专用)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| tracker | str | null | 跟踪器配置文件: botsort.yaml 或 bytetrack.yaml |

## 快速开始

### 1. 检测任务

```bash
python yolo.py predict --config configs/predict/example/detect_example.yaml
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

### 6. 目标跟踪

```bash
python yolo.py predict --config configs/predict/example/track_example.yaml
```

## 注意事项

1. **配置文件优先级**: 命令行参数 > 配置文件 > 默认值
2. **任务匹配**: 确保配置文件与模型任务类型匹配
3. **GPU 内存**: OBB 和分割任务内存占用大，适当减小 batch
4. **精度选择**: half=true 提速明显且精度损失小，推荐开启
5. **视频处理**: 大视频务必开启 stream=true 避免内存溢出
