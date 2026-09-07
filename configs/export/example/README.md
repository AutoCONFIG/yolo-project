# YOLO Export Configuration Examples

This directory contains example export configurations for different deployment formats.

## Directory Structure

```
export/
├── example/
│   ├── engine/          # TensorRT (`.engine`) export configs
│   ├── onnx/            # ONNX export configs
│   ├── openvino/        # OpenVINO export configs
│   └── torchscript/     # TorchScript export configs
```

## Usage

Each subdirectory should contain YAML configuration files specifying export parameters
for the corresponding format. Example:

```yaml
# configs/export/example/onnx/detect_example.yaml
model:
  path: best.pt
  format: onnx
  imgsz: 640

export:
  simplify: true
  dynamic: false
  quantize: null
  opset: 17
```

## Supported Export Formats

| Format | Suffix | Description |
|--------|--------|-------------|
| onnx | `.onnx` | ONNX (default) |
| torchscript | `.torchscript` | TorchScript |
| openvino | `_openvino_model` | OpenVINO |
| engine | `.engine` | TensorRT |
| coreml | `.mlpackage` | CoreML |
| saved_model | `_saved_model` | TensorFlow SavedModel |
| tflite | `.tflite` | TensorFlow Lite |
| ncnn | `_ncnn_model` | NCNN |

## Creating Custom Export Configs

1. Copy an example config from the appropriate subdirectory
2. Modify `model.path` to point to your trained model
3. Adjust export options (simplify, dynamic, quantize, etc.)
4. Run: `python yolo.py your_config.yaml`

## Common Options

- `simplify`: Simplify ONNX graph (recommended, default: true)
- `dynamic`: Dynamic input shapes (default: false)
- `quantize`: Export precision — `8` (INT8, requires calibration data), `16` (FP16), `32` (FP32), `w8a16`/`w8a32` (mixed)
- `opset`: ONNX opset version (auto-detected if not specified)

## Note

Example configuration files are not yet populated. Please create your own configs
based on the format-specific requirements above.
