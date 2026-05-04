# YOLO Project Frontend Configuration Guide

This document describes the custom frontend (non-ultralytics) configuration system, including frontend-only keys, structure conventions, and which parameters are passed through to the ultralytics backend.

## Architecture Overview

- **Backend**: `ultralytics/` submodule (v8.4.45) — do not modify.
- **Frontend**: Everything else (`commands/`, `core/`, `utils/`, `configs/`, `tools/`) — our custom CLI and inference pipeline.
- **Config files**: YAML files under `configs/` are read by frontend command modules (`train.py`, `val.py`, `predict.py`, `export.py`), which parse them and forward relevant args to ultralytics APIs.

---

## Config File Structure

All config files use a nested dictionary structure. CLI arguments are merged into (and override) YAML config via `utils.config.merge_configs`.

Common top-level sections:

```yaml
model:        # Model path, format, device, imgsz, batch, etc.
data:         # Dataset config path and split
train:        # Training hyperparameters (train-only)
validation:   # Validation parameters (val-only / train validation loop)
augmentation: # Augmentation hyperparameters (train-only)
export:       # Export options (export-only)
io:           # Input/output paths and save flags (predict-only)
nms:          # NMS thresholds (predict-only)
visualization:# Drawing settings (predict-only)
video:        # Video codec / fps (predict-only)
output:       # Project/name/exist_ok/verbose/save_period
```

---

## Frontend-Only Keys (Not Passed to Ultralytics)

These keys are consumed exclusively by the frontend code and never forwarded to `ultralytics` APIs.

### Predict / Inference

| Key | Type | Description |
|-----|------|-------------|
| `io.input` | `str` | Input image/video/directory path |
| `io.output` | `str` | Output directory path |
| `io.save_vis` | `bool` | Save annotated visualization images |
| `io.save_json` | `bool` | Save results as JSON (frontend implementation) |
| `io.save_txt` | `bool` | Save YOLO-format txt labels (frontend implementation) |
| `io.save_crop` | `bool` | Save cropped detections (frontend implementation) |
| `visualization.box_thickness` | `int` | Bounding box line thickness for **frontend drawing** |
| `visualization.font_scale` | `float` | Label font scale for **frontend drawing** |
| `visualization.show_labels` | `bool` | Show class labels on visualization |
| `visualization.show_conf` | `bool` | Show confidence scores on visualization |
| `visualization.skeleton` | `List[List[int]]` | Pose keypoint connections (e.g. `[[0,1], [1,2]]`) |
| `visualization.kpt_names` | `Dict[int, List[str]]` | **Must be dict**: `{class_id: [name1, name2, ...]}` |
| `visualization.mask_alpha` | `float` | Segmentation mask overlay alpha |
| `visualization.kpt_radius` | `int` | Keypoint circle radius |
| `visualization.kpt_line` | `bool` | Draw skeleton lines |
| `video.codec` | `str` | OpenCV fourcc codec (default: `mp4v`) |
| `video.fps` | `float` | Output video FPS override |
| `verbose` / `output.verbose` | `bool` | Frontend console verbosity |

> **Important**: `model.line_width` is passed to the ultralytics backend (`predict()`), while `visualization.box_thickness` is used by the **frontend** `draw_detections()` function. Do not confuse the two.

### Train / Val / Export

Most keys in `train/`, `validation/`, `export/`, `augmentation/`, and `model/` are **directly forwarded** to ultralytics. The frontend mainly acts as a config parser and wrapper.

Frontend-only organizational keys:

| Key | Type | Description |
|-----|------|-------------|
| `output.project` | `str` | Results root directory |
| `output.name` | `str` | Experiment run name |
| `output.exist_ok` | `bool` | Overwrite existing run directory |
| `output.save_period` | `int` | Save checkpoint every N epochs |
| `output.verbose` | `bool` | Verbose logging during train/val |

---

## Backend-Passthrough Keys (Forwarded to Ultralytics)

The following keys are extracted from config and passed directly to ultralytics `model.train()`, `model.val()`, `model.predict()`, or `model.export()`.

### Predict passthrough

All keys under `model.*` (except `model.path` which is used to load the model) are passed as kwargs to `model.predict()` via `core.engine.YOLOInference._predict_kwargs()`.

Examples: `imgsz`, `conf`, `iou`, `max_det`, `agnostic_nms`, `classes`, `stream`, `half`, `augment`, `vid_stride`, `retina_masks`, `visualize`, `embed`, `int8`, `save_conf`, `save_frames`, `stream_buffer`, `dnn`, `end2end`, `line_width`, `show`, `kpt_thres`, `topk`.

### Train passthrough

All keys under `train.*`, `augmentation.*`, `validation.*`, `output.*`, plus `model.task`, `model.pretrained`, `model.classes`, `cfg`, `tracker` are forwarded to `model.train()`.

### Val passthrough

All keys under `validation.*`, `model.*` (imgsz, batch, device, classes, task), `data.*`, and `output.*` are forwarded to `model.val()`.

### Export passthrough

All keys under `export.*`, `model.*` (format, imgsz, batch, device), `output.verbose`, and `verify.*` are forwarded to `model.export()`.

---

## Configuration Conventions & Pitfalls

### 1. `kpt_names` must be a **dict**, not a list

```yaml
# CORRECT
visualization:
  kpt_names:
    0: [front_left, front_right, rear_right, rear_left]

# WRONG — will be silently ignored
visualization:
  kpt_names:
    - front_left
    - front_right
```

### 2. Pose skeleton belongs under `visualization`, not `pose`

```yaml
# CORRECT
visualization:
  skeleton: [[0, 1], [1, 2], [2, 3], [3, 0]]

# WRONG — code reads from visualization.skeleton, not pose.skeleton
pose:
  skeleton: ...
```

### 3. `cfg` must be a **string path** (or null), never a dict

```yaml
# CORRECT
cfg: configs/custom.yaml   # string path or omitted
tracker: botsort.yaml

# WRONG — backend expects cfg to be a string, not a nested dict
cfg:
  tracker: botsort.yaml
```

### 4. `pretrained` semantics in train configs

- In **YAML**: `pretrained: false` (bool) means train from scratch; `pretrained: true` (bool) means use default pretrained weights; `pretrained: path/to/weights.pt` (str) means load specific weights.
- On **CLI**: use `--pretrained-bool true/false` for boolean control, or `--pretrained <path>` for a weight path. Do **not** pass `--pretrained false` as a string — it will be treated as a filename.

### 5. CLI defaults must not strip user overrides

Never delete a CLI argument from the config dict just because it matches a hardcoded default. If YAML has `batch: 64` and user runs `--batch 1`, the `1` must survive into the merged config to override YAML.

### 6. Boolean CLI arguments

Use `utils.config.set_boolean_argument(parser, "flag", "flag")` to generate `--flag` / `--no-flag` pairs. This yields `None` when omitted, allowing YAML defaults to take effect. Avoid `type=str, choices=["true", "false"]` for booleans.

---

## File Responsibilities

| File | Role |
|------|------|
| `yolo.py` | Unified entry point; routes `train/val/predict/export` to `commands.*` |
| `commands/train.py` | Parses train config, loads model, calls `model.train()` |
| `commands/val.py` | Parses val config, loads model, calls `model.val()` |
| `commands/predict.py` | Parses predict config, runs `core.engine.YOLOInference`, saves results |
| `commands/export.py` | Parses export config, calls `model.export()`, optionally verifies |
| `core/engine.py` | `YOLOInference` class: PyTorch / ONNX inference, preprocessing, NMS |
| `core/parser.py` | Converts ultralytics `Results` objects to frontend `ImageResult` |
| `core/visualization.py` | `draw_detections()`: frontend OpenCV rendering for all task types |
| `core/video.py` | Video file collection, video inference with frame-by-frame processing |
| `utils/config.py` | YAML loading, config merging, CLI boolean helpers, path setup |
| `utils/constants.py` | Default values, color palettes, file extension sets |

---

## Model Config Keys Supported by Ultralytics Default YAML

Reference: `ultralytics/ultralytics/cfg/default.yaml`

All keys listed there are valid backend arguments. The frontend config system is a wrapper around these; any key present in `default.yaml` can be placed in the appropriate frontend config section and will be forwarded.

Key frontend sections mapping to backend args:
- `train.*` -> `model.train()` kwargs
- `validation.*` -> `model.val()` kwargs (also forwarded during training loop)
- `augmentation.*` -> `model.train()` kwargs (augmentation hyperparameters)
- `export.*` -> `model.export()` kwargs
- `model.*` -> varies by command (predict/val/train/export all read model settings)
