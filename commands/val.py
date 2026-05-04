"""
YOLO Validation Module
========================
Standalone validation module invoked via ``python yolo.py val`` or
``python -m commands.val``.

Uses YAML config files and/or CLI arguments.  CLI takes precedence over YAML.

Typical usage::

    # Via unified entry point
    python yolo.py val --config configs/validate/val.yaml
    python yolo.py val --model runs/detect/train/weights/best.pt --data coco8.yaml

    # Direct invocation
    python -m commands.val --config configs/validate/val.yaml
"""

import argparse
import sys
import traceback
from typing import Any, Dict

from utils.config import (
    config_from_args,
    get_nested_value,
    load_yaml_config,
    merge_configs,
    resolve_config_value,
    set_boolean_argument,
    setup_ultralytics_path,
    to_bool,
)

setup_ultralytics_path()
from ultralytics import YOLO

# ─── Argument parser ─────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate with config file
    python -m commands.val --config configs/validate/val.yaml

    # Validate with CLI args
    python -m commands.val --model runs/detect/train/weights/best.pt --data coco8.yaml

    # Validate with custom split and batch size
    python -m commands.val --model best.pt --data coco8.yaml --split test --batch 32
        """,
    )

    # ── Config ────────────────────────────────────────────────────────
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")

    # ── Model ─────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default=None, help="模型路径 (如 best.pt)")
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "segment", "classify", "pose", "obb"],
        default=None,
        help="任务类型",
    )

    # ── Data ──────────────────────────────────────────────────────────
    parser.add_argument("--data", type=str, default=None, help="数据集配置 YAML 文件")
    parser.add_argument("--split", type=str, default=None, help="数据集划分: val / test / train")

    # ── Core validation ───────────────────────────────────────────────
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸 (默认 640)")
    parser.add_argument("--batch", type=int, default=None, help="批大小 (默认 16)")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu, 0, 0,1, mps")
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值 (默认 0.25)")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值 (默认 0.7)")
    parser.add_argument("--max-det", type=int, default=None, help="每张图最大检测数 (默认 300)")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="按类别 ID 过滤 (如 0 或 0 1 2)")
    set_boolean_argument(parser, "half", "half", help_true="FP16 半精度验证", help_false="全精度验证")
    set_boolean_argument(parser, "plots", "plots", help_true="保存验证图表", help_false="不保存图表")
    set_boolean_argument(
        parser, "save_json", "save-json", help_true="保存 COCO JSON 结果", help_false="不保存 JSON"
    )
    set_boolean_argument(
        parser, "dnn", "dnn", help_true="使用 OpenCV DNN 进行 ONNX 推理", help_false="不使用 DNN"
    )
    set_boolean_argument(
        parser, "agnostic_nms", "agnostic-nms",
        help_true="类别无关 NMS", help_false="类别特定 NMS"
    )
    set_boolean_argument(
        parser, "augment", "augment",
        help_true="测试时增强 (TTA)", help_false="不使用 TTA"
    )
    set_boolean_argument(
        parser, "rect", "rect",
        help_true="矩形验证 (更快但精度略低)", help_false="正常方形验证"
    )
    set_boolean_argument(
        parser, "save_conf", "save-conf",
        help_true="在 txt 标签中保存置信度分数", help_false="不保存置信度"
    )
    set_boolean_argument(
        parser, "int8", "int8",
        help_true="INT8 量化推理验证", help_false="不使用 INT8"
    )
    set_boolean_argument(
        parser, "end2end", "end2end",
        help_true="端到端检测头验证 (YOLO26/YOLOv10)", help_false="标准验证"
    )

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=None, help="结果根目录的项目名")
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--save-period", type=int, default=None, help="每 N 轮保存检查点 (-1 = 关闭)")
    set_boolean_argument(parser, "exist_ok", "exist-ok", help_true="覆盖已有项目/名称", help_false="不覆盖")
    set_boolean_argument(parser, "verbose", "verbose", help_true="详细输出", help_false="简洁输出")

    return parser.parse_args()


# ─── CLI → nested config ──────────────────────────────────────────────


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config = {}

    # Model (imgsz/batch/device 也属于 model 节，与 validate() 读取一致)
    model_cfg = config_from_args(
        args, plain=("model", "task", "classes", "imgsz", "batch", "device"),
        rename={"model": "name"}
    )
    if model_cfg:
        config["model"] = model_cfg

    # Data
    data_cfg = config_from_args(
        args, plain=("data", "split"), rename={"data": "config"}
    )
    if data_cfg:
        config["data"] = {**config.get("data", {}), **data_cfg}

    # Validation
    val_cfg = config_from_args(
        args,
        boolean=("half", "plots", "save_json", "dnn", "agnostic_nms",
                 "augment", "rect", "save_conf", "int8", "end2end"),
        plain=("conf", "iou", "max_det"),
    )
    if val_cfg:
        config["validation"] = {**config.get("validation", {}), **val_cfg}

    # Output
    out_cfg = config_from_args(
        args,
        boolean=("exist_ok", "verbose"),
        plain=("project", "name", "save_period"),
    )
    if out_cfg:
        config["output"] = {**config.get("output", {}), **out_cfg}

    return config


# ─── Validate ─────────────────────────────────────────────────────────


def validate(config: Dict):
    """验证 YOLO 模型。"""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")

    val_args = {
        "data": get_nested_value(config, "data", "config", default="coco8.yaml"),
        "split": get_nested_value(config, "data", "split", default="val"),
        "imgsz": resolve_config_value(config, ("model", "imgsz"), ("train", "imgsz"), default=640),
        "batch": resolve_config_value(config, ("model", "batch"), ("train", "batch"), default=16),
        "iou": get_nested_value(config, "validation", "iou", default=0.7),
        "plots": get_nested_value(config, "validation", "plots", default=True),
        "save_json": get_nested_value(config, "validation", "save_json", default=False),
        "dnn": get_nested_value(config, "validation", "dnn", default=False),
        "agnostic_nms": get_nested_value(config, "validation", "agnostic_nms", default=False),
        "augment": get_nested_value(config, "validation", "augment", default=False),
        "half": get_nested_value(config, "validation", "half", default=False),
        "max_det": get_nested_value(config, "validation", "max_det", default=300),
        "verbose": get_nested_value(config, "output", "verbose", default=True),
    }

    # conf 默认不传递，让后端使用自己的默认值 (val 时默认 0.001)
    conf = get_nested_value(config, "validation", "conf")
    if conf is not None:
        val_args["conf"] = conf

    # 可选参数: 仅在配置中显式设置时才传递
    device = resolve_config_value(config, ("model", "device"), ("train", "device"))
    if device is not None:
        val_args["device"] = device

    classes = get_nested_value(config, "model", "classes")
    if classes is not None:
        val_args["classes"] = classes

    # 矩形验证 (更快但精度略低)
    rect = get_nested_value(config, "validation", "rect")
    if rect is not None:
        val_args["rect"] = rect

    # 在 txt 标签中保存置信度分数
    save_conf = get_nested_value(config, "validation", "save_conf")
    if save_conf is not None:
        val_args["save_conf"] = save_conf

    # INT8 量化推理验证
    int8 = get_nested_value(config, "validation", "int8")
    if int8 is not None:
        val_args["int8"] = int8

    # 端到端检测头 (YOLO26/YOLOv10, 无 NMS 推理)
    end2end = get_nested_value(config, "validation", "end2end")
    if end2end is not None:
        val_args["end2end"] = end2end

    for key in ("project", "name", "save_period", "exist_ok"):
        v = get_nested_value(config, "output", key)
        if v is not None:
            val_args[key] = v

    print(f"\n{'='*60}")
    print("YOLO 验证配置")
    print(f"{'='*60}")
    print(f"模型:      {model_name}")
    print(f"数据集:    {val_args['data']}")
    print(f"划分:      {val_args['split']}")
    print(f"图像尺寸:  {val_args['imgsz']}")
    print(f"批大小:    {val_args['batch']}")
    print(f"设备:      {val_args.get('device', 'auto')}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)
    metrics = model.val(**val_args)

    # 根据任务类型打印对应的指标
    task = model.task if hasattr(model, "task") else "detect"

    print(f"\n{'='*60}")
    print("验证结果")
    print(f"{'='*60}")

    if task == "classify":
        # 分类任务: top-1 / top-5 准确率
        print(f"Top-1 Accuracy: {metrics.top1:.4f}")
        print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    elif task == "segment":
        # 分割任务: box + mask mAP
        print(f"Box mAP50-95:  {metrics.box.map:.4f}")
        print(f"Box mAP50:     {metrics.box.map50:.4f}")
        print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
        print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
    elif task == "pose":
        # 姿态任务: box + pose mAP
        print(f"Box mAP50-95:  {metrics.box.map:.4f}")
        print(f"Box mAP50:     {metrics.box.map50:.4f}")
        print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
        print(f"Pose mAP50:    {metrics.pose.map50:.4f}")
    else:
        # detect / obb: box mAP
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP75:    {metrics.box.map75:.4f}")

    print(f"{'='*60}\n")

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)  # YAML in base, CLI in override = CLI wins

        validate(config)
    except KeyboardInterrupt:
        print("\n验证被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"错误: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()