"""
YOLO Training Module
=====================
Standalone training module invoked via ``python yolo.py train`` or
``python -m commands.train``.

Uses YAML config files and/or CLI arguments.  CLI takes precedence over YAML.

Typical usage::

    # Via unified entry point
    python yolo.py train --config configs/train/chaoyuan.yaml
    python yolo.py train --model yolo26n.pt --data coco8.yaml --epochs 50

    # Direct invocation
    python -m commands.train --config configs/train/chaoyuan.yaml
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from utils.config import (
    PROJECT_ROOT,
    config_from_args,
    get_nested_value,
    load_yaml_config,
    merge_configs,
    set_boolean_argument,
    setup_ultralytics_path,
    to_bool,
)

setup_ultralytics_path()
from ultralytics import YOLO


# ─── Argument parser ─────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with config file
    python -m commands.train --config configs/train/chaoyuan.yaml

    # Train with CLI overrides
    python -m commands.train --config configs/train/chaoyuan.yaml --epochs 50 --batch 32

    # Quick training without config
    python -m commands.train --model yolo26n.pt --data coco8.yaml --epochs 100

    # Resume training
    python -m commands.train --model runs/detect/train/weights/last.pt --resume
        """,
    )

    # ── Config ────────────────────────────────────────────────────────
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")

    # ── Model ─────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default=None, help="模型路径或名称 (如 yolo26n.pt)")
    parser.add_argument("--model-yaml", type=str, default=None, help="从 YAML 配置从零构建模型")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="预训练权重路径 (如 yolo26n.pt); 布尔控制请用 --pretrained-bool",
    )
    parser.add_argument(
        "--pretrained-bool",
        type=str,
        default=None,
        choices=["true", "false"],
        help="是否使用预训练模型 (布尔字符串); 优先于 --pretrained",
        dest="pretrained_bool",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "segment", "classify", "pose", "obb"],
        default=None,
        help="任务类型",
    )
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="按类别 ID 过滤训练 (如 0 或 0 1 2)")

    # ── Data ──────────────────────────────────────────────────────────
    parser.add_argument("--data", type=str, default=None, help="数据集配置 YAML 文件")
    parser.add_argument("--split", type=str, default=None, help="数据集划分: val / test / train")

    # ── Core training ────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认 100)")
    parser.add_argument("--time", type=float, default=None, help="最大训练时长(小时), 设置后覆盖 --epochs")
    parser.add_argument("--batch", type=int, default=None, help="批大小 (默认 16)")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸 (默认 640)")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu, 0, 0,1, mps")
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数 (默认 8)")
    set_boolean_argument(parser, "resume", "resume", help_true="从上次检查点恢复训练")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心轮数 (默认 100)")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=["ram", "disk"],
        help="缓存图像到内存或磁盘以加速训练",
    )
    set_boolean_argument(parser, "save", "save", help_true="保存训练检查点", help_false="不保存检查点")
    set_boolean_argument(parser, "amp", "amp", help_true="自动混合精度训练", help_false="禁用 AMP")
    set_boolean_argument(parser, "rect", "rect", help_true="矩形批量训练", help_false="正常方形训练")
    set_boolean_argument(parser, "single_cls", "single-cls", help_true="单类别模式", help_false="多类别模式")
    parser.add_argument("--fraction", type=float, default=None, help="使用数据集的比例 (0.0-1.0)")
    parser.add_argument("--freeze", type=int, nargs="+", default=None, help="冻结前 N 层 (整数或列表)")
    parser.add_argument(
        "--multi-scale",
        type=float,
        default=None,
        help="多尺度训练范围, 以 imgsz 的比例表示 (0.0 = 关闭)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default=None,
        choices=["true", "false", "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
        help="torch.compile 模式",
    )
    set_boolean_argument(parser, "end2end", "end2end", help_true="端到端检测头 (YOLO26/YOLOv10)", help_false="标准检测头")
    parser.add_argument("--nbs", type=int, default=None, help="损失归一化的标称批大小 (默认 64)")
    set_boolean_argument(parser, "profile", "profile", help_true="性能分析 ONNX/TensorRT 速度", help_false="不进行性能分析")

    # ── Optimizer ─────────────────────────────────────────────────────
    parser.add_argument("--optimizer", type=str, default=None, help="优化器: SGD/Adam/AdamW/auto (默认 auto)")
    parser.add_argument("--lr0", type=float, default=None, help="初始学习率 (默认 0.01)")
    parser.add_argument("--lrf", type=float, default=None, help="最终学习率比例 (默认 0.01)")
    parser.add_argument("--momentum", type=float, default=None, help="SGD 动量 / Adam beta1 (默认 0.937)")
    parser.add_argument("--weight-decay", type=float, default=None, help="权重衰减 L2 (默认 0.0005)")
    set_boolean_argument(
        parser, "cos_lr", "cos-lr",
        help_true="使用余弦学习率调度",
        help_false="禁用余弦学习率 (使用阶梯衰减)",
    )
    parser.add_argument("--close-mosaic", type=int, default=None, help="最后 N 轮禁用马赛克增强 (默认 10, 0 = 保持)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认 0)")
    set_boolean_argument(parser, "deterministic", "deterministic", help_true="确定性操作以保证可复现性", help_false="非确定性模式")

    # ── Warmup ────────────────────────────────────────────────────────
    parser.add_argument("--warmup-epochs", type=float, default=None, help="预热轮数 (默认 3.0)")
    parser.add_argument("--warmup-momentum", type=float, default=None, help="预热初始动量 (默认 0.8)")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, help="预热偏置学习率 (默认 0.1)")

    # ── Loss gains ────────────────────────────────────────────────────
    parser.add_argument("--box", type=float, default=None, help="边框损失增益 (默认 7.5)")
    parser.add_argument("--cls", type=float, default=None, help="类别损失增益 (默认 0.5)")
    parser.add_argument("--cls-pw", type=float, default=None, help="类别权重幂, 用于类别不平衡 (默认 0.0)")
    parser.add_argument("--dfl", type=float, default=None, help="分布焦点损失增益 (默认 1.5)")
    parser.add_argument("--pose", type=float, default=None, help="姿态损失增益 (默认 12.0)")
    parser.add_argument("--kobj", type=float, default=None, help="关键点目标损失增益 (默认 1.0)")
    parser.add_argument("--rle", type=float, default=None, help="RLE 损失增益 (默认 1.0)")
    parser.add_argument("--angle", type=float, default=None, help="OBB 角度损失增益 (默认 1.0)")

    # ── Task-specific ─────────────────────────────────────────────────
    set_boolean_argument(parser, "overlap_mask", "overlap-mask", help_true="训练时合并实例掩码 (仅 segment)", help_false="不合并实例掩码")
    parser.add_argument("--mask-ratio", type=int, default=None, help="掩码下采样比 (仅 segment, 默认 4)")
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="分类头 Dropout (仅 classify, 0.0-1.0)",
    )

    # ── Augmentation ──────────────────────────────────────────────────
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV 色调增强 (默认 0.015)")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV 饱和度增强 (默认 0.7)")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV 明度增强 (默认 0.4)")
    parser.add_argument("--degrees", type=float, default=None, help="旋转角度 ± (默认 0.0)")
    parser.add_argument("--translate", type=float, default=None, help="平移比例 ± (默认 0.1)")
    parser.add_argument("--scale", type=float, default=None, help="缩放增益 ± (默认 0.5)")
    parser.add_argument("--shear", type=float, default=None, help="剪切角度 ± (默认 0.0)")
    parser.add_argument("--perspective", type=float, default=None, help="透视变换比例 (默认 0.0)")
    parser.add_argument("--flipud", type=float, default=None, help="垂直翻转概率 (默认 0.0)")
    parser.add_argument("--fliplr", type=float, default=None, help="水平翻转概率 (默认 0.5)")
    parser.add_argument("--bgr", type=float, default=None, help="RGB↔BGR 通道交换概率 (默认 0.0)")
    parser.add_argument("--mosaic", type=float, default=None, help="马赛克增强概率 (默认 1.0)")
    parser.add_argument("--mixup", type=float, default=None, help="混合增强概率 (默认 0.0)")
    parser.add_argument("--cutmix", type=float, default=None, help="CutMix 增强概率 (默认 0.0)")
    parser.add_argument("--copy-paste", type=float, default=None, help="复制粘贴概率 (仅 segment, 默认 0.0)")
    parser.add_argument(
        "--copy-paste-mode",
        type=str,
        default=None,
        choices=["flip", "mixup"],
        help="分割复制粘贴策略 (默认 flip)",
    )
    parser.add_argument(
        "--auto-augment",
        type=str,
        default=None,
        choices=["randaugment", "autoaugment", "augmix"],
        help="自动增强策略 (仅 classify, 默认 randaugment)",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=None,
        help="随机擦除概率 (仅 classify, 默认 0.4)",
    )

    # ── Validation during training ────────────────────────────────────
    set_boolean_argument(parser, "val", "val", help_true="训练期间运行验证", help_false="训练期间不验证")
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值 (默认 0.7)")
    parser.add_argument("--max-det", type=int, default=None, help="每张图最大检测数 (默认 300)")

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=None, help="结果根目录的项目名")
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--save-period", type=int, default=None, help="每 N 轮保存检查点 (-1 = 关闭)")
    set_boolean_argument(parser, "exist_ok", "exist-ok", help_true="覆盖已有项目/名称", help_false="不覆盖")
    set_boolean_argument(parser, "verbose", "verbose", help_true="详细输出", help_false="简洁输出")
    set_boolean_argument(parser, "plots", "plots", help_true="训练/验证时保存图表", help_false="不保存图表")
    set_boolean_argument(
        parser, "half", "half", help_true="FP16 半精度验证", help_false="全精度验证"
    )
    set_boolean_argument(
        parser, "dnn", "dnn", help_true="使用 OpenCV DNN 进行 ONNX 推理", help_false="不使用 DNN"
    )

    return parser.parse_args()


_AUG_KEYS = [
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic",
    "mixup", "cutmix", "copy_paste", "copy_paste_mode",
    "auto_augment", "erasing",
]


# ─── CLI → nested config ──────────────────────────────────────────────


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config = {}

    # Model
    model_cfg = config_from_args(
        args,
        plain=("model_yaml", "task", "classes"),
        rename={"model_yaml": "yaml"},
    )
    if args.model:
        model_cfg["name"] = args.model
    if args.pretrained_bool is not None:
        model_cfg["pretrained"] = to_bool(args.pretrained_bool)
    elif args.pretrained is not None:
        # 尝试将 true/false 字符串转为布尔，否则保留为权重路径字符串
        pretrained_val = to_bool(args.pretrained)
        model_cfg["pretrained"] = pretrained_val if pretrained_val is not None else args.pretrained
    if model_cfg:
        config["model"] = model_cfg

    # Data
    data_cfg = config_from_args(
        args, plain=("data", "split"), rename={"data": "config"}
    )
    if data_cfg:
        config["data"] = {**config.get("data", {}), **data_cfg}

    # Training
    train_plain = (
        "epochs", "time", "batch", "imgsz", "device", "workers", "patience",
        "cache", "fraction", "freeze", "multi_scale", "nbs", "optimizer",
        "lr0", "lrf", "momentum", "weight_decay", "close_mosaic", "seed",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        "box", "cls", "cls_pw", "dfl", "pose", "kobj", "rle", "angle",
        "mask_ratio", "dropout", "auto_augment", "erasing",
    )
    train_bool = (
        "save", "amp", "rect", "single_cls", "end2end", "profile",
        "deterministic", "overlap_mask",
    )
    train_cfg = config_from_args(args, plain=train_plain, boolean=train_bool)
    # 特殊处理: resume/cos_lr/compile 保持原值语义
    if args.resume is not None:
        train_cfg["resume"] = args.resume
    if args.cos_lr is not None:
        train_cfg["cos_lr"] = args.cos_lr
    if args.compile is not None:
        b = to_bool(args.compile)
        train_cfg["compile"] = b if b is not None else args.compile
    if train_cfg:
        config["train"] = {**config.get("train", {}), **train_cfg}

    # Augmentation
    aug_cfg = {k: getattr(args, k) for k in _AUG_KEYS if getattr(args, k) is not None}
    if aug_cfg:
        config["augmentation"] = {**config.get("augmentation", {}), **aug_cfg}

    # Validation
    val_cfg = config_from_args(
        args, boolean=("val", "half", "plots", "dnn"), plain=("conf", "iou", "max_det")
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


# ─── Train ────────────────────────────────────────────────────────────


def train(config: Dict):
    """训练 YOLO 模型。"""
    model_name = get_nested_value(config, "model", "name", default="yolo26n.pt")
    model_yaml = get_nested_value(config, "model", "yaml")
    pretrained = get_nested_value(config, "model", "pretrained")
    data_config = get_nested_value(config, "data", "config", default="coco8.yaml")

    train_args = {
        "data": data_config,
        "epochs": get_nested_value(config, "train", "epochs", default=100),
        "batch": get_nested_value(config, "train", "batch", default=16),
        "imgsz": get_nested_value(config, "train", "imgsz", default=640),
        "workers": get_nested_value(config, "train", "workers", default=8),
        "optimizer": get_nested_value(config, "train", "optimizer", default="auto"),
        "lr0": get_nested_value(config, "train", "lr0", default=0.01),
        "lrf": get_nested_value(config, "train", "lrf", default=0.01),
        "momentum": get_nested_value(config, "train", "momentum", default=0.937),
        "weight_decay": get_nested_value(config, "train", "weight_decay", default=0.0005),
        "patience": get_nested_value(config, "train", "patience", default=100),
        "amp": get_nested_value(config, "train", "amp", default=True),
        "save": get_nested_value(config, "train", "save", default=True),
        "cos_lr": get_nested_value(config, "train", "cos_lr", default=False),
        "close_mosaic": get_nested_value(config, "train", "close_mosaic", default=10),
        "seed": get_nested_value(config, "train", "seed", default=0),
        "deterministic": get_nested_value(config, "train", "deterministic", default=True),
        "warmup_epochs": get_nested_value(config, "train", "warmup_epochs", default=3.0),
        "warmup_momentum": get_nested_value(config, "train", "warmup_momentum", default=0.8),
        "warmup_bias_lr": get_nested_value(config, "train", "warmup_bias_lr", default=0.1),
        "box": get_nested_value(config, "train", "box", default=7.5),
        "cls": get_nested_value(config, "train", "cls", default=0.5),
        "dfl": get_nested_value(config, "train", "dfl", default=1.5),
        "nbs": get_nested_value(config, "train", "nbs", default=64),
        "save_period": get_nested_value(config, "output", "save_period", default=-1),
        "exist_ok": get_nested_value(config, "output", "exist_ok", default=False),
        "verbose": get_nested_value(config, "output", "verbose", default=True),
        "resume": get_nested_value(config, "train", "resume", default=False),
        "fraction": get_nested_value(config, "train", "fraction", default=1.0),
        "multi_scale": get_nested_value(config, "train", "multi_scale", default=0.0),
    }

    # 可选参数: 仅在配置中显式设置时才传递
    for key in (
        "device", "cache", "time", "rect", "single_cls", "freeze",
        "compile", "end2end", "profile",
        "cls_pw", "pose", "kobj", "rle", "angle", "overlap_mask",
        "mask_ratio", "dropout",
    ):
        v = get_nested_value(config, "train", key)
        if v is not None:
            train_args[key] = v

    # classes 过滤器: 从 model 配置节读取 (与 val/predict 一致)
    classes = get_nested_value(config, "model", "classes")
    if classes is not None:
        train_args["classes"] = classes

    if pretrained is not None:
        train_args["pretrained"] = pretrained
    task = get_nested_value(config, "model", "task")
    if task is not None:
        train_args["task"] = task

    # Validation 节参数 -> 传给 train (ultralytics model.train 统一接受)
    # 注意: save_json 是验证参数，不应传给 train
    for key in ("val", "conf", "iou", "max_det", "half", "plots", "dnn",
                "agnostic_nms", "augment", "save_conf", "int8"):
        v = get_nested_value(config, "validation", key)
        if v is not None:
            train_args[key] = v

    for key in ("project", "name"):
        v = get_nested_value(config, "output", key)
        if v is not None:
            train_args[key] = v

    for key in _AUG_KEYS:
        v = get_nested_value(config, "augmentation", key)
        if v is not None:
            train_args[key] = v

    if config.get("cfg") is not None:
        train_args["cfg"] = config["cfg"]
    if config.get("tracker") is not None:
        train_args["tracker"] = config["tracker"]

    resume = train_args["resume"]
    project = train_args.get("project")
    name = train_args.get("name")

    project_root = PROJECT_ROOT
    save_dir = (project_root / str(project) / str(name)).resolve() if project or name else None

    print(f"\n{'='*60}")
    print("YOLO 训练配置")
    print(f"{'='*60}")
    print(f"模型:      {model_name}")
    print(f"预训练:    {pretrained}")
    print(f"数据集:    {data_config}")
    print(f"轮数:      {train_args['epochs']}")
    print(f"批大小:    {train_args['batch']}")
    print(f"图像尺寸:  {train_args['imgsz']}")
    print(f"设备:      {train_args.get('device', 'auto')}")
    print(f"缓存:      {train_args.get('cache', 'False')}")
    print(f"优化器:    {train_args['optimizer']}")
    print(f"学习率:    {train_args['lr0']}")
    if resume and save_dir:
        print(f"恢复训练:  {save_dir / 'weights' / 'last.pt'}")
    print(f"{'='*60}\n")

    last_pt = (save_dir / "weights" / "last.pt") if save_dir else None
    if resume and last_pt and last_pt.exists():
        print(f"从检查点恢复: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        if resume:
            print(f"检查点未找到: {last_pt}, 从零开始训练")
        if model_yaml:
            model = YOLO(model_yaml, task=task) if task else YOLO(model_yaml)
            if isinstance(pretrained, str):
                model.load(pretrained)
        else:
            model = YOLO(model_name)

    results = model.train(**train_args)

    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"结果保存至: {results.save_dir}")
    print(f"最佳权重: {results.save_dir}/weights/best.pt")
    print(f"最后权重: {results.save_dir}/weights/last.pt")
    print(f"{'='*60}\n")

    return results


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)  # YAML in base, CLI in override = CLI wins

        train(config)
    except KeyboardInterrupt:
        print("\n训练被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"错误: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()