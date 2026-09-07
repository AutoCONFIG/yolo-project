"""
Shared configuration utilities for YOLO project.

This module provides common functions used across all command modules
(train, val, predict, export) and tools to avoid code duplication and ensure consistency.

Functions:
    load_yaml_config: Load configuration from YAML file.
    merge_configs: Merge override args into base config.
    get_nested_value: Safely get nested value from config dict.
    to_bool: Convert 'true'/'false' string to bool.
    set_boolean_argument: Add paired --flag/--no-flag CLI arguments.
    setup_ultralytics_path: Add local ultralytics submodule to sys.path.

Constants:
    PROJECT_ROOT: Absolute path to the project root directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import copy

import yaml

# ─── Project root & path setup ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_ultralytics_path() -> None:
    """Prefer the local ultralytics submodule in this process and its children.

    Updating ``sys.path`` makes imports in the current process resolve to the
    submodule. Updating ``PYTHONPATH`` is also required because Ultralytics
    starts fresh Python processes for multi-GPU DDP training.
    """
    ult_path = PROJECT_ROOT / "ultralytics"
    if not ult_path.exists():
        return

    ult_path_str = str(ult_path)
    if ult_path_str not in sys.path:
        sys.path.insert(0, ult_path_str)

    pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    if ult_path_str not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([ult_path_str, *entries])


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary. Returns empty dict if file is empty or unreadable.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def merge_configs(base_config: Dict, override_args: Dict) -> Dict:
    """Merge override args into base config (deep merge for nested dicts).

    Args:
        base_config: Base configuration dictionary.
        override_args: Override arguments to merge in. CLI takes precedence.

    Returns:
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base_config)
    for key, value in override_args.items():
        if value is not None:
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def get_nested_value(config: Dict, *keys, default=None):
    """Safely get nested value from config dict.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse.
        default: Default value if any key is missing.

    Returns:
        The nested value, or default if any key is missing.
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def resolve_config_value(config: Dict, *chain, default=None):
    """按优先级链查找配置值，返回第一个非 None 的结果。

    chain 中每个元素为 (section, key) 元组，按传入顺序依次查找。
    若全部未命中则返回 default。
    """
    for section, key in chain:
        v = get_nested_value(config, section, key)
        if v is not None:
            return v
    return default


def to_bool(value: str | bool | None) -> bool | None:
    """Convert 'true'/'false' string or native bool to bool.

    Args:
        value: String or bool value to convert. Case-insensitive for strings.

    Returns:
        True for 'true'/True, False for 'false'/False, None for None or unknown values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def parse_quantize(value: str | int | None) -> int | str | None:
    """Parse the unified Ultralytics precision selector."""
    if value is None or isinstance(value, int):
        return value
    value = value.strip().lower()
    if value in {"none", "null", "fp32", "float32"}:
        return None if value in {"none", "null"} else 32
    if value in {"fp16", "float16"}:
        return 16
    if value.isdigit() and int(value) in {8, 16, 32}:
        return int(value)
    if value in {"w8a8", "w16a16", "w8a16", "w8a32"}:
        return value
    raise ValueError("quantize 必须是 8、16、32、w8a8、w16a16、w8a16、w8a32 或 null")


def set_boolean_argument(
    parser: argparse.ArgumentParser,
    dest: str,
    flag_name: str | None = None,
    *,
    neg_prefix: str = "no-",
    help_true: str = "",
    help_false: str = "",
) -> None:
    """Add a paired boolean argument (e.g. --amp / --no-amp) to a parser.

    Omitting both flags yields None, allowing YAML defaults to be preserved.

    Parameters
    ----------
    parser :
        The argparse.ArgumentParser to add arguments to.
    dest :
        Destination attribute name in the parsed Namespace.
    flag_name :
        The positive flag text (default: *dest* with underscores replaced by hyphens).
    neg_prefix :
        Prefix for the negative flag (default ``"no-"``).
    help_true / help_false :
        Help text for the positive / negative flags.
    """
    flag = flag_name or dest.replace("_", "-")

    positive = f"--{flag}"
    negative = f"--{neg_prefix}{flag}"

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        positive,
        dest=dest,
        action="store_const",
        const=True,
        default=None,
        help=help_true or f"Enable {flag}",
    )
    group.add_argument(
        negative,
        dest=dest,
        action="store_const",
        const=False,
        default=None,
        help=help_false or f"Disable {flag}",
    )


def config_from_args(
    args: argparse.Namespace,
    plain: tuple = (),
    boolean: tuple = (),
    rename: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """从 argparse 命名空间提取配置字典。

    消除 ``for field in (...): v = getattr(args, field)`` 的重复样板代码。

    Args:
        args: 解析后的命令行参数
        plain: 原样传递的字段名 (直接 getattr, 排除 None)
        boolean: 通过 ``to_bool`` 转换的字段名 (排除 None)
        rename: {arg字段名: config键名} 重映射, 例如 {"model": "name", "data": "config"}

    Returns:
        非空字段组成的配置字典

    Example::

        model_cfg = config_from_args(
            args,
            plain=("model", "task", "classes"),
            rename={"model": "name"},
        )
        # args.model="yolo.pt" → {"name": "yolo.pt"}
    """
    cfg: Dict[str, Any] = {}
    rename = rename or {}

    for field in plain:
        v = getattr(args, field, None)
        if v is not None:
            cfg[rename.get(field, field)] = v

    for field in boolean:
        v = to_bool(getattr(args, field, None))
        if v is not None:
            cfg[rename.get(field, field)] = v

    return cfg


