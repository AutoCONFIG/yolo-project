"""
Shared configuration utilities for YOLO project.

This module provides common functions used across train.py, export.py, and inference.py
to avoid code duplication and ensure consistency.

Functions:
    load_yaml_config: Load configuration from YAML file.
    merge_configs: Merge override args into base config.
    get_nested_value: Safely get nested value from config dict.
    to_bool: Convert 'true'/'false' string to bool.
    set_boolean_argument: Add paired --flag/--no-flag CLI arguments.
    setup_logging: Configure logging with optional level and file output.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
    result = base_config.copy()
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


def to_bool(value: str | None) -> bool | None:
    """Convert 'true'/'false' string to bool.

    Args:
        value: String value to convert. Case-insensitive.

    Returns:
        True for 'true', False for 'false', None for None or unknown values.
    """
    if value is None:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return None


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


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_str: str = None,
) -> None:
    """Configure Python logging.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to log file.
        format_str: Custom log format string.
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        handlers=handlers,
        force=True,
    )
