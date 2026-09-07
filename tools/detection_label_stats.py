#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML>=6.0"]
# ///
# ─── How to run ───
# 1. 修改下面的 CONFIG_PATH
# 2. 在项目根目录运行: python tools/detection_label_stats.py
"""统计 YOLO 检测数据集的图片数量、目标数量和标签类别分布。"""

from pathlib import Path
from typing import Final

from classification_stats import run

# 推荐填写 configs 下的数据集 YAML，也兼容数据集根目录或单个 TXT 图片清单。
CONFIG_PATH: Final[Path] = Path(r"configs/datasets/11classes/11classes.yaml")


if __name__ == "__main__":
    raise SystemExit(run(CONFIG_PATH))
