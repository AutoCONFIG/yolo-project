"""Shared I/O utilities for the YOLO project."""

from pathlib import Path


def read_text_robust(path: Path) -> str:
    """读取文本文件，自动探测编码 (utf-8-sig → utf-8 → gbk → gb2312 → gb18030 → latin-1)。"""
    raw = path.read_bytes()
    if not raw:
        return ""

    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue

    return raw.decode("latin-1")
