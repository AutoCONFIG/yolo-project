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


def contained_path(root: Path, *parts) -> Path:
    """拼接 root 下的路径并校验最终位置不越出 root，防止路径穿越写入。"""
    resolved_root = Path(root).resolve()
    target = resolved_root.joinpath(*map(str, parts)).resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError(f"检测到路径穿越: {target} 超出目录 {resolved_root}")
    return target
