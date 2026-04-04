"""
Custom utilities for YOLO project.
"""
from .downloads import attempt_download_asset, patch_ultralytics_downloads, WEIGHTS_DIR

__all__ = ["attempt_download_asset", "patch_ultralytics_downloads", "WEIGHTS_DIR"]
