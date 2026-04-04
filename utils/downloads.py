"""
Custom utilities for YOLO project.
Overrides ultralytics download behavior to use weights_dir.
"""
import sys
from pathlib import Path

# Add ultralytics submodule to path
ULTRALYTICS_PATH = Path(__file__).parent.parent / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from urllib import parse

from ultralytics.utils import SETTINGS, LOGGER, checks, clean_url, url2file
from ultralytics.utils.downloads import (
    GITHUB_ASSETS_REPO,
    GITHUB_ASSETS_NAMES,
    get_github_assets,
    safe_download,
)

# Ensure weights_dir exists
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def attempt_download_asset(
    file: str | Path,
    repo: str = "ultralytics/assets",
    release: str = "v8.4.0",
    **kwargs,
) -> str:
    """
    Download a file from GitHub release assets if it is not found locally.
    Always downloads to weights_dir instead of current directory.

    Args:
        file: The filename or file path to be downloaded.
        repo: The GitHub repository in the format 'owner/repo'.
        release: The specific release version to be downloaded.
        **kwargs: Additional keyword arguments for the download process.

    Returns:
        The path to the downloaded file.
    """
    # YOLOv3/5u updates
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))

    # Check if file exists in current directory or weights_dir
    if file.exists():
        return str(file)
    elif (WEIGHTS_DIR / file).exists():
        return str(WEIGHTS_DIR / file)
    else:
        # URL specified
        name = Path(parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        download_url = f"https://github.com/{repo}/releases/download"

        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = url2file(name)  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # file already exists
            else:
                safe_download(url=url, file=file, dir=WEIGHTS_DIR, min_bytes=1e5, **kwargs)
                return str(WEIGHTS_DIR / file)

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(
                url=f"{download_url}/{release}/{name}",
                file=name,
                dir=WEIGHTS_DIR,
                min_bytes=1e5,
                **kwargs
            )
            return str(WEIGHTS_DIR / name)

        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)  # latest release
            if name in assets:
                safe_download(
                    url=f"{download_url}/{tag}/{name}",
                    file=name,
                    dir=WEIGHTS_DIR,
                    min_bytes=1e5,
                    **kwargs
                )
                return str(WEIGHTS_DIR / name)

        return str(file)


def patch_ultralytics_downloads():
    """
    Monkey patch ultralytics downloads module to use our custom download function.
    Call this at the start of your script to ensure downloads go to weights_dir.
    """
    import ultralytics.utils.downloads as dl_module
    dl_module.attempt_download_asset = attempt_download_asset

    # Also patch the import in nn/tasks.py
    import ultralytics.nn.tasks as tasks_module
    tasks_module.attempt_download_asset = attempt_download_asset

    LOGGER.info(f"✅ Patched ultralytics downloads to use weights_dir: {WEIGHTS_DIR}")


# Auto-patch when this module is imported
patch_ultralytics_downloads()
