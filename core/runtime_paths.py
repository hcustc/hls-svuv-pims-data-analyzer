from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_NAME = "BL03U_MassSpectrumTool"
REPO_ROOT = Path(__file__).resolve().parents[1]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return REPO_ROOT


def resource_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else resource_root() / path


def user_data_root() -> Path:
    override = os.environ.get("BL03U_USER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if not is_frozen():
        return REPO_ROOT

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return Path(base or Path.home() / "AppData" / "Local") / APP_NAME

    base = os.environ.get("XDG_DATA_HOME")
    return Path(base).expanduser() / APP_NAME if base else Path.home() / ".local" / "share" / APP_NAME


def user_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else user_data_root() / path


def writable_path(path: str | Path) -> Path:
    return user_path(path) if is_frozen() else resource_path(path)


def runtime_read_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    user_candidate = user_path(path)
    if is_frozen() and user_candidate.exists():
        return user_candidate
    return resource_path(path)


def ensure_user_copy(path: str | Path) -> Path:
    destination = user_path(path)
    if destination.exists() or not is_frozen():
        return destination

    source = resource_path(path)
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination
