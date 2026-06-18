from __future__ import annotations

import logging
import os
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configure application-wide logging once.

    Environment variables:
    - ``BL03U_LOG_LEVEL``: logging level, defaults to ``INFO``.
    - ``BL03U_LOG_FILE``: optional file path for persistent logs.
    """

    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        return

    resolved_level = _resolve_level(level or os.environ.get("BL03U_LOG_LEVEL", "INFO"))
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    raw_log_file = log_file or os.environ.get("BL03U_LOG_FILE")
    if raw_log_file:
        file_path = Path(raw_log_file).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))

    logging.basicConfig(
        level=resolved_level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=force,
    )


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(level.upper())
    return value if isinstance(value, int) else logging.INFO
