from __future__ import annotations

import logging
import sys
from types import TracebackType


def install_global_exception_hook(logger_name: str = "bl03u_masstool") -> None:
    """Route uncaught exceptions through logging before Python renders them."""

    logger = logging.getLogger(logger_name)
    original_hook = sys.excepthook

    def log_exception(
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            original_hook(exc_type, exc, traceback)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, traceback))
        original_hook(exc_type, exc, traceback)

    sys.excepthook = log_exception
