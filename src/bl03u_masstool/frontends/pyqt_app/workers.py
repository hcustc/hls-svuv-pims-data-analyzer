from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from PyQt6 import QtCore


logger = logging.getLogger(__name__)


class WorkerThread(QtCore.QThread):
    finished_with_result = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, str)

    def __init__(self, task: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        try:
            self.finished_with_result.emit(self._task())
        except Exception as exc:
            logger.exception("Background worker task failed")
            self.failed.emit(str(exc))

    def report_progress(self, value: int, message: str = "") -> None:
        """Thread-safe callback for tasks that can expose coarse real progress."""
        self.progress.emit(max(0, min(100, int(value))), str(message))
