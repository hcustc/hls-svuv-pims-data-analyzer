"""Manages background worker threads and their lifecycle.

Ensures only one worker runs at a time and handles proper
cleanup of threads to prevent resource leaks.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QThread

from .worker import ProjectWorker


class WorkerManager(QObject):
    """Manages the lifecycle of background project workers.

    Ensures:
    - Only one worker runs at a time
    - Proper thread cleanup
    - Signal connections are properly established
    - Cancellation works correctly
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.current_worker: ProjectWorker | None = None
        self.current_thread: QThread | None = None

    def run_worker(self, worker: ProjectWorker):
        """Start a new worker (cancels any running worker first)."""
        # Cancel any existing worker
        if self.current_worker:
            self.current_worker.cancel()
            if self.current_thread:
                self.current_thread.quit()
                self.current_thread.wait()

        # Create new thread
        self.current_thread = QThread()
        self.current_worker = worker

        # Move worker to thread (important for Qt signal/slot mechanism)
        worker.moveToThread(self.current_thread)

        # Connect signals
        # When thread starts, call worker.run()
        self.current_thread.started.connect(worker.run)

        # When worker finishes, clean up
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        worker.cancelled.connect(self._on_worker_cancelled)

        # When worker finishes, stop the thread
        worker.finished.connect(self.current_thread.quit)
        worker.error.connect(self.current_thread.quit)
        worker.cancelled.connect(self.current_thread.quit)

        # Start the thread
        self.current_thread.start()

    def cancel(self):
        """Cancel current worker if any."""
        if self.current_worker:
            self.current_worker.cancel()

    def request_shutdown(self) -> None:
        """Ask the managed worker and thread to finish without blocking the UI."""
        self.cancel()
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.requestInterruption()
            self.current_thread.quit()

    def is_running(self) -> bool:
        """Check if a worker is currently running."""
        return self.current_thread is not None and self.current_thread.isRunning()

    def _on_worker_finished(self, result):
        """Called when worker finishes successfully."""
        self._cleanup()

    def _on_worker_error(self, message):
        """Called when worker encounters error."""
        self._cleanup()

    def _on_worker_cancelled(self):
        """Called when worker is cancelled."""
        self._cleanup()

    def _cleanup(self):
        """Clean up thread resources."""
        if self.current_thread:
            # Thread may have already quit due to signal connection
            self.current_thread.quit()
            self.current_thread.wait()
            self.current_thread = None
            self.current_worker = None

    def __del__(self):
        """Ensure thread is cleaned up on manager deletion."""
        if self.current_thread and self.current_thread.isRunning():
            if self.current_worker:
                self.current_worker.cancel()
            self.current_thread.quit()
            self.current_thread.wait()
