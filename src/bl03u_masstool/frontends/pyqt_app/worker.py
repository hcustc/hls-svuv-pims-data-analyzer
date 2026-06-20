"""Background worker threads for long-running file operations.

Implements Qt thread-safe workers for:
- Creating project snapshots
- Exporting projects to archives
- Importing project sources
- Scanning project artifacts

All workers emit progress signals to keep UI responsive.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal as Signal, QThread

from bl03u_masstool.core.project_lifecycle import (
    ProjectSettings,
    create_project_snapshot,
    export_project_archive,
    import_project_source,
    scan_project_artifacts,
)


class OperationCancelledError(Exception):
    """Raised when user cancels a background operation."""

    pass


class ProjectWorker(QObject):
    """Base class for all project background operations.

    Subclasses must implement run() to perform their specific operation.
    They should regularly call _check_cancelled() to support cancellation.

    Signals:
        progress: (percentage: int, message: str)
        finished: (result: object)
        warning: (message: str)
        error: (message: str)
        cancelled: ()
    """

    progress = Signal(int, str)  # (percentage, message)
    finished = Signal(object)  # result
    warning = Signal(str)  # warning message
    error = Signal(str)  # error message
    cancelled = Signal()  # operation cancelled

    def __init__(self):
        super().__init__()
        self._is_cancelled = False

    def run(self):
        """Execute the operation. Must be implemented by subclasses."""
        raise NotImplementedError

    def cancel(self):
        """Request cancellation and perform cleanup."""
        self._is_cancelled = True
        self.on_cancel()

    def on_cancel(self):
        """Cleanup logic when operation is cancelled. Override in subclasses."""
        pass

    def _check_cancelled(self):
        """Check if cancellation was requested. Raises OperationCancelledError."""
        if self._is_cancelled:
            raise OperationCancelledError("Operation cancelled by user")

    def _update_progress(self, current: int, total: int, message: str = ""):
        """Update progress with automatic percentage calculation."""
        if total > 0:
            percent = int((current / total) * 100)
        else:
            percent = 0
        self.progress.emit(percent, message)


class SnapshotWorker(ProjectWorker):
    """Creates a project snapshot (backup) in background."""

    def __init__(self, settings: ProjectSettings, note: str = ""):
        super().__init__()
        self.settings = settings
        self.note = note
        self.snapshot_path: Path | None = None

    def run(self):
        """Create snapshot and emit signals."""
        try:
            self.progress.emit(10, "Starting snapshot creation...")

            self._check_cancelled()

            # Create snapshot using core function
            self.progress.emit(25, "Scanning project contents...")
            snapshot_path = create_project_snapshot(self.settings, self.note)

            self._check_cancelled()

            self.snapshot_path = snapshot_path
            size = snapshot_path.stat().st_size if snapshot_path.exists() else 0

            self.progress.emit(95, "Finalizing snapshot...")
            self._check_cancelled()

            self.finished.emit(
                {
                    "success": True,
                    "path": str(snapshot_path),
                    "size": size,
                    "note": self.note,
                }
            )
        except OperationCancelledError:
            self.progress.emit(100, "Snapshot creation cancelled")
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"Snapshot creation failed: {str(e)}")

    def on_cancel(self):
        """Clean up any temporary snapshot files."""
        # Temporary files are created in versions/ directory
        # The core function handles cleanup on exception
        pass


class ExportWorker(ProjectWorker):
    """Exports project to archive in background."""

    def __init__(
        self, settings: ProjectSettings, destination: str | Path, include_metadata: bool = True
    ):
        super().__init__()
        self.settings = settings
        self.destination = Path(destination)
        self.include_metadata = include_metadata
        self.export_path: Path | None = None

    def run(self):
        """Export project and emit signals."""
        staging = self.destination.with_suffix(".tmp")
        try:
            self.progress.emit(5, "Validating export destination...")
            self._check_cancelled()

            self.progress.emit(15, "Preparing export...")
            self._check_cancelled()

            # Export to temporary location first (safe atomic write)
            export_path = export_project_archive(
                self.settings,
                staging,
                include_metadata=self.include_metadata,
            )

            self._check_cancelled()

            self.progress.emit(90, "Finalizing export...")

            # Atomic replace
            if self.destination.exists():
                self.destination.unlink()
            staging.replace(export_path)

            self._check_cancelled()

            self.export_path = export_path
            size = export_path.stat().st_size if export_path.exists() else 0

            self.finished.emit(
                {
                    "success": True,
                    "path": str(export_path),
                    "size": size,
                }
            )
        except OperationCancelledError:
            shutil.rmtree(staging, ignore_errors=True)
            self.cancelled.emit()
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            self.error.emit(f"Export failed: {str(e)}")

    def on_cancel(self):
        """Clean up staging directory on cancellation."""
        staging = self.destination.with_suffix(".tmp")
        shutil.rmtree(staging, ignore_errors=True)


class ImportWorker(ProjectWorker):
    """Imports data source to project in background."""

    def __init__(self, settings: ProjectSettings, source: str | Path, source_key: str):
        super().__init__()
        self.settings = settings
        self.source = Path(source)
        self.source_key = source_key
        self.result: dict[str, Any] | None = None

    def run(self):
        """Import source and emit signals."""
        try:
            self.progress.emit(10, f"Validating source: {self.source.name}...")
            self._check_cancelled()

            self.progress.emit(20, "Checking for conflicts...")
            self._check_cancelled()

            # Import using core function
            result = import_project_source(self.settings, self.source, self.source_key)

            self._check_cancelled()

            self.progress.emit(90, "Finalizing import...")
            self._check_cancelled()

            self.result = {
                "success": True,
                "source": str(result.source),
                "destination": str(result.destination),
                "file_count": result.file_count if hasattr(result, "file_count") else 0,
            }

            self.finished.emit(self.result)
        except OperationCancelledError:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"Import failed: {str(e)}")

    def on_cancel(self):
        """Clean up any partial imports."""
        # The core function handles cleanup on exception
        pass


class ScanWorker(ProjectWorker):
    """Scans project for artifacts in background."""

    def __init__(self, settings: ProjectSettings):
        super().__init__()
        self.settings = settings
        self.artifacts = []

    def run(self):
        """Scan artifacts and emit signals."""
        try:
            self.progress.emit(5, "Scanning artifact directories...")
            self._check_cancelled()

            artifacts = scan_project_artifacts(self.settings)

            self._check_cancelled()

            self.progress.emit(95, "Finalizing scan...")
            self._check_cancelled()

            self.artifacts = artifacts

            self.finished.emit(
                {
                    "success": True,
                    "count": len(artifacts),
                    "artifacts": artifacts,
                }
            )
        except OperationCancelledError:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"Scan failed: {str(e)}")

    def on_cancel(self):
        """No cleanup needed for scan operation."""
        pass
