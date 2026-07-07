"""Tests for background worker threads and lifecycle management."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

# Set offscreen platform BEFORE importing PyQt6
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.project_lifecycle import ProjectSettings, ensure_project_structure
from bl03u_masstool.frontends.pyqt_app.progress_dialog import ProgressDialog
from bl03u_masstool.frontends.pyqt_app.worker import (
    ExportWorker,
    ImportWorker,
    OperationCancelledError,
    ProjectWorker,
    ScanWorker,
    SnapshotWorker,
)
from bl03u_masstool.frontends.pyqt_app.worker_manager import WorkerManager


# Ensure QApplication exists for PyQt tests
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestProjectWorkerBase:
    """Test ProjectWorker base class."""

    def test_worker_initialization(self):
        """Test worker initializes with proper state."""
        worker = SnapshotWorker(
            ProjectSettings(output_dir="/tmp", project_name="test", system="test"),
            "test note",
        )
        assert worker is not None
        assert not worker._is_cancelled

    def test_check_cancelled_when_not_cancelled(self):
        """Test _check_cancelled doesn't raise when not cancelled."""
        worker = SnapshotWorker(
            ProjectSettings(output_dir="/tmp", project_name="test", system="test"),
        )
        # Should not raise
        worker._check_cancelled()

    def test_check_cancelled_when_cancelled(self):
        """Test _check_cancelled raises when cancelled."""
        worker = SnapshotWorker(
            ProjectSettings(output_dir="/tmp", project_name="test", system="test"),
        )
        worker.cancel()
        with pytest.raises(OperationCancelledError):
            worker._check_cancelled()

    def test_cancel_sets_flag(self):
        """Test cancel() sets _is_cancelled flag."""
        worker = SnapshotWorker(
            ProjectSettings(output_dir="/tmp", project_name="test", system="test"),
        )
        assert not worker._is_cancelled
        worker.cancel()
        assert worker._is_cancelled

    def test_update_progress_calculates_percentage(self):
        """Test _update_progress calculates percentage correctly."""
        worker = SnapshotWorker(
            ProjectSettings(output_dir="/tmp", project_name="test", system="test"),
        )

        # Mock the signal
        worker.progress = MagicMock()
        worker._update_progress(50, 100, "test message")
        worker.progress.emit.assert_called_once_with(50, "test message")

    def test_update_progress_with_zero_total(self):
        """Test _update_progress handles zero total gracefully."""
        worker = SnapshotWorker(
            ProjectSettings(output_dir="/tmp", project_name="test", system="test"),
        )

        # Mock the signal
        worker.progress = MagicMock()
        worker._update_progress(0, 0, "test message")
        worker.progress.emit.assert_called_once_with(0, "test message")


class TestSnapshotWorker:
    """Test SnapshotWorker implementation."""

    def test_snapshot_worker_initialization(self, qapp):
        """Test SnapshotWorker initializes correctly."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = SnapshotWorker(settings, "test note")
            assert worker.settings == settings
            assert worker.note == "test note"
            assert worker.snapshot_path is None

    def test_snapshot_worker_run_success(self, qapp):
        """Test successful snapshot creation."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = SnapshotWorker(settings, "test note")

            # Mock signals
            worker.progress = MagicMock()
            worker.finished = MagicMock()
            worker.error = MagicMock()

            # Run the worker
            worker.run()

            # Check that finished was called
            assert worker.finished.emit.called
            result = worker.finished.emit.call_args[0][0]
            assert result["success"] is True
            assert "path" in result
            assert "size" in result

    def test_snapshot_worker_run_with_cancellation(self, qapp):
        """Test snapshot creation with cancellation request."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = SnapshotWorker(settings, "test note")

            # Set cancellation flag before running
            worker._is_cancelled = True

            # Mock signals
            worker.progress = MagicMock()
            worker.cancelled = MagicMock()
            worker.error = MagicMock()

            # Run the worker
            worker.run()

            # Check that cancelled was called
            assert worker.cancelled.emit.called

    def test_snapshot_worker_on_cancel(self, qapp):
        """Test on_cancel cleanup."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            worker = SnapshotWorker(settings, "test note")
            # Should not raise
            worker.on_cancel()


class TestExportWorker:
    """Test ExportWorker implementation."""

    def test_export_worker_initialization(self, qapp):
        """Test ExportWorker initializes correctly."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            export_path = Path(tmpdir) / "export.zip"
            worker = ExportWorker(settings, export_path)

            assert worker.settings == settings
            assert worker.destination == export_path
            assert worker.export_path is None

    def test_export_worker_run_success(self, qapp):
        """Test successful project export."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            export_path = Path(tmpdir) / "export.zip"
            worker = ExportWorker(settings, export_path)

            # Use real signal handlers instead of mocks
            results = {"finished": None, "error": None}

            def on_finished(result):
                results["finished"] = result

            def on_error(message):
                results["error"] = message

            worker.finished.connect(on_finished)
            worker.error.connect(on_error)

            # Run the worker
            worker.run()

            assert results["error"] is None
            assert results["finished"] is not None
            assert results["finished"]["success"] is True
            assert results["finished"]["path"] == str(export_path)
            assert export_path.exists()
            assert worker.export_path == export_path
            assert results["finished"]["size"] == export_path.stat().st_size
            assert not export_path.with_name(f"{export_path.name}.tmp.zip").exists()
            with zipfile.ZipFile(export_path) as archive:
                names = set(archive.namelist())
            assert f"{Path(tmpdir).name}/project_state.yaml" in names

    def test_export_worker_honors_include_metadata_false(self, qapp):
        """Test project export can omit metadata when requested."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            export_path = Path(tmpdir) / "export.zip"
            worker = ExportWorker(settings, export_path, include_metadata=False)
            results = {"finished": None, "error": None}
            worker.finished.connect(lambda result: results.__setitem__("finished", result))
            worker.error.connect(lambda message: results.__setitem__("error", message))

            worker.run()

            assert results["error"] is None
            assert results["finished"] is not None
            assert export_path.exists()
            with zipfile.ZipFile(export_path) as archive:
                names = set(archive.namelist())
            assert not any(name.endswith("/project_state.yaml") for name in names)

    def test_export_worker_on_cancel(self, qapp):
        """Test on_cancel cleanup removes staging directory."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            export_path = Path(tmpdir) / "export.zip"
            worker = ExportWorker(settings, export_path)

            # Create mock staging directory
            staging = export_path.with_name(f"{export_path.name}.tmp.zip")
            staging.write_text("test")

            assert staging.exists()

            # Cancel should clean up staging
            worker.on_cancel()
            assert not staging.exists()


class TestImportWorker:
    """Test ImportWorker implementation."""

    def test_import_worker_run_success_returns_registration_fields(self, qapp):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = ProjectSettings(
                output_dir=str(root / "project"),
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)
            source = root / "external_temp"
            source.mkdir()
            (source / "temp.txt").write_text("tof intensity\n1 2\n", encoding="utf-8")

            worker = ImportWorker(settings, source, "temperature_scan")
            worker.progress = MagicMock()
            worker.finished = MagicMock()
            worker.error = MagicMock()

            worker.run()

            assert worker.finished.emit.called
            result = worker.finished.emit.call_args[0][0]
            assert result["success"] is True
            assert result["field_name"] == "temperature_scan_folder"
            assert result["label"] == "温度扫描目录"
            assert result["mode"] == "link"
            assert result["source_key"] == "temperature_scan"
            assert Path(result["destination"]).exists()
            assert Path(result["destination"]).is_symlink()

    def test_import_worker_accepts_copy_mode_for_project_managed_sources(self, qapp):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = ProjectSettings(
                output_dir=str(root / "project"),
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)
            source = root / "external_temp"
            source.mkdir()
            (source / "temp.txt").write_text("tof intensity\n1 2\n", encoding="utf-8")

            worker = ImportWorker(settings, source, "temperature_scan", mode="copy")
            worker.progress = MagicMock()
            worker.finished = MagicMock()
            worker.error = MagicMock()

            worker.run()

            assert worker.finished.emit.called
            result = worker.finished.emit.call_args[0][0]
            assert result["success"] is True
            assert result["mode"] == "copy"
            destination = Path(result["destination"])
            assert destination.exists()
            assert not destination.is_symlink()
            assert (destination / "temp.txt").read_text(encoding="utf-8") == "tof intensity\n1 2\n"


class TestScanWorker:
    """Test ScanWorker implementation."""

    def test_scan_worker_initialization(self, qapp):
        """Test ScanWorker initializes correctly."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = ScanWorker(settings)
            assert worker.settings == settings
            assert worker.artifacts == []

    def test_scan_worker_run_success(self, qapp):
        """Test successful artifact scan."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = ScanWorker(settings)

            # Mock signals
            worker.progress = MagicMock()
            worker.finished = MagicMock()
            worker.error = MagicMock()

            # Run the worker
            worker.run()

            # Check that finished was called
            assert worker.finished.emit.called
            result = worker.finished.emit.call_args[0][0]
            assert result["success"] is True
            assert "count" in result
            assert "artifacts" in result


class TestWorkerManager:
    """Test WorkerManager lifecycle management."""

    def test_manager_initialization(self, qapp):
        """Test WorkerManager initializes correctly."""
        manager = WorkerManager()
        assert manager.current_worker is None
        assert manager.current_thread is None
        assert not manager.is_running()

    def test_manager_is_running(self, qapp):
        """Test is_running reports state correctly."""
        manager = WorkerManager()
        assert not manager.is_running()

    def test_manager_cancel_with_no_worker(self, qapp):
        """Test cancel with no active worker doesn't raise."""
        manager = WorkerManager()
        # Should not raise
        manager.cancel()

    def test_manager_cleanup_sets_none(self, qapp):
        """Test _cleanup properly resets worker and thread references."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            manager = WorkerManager()
            worker = SnapshotWorker(settings, "test")

            # Mock the thread
            mock_thread = MagicMock()
            mock_thread.isRunning.return_value = False
            manager.current_thread = mock_thread
            manager.current_worker = worker

            # Cleanup
            manager._cleanup()

            # Verify cleanup
            assert manager.current_worker is None
            assert manager.current_thread is None


class TestProgressDialog:
    """Test ProgressDialog UI component."""

    def test_progress_dialog_initialization(self, qapp):
        """Test ProgressDialog initializes correctly."""
        dialog = ProgressDialog(None, "Test Operation")
        assert dialog.progress_bar.value() == 0
        assert dialog.progress_bar.maximum() == 100

    def test_progress_dialog_update(self, qapp):
        """Test progress dialog update method."""
        dialog = ProgressDialog(None, "Test Operation")
        dialog.update(50, "Test message")

        assert dialog.progress_bar.value() == 50
        assert dialog.message_label.text() == "Test message"

    def test_progress_dialog_reset(self, qapp):
        """Test progress dialog reset method."""
        dialog = ProgressDialog(None, "Test Operation")
        dialog.update(50, "Test message")
        dialog.reset()

        assert dialog.progress_bar.value() == 0
        assert dialog.message_label.text() == ""

    def test_progress_dialog_set_message(self, qapp):
        """Test setting message without updating progress."""
        dialog = ProgressDialog(None, "Test Operation")
        dialog.update(50, "Old message")
        dialog.set_message("New message")

        assert dialog.progress_bar.value() == 50
        assert dialog.message_label.text() == "New message"


class TestWorkerSignals:
    """Test worker signals are properly connected and emitted."""

    def test_worker_finished_signal(self, qapp):
        """Test finished signal is emitted with result."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = SnapshotWorker(settings, "test")
            finished_called = []

            def on_finished(result):
                finished_called.append(result)

            worker.finished.connect(on_finished)
            worker.run()

            assert len(finished_called) == 1
            assert finished_called[0]["success"] is True

    def test_worker_error_signal(self, qapp):
        """Test error signal is emitted on failure."""
        # Use a path inside a regular file as output_dir – creating subdirs
        # under a file always fails on all platforms.
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pass  # keep the file as a blocker
        try:
            settings = ProjectSettings(
                output_dir=os.path.join(tf.name, "subdir"),
                project_name="test",
                system="test",
            )

            worker = SnapshotWorker(settings, "test")
            error_called = []

            def on_error(message):
                error_called.append(message)

            worker.error.connect(on_error)
            worker.run()

            assert len(error_called) == 1
            assert "fail" in error_called[0].lower() or "error" in error_called[0].lower()
        finally:
            os.unlink(tf.name)

    def test_worker_progress_signal(self, qapp):
        """Test progress signal is emitted during execution."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = SnapshotWorker(settings, "test")
            progress_updates = []

            def on_progress(percent, message):
                progress_updates.append((percent, message))

            worker.progress.connect(on_progress)
            worker.run()

            # Should have multiple progress updates
            assert len(progress_updates) > 0
            # Progress should start at 10 and increase
            assert progress_updates[0][0] >= 10


class TestWorkerErrorHandling:
    """Test error handling in workers."""

    def test_worker_handles_cancelled_exception(self, qapp):
        """Test worker properly handles OperationCancelledError."""
        with TemporaryDirectory() as tmpdir:
            settings = ProjectSettings(
                output_dir=tmpdir,
                project_name="test",
                system="test",
            )
            ensure_project_structure(settings)

            worker = SnapshotWorker(settings, "test")
            worker._is_cancelled = True

            cancelled_emitted = []
            worker.cancelled.connect(lambda: cancelled_emitted.append(True))

            worker.run()

            assert len(cancelled_emitted) == 1

    def test_worker_emits_error_on_exception(self, qapp):
        """Test worker emits error signal on unexpected exception."""
        # Use a path inside a regular file as output_dir – creating subdirs
        # under a file always fails on all platforms.
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pass  # keep the file as a blocker
        try:
            settings = ProjectSettings(
                output_dir=os.path.join(tf.name, "subdir"),
                project_name="test",
                system="test",
            )

            worker = SnapshotWorker(settings, "test")
            error_messages = []
            worker.error.connect(lambda msg: error_messages.append(msg))

            worker.run()

            assert len(error_messages) == 1
        finally:
            os.unlink(tf.name)
