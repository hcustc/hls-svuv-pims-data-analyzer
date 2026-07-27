from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets

from bl03u_masstool.core.peak_sets import create_peak_set, verify_peak_set
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.frontends.pyqt_app.project_peak_workflow import (
    prepare_project_peak_set,
)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_valid_project_peak_set_never_prompts(qapp, tmp_path, monkeypatch):
    project = tmp_path / "project"
    record = create_peak_set(
        project,
        content=b"label,peak_index,mz,left_bound,right_bound\nA,10,10,8,12\n",
        extension=".csv",
        label="approved",
        origin="imported",
    )
    settings = ProjectSettings(
        output_dir=str(project),
        active_peak_set_id=record.peak_set_id,
        manual_peak_file=str(verify_peak_set(project, record)),
    )
    owner = QtWidgets.QWidget()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: pytest.fail("有效卡峰集不应再次提示自动寻峰"),
    )
    try:
        assert prepare_project_peak_set(
            owner,
            settings,
            retry=lambda: None,
            set_busy=lambda *_args: None,
            show_error=lambda message: pytest.fail(message),
        )
    finally:
        owner.deleteLater()


def test_cancelled_project_peak_generation_has_no_side_effects(
    qapp,
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    settings = ProjectSettings(output_dir=str(project))
    owner = QtWidgets.QWidget()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    try:
        assert not prepare_project_peak_set(
            owner,
            settings,
            retry=lambda: pytest.fail("取消后不应重试分析"),
            set_busy=lambda *_args: pytest.fail("取消后不应启动后台任务"),
            show_error=lambda message: pytest.fail(message),
        )
        assert settings.active_peak_set_id == ""
        assert not (project / "analysis" / "spectrum" / "manual_peaks").exists()
    finally:
        owner.deleteLater()
