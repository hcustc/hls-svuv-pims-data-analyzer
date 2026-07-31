from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("matplotlib")
pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

from bl03u_masstool.frontends.pyqt_app.common.plot_spec import (
    CurveSeries,
    PlotProfile,
    ScientificPlotSpec,
    SeriesRole,
    VerticalReference,
)
from bl03u_masstool.frontends.pyqt_app.common.static_plot import (
    StaticCurvePlot,
    load_plot_profile,
)
from bl03u_masstool.frontends.pyqt_app.common.widgets import AnalysisProgressState


pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _pie_spec() -> ScientificPlotSpec:
    x = np.array([7.2, 7.1, 7.3])
    return ScientificPlotSpec(
        title="m/z 40 PIE物种识别拟合",
        xlabel="光子能量 / eV",
        ylabel="相对信号强度",
        show_legend=True,
        legend_loc="upper left",
        series=(
            CurveSeries("exp", SeriesRole.EXPERIMENTAL, x, [1.0, 0.5, 2.0], "实验数据"),
            CurveSeries("fit", SeriesRole.TOTAL_FIT, x, [0.9, 0.6, 1.9], "总拟合"),
            CurveSeries(
                "allene",
                SeriesRole.COMPONENT,
                x,
                [0.2, 0.1, 0.3],
                "Allene",
                contribution_percent=20.0,
            ),
            CurveSeries(
                "propyne",
                SeriesRole.COMPONENT,
                x,
                [0.7, 0.5, 1.6],
                "Propyne",
                contribution_percent=80.0,
            ),
            CurveSeries(
                "zero",
                SeriesRole.COMPONENT,
                x,
                [0.0, 0.0, 0.0],
                "Zero contribution",
                contribution_percent=0.0,
            ),
        ),
    )


def test_packaged_profiles_are_local_and_disable_latex():
    for profile in PlotProfile:
        params = load_plot_profile(profile)
        assert params
        assert params["text.usetex"] is False
        assert params["axes.unicode_minus"] is False
    screen = load_plot_profile(PlotProfile.SCREEN)
    assert screen["axes.grid"] is False
    assert screen["xtick.direction"] == "in"
    assert screen["ytick.direction"] == "in"
    assert screen["axes.prop_cycle"].by_key()["color"][:4] == [
        "#0C5DA5",
        "#00B945",
        "#FF9500",
        "#FF2C00",
    ]
    publication = load_plot_profile(PlotProfile.PUBLICATION)
    assert publication["figure.figsize"] == [3.5, 2.625]
    assert publication["axes.linewidth"] == pytest.approx(0.5)
    assert publication["savefig.pad_inches"] == pytest.approx(0.05)


def test_renderer_preserves_order_and_hides_zero_components(qapp):
    widget = StaticCurvePlot("x", "y")
    widget.render_spec(_pie_spec())

    labels = [line.get_label() for line in widget.axes.lines]
    assert labels == ["实验数据", "总拟合", "Allene", "Propyne"]
    np.testing.assert_array_equal(widget.axes.lines[0].get_xdata(), [7.2, 7.1, 7.3])
    assert widget.axes.lines[2].get_linestyle() != widget.axes.lines[3].get_linestyle()
    assert [text.get_text() for text in widget.axes.get_legend().get_texts()] == labels
    assert not any(line.get_visible() for line in widget.axes.get_xgridlines())
    assert not any(line.get_visible() for line in widget.axes.get_ygridlines())
    assert widget.axes.spines["top"].get_visible()
    assert widget.axes.spines["right"].get_visible()


def test_publication_export_restores_screen_plot(qapp, tmp_path):
    widget = StaticCurvePlot("x", "y")
    widget.render_spec(_pie_spec())
    output = tmp_path / "pie-preview.png"
    original_size = widget.figure.get_size_inches().copy()

    assert widget.save_plot(output)
    assert output.is_file()
    assert output.stat().st_size > 0
    np.testing.assert_allclose(widget.figure.get_size_inches(), original_size)
    assert widget._last_profile is PlotProfile.SCREEN
    assert [line.get_label() for line in widget.axes.lines] == [
        "实验数据",
        "总拟合",
        "Allene",
        "Propyne",
    ]


def test_vertical_reference_and_nearest_point_cursor(qapp):
    widget = StaticCurvePlot("x", "y")
    spec = replace(
        _pie_spec(),
        vertical_references=(
            VerticalReference(
                key="candidate-ie",
                x=7.2,
                label="IE · CH2O 7.2000 eV",
                emphasized=True,
            ),
        ),
    )
    widget.render_spec(spec)

    reference = next(
        line for line in widget.axes.lines
        if line.get_label() == "IE · CH2O 7.2000 eV"
    )
    assert list(reference.get_xdata()) == pytest.approx([7.2, 7.2])
    assert reference.get_linewidth() > 1.0

    widget.set_nearest_point_cursor(
        [7.0, 7.2, 7.4],
        [0.0, 0.25, 0.5],
        x_label="E",
        y_label="I",
        x_unit="eV",
    )
    widget._on_cursor_motion(
        SimpleNamespace(inaxes=widget.axes, xdata=7.19)
    )
    assert widget._cursor_index == 1
    assert widget._cursor_marker.get_visible()
    assert "E = 7.2000 eV" in widget._cursor_annotation.get_text()
    assert "点击锁定" in widget._cursor_annotation.get_text()

    widget._on_cursor_click(
        SimpleNamespace(inaxes=widget.axes, xdata=7.19, button=1)
    )
    assert widget._cursor_locked
    widget._on_cursor_motion(
        SimpleNamespace(inaxes=widget.axes, xdata=7.4)
    )
    assert widget._cursor_index == 1
    widget._on_cursor_click(
        SimpleNamespace(inaxes=widget.axes, xdata=7.4, button=1)
    )
    assert not widget._cursor_locked
    assert widget._cursor_index == 2


def test_analysis_progress_state_reports_real_stage(qapp):
    widget = AnalysisProgressState(title="正在处理")
    widget.start(title="正在生成曲线", detail="正在读取数据…")
    widget.set_progress(42, "正在处理第 2 个数据目录")

    assert widget.title_label.text() == "正在生成曲线"
    assert widget.progress_bar.value() == 42
    assert widget.progress_bar.maximum() == 100
    assert widget.detail_label.text() == "正在处理第 2 个数据目录"
