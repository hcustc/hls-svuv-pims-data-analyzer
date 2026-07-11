from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow


pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_workspace_uses_task_oriented_formula_mass_name(qapp):
    window = MainWindow()
    try:
        assert window.page_buttons["isotope"].text() == "分子式与质量分析"
        assert "分子式" in window.page_buttons["isotope"].toolTip()
        assert "质量" in window.page_buttons["isotope"].toolTip()
    finally:
        window.deleteLater()


def test_formula_tab_calculates_summary_and_isotope_rows(qapp):
    widget = IsotopeAbundanceDialog()
    try:
        assert widget.tabs.tabText(0) == "分子式 → 质量与同位素"
        widget.formula_edit.setText("H2O")
        widget.calculate()

        assert "18" in widget.mass_label.text()
        assert "H × 2" in widget.composition_label.text()
        assert widget.table.rowCount() > 0
        assert widget.formula_status_label.property("status") == "success"
    finally:
        widget.deleteLater()


def test_candidate_tab_defaults_to_nominal_mass_search(qapp):
    widget = IsotopeAbundanceDialog()
    try:
        assert widget.tabs.tabText(1) == "质量数 → 候选分子式"
        assert widget.mass_mode_combo.currentData() == "nominal"
        assert widget.target_mass_edit.value() == pytest.approx(128)
        assert widget.tolerance_edit.value() == pytest.approx(0)
        assert not widget.tolerance_unit_combo.isEnabled()
        assert widget.chemical_rules_check.isChecked()
        assert {element for element, check in widget.element_checks.items() if check.isChecked()} == {"C", "H", "N", "O"}

        widget.mass_mode_combo.setCurrentIndex(1)
        assert widget.tolerance_unit_combo.isEnabled()
        assert widget.tolerance_unit_combo.currentData() == "ppm"
        assert widget.tolerance_edit.value() == pytest.approx(5)
    finally:
        widget.deleteLater()


def test_candidate_elements_follow_project_management_selection(qapp):
    widget = IsotopeAbundanceDialog()
    try:
        widget.set_project_settings(ProjectSettings(selected_elements=["H", "C", "F"]))

        selected = {element for element, check in widget.element_checks.items() if check.isChecked()}
        assert selected == {"H", "C", "F"}
        assert "项目管理" in widget.element_source_label.text()

        ranges = widget._candidate_element_ranges(128, 0, "Da")
        assert set(ranges) == {"H", "C", "F"}
        assert all(minimum == 0 and maximum > 0 for minimum, maximum in ranges.values())
    finally:
        widget.deleteLater()
