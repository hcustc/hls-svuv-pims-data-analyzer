from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
try:
    from PyQt6 import QtCore, QtWidgets
except ImportError as e:
    pytest.skip(f"PyQt6 display libraries not available: {e}", allow_module_level=True)

pytestmark = pytest.mark.gui

pytest.importorskip("matplotlib")

from bl03u_masstool.frontends.pyqt_app.common import static_plot


def test_windows_system_chinese_font_is_registered_and_selected(tmp_path, monkeypatch):
    fonts_dir = tmp_path / "Fonts"
    fonts_dir.mkdir()
    yahei_path = fonts_dir / "msyh.ttc"
    yahei_path.touch()

    class _FontEntry:
        def __init__(self, name):
            self.name = name

    class _FontManager:
        def __init__(self):
            self.ttflist = [_FontEntry("DejaVu Sans")]
            self.registered = []

        def addfont(self, path):
            self.registered.append(path)
            self.ttflist.append(_FontEntry("Microsoft YaHei"))

    fake_manager = _FontManager()
    monkeypatch.setattr(static_plot.font_manager, "fontManager", fake_manager)

    selected = static_plot.configure_matplotlib_fonts(
        platform_name="win32",
        windows_dir=tmp_path,
    )

    assert fake_manager.registered == [str(yahei_path)]
    assert selected == "Microsoft YaHei"
    assert static_plot.rcParams["font.family"] == ["sans-serif"]
    assert static_plot.rcParams["font.sans-serif"][0] == "Microsoft YaHei"
