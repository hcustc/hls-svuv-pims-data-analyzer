from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from bl03u_masstool.core.config import load_calibration_config
from bl03u_masstool.core.output_paths import ensure_output_structure
from bl03u_masstool.frontends.pyqt_app.core_tools.dialog import CoreToolsDialog
from bl03u_masstool.frontends.pyqt_app.spectrum.workbench import MainWindow
from bl03u_masstool.frontends.pyqt_app.theme import apply_application_theme


def show_help() -> None:
    help_text = """
BL03U 数据分析仪启动脚本

用法:
    python main.py [选项]

选项:
    -h, --help          显示此帮助信息
    --pics              直接启动PICS计算工具
    --tools             直接启动核心处理工具集

示例:
    python main.py              # 启动主程序
    python main.py --pics       # 直接打开PICS计算工具
    python main.py --tools      # 直接打开核心处理工具集
    """
    print(help_text)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "-h" in args or "--help" in args:
        show_help()
        return 0

    ensure_output_structure()
    app = QtWidgets.QApplication(sys.argv)
    apply_application_theme(app)

    if "--pics" in args:
        calibration = load_calibration_config()
        dialog = CoreToolsDialog(calibration, initial_tab="pics")
        dialog.exec()
        return 0

    if "--tools" in args:
        calibration = load_calibration_config()
        dialog = CoreToolsDialog(calibration)
        dialog.exec()
        return 0

    ui = MainWindow()
    ui.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
