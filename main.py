import sys
from core.output_paths import ensure_output_structure
from frontends.pyqt_app.login_dialog import LoginDialog
from frontends.pyqt_app.main_window import MainWindow
from frontends.pyqt_app.theme import apply_application_theme
from frontends.pyqt_app.dialogs import PICSCalculatorDialog, CoreToolsDialog
from core.config import load_calibration_config
from core.normalization import load_normalization_settings
from PyQt6 import QtWidgets


def show_help():
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


if __name__ == "__main__":
    ensure_output_structure()
    app = QtWidgets.QApplication(sys.argv)
    apply_application_theme(app)
    
    args = sys.argv[1:]
    
    if "-h" in args or "--help" in args:
        show_help()
        sys.exit(0)
    
    elif "--pics" in args:
        calibration = load_calibration_config()
        normalization_settings = load_normalization_settings()
        dialog = CoreToolsDialog(calibration, initial_tab="pics")
        dialog.exec()
    
    elif "--tools" in args:
        calibration = load_calibration_config()
        dialog = CoreToolsDialog(calibration)
        dialog.exec()
    
    else:
        ui = MainWindow()
        ui.show()
        sys.exit(app.exec())
