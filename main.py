import sys
from core.output_paths import ensure_output_structure
from frontends.pyqt_app.login_dialog import LoginDialog
from frontends.pyqt_app.main_window import MainWindow
from frontends.pyqt_app.theme import apply_application_theme
from PyQt6 import QtWidgets


if __name__ == "__main__":
    ensure_output_structure()
    app = QtWidgets.QApplication(sys.argv)
    apply_application_theme(app)
    # login = LoginDialog()
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec())
    # if login.exec() == QtWidgets.QDialog.DialogCode.Accepted:
    #     ui = MainWindow()
    #     ui.show()
    #     sys.exit(app.exec())
    # else:
    #     sys.exit()
