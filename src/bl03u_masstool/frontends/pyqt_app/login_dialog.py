from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QMessageBox, QPushButton, QVBoxLayout

class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册")
        self.resize(300, 150)
        layout = QFormLayout(self)

        self.username_edit = QLineEdit(self)
        self.password_edit = QLineEdit(self)
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_password_edit = QLineEdit(self)
        self.confirm_password_edit.setEchoMode(QLineEdit.EchoMode.Password)

        layout.addRow("用户名：", self.username_edit)
        layout.addRow("密码：", self.password_edit)
        layout.addRow("确认密码：", self.confirm_password_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.button_box.accepted.connect(self.register)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def register(self):
        username = self.username_edit.text()
        password = self.password_edit.text()
        confirm_password = self.confirm_password_edit.text()
        if not username or not password or not confirm_password:
            QMessageBox.warning(self, "注册失败", "请填写所有字段！")
            return
        if password != confirm_password:
            QMessageBox.warning(self, "注册失败", "两次输入的密码不一致！")
            return
        # 这里可以添加注册逻辑，例如保存到数据库或文件
        QMessageBox.information(self, "注册成功", "注册成功！")
        self.accept()

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("登录")
        self.resize(300, 100)
        layout = QFormLayout(self)

        self.username_edit = QLineEdit(self)
        self.password_edit = QLineEdit(self)
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)

        layout.addRow("用户名：", self.username_edit)
        layout.addRow("密码：", self.password_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.button_box.accepted.connect(self.check_login)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.register_button = QPushButton("注册", self)
        self.register_button.clicked.connect(self.show_register_dialog)
        layout.addWidget(self.register_button)

    def check_login(self):
        username = self.username_edit.text()
        password = self.password_edit.text()
        if username == "zhaolong" and password == "123456":
            self.accept()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误！")

    def show_register_dialog(self):
        register_dialog = RegisterDialog(self)
        register_dialog.exec() 