"""Progress dialog for background operations.

Shows operation progress with a progress bar, message label,
and cancel button. Automatically styled with theme colors.
"""

from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class ProgressDialog(QtWidgets.QDialog):
    """Modal progress dialog for background operations.

    Shows:
    - Progress bar (0-100%)
    - Status message
    - Cancel button

    Signals:
        rejected(): Emitted when user clicks Cancel
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, title: str = "Operation in Progress"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        # Main layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Message label
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setObjectName("ProjectHint")
        layout.addWidget(self.message_label)

        # Spacer
        layout.addStretch()

        # Cancel button
        cancel_button = QtWidgets.QPushButton("取消")
        cancel_button.setMaximumWidth(100)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

    def update(self, percent: int, message: str = ""):
        """Update progress bar and message.

        Args:
            percent: Progress percentage (0-100)
            message: Status message to display
        """
        self.progress_bar.setValue(percent)
        if message:
            self.message_label.setText(message)

    def set_message(self, message: str):
        """Set message without updating progress."""
        self.message_label.setText(message)

    def reset(self):
        """Reset dialog to initial state."""
        self.progress_bar.setValue(0)
        self.message_label.setText("")
