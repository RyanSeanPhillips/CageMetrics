"""
First Launch Dialog for CageMetrics.

Shows on first app launch to welcome users and request telemetry consent.
"""

import sys

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from version_info import VERSION_STRING


class FirstLaunchDialog(QDialog):
    """Dialog shown on first launch to welcome user and get telemetry consent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to CageMetrics")
        self.setMinimumSize(550, 500)
        self.resize(550, 500)
        self.setModal(True)

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(40, 30, 40, 30)

        # Welcome header
        header = QLabel("Welcome to CageMetrics")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #3daee9;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Version
        version_label = QLabel(f"Version {VERSION_STRING}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #888888; font-size: 11pt;")
        layout.addWidget(version_label)

        layout.addSpacing(10)

        # Description
        desc = QLabel(
            "CageMetrics helps you analyze behavioral data from\n"
            "Allentown cage monitoring systems.\n\n"
            "Features include consolidation of animal data,\n"
            "filtering, visualization, and statistical analysis."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 10pt;")
        layout.addWidget(desc)

        layout.addSpacing(15)

        # Telemetry section
        telemetry_frame = QFrame()
        telemetry_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        telemetry_layout = QVBoxLayout(telemetry_frame)
        telemetry_layout.setContentsMargins(15, 15, 15, 15)
        telemetry_layout.setSpacing(8)

        telemetry_header = QLabel("Help Improve CageMetrics")
        telemetry_header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        telemetry_header.setStyleSheet("color: #3daee9; border: none; background: transparent;")
        telemetry_layout.addWidget(telemetry_header)

        telemetry_desc = QLabel(
            "Share anonymous usage statistics to help improve the app.\n"
            "No personal data or file contents are ever collected."
        )
        telemetry_desc.setStyleSheet("font-size: 9pt; color: #cccccc; border: none; background: transparent;")
        telemetry_desc.setWordWrap(True)
        telemetry_layout.addWidget(telemetry_desc)

        # Checkboxes
        self.telemetry_checkbox = QCheckBox("Share anonymous usage statistics")
        self.telemetry_checkbox.setChecked(True)
        self.telemetry_checkbox.setStyleSheet("border: none; background: transparent;")
        telemetry_layout.addWidget(self.telemetry_checkbox)

        self.crash_checkbox = QCheckBox("Send crash reports")
        self.crash_checkbox.setChecked(True)
        self.crash_checkbox.setStyleSheet("border: none; background: transparent;")
        telemetry_layout.addWidget(self.crash_checkbox)

        layout.addWidget(telemetry_frame)

        # Note about changing settings
        note = QLabel("You can change these settings anytime in Help > About")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("font-size: 9pt; color: #888888;")
        layout.addWidget(note)

        layout.addStretch()

        # Continue button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        continue_btn = QPushButton("Get Started")
        continue_btn.setFixedSize(120, 36)
        continue_btn.clicked.connect(self.accept)
        button_layout.addWidget(continue_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def get_telemetry_enabled(self) -> bool:
        """Return whether telemetry was enabled."""
        return self.telemetry_checkbox.isChecked()

    def get_crash_reports_enabled(self) -> bool:
        """Return whether crash reports were enabled."""
        return self.crash_checkbox.isChecked()

    def _apply_dark_theme(self):
        """Apply dark theme to dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #5a5a5a;
                background-color: #2a2a2a;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #3daee9;
                background-color: #3daee9;
                border-radius: 3px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4db8ff, stop:1 #2196F3);
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6ec6ff, stop:1 #42a5f5);
            }
        """)

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value)
                )
            except Exception:
                pass
