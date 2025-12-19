#!/usr/bin/env python3
"""
Theme Preview - Compare button and progress bar color themes
Run this to see all themes side by side without modifying the main app.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QLabel, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt

# Base dark theme (shared across all previews)
BASE_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLabel {
    color: #cccccc;
}
"""

# Theme definitions
THEMES = {
    "Current (Flat Blue)": """
        QPushButton {
            background-color: #3daee9;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #4dc0ff;
        }
        QPushButton:pressed {
            background-color: #2d9ed9;
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #3daee9;
            border-radius: 3px;
        }
    """,

    "Gradient Blue (Modern)": """
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
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1976D2, stop:1 #1565C0);
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #2196F3, stop:1 #4db8ff);
            border-radius: 3px;
        }
    """,

    "Teal/Cyan (Fresh)": """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #26c6da, stop:1 #00acc1);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4dd0e1, stop:1 #26c6da);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #00838f, stop:1 #006064);
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00acc1, stop:0.5 #26c6da, stop:1 #4dd0e1);
            border-radius: 3px;
        }
    """,

    "Purple/Violet (Vibrant)": """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ab47bc, stop:1 #8e24aa);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ce93d8, stop:1 #ab47bc);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #7b1fa2, stop:1 #6a1b9a);
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #8e24aa, stop:0.5 #ab47bc, stop:1 #ce93d8);
            border-radius: 3px;
        }
    """,

    "Green (Nature)": """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #66bb6a, stop:1 #43a047);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #81c784, stop:1 #66bb6a);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2e7d32, stop:1 #1b5e20);
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #43a047, stop:0.5 #66bb6a, stop:1 #81c784);
            border-radius: 3px;
        }
    """,

    "Orange/Amber (Warm)": """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffa726, stop:1 #fb8c00);
            color: #1e1e1e;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffb74d, stop:1 #ffa726);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ef6c00, stop:1 #e65100);
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #fb8c00, stop:0.5 #ffa726, stop:1 #ffb74d);
            border-radius: 3px;
        }
    """,

    "3D Classic (Subtle)": """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5c5c5c, stop:0.4 #4a4a4a, stop:1 #3a3a3a);
            color: #ffffff;
            border: 1px solid #2a2a2a;
            border-bottom: 2px solid #1a1a1a;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #6c6c6c, stop:0.4 #5a5a5a, stop:1 #4a4a4a);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3a3a3a, stop:1 #4a4a4a);
            border-bottom: 1px solid #1a1a1a;
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5dade2, stop:1 #3498db);
            border-radius: 3px;
        }
    """,

    "Rose/Pink (Soft)": """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f06292, stop:1 #e91e63);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f48fb1, stop:1 #f06292);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #c2185b, stop:1 #ad1457);
        }
        QProgressBar {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e91e63, stop:0.5 #f06292, stop:1 #f48fb1);
            border-radius: 3px;
        }
    """,
}


class ThemePreviewWidget(QGroupBox):
    """Widget showing a single theme preview."""

    def __init__(self, theme_name: str, theme_style: str):
        super().__init__(theme_name)
        self.setStyleSheet(theme_style)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Buttons row
        btn_layout = QHBoxLayout()

        btn1 = QPushButton("Browse...")
        btn2 = QPushButton("Save Data")
        btn3 = QPushButton("Action")

        btn_layout.addWidget(btn1)
        btn_layout.addWidget(btn2)
        btn_layout.addWidget(btn3)

        layout.addLayout(btn_layout)

        # Progress bars at different levels
        for pct in [25, 50, 75, 100]:
            prog = QProgressBar()
            prog.setRange(0, 100)
            prog.setValue(pct)
            prog.setFormat(f"{pct}%")
            layout.addWidget(prog)


class ThemePreviewWindow(QMainWindow):
    """Main window showing all theme previews."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CageMetrics Theme Preview")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(BASE_STYLE)

        # Grid layout for themes
        layout = QGridLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Add title
        title = QLabel("Click buttons and hover to see interactions. Close window when done.")
        title.setStyleSheet("font-size: 12pt; color: #ffffff; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, 0, 1, 4)

        # Add theme previews in a grid (2 rows x 4 columns)
        row, col = 1, 0
        for theme_name, theme_style in THEMES.items():
            preview = ThemePreviewWidget(theme_name, theme_style)
            layout.addWidget(preview, row, col)

            col += 1
            if col >= 4:
                col = 0
                row += 1


def main():
    app = QApplication(sys.argv)
    window = ThemePreviewWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
