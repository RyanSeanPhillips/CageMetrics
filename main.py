#!/usr/bin/env python3
"""
CageMetrics - Behavioral Analysis App

A PyQt6 application for analyzing behavioral data from Allentown cage monitoring systems.
Processes individual animal data, generates visualizations, and exports extracted datasets.

Part of the PhysioMetrics ecosystem.
"""

import sys
import os
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QFileDialog,
    QScrollArea, QFrame, QStatusBar, QMessageBox,
    QProgressBar, QSizePolicy, QScroller
)
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, QThread, QEvent
from PyQt6.QtGui import QFont, QWheelEvent

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Local imports
from core.data_loader import DataLoader
from core.analysis import BehaviorAnalyzer
from core.figure_generator import FigureGenerator


# Dark theme stylesheet (matching PhysioMetrics)
class WheelScrollArea(QScrollArea):
    """Custom QScrollArea that captures wheel events from all child widgets.

    This fixes the issue where scrolling only works when hovering over the scrollbar,
    by intercepting wheel events from child widgets (like matplotlib canvases) and
    forwarding them to the scroll area.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._content_widget = None

    def setWidget(self, widget):
        """Override to store reference to content widget."""
        super().setWidget(widget)
        self._content_widget = widget
        # Install event filter on the widget and existing children
        self._install_event_filter_recursive(widget)

    def _install_event_filter_recursive(self, widget):
        """Install event filter on widget and all its children."""
        if widget:
            widget.installEventFilter(self)
            for child in widget.findChildren(QWidget):
                child.installEventFilter(self)

    def install_filter_on_new_widgets(self):
        """Call this after adding new widgets to ensure they get the event filter."""
        if self._content_widget:
            self._install_event_filter_recursive(self._content_widget)

    def eventFilter(self, obj, event):
        """Intercept wheel events and handle scrolling."""
        if event.type() == QEvent.Type.Wheel:
            # Scroll the viewport
            delta = event.angleDelta().y()
            scroll_bar = self.verticalScrollBar()
            # Adjust scroll step (divide delta for smoother scrolling)
            scroll_bar.setValue(scroll_bar.value() - delta)
            return True  # Event handled
        return super().eventFilter(obj, event)


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
}

QTabWidget::pane {
    border: 1px solid #3d3d3d;
    background-color: #2d2d2d;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #ffffff;
    padding: 8px 16px;
    margin-right: 2px;
    border: 1px solid #3d3d3d;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #3daee9;
    color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #3d3d3d;
}

QPushButton {
    background-color: #3daee9;
    color: #ffffff;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #2d9ed9;
}

QPushButton:pressed {
    background-color: #1d8ec9;
}

QPushButton:disabled {
    background-color: #4d4d4d;
    color: #8d8d8d;
}

QLineEdit {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #3d3d3d;
    padding: 6px;
    border-radius: 4px;
}

QLineEdit:focus {
    border: 1px solid #3daee9;
}

QLabel {
    color: #ffffff;
}

QScrollArea {
    border: none;
    background-color: #2d2d2d;
}

QFrame {
    background-color: #2d2d2d;
    border-radius: 4px;
}

QStatusBar {
    background-color: #2d2d2d;
    color: #ffffff;
    border-top: 1px solid #3d3d3d;
}

QProgressBar {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    background-color: #2d2d2d;
    text-align: center;
    color: #ffffff;
}

QProgressBar::chunk {
    background-color: #3daee9;
    border-radius: 3px;
}

QMessageBox {
    background-color: #2d2d2d;
}

NavigationToolbar2QT {
    background-color: #2d2d2d;
    border: none;
}
"""


class ScrollableFigureWidget(QWidget):
    """Widget displaying all figures for an animal in a scrollable view with toolbar."""

    def __init__(self, animal_id: str, animal_data: dict, parent=None):
        super().__init__(parent)
        self.animal_id = animal_id
        self.animal_data = animal_data
        self.figure_generator = FigureGenerator()
        self.canvases = []

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)

        # Scrollable area for all figures - use custom class for wheel event handling
        self.scroll_area = WheelScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        # Container for all figures
        self.figures_container = QWidget()
        self.figures_layout = QVBoxLayout(self.figures_container)
        self.figures_layout.setContentsMargins(5, 5, 5, 5)
        self.figures_layout.setSpacing(10)

        self.scroll_area.setWidget(self.figures_container)
        layout.addWidget(self.scroll_area)

    def generate_and_display_figures(self, progress_callback=None):
        """Generate all figures and add them to the scrollable view."""
        pages = self.figure_generator.generate_all_pages(self.animal_id, self.animal_data)

        for i, (title, fig) in enumerate(pages):
            if progress_callback:
                progress_callback(f"Rendering {title}...")

            # Create frame for this figure
            fig_frame = QFrame()
            fig_frame.setStyleSheet("QFrame { background-color: #252525; border: 1px solid #3d3d3d; }")
            fig_layout = QVBoxLayout(fig_frame)
            fig_layout.setContentsMargins(5, 5, 5, 5)
            fig_layout.setSpacing(2)

            # Title label
            title_label = QLabel(f"{title}")
            title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fig_layout.addWidget(title_label)

            # Canvas
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(600)
            self.canvases.append(canvas)

            # Toolbar
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setStyleSheet("background-color: #2d2d2d;")

            fig_layout.addWidget(toolbar)
            fig_layout.addWidget(canvas)

            self.figures_layout.addWidget(fig_frame)

        # Add stretch at the end
        self.figures_layout.addStretch()

        # Install event filters on newly added widgets for scroll wheel support
        self.scroll_area.install_filter_on_new_widgets()


class AnalysisTab(QWidget):
    """Main analysis tab for processing individual data files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.current_file = None
        self.analyzed_data = None
        self.data_loader = DataLoader()
        self.analyzer = BehaviorAnalyzer()
        self.timing_info = {}

        self.setup_ui()

    def setup_ui(self):
        """Set up the analysis tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # === Top Bar: Browse, Path, Analyze, Save - all in one row ===
        top_frame = QFrame()
        top_frame.setStyleSheet("QFrame { background-color: #2d2d2d; padding: 5px; }")
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(10, 8, 10, 8)
        top_layout.setSpacing(10)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setFixedWidth(90)
        self.browse_btn.clicked.connect(self.browse_file)
        top_layout.addWidget(self.browse_btn)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a data file to analyze...")
        self.file_path_edit.setReadOnly(True)
        top_layout.addWidget(self.file_path_edit)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setFixedWidth(90)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_file)
        top_layout.addWidget(self.analyze_btn)

        self.save_btn = QPushButton("Save Data")
        self.save_btn.setFixedWidth(100)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_data)
        top_layout.addWidget(self.save_btn)

        layout.addWidget(top_frame)

        # === Progress Bar ===
        self.progress_frame = QFrame()
        self.progress_frame.setStyleSheet("QFrame { background-color: #2d2d2d; }")
        progress_layout = QHBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(10, 5, 10, 5)

        self.progress_label = QLabel("Ready")
        self.progress_label.setFixedWidth(200)
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.timing_label = QLabel("")
        self.timing_label.setFixedWidth(150)
        self.timing_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        progress_layout.addWidget(self.timing_label)

        layout.addWidget(self.progress_frame)

        # === Animal Tabs (main viewing area) ===
        self.animal_tabs = QTabWidget()
        self.animal_tabs.setDocumentMode(True)

        # Placeholder when no data loaded
        placeholder = QLabel("Load and analyze a data file to view results")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #8d8d8d; font-size: 14px;")
        self.animal_tabs.addTab(placeholder, "No Data")

        layout.addWidget(self.animal_tabs)

    def update_progress(self, message: str, value: int = None):
        """Update progress bar and label."""
        self.progress_label.setText(message)
        if value is not None:
            self.progress_bar.setValue(value)
        QApplication.processEvents()

    def browse_file(self):
        """Open file browser to select data file."""
        settings = QSettings("PhysioMetrics", "CageMetrics")
        last_dir = settings.value("last_data_dir", str(Path.home()))

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            last_dir,
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            settings.setValue("last_data_dir", str(Path(file_path).parent))

            self.current_file = file_path
            self.file_path_edit.setText(file_path)
            self.analyze_btn.setEnabled(True)
            self.save_btn.setEnabled(False)

            self.clear_animal_tabs()
            self.update_progress(f"Selected: {Path(file_path).name}", 0)

    def analyze_file(self):
        """Analyze the selected data file with timing."""
        if not self.current_file:
            return

        self.timing_info = {}
        total_start = time.time()

        try:
            # Step 1: Load data
            self.update_progress("Loading Excel file...", 10)
            load_start = time.time()

            raw_data = self.data_loader.load_file(self.current_file)
            self.timing_info['load'] = time.time() - load_start
            print(f"[Timing] Load file: {self.timing_info['load']:.2f}s")

            if raw_data is None:
                QMessageBox.warning(self, "Error", "Failed to load data file.")
                self.update_progress("Load failed", 0)
                return

            self.update_progress(f"Loaded {len(raw_data):,} rows", 30)

            # Step 2: Analyze data
            cohort_name = Path(self.current_file).stem.replace("_1MinuteBouts", "")

            self.update_progress("Analyzing animals...", 40)
            analyze_start = time.time()

            self.analyzed_data = self.analyzer.analyze_all_animals(raw_data, cohort_name)
            self.timing_info['analyze'] = time.time() - analyze_start
            print(f"[Timing] Analyze: {self.timing_info['analyze']:.2f}s")

            if not self.analyzed_data:
                QMessageBox.warning(self, "Error", "No valid animals found.")
                self.update_progress("No animals found", 0)
                return

            n_animals = len(self.analyzed_data)
            self.update_progress(f"Analyzed {n_animals} animals", 60)

            # Step 3: Generate figures
            self.update_progress("Generating figures...", 70)
            figure_start = time.time()

            self.populate_animal_tabs()

            self.timing_info['figures'] = time.time() - figure_start
            print(f"[Timing] Figures: {self.timing_info['figures']:.2f}s")

            # Done
            self.timing_info['total'] = time.time() - total_start
            print(f"[Timing] Total: {self.timing_info['total']:.2f}s")

            self.save_btn.setEnabled(True)
            self.update_progress(f"Complete: {n_animals} animals", 100)
            self.timing_label.setText(f"Total: {self.timing_info['total']:.1f}s")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{str(e)}")
            self.update_progress("Analysis failed", 0)
            import traceback
            traceback.print_exc()

    def clear_animal_tabs(self):
        """Clear all animal tabs."""
        self.animal_tabs.clear()
        placeholder = QLabel("Load and analyze a data file to view results")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #8d8d8d; font-size: 14px;")
        self.animal_tabs.addTab(placeholder, "No Data")

    def populate_animal_tabs(self):
        """Create tabs for each analyzed animal with scrollable figures."""
        self.animal_tabs.clear()

        total_animals = len(self.analyzed_data)
        for idx, (animal_id, animal_data) in enumerate(self.analyzed_data.items()):
            genotype = animal_data.get('genotype', 'Unknown')
            tab_title = f"{animal_id} ({genotype})"

            self.update_progress(f"Rendering {animal_id}...", 70 + int(25 * idx / total_animals))

            animal_widget = ScrollableFigureWidget(animal_id, animal_data)
            animal_widget.generate_and_display_figures(
                progress_callback=lambda msg: self.update_progress(msg)
            )

            self.animal_tabs.addTab(animal_widget, tab_title)

    def save_data(self):
        """Save extracted data for all animals."""
        if not self.analyzed_data:
            return

        settings = QSettings("PhysioMetrics", "CageMetrics")
        last_dir = settings.value("last_save_dir", str(Path.home()))

        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", last_dir
        )

        if save_dir:
            settings.setValue("last_save_dir", save_dir)

            try:
                self.update_progress("Saving data...", 50)

                from core.data_exporter import DataExporter
                exporter = DataExporter()
                saved_files = exporter.export_all_animals(self.analyzed_data, save_dir)

                self.update_progress(f"Saved {len(saved_files)} files", 100)

                QMessageBox.information(
                    self, "Save Complete",
                    f"Saved {len(saved_files)} files to:\n{save_dir}"
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")
                import traceback
                traceback.print_exc()


class ConsolidationTab(QWidget):
    """Consolidation tab for combining multiple animal datasets (future)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up placeholder UI."""
        layout = QVBoxLayout(self)

        placeholder = QLabel("Consolidation features coming soon...\n\n"
                           "This tab will allow you to:\n"
                           "- Load multiple extracted animal datasets\n"
                           "- Group by genotype/treatment\n"
                           "- Compute population statistics\n"
                           "- Generate comparison figures")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #8d8d8d; font-size: 14px;")

        layout.addWidget(placeholder)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CageMetrics - Behavioral Analysis")
        self.setMinimumSize(1200, 800)

        settings = QSettings("PhysioMetrics", "CageMetrics")
        geometry = settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1400, 900)

        self.setup_ui()

    def setup_ui(self):
        """Set up the main window UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main tab widget
        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)

        # Analysis tab
        self.analysis_tab = AnalysisTab(self)
        self.main_tabs.addTab(self.analysis_tab, "Analysis")

        # Consolidation tab (placeholder)
        self.consolidation_tab = ConsolidationTab(self)
        self.main_tabs.addTab(self.consolidation_tab, "Consolidation")

        main_layout.addWidget(self.main_tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def closeEvent(self, event):
        """Save window state on close."""
        settings = QSettings("PhysioMetrics", "CageMetrics")
        settings.setValue("window_geometry", self.saveGeometry())
        event.accept()


def main():
    """Main entry point."""
    plt.style.use('dark_background')

    app = QApplication(sys.argv)
    app.setApplicationName("CageMetrics")
    app.setOrganizationName("PhysioMetrics")

    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
