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
    QScrollArea, QFrame, QStatusBar, QMessageBox, QDialog,
    QProgressBar, QSizePolicy, QScroller, QListWidget, QListWidgetItem,
    QAbstractItemView, QGroupBox, QTextEdit, QDialogButtonBox, QSplitter,
    QCheckBox, QDoubleSpinBox, QPlainTextEdit, QToolTip, QComboBox
)
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, QThread, QEvent
from PyQt6.QtGui import QFont, QWheelEvent, QShortcut, QKeySequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Local imports
from core.data_loader import DataLoader
from core.analysis import BehaviorAnalyzer
from core.figure_generator import FigureGenerator
from core import config as app_config
from core import telemetry
from core import update_checker
from core import sleep_analysis
from version_info import VERSION_STRING


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
    color: #cccccc;
    padding: 8px 16px;
    margin-right: 2px;
    border: 1px solid #3d3d3d;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #2d2d2d;
    border: 2px solid #3daee9;
    border-bottom: none;
    color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #3d3d3d;
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

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1976D2, stop:1 #1565C0);
}

QPushButton:disabled {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5d5d5d, stop:1 #4d4d4d);
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
    background-color: #1e1e1e;
    text-align: center;
    color: #ffffff;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2196F3, stop:1 #4db8ff);
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
        self.current_files = []  # List of selected files
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

        self.metadata_btn = QPushButton("Edit Metadata")
        self.metadata_btn.setFixedWidth(110)
        self.metadata_btn.setEnabled(False)
        self.metadata_btn.clicked.connect(self.edit_metadata)
        top_layout.addWidget(self.metadata_btn)

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

        # === Sleep Analysis Options ===
        self.sleep_options_frame = QFrame()
        self.sleep_options_frame.setStyleSheet("QFrame { background-color: #353535; border-radius: 4px; }")
        sleep_layout = QHBoxLayout(self.sleep_options_frame)
        sleep_layout.setContentsMargins(10, 5, 10, 5)
        sleep_layout.setSpacing(15)

        # Enable checkbox
        self.sleep_analysis_checkbox = QCheckBox("Sleep Bout Analysis")
        self.sleep_analysis_checkbox.setChecked(True)
        self.sleep_analysis_checkbox.setToolTip("Analyze sleep fragmentation from Sleeping % data")
        sleep_layout.addWidget(self.sleep_analysis_checkbox)

        # Threshold
        sleep_layout.addWidget(QLabel("Threshold:"))
        self.sleep_threshold_spin = QDoubleSpinBox()
        self.sleep_threshold_spin.setRange(0.1, 0.9)
        self.sleep_threshold_spin.setSingleStep(0.1)
        self.sleep_threshold_spin.setValue(0.5)
        self.sleep_threshold_spin.setToolTip("Sleep detection threshold (0-1)")
        self.sleep_threshold_spin.setFixedWidth(60)
        sleep_layout.addWidget(self.sleep_threshold_spin)

        # Bin width
        sleep_layout.addWidget(QLabel("Bin Width:"))
        self.sleep_bin_spin = QDoubleSpinBox()
        self.sleep_bin_spin.setRange(1.0, 30.0)
        self.sleep_bin_spin.setSingleStep(1.0)
        self.sleep_bin_spin.setValue(5.0)
        self.sleep_bin_spin.setSuffix(" min")
        self.sleep_bin_spin.setToolTip("Histogram bin width in minutes")
        self.sleep_bin_spin.setFixedWidth(80)
        sleep_layout.addWidget(self.sleep_bin_spin)

        # Re-analyze button
        self.reanalyze_sleep_btn = QPushButton("Re-analyze Sleep")
        self.reanalyze_sleep_btn.setFixedWidth(140)
        self.reanalyze_sleep_btn.setEnabled(False)
        self.reanalyze_sleep_btn.clicked.connect(self.reanalyze_sleep)
        self.reanalyze_sleep_btn.setToolTip("Re-run sleep analysis with new parameters (uses cached data)")
        sleep_layout.addWidget(self.reanalyze_sleep_btn)

        sleep_layout.addStretch()
        layout.addWidget(self.sleep_options_frame)

        # === Animal Tabs (main viewing area) ===
        self.animal_tabs = QTabWidget()
        self.animal_tabs.setObjectName("animalTabs")  # For stylesheet targeting
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
        """Open file browser to select one or more data files."""
        settings = QSettings("PhysioMetrics", "CageMetrics")
        last_dir = settings.value("last_data_dir", str(Path.home()))

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Data File(s)",
            last_dir,
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        )

        if file_paths:
            settings.setValue("last_data_dir", str(Path(file_paths[0]).parent))

            self.current_files = file_paths
            # Show file count or single filename
            if len(file_paths) == 1:
                self.file_path_edit.setText(file_paths[0])
            else:
                self.file_path_edit.setText(f"{len(file_paths)} files selected")
            self.save_btn.setEnabled(False)

            self.clear_animal_tabs()

            # Auto-analyze all files
            self.analyze_files()

    def analyze_files(self):
        """Analyze all selected data files with timing."""
        if not self.current_files:
            return

        self.timing_info = {}
        total_start = time.time()

        try:
            # Merge analyzed data from all files
            self.analyzed_data = {}
            total_rows = 0
            n_files = len(self.current_files)

            # Step 1: Load and analyze each file
            load_start = time.time()

            for file_idx, file_path in enumerate(self.current_files):
                file_name = Path(file_path).name
                progress_base = int(10 + (50 * file_idx / n_files))

                self.update_progress(f"Loading {file_name}... ({file_idx + 1}/{n_files})", progress_base)

                raw_data = self.data_loader.load_file(file_path)

                if raw_data is None:
                    print(f"[Warning] Failed to load: {file_path}")
                    continue

                total_rows += len(raw_data)
                self.update_progress(f"Analyzing {file_name}...", progress_base + 10)

                # Extract cohort name from filename
                cohort_name = Path(file_path).stem.replace("_1MinuteBouts", "")

                # Analyze this file's animals
                file_analyzed = self.analyzer.analyze_all_animals(raw_data, cohort_name)

                if file_analyzed:
                    # Merge into main dict (handle duplicate animal IDs by prefixing with cohort)
                    for animal_id, animal_data in file_analyzed.items():
                        # Create unique key if needed
                        unique_id = animal_id
                        if unique_id in self.analyzed_data:
                            unique_id = f"{cohort_name}_{animal_id}"

                        self.analyzed_data[unique_id] = animal_data

            self.timing_info['load_analyze'] = time.time() - load_start
            print(f"[Timing] Load & Analyze {n_files} files: {self.timing_info['load_analyze']:.2f}s")

            if not self.analyzed_data:
                QMessageBox.warning(self, "Error", "No valid animals found in any file.")
                self.update_progress("No animals found", 0)
                return

            n_animals = len(self.analyzed_data)
            self.update_progress(f"Analyzed {n_animals} animals from {n_files} file(s)", 60)

            # Step 1.5: Sleep bout analysis (if enabled)
            if self.sleep_analysis_checkbox.isChecked():
                self.update_progress("Running sleep bout analysis...", 65)
                self.run_sleep_analysis()

            # Enable re-analyze button now that we have data
            self.reanalyze_sleep_btn.setEnabled(True)

            # Step 2: Generate figures
            self.update_progress("Generating figures...", 70)
            figure_start = time.time()

            self.populate_animal_tabs()

            self.timing_info['figures'] = time.time() - figure_start
            print(f"[Timing] Figures: {self.timing_info['figures']:.2f}s")

            # Done
            self.timing_info['total'] = time.time() - total_start
            print(f"[Timing] Total: {self.timing_info['total']:.2f}s")

            self.save_btn.setEnabled(True)
            self.metadata_btn.setEnabled(True)
            self.update_progress(f"Complete: {n_animals} animals from {n_files} file(s)", 100)
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

    def run_sleep_analysis(self):
        """Run sleep bout analysis on all animals with Sleeping % data."""
        if not self.analyzed_data:
            return

        threshold = self.sleep_threshold_spin.value()
        bin_width = self.sleep_bin_spin.value()

        import numpy as np

        for animal_id, animal_data in self.analyzed_data.items():
            metrics = animal_data.get('metrics', {})

            # Find sleeping data
            sleeping_data = None
            for metric_name in ['Sleeping %', 'sleeping', 'sleep']:
                if metric_name in metrics:
                    sleeping_data = metrics[metric_name]
                    break

            if sleeping_data is None:
                # No sleeping data for this animal
                animal_data['sleep_analysis'] = None
                continue

            # Get daily data
            daily_data = sleeping_data.get('daily_data', [])
            if not daily_data:
                animal_data['sleep_analysis'] = None
                continue

            # Convert to numpy array
            daily_array = np.array(daily_data)

            # Run sleep analysis
            result = sleep_analysis.analyze_sleep(daily_array, threshold, bin_width)
            animal_data['sleep_analysis'] = result

        print(f"[Sleep Analysis] Completed for {len(self.analyzed_data)} animals (threshold={threshold}, bin_width={bin_width})")

    def reanalyze_sleep(self):
        """Re-run sleep analysis with new parameters and refresh figures."""
        if not self.analyzed_data:
            QMessageBox.warning(self, "No Data", "Load and analyze data first.")
            return

        self.update_progress("Re-analyzing sleep bouts...", 30)

        # Run sleep analysis with new parameters
        self.run_sleep_analysis()

        # Regenerate figures
        self.update_progress("Regenerating figures...", 60)
        self.populate_animal_tabs()

        self.update_progress("Sleep re-analysis complete", 100)
        self.timing_label.setText("")

    def populate_animal_tabs(self):
        """Create tabs for each analyzed animal with scrollable figures."""
        self.animal_tabs.clear()

        # Build cage-to-color mapping for rainbow tab colors
        cage_colors = self._get_cage_color_map()

        # Create cage-to-index mapping for shape selection
        cage_indices = {cage_id: idx for idx, cage_id in enumerate(cage_colors.keys())}

        # Sort animals by cage ID so cagemates are adjacent and in rainbow order
        sorted_animals = self._sort_animals_by_cage(cage_colors)

        # Collect tab info for color/shape icons: (tab_idx, color, shape_idx)
        tab_icons = []

        total_animals = len(sorted_animals)
        for idx, (animal_id, animal_data) in enumerate(sorted_animals):
            genotype = animal_data.get('genotype', 'Unknown')
            tab_title = f"{animal_id} ({genotype})"

            self.update_progress(f"Rendering {animal_id}...", 70 + int(25 * idx / total_animals))

            animal_widget = ScrollableFigureWidget(animal_id, animal_data)
            animal_widget.generate_and_display_figures(
                progress_callback=lambda msg: self.update_progress(msg)
            )

            tab_idx = self.animal_tabs.addTab(animal_widget, tab_title)

            # Collect cage color and shape index for this tab
            metadata = animal_data.get('metadata', {})
            cage_id = str(metadata.get('cage_id', 'Unknown'))
            if cage_id in cage_colors:
                color = cage_colors[cage_id]
                shape_idx = cage_indices.get(cage_id, 0)
                tab_icons.append((tab_idx, color, shape_idx))

        # Apply color/shape icons to tabs
        self._apply_tab_icons(tab_icons)

    def _get_cage_color_map(self):
        """Create a mapping of cage IDs to rainbow colors."""
        from PyQt6.QtGui import QColor

        # Rainbow colors (saturated, visible on dark background)
        rainbow = [
            QColor(255, 100, 100),   # Red
            QColor(255, 180, 100),   # Orange
            QColor(255, 255, 100),   # Yellow
            QColor(100, 255, 100),   # Green
            QColor(100, 255, 255),   # Cyan
            QColor(100, 150, 255),   # Blue
            QColor(180, 100, 255),   # Purple
            QColor(255, 100, 200),   # Pink
        ]

        # Get unique cage IDs
        cage_ids = set()
        for animal_data in self.analyzed_data.values():
            metadata = animal_data.get('metadata', {})
            cage_id = metadata.get('cage_id')
            if cage_id and cage_id != 'Unknown':
                cage_ids.add(str(cage_id))  # Convert to string for consistent comparison

        # Map each cage to a color
        cage_colors = {}
        sorted_cages = sorted(cage_ids, key=str)
        for i, cage_id in enumerate(sorted_cages):
            cage_colors[cage_id] = rainbow[i % len(rainbow)]

        print(f"[Debug] Cage color map: {list(cage_colors.keys())}")
        return cage_colors

    def _sort_animals_by_cage(self, cage_colors):
        """Sort animals by cage ID so cagemates are adjacent and in rainbow order.

        Args:
            cage_colors: Dict mapping cage_id to QColor (determines order)

        Returns:
            List of (animal_id, animal_data) tuples sorted by cage
        """
        # Get the cage order from cage_colors (already sorted)
        cage_order = {cage_id: idx for idx, cage_id in enumerate(cage_colors.keys())}

        # Build list with sort keys
        animals_with_keys = []
        for animal_id, animal_data in self.analyzed_data.items():
            metadata = animal_data.get('metadata', {})
            cage_id = str(metadata.get('cage_id', 'Unknown'))

            # Sort key: (cage_order, animal_id for stable secondary sort)
            sort_key = (cage_order.get(cage_id, 999), animal_id)
            animals_with_keys.append((sort_key, animal_id, animal_data))

        # Sort by cage order, then by animal_id within each cage
        animals_with_keys.sort(key=lambda x: x[0])

        # Return just (animal_id, animal_data) pairs
        sorted_animals = [(item[1], item[2]) for item in animals_with_keys]

        print(f"[Debug] Tab order: {[a[0] for a in sorted_animals]}")
        return sorted_animals

    def _apply_tab_icons(self, tab_icons):
        """Apply colored shape icons to tabs for cage identification.

        Args:
            tab_icons: List of (tab_index, QColor, shape_index) tuples

        Each cage gets a unique color AND shape for easy visual identification.
        """
        if not tab_icons:
            return

        from PyQt6.QtGui import QPixmap, QIcon, QPainter, QBrush, QPen, QPolygon
        from PyQt6.QtCore import QPoint, Qt

        tab_bar = self.animal_tabs.tabBar()

        # Shape drawing functions
        def draw_circle(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            painter.drawEllipse(1, 1, size - 2, size - 2)

        def draw_square(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            painter.drawRect(1, 1, size - 3, size - 3)

        def draw_triangle(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            points = QPolygon([
                QPoint(size // 2, 1),
                QPoint(1, size - 2),
                QPoint(size - 2, size - 2)
            ])
            painter.drawPolygon(points)

        def draw_diamond(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            mid = size // 2
            points = QPolygon([
                QPoint(mid, 0),
                QPoint(size - 1, mid),
                QPoint(mid, size - 1),
                QPoint(0, mid)
            ])
            painter.drawPolygon(points)

        def draw_star(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            # 4-pointed star
            mid = size // 2
            points = QPolygon([
                QPoint(mid, 0),
                QPoint(mid + 2, mid - 2),
                QPoint(size - 1, mid),
                QPoint(mid + 2, mid + 2),
                QPoint(mid, size - 1),
                QPoint(mid - 2, mid + 2),
                QPoint(0, mid),
                QPoint(mid - 2, mid - 2)
            ])
            painter.drawPolygon(points)

        def draw_hexagon(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            # Regular hexagon
            mid = size // 2
            q = size // 4
            points = QPolygon([
                QPoint(q, 1),
                QPoint(size - q - 1, 1),
                QPoint(size - 2, mid),
                QPoint(size - q - 1, size - 2),
                QPoint(q, size - 2),
                QPoint(1, mid)
            ])
            painter.drawPolygon(points)

        def draw_cross(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            # Plus/cross shape
            t = size // 3
            points = QPolygon([
                QPoint(t, 0),
                QPoint(2 * t, 0),
                QPoint(2 * t, t),
                QPoint(size - 1, t),
                QPoint(size - 1, 2 * t),
                QPoint(2 * t, 2 * t),
                QPoint(2 * t, size - 1),
                QPoint(t, size - 1),
                QPoint(t, 2 * t),
                QPoint(0, 2 * t),
                QPoint(0, t),
                QPoint(t, t)
            ])
            painter.drawPolygon(points)

        def draw_inv_triangle(painter, color, size):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 1))
            points = QPolygon([
                QPoint(1, 1),
                QPoint(size - 2, 1),
                QPoint(size // 2, size - 2)
            ])
            painter.drawPolygon(points)

        # List of shape drawing functions
        shapes = [
            draw_circle,
            draw_square,
            draw_triangle,
            draw_diamond,
            draw_star,
            draw_hexagon,
            draw_cross,
            draw_inv_triangle
        ]

        icon_size = 14

        for tab_idx, color, shape_idx in tab_icons:
            # Create transparent pixmap
            pixmap = QPixmap(icon_size, icon_size)
            pixmap.fill(Qt.GlobalColor.transparent)

            # Draw the shape
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Select shape based on cage index
            shape_func = shapes[shape_idx % len(shapes)]
            shape_func(painter, color, icon_size)

            painter.end()

            icon = QIcon(pixmap)
            tab_bar.setTabIcon(tab_idx, icon)

    def edit_metadata(self):
        """Open the metadata editor dialog."""
        if not self.analyzed_data:
            return

        from dialogs.metadata_editor_dialog import MetadataEditorDialog

        dialog = MetadataEditorDialog(self.analyzed_data, self)

        # Connect save signal to our save method
        dialog.save_requested.connect(self._on_metadata_save_requested)

        dialog.exec()

        # Get potentially modified data back
        self.analyzed_data = dialog.get_analyzed_data()

    def _on_metadata_save_requested(self):
        """Handle save request from metadata editor."""
        self.save_data()

    def save_data(self):
        """Save extracted data for all animals (PDF figures, Excel workbooks, NPZ)."""
        if not self.analyzed_data:
            return

        # Confirm save - files will go to "analysis" folder next to source data
        source_dir = Path(self.current_files[0]).parent if self.current_files else Path.home()
        analysis_dir = source_dir / "analysis"

        # Get preview of files that will be created
        from core.data_exporter import DataExporter
        exporter = DataExporter()
        filenames = exporter.get_filenames_preview(self.analyzed_data)

        # Show confirmation dialog with details button
        n_animals = len(self.analyzed_data)
        n_files = len(filenames)

        # Create custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Data")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Summary message
        summary_label = QLabel(
            f"Save data for <b>{n_animals} animals</b> to:\n"
            f"<code>{analysis_dir}</code>\n\n"
            f"This will create <b>{n_files} files</b>:\n"
            f"  • {n_animals} PDF figure files\n"
            f"  • {n_animals} Excel workbooks\n"
            f"  • {n_animals} NPZ data files"
        )
        summary_label.setTextFormat(Qt.TextFormat.RichText)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        # Expandable details section
        details_group = QGroupBox("File Details")
        details_group.setCheckable(True)
        details_group.setChecked(False)
        details_layout = QVBoxLayout(details_group)

        # Scrollable file list
        file_list = QTextEdit()
        file_list.setReadOnly(True)
        file_list.setMaximumHeight(200)

        # Build file list text
        file_list_text = ""
        for i in range(0, len(filenames), 3):
            base = filenames[i].replace('_Figures.pdf', '')
            file_list_text += f"{base}:\n"
            file_list_text += f"    _Figures.pdf\n"
            file_list_text += f"    _Data.xlsx\n"
            file_list_text += f"    _Data.npz\n\n"

        file_list.setPlainText(file_list_text.strip())
        details_layout.addWidget(file_list)

        # Hide details initially
        file_list.setVisible(False)
        details_group.toggled.connect(file_list.setVisible)

        layout.addWidget(details_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            self.update_progress("Exporting data...", 10)
            export_start = time.time()

            # Create figure generator for PDF export
            figure_generator = FigureGenerator()

            self.update_progress("Generating figures and saving files...", 30)

            saved_files = exporter.export_all_animals(
                self.analyzed_data,
                str(source_dir),
                source_file=self.current_files[0] if self.current_files else None,
                figure_generator=figure_generator,
                parallel=True,
                progress_callback=self.update_progress
            )

            export_time = time.time() - export_start
            print(f"[Timing] Export: {export_time:.2f}s")

            self.update_progress(f"Saved {len(saved_files)} files", 100)

            # Group files by type for summary
            pdf_count = sum(1 for f in saved_files if f.endswith('.pdf'))
            excel_count = sum(1 for f in saved_files if f.endswith('.xlsx'))
            npz_count = sum(1 for f in saved_files if f.endswith('.npz'))

            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(self.analyzed_data)} animals to:\n{analysis_dir}\n\n"
                f"Files created:\n"
                f"  • {pdf_count} PDF figure files\n"
                f"  • {excel_count} Excel workbooks\n"
                f"  • {npz_count} NPZ data files"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
            import traceback
            traceback.print_exc()


class ClickableLabel(QLabel):
    """A QLabel that emits a clicked signal when clicked."""
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ConsolidateListDialog(QDialog):
    """Dialog for viewing and managing files in the consolidate list."""

    def __init__(self, items: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Files to Consolidate")
        self.setMinimumSize(500, 400)
        self.removed_indices = []

        # Dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #404040;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton#removeBtn {
                color: #f44336;
            }
            QPushButton#removeBtn:hover {
                background-color: #5a3030;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        layout = QVBoxLayout(self)

        # Header with count
        self.header_label = QLabel(f"Files in consolidate list: {len(items)}")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.header_label)

        # List widget showing animal IDs
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        for i, item_data in enumerate(items):
            animal_id = item_data.get('metadata', {}).get('animal_id', 'Unknown')
            genotype = item_data.get('metadata', {}).get('genotype', '')
            display = f"{animal_id}"
            if genotype:
                display += f" ({genotype})"
            list_item = QListWidgetItem(display)
            list_item.setData(Qt.ItemDataRole.UserRole, i)  # Store original index
            self.list_widget.addItem(list_item)

        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setObjectName("removeBtn")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_layout.addWidget(self.remove_btn)

        btn_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _remove_selected(self):
        """Remove selected items and track which indices were removed."""
        selected = self.list_widget.selectedItems()
        for item in selected:
            idx = item.data(Qt.ItemDataRole.UserRole)
            self.removed_indices.append(idx)
            self.list_widget.takeItem(self.list_widget.row(item))

        # Update header
        self.header_label.setText(f"Files in consolidate list: {self.list_widget.count()}")

    def get_removed_indices(self) -> list:
        """Return list of indices that were removed."""
        return sorted(self.removed_indices, reverse=True)


class ConsolidationTab(QWidget):
    """Consolidation tab for combining multiple animal NPZ datasets."""

    # Dark theme stylesheets
    LIST_STYLE = """
        QListWidget {
            background-color: #2b2b2b;
            border: 2px solid #555555;
            border-radius: 6px;
            color: #ffffff;
            padding: 4px;
            font-size: 12px;
            outline: none;
        }
        QListWidget::item {
            background-color: transparent;
            padding: 6px 8px;
            margin: 1px 0px;
            border-radius: 3px;
            color: #ffffff;
        }
        QListWidget::item:hover {
            background-color: #404040;
        }
        QListWidget::item:selected {
            background-color: #0078d4;
            color: #ffffff;
        }
        QListWidget::item:selected:!active {
            background-color: #6c6c6c;
        }
        QListWidget:focus {
            border-color: #0078d4;
        }
        QListWidget QScrollBar:vertical {
            background-color: #3c3c3c;
            width: 12px;
            border-radius: 6px;
        }
        QListWidget QScrollBar::handle:vertical {
            background-color: #666666;
            border-radius: 6px;
            min-height: 20px;
        }
        QListWidget QScrollBar::handle:vertical:hover {
            background-color: #777777;
        }
    """

    BUTTON_STYLE = """
        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: 1px solid #0b5ed7;
            border-radius: 6px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #0b5ed7;
        }
        QPushButton:pressed {
            background-color: #0a58ca;
        }
        QPushButton:disabled {
            background-color: #9ec5fe;
            border-color: #9ec5fe;
            color: #eef4ff;
        }
    """

    MOVE_RIGHT_STYLE = """
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 14px;
            color: #4caf50;
        }
        QPushButton:hover {
            background-color: #505050;
            border-color: #777777;
            color: #66bb6a;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
    """

    MOVE_LEFT_STYLE = """
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 14px;
            color: #f44336;
        }
        QPushButton:hover {
            background-color: #505050;
            border-color: #777777;
            color: #ef5350;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._scan_dir = None
        self._cagemate_cache = None
        self._filter_criteria = None
        self._metadata_discovery = None
        self._filter_checkboxes = {}  # field_name -> {value: QCheckBox}
        self._animal_id_filter_terms = []  # Animal ID filter search terms
        self._cagemate_id_filter_terms = []  # Cagemate ID filter search terms
        self._preview_figures = []
        self.setup_ui()
        self._connect_signals()
        self._init_filter_criteria()

    def _match_wildcard(self, pattern: str, text: str) -> bool:
        """
        Match a pattern against text, supporting wildcards and anchors.

        Args:
            pattern: Search pattern with optional modifiers
            text: Text to match against

        Syntax:
            - Simple text: substring match (e.g., "_wt" matches "mouse_wt_01")
            - * wildcard: matches any text (e.g., "_*wt" matches "_anywt")
            - "text": exact match (e.g., "\"_wt\"" only matches "_wt" exactly)
            - text$: ends with (e.g., "_wt$" matches "mouse_wt" but not "mouse_wt_01")
            - ^text: starts with (e.g., "^mouse" matches "mouse_01" but not "test_mouse")

        Returns:
            True if pattern matches text
        """
        import re
        pattern_orig = pattern
        pattern = pattern.lower().strip()
        text = text.lower()

        # Check for exact match syntax: "term"
        if pattern.startswith('"') and pattern.endswith('"') and len(pattern) > 2:
            exact_term = pattern[1:-1]
            return text == exact_term

        # Check for ends-with syntax: term$
        if pattern.endswith('$') and not pattern.endswith('\\$'):
            ends_pattern = pattern[:-1]
            if '*' in ends_pattern:
                # Wildcard + ends-with: e.g., "*_wt$"
                regex_pattern = re.escape(ends_pattern).replace(r'\*', '.*') + '$'
                return bool(re.search(regex_pattern, text))
            else:
                return text.endswith(ends_pattern)

        # Check for starts-with syntax: ^term
        if pattern.startswith('^'):
            starts_pattern = pattern[1:]
            if '*' in starts_pattern:
                # Wildcard + starts-with: e.g., "^mouse*"
                regex_pattern = '^' + re.escape(starts_pattern).replace(r'\*', '.*')
                return bool(re.search(regex_pattern, text))
            else:
                return text.startswith(starts_pattern)

        # Check for wildcard
        if '*' in pattern:
            # Convert wildcard pattern to regex: * becomes .*
            # Escape other special regex characters first
            regex_pattern = re.escape(pattern).replace(r'\*', '.*')
            # Match anywhere in the string (not just from start)
            return bool(re.search(regex_pattern, text))
        else:
            # Simple substring match (original behavior)
            return pattern in text or text == pattern

    def _show_filter_help(self):
        """Show a dialog with filter syntax help and examples."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Syntax Help")
        dialog.setMinimumWidth(650)
        dialog.setMinimumHeight(400)

        layout = QVBoxLayout(dialog)

        # Title
        title_label = QLabel("Filter Pattern Syntax")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "Enter one or more patterns separated by commas. Use the dropdown to select "
            "OR (match any) or AND (match all) logic."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("margin-bottom: 15px; color: #aaaaaa;")
        layout.addWidget(desc_label)

        # Create help text with examples table
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("""
            QTextEdit {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 10px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)

        # HTML table with examples
        html_content = """
        <style>
            table { border-collapse: collapse; width: 100%; }
            th { background-color: #4a4a4a; padding: 8px; text-align: left; border-bottom: 2px solid #666; }
            td { padding: 6px 8px; border-bottom: 1px solid #444; }
            tr:hover { background-color: #454545; }
            .pattern { color: #88ccff; font-weight: bold; }
            .type { color: #aaddaa; }
            .match { color: #88ff88; }
            .nomatch { color: #ff8888; }
        </style>
        <table>
            <tr>
                <th>Pattern</th>
                <th>Type</th>
                <th>Matches</th>
                <th>Doesn't Match</th>
            </tr>
            <tr>
                <td class="pattern">_wt</td>
                <td class="type">Substring</td>
                <td class="match">mouse_wt_01, abc_wt</td>
                <td class="nomatch">mouse_ko</td>
            </tr>
            <tr>
                <td class="pattern">"_wt"</td>
                <td class="type">Exact</td>
                <td class="match">_wt</td>
                <td class="nomatch">mouse_wt, _wt_01</td>
            </tr>
            <tr>
                <td class="pattern">_wt$</td>
                <td class="type">Ends with</td>
                <td class="match">mouse_wt, abc_wt</td>
                <td class="nomatch">_wt_01, mouse_wt_x</td>
            </tr>
            <tr>
                <td class="pattern">^mouse</td>
                <td class="type">Starts with</td>
                <td class="match">mouse_01, mouse_wt</td>
                <td class="nomatch">test_mouse, my_mouse</td>
            </tr>
            <tr>
                <td class="pattern">_*wt</td>
                <td class="type">Wildcard</td>
                <td class="match">mouse_abc_wt, _anywt</td>
                <td class="nomatch">mousetrap</td>
            </tr>
            <tr>
                <td class="pattern">^m*_wt$</td>
                <td class="type">Combined</td>
                <td class="match">mouse_wt, m123_wt</td>
                <td class="nomatch">test_wt, mouse_wt_x</td>
            </tr>
        </table>
        <br>
        <b>Multiple Patterns:</b><br>
        <span style="color: #88ccff;">_wt, _ko</span> &nbsp;→&nbsp; Match animals with _wt <b>OR</b> _ko (using OR mode)<br>
        <span style="color: #88ccff;">_wt, _01</span> &nbsp;→&nbsp; Match animals with _wt <b>AND</b> _01 (using AND mode)<br>
        <br>
        <b>Quick Reference:</b><br>
        • <span class="pattern">text</span> = substring (contains)<br>
        • <span class="pattern">"text"</span> = exact match only<br>
        • <span class="pattern">text$</span> = ends with text<br>
        • <span class="pattern">^text</span> = starts with text<br>
        • <span class="pattern">*</span> = wildcard (any characters)
        """
        help_text.setHtml(html_content)
        layout.addWidget(help_text)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.close)
        layout.addWidget(button_box)

        dialog.exec()

    def setup_ui(self):
        """Set up the consolidation UI with full-tab scrolling."""
        # Root layout for the tab - just holds the scroll area
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Main scroll area for entire tab content
        self.main_scroll = WheelScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2d2d2d;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 4px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        # Container widget for all tab content
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #2d2d2d;")
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top row: Select folder button
        top_layout = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.setFixedWidth(120)
        self.select_folder_btn.setStyleSheet(self.BUTTON_STYLE)
        top_layout.addWidget(self.select_folder_btn)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: #8d8d8d;")
        top_layout.addWidget(self.folder_label)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Filters Panel
        self._setup_filter_panel(main_layout)

        # Main content: two list boxes with move buttons in a vertical splitter
        # Create a container for the lists section that can be resized
        lists_container = QWidget()
        lists_container_layout = QHBoxLayout(lists_container)
        lists_container_layout.setContentsMargins(0, 0, 0, 0)
        lists_container_layout.setSpacing(10)

        # Left side: Files Detected
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_label = QLabel("NPZ Files Detected")
        self.left_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #ffffff;")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.left_label)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet(self.LIST_STYLE)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.file_list, stretch=1)

        # Search/filter box
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        filter_layout.addWidget(filter_label)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by keywords (e.g., 'WT male' or 'WT, KO')...")
        filter_layout.addWidget(self.filter_edit)
        left_layout.addLayout(filter_layout)

        # Edit Metadata button for all detected files
        self.edit_all_metadata_btn = QPushButton("Edit Metadata")
        self.edit_all_metadata_btn.setStyleSheet(self.BUTTON_STYLE)
        self.edit_all_metadata_btn.setEnabled(False)
        self.edit_all_metadata_btn.setToolTip("Edit metadata for all detected NPZ files")
        left_layout.addWidget(self.edit_all_metadata_btn)

        lists_container_layout.addWidget(left_widget, stretch=1)

        # Middle: Move buttons
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()

        self.move_all_right_btn = QPushButton(">>")
        self.move_all_right_btn.setStyleSheet(self.MOVE_RIGHT_STYLE)
        self.move_all_right_btn.setToolTip("Move all visible items to consolidate list")
        button_layout.addWidget(self.move_all_right_btn)

        self.move_right_btn = QPushButton(">")
        self.move_right_btn.setStyleSheet(self.MOVE_RIGHT_STYLE)
        self.move_right_btn.setToolTip("Move selected items to consolidate list")
        button_layout.addWidget(self.move_right_btn)

        self.move_left_btn = QPushButton("<")
        self.move_left_btn.setStyleSheet(self.MOVE_LEFT_STYLE)
        self.move_left_btn.setToolTip("Remove selected items from consolidate list")
        button_layout.addWidget(self.move_left_btn)

        self.move_all_left_btn = QPushButton("<<")
        self.move_all_left_btn.setStyleSheet(self.MOVE_LEFT_STYLE)
        self.move_all_left_btn.setToolTip("Remove all items from consolidate list")
        button_layout.addWidget(self.move_all_left_btn)

        button_layout.addStretch()
        lists_container_layout.addWidget(button_widget)

        # Right side: Files to Consolidate
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Clickable count label that shows animal IDs on hover and opens dialog on click
        self.right_label = ClickableLabel("Files to Consolidate (n=0)")
        self.right_label.setStyleSheet("""
            QLabel {
                font-size: 10pt;
                font-weight: bold;
                color: #ffffff;
            }
            QLabel:hover {
                color: #3daee9;
                text-decoration: underline;
            }
        """)
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.right_label.clicked.connect(self._on_consolidate_label_clicked)
        right_layout.addWidget(self.right_label)

        self.consolidate_list = QListWidget()
        self.consolidate_list.setStyleSheet(self.LIST_STYLE)
        self.consolidate_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        right_layout.addWidget(self.consolidate_list, stretch=1)

        # Action buttons row
        action_layout = QHBoxLayout()

        self.preview_btn = QPushButton("Generate Preview")
        self.preview_btn.setStyleSheet(self.BUTTON_STYLE)
        self.preview_btn.setEnabled(False)
        action_layout.addWidget(self.preview_btn)

        self.consolidate_btn = QPushButton("Consolidate && Save")
        self.consolidate_btn.setStyleSheet(self.BUTTON_STYLE)
        self.consolidate_btn.setEnabled(False)
        action_layout.addWidget(self.consolidate_btn)

        right_layout.addLayout(action_layout)

        lists_container_layout.addWidget(right_widget, stretch=1)

        # Add lists container directly to main layout (no splitter - everything scrolls together)
        lists_container.setMinimumHeight(200)
        lists_container.setMaximumHeight(350)  # Constrain height so figures get space
        main_layout.addWidget(lists_container)

        # Create preview container and add directly to main layout
        self._setup_preview_panel_widget()  # Creates self.preview_widget
        main_layout.addWidget(self.preview_widget)

        # Set scroll content
        self.main_scroll.setWidget(scroll_content)
        root_layout.addWidget(self.main_scroll)

    def _connect_signals(self):
        """Connect button signals."""
        self.select_folder_btn.clicked.connect(self.on_select_folder)
        self.move_all_right_btn.clicked.connect(self.on_move_all_right)
        self.move_right_btn.clicked.connect(self.on_move_right)
        self.move_left_btn.clicked.connect(self.on_move_left)
        self.move_all_left_btn.clicked.connect(self.on_move_all_left)
        self.filter_edit.textChanged.connect(self.on_filter_changed)
        self.edit_all_metadata_btn.clicked.connect(self.on_edit_all_metadata)
        self.preview_btn.clicked.connect(self.on_generate_preview)
        self.consolidate_btn.clicked.connect(self.on_consolidate)
        self.consolidate_list.model().rowsInserted.connect(self._update_action_buttons)
        self.consolidate_list.model().rowsRemoved.connect(self._update_action_buttons)

    def _update_action_buttons(self):
        """Enable action buttons only if there are items to consolidate."""
        has_items = self.consolidate_list.count() > 0
        self.preview_btn.setEnabled(has_items)
        self.consolidate_btn.setEnabled(has_items)
        self._update_consolidate_list_header()

    def _update_consolidate_list_header(self):
        """Update the 'Files to Consolidate' header with count and tooltip."""
        count = self.consolidate_list.count()
        self.right_label.setText(f"Files to Consolidate (n={count})")

        # Build tooltip with list of animal IDs
        if count > 0:
            animal_ids = []
            for i in range(count):
                item = self.consolidate_list.item(i)
                if item:
                    item_data = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(item_data, dict):
                        animal_id = item_data.get('metadata', {}).get('animal_id', 'Unknown')
                        animal_ids.append(animal_id)
                    else:
                        animal_ids.append(item.text().split(' | ')[0])

            # Limit tooltip to first 20 IDs
            if len(animal_ids) > 20:
                tooltip_text = "Animal IDs:\n" + "\n".join(animal_ids[:20]) + f"\n... and {len(animal_ids) - 20} more"
            else:
                tooltip_text = "Animal IDs:\n" + "\n".join(animal_ids)
            tooltip_text += "\n\nClick to manage list"
            self.right_label.setToolTip(tooltip_text)
        else:
            self.right_label.setToolTip("No files selected")

    def _on_consolidate_label_clicked(self):
        """Open dialog to view and manage files in consolidate list."""
        if self.consolidate_list.count() == 0:
            return

        # Gather item data
        items = []
        for i in range(self.consolidate_list.count()):
            item = self.consolidate_list.item(i)
            if item:
                item_data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(item_data, dict):
                    items.append(item_data)
                else:
                    items.append({'metadata': {'animal_id': item.text()}})

        # Show dialog
        dialog = ConsolidateListDialog(items, self)
        dialog.exec()

        # Process removals
        removed_indices = dialog.get_removed_indices()
        if removed_indices:
            for idx in removed_indices:
                if idx < self.consolidate_list.count():
                    taken = self.consolidate_list.takeItem(idx)
                    if taken:
                        # Move back to file list
                        self.file_list.addItem(taken)

            self.file_list.sortItems()
            self._update_consolidate_list_header()

    def on_select_folder(self):
        """Select folder to scan recursively for CageMetrics NPZ files."""
        settings = QSettings("PhysioMetrics", "CageMetrics")
        last_dir = settings.value("consolidation_last_dir", str(Path.home()))

        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Scan (searches recursively)", last_dir
        )

        if folder:
            settings.setValue("consolidation_last_dir", folder)
            self._scan_dir = Path(folder)
            self._scan_for_npz_files()  # This also updates the folder label

    def _scan_for_npz_files(self):
        """Scan the selected folder recursively for CageMetrics NPZ data files."""
        import json
        import numpy as np
        from core.consolidation_filters import CagemateGenotypeCache

        if not self._scan_dir:
            return

        self.file_list.clear()

        # Recursively find all CageMetrics_*_Data.npz files
        npz_files = list(self._scan_dir.rglob("CageMetrics_*_Data.npz"))

        # First pass: collect all metadata for cagemate cache
        npz_items = []
        for npz_path in sorted(npz_files):
            metadata = self._load_npz_metadata(npz_path)
            npz_items.append({
                'path': str(npz_path),
                'metadata': metadata or {}
            })

        # Build cagemate genotype cache
        self._cagemate_cache = CagemateGenotypeCache()
        self._cagemate_cache.build_from_files(npz_items)

        # Second pass: create list items with cagemate genotype resolved
        for item_info in npz_items:
            npz_path = Path(item_info['path'])
            metadata = item_info['metadata']

            # Resolve cagemate genotype from cache
            cagemate_genotype = self._cagemate_cache.get_cagemate_genotype(metadata)

            if metadata:
                # Build display name from metadata
                animal_id = metadata.get('animal_id', 'Unknown')
                genotype = metadata.get('genotype', 'Unknown')
                sex = metadata.get('sex', 'Unknown')
                cohort = metadata.get('cohort', 'Unknown')
                cage_id = metadata.get('cage_id', 'Unknown')

                # Get cagemate info
                companion = metadata.get('companion', '')
                if isinstance(companion, list):
                    cagemate_id = ', '.join(str(c) for c in companion) if companion else 'Single'
                else:
                    cagemate_id = str(companion) if companion else 'Single'
                cagemate_sex = metadata.get('cagemate_sex', '')

                # Build display: ID | Geno | Sex | Cohort | Cage | Cagemate (Sex)
                display_name = f"{animal_id} | {genotype} | {sex} | {cohort} | Cage {cage_id}"
                if cagemate_id != 'Single':
                    cagemate_display = f"w/ {cagemate_id}"
                    if cagemate_sex:
                        cagemate_display += f" ({cagemate_sex})"
                    display_name += f" | {cagemate_display}"
                else:
                    display_name += " | Single"

                # Add subfolder context
                try:
                    rel_path = npz_path.relative_to(self._scan_dir)
                    if rel_path.parent != Path('.'):
                        display_name += f"  [{rel_path.parent}]"
                except ValueError:
                    pass

                # Build searchable text from all metadata
                search_text = ' '.join([
                    str(v).lower() for v in metadata.values() if v
                ])
            else:
                # Fallback to filename parsing
                name = npz_path.stem.replace("CageMetrics_", "").replace("_Data", "")
                display_name = name
                search_text = name.lower()

            item = QListWidgetItem(display_name)

            # Build detailed tooltip with cagemate info
            cagemate_geno_str = cagemate_genotype if cagemate_genotype else "Unknown"
            cagemate_sex_str = metadata.get('cagemate_sex', 'Unknown') if metadata else 'Unknown'
            if metadata:
                companion_val = metadata.get('companion', 'None')
                if isinstance(companion_val, list):
                    companion_str = ', '.join(str(c) for c in companion_val) if companion_val else 'None'
                else:
                    companion_str = str(companion_val) if companion_val else 'None'

                tooltip_lines = [
                    f"Animal ID: {metadata.get('animal_id', 'Unknown')}",
                    f"Genotype: {metadata.get('genotype', 'Unknown')}",
                    f"Sex: {metadata.get('sex', 'Unknown')}",
                    f"Cohort: {metadata.get('cohort', 'Unknown')}",
                    f"Cage ID: {metadata.get('cage_id', 'Unknown')}",
                    f"Cagemate ID: {companion_str}",
                    f"Cagemate Genotype: {cagemate_geno_str}",
                    f"Cagemate Sex: {cagemate_sex_str}",
                    f"Days: {metadata.get('n_days_analyzed', 'Unknown')}",
                    f"",
                    f"File: {npz_path}"
                ]
                item.setToolTip('\n'.join(tooltip_lines))
            else:
                item.setToolTip(str(npz_path))

            # Store path, searchable metadata, and cagemate genotype
            item.setData(Qt.ItemDataRole.UserRole, {
                'path': str(npz_path),
                'search_text': search_text,
                'metadata': metadata,
                'cagemate_genotype': cagemate_genotype
            })

            self.file_list.addItem(item)

        self.file_list.sortItems()

        # Update folder label with count
        count = self.file_list.count()
        folder_str = str(self._scan_dir)
        if len(folder_str) > 40:
            folder_str = "..." + folder_str[-37:]
        self.folder_label.setText(f"{folder_str}  ({count} files found)")
        self.folder_label.setToolTip(str(self._scan_dir))

        # Update the file list header with count
        self._update_file_list_header(count, count)

        # Enable/disable Edit Metadata button based on file count
        self.edit_all_metadata_btn.setEnabled(count > 0)

        # Discover available metadata values for dynamic filters
        if self._metadata_discovery:
            self._metadata_discovery.scan_files(npz_items, self._cagemate_cache)
            self._populate_filter_checkboxes()

    def _load_npz_metadata(self, npz_path: Path) -> dict:
        """Load metadata from an NPZ file."""
        import json
        import numpy as np

        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'metadata_json' in data:
                return json.loads(str(data['metadata_json']))
        except Exception as e:
            print(f"Warning: Could not load metadata from {npz_path}: {e}")
        return None

    def on_filter_changed(self, text: str):
        """Filter file list based on search text (searches metadata)."""
        search_text = text.strip().lower()

        # Determine search mode
        if ',' in search_text:
            keywords = [k.strip() for k in search_text.split(',') if k.strip()]
            search_mode = 'OR'
        else:
            keywords = [k.strip() for k in search_text.split() if k.strip()]
            search_mode = 'AND'

        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if not item:
                continue

            # Get searchable text from item data
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(item_data, dict):
                searchable = item_data.get('search_text', '')
            else:
                searchable = item.text().lower()

            # Also include display text
            combined = f"{item.text().lower()} {searchable}"

            if not keywords:
                item.setHidden(False)
                continue

            if search_mode == 'AND':
                matches = all(kw in combined for kw in keywords)
            else:
                matches = any(kw in combined for kw in keywords)

            item.setHidden(not matches)

    def _get_item_path(self, item) -> str:
        """Extract file path from item data (handles both dict and string formats)."""
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(item_data, dict):
            return item_data.get('path', '')
        return str(item_data) if item_data else ''

    def _list_has_path(self, list_widget, path: str) -> bool:
        """Check if path already exists in list."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item and self._get_item_path(item) == path:
                return True
        return False

    def _move_items(self, src, dst, rows: list):
        """Move items from source to destination list."""
        moved = 0
        skipped = 0

        # Process in reverse order to avoid index shifting
        for row in sorted(rows, reverse=True):
            item = src.item(row)
            if not item:
                continue

            path = self._get_item_path(item)
            if self._list_has_path(dst, path):
                skipped += 1
                continue

            taken = src.takeItem(row)
            if taken:
                dst.addItem(taken)
                moved += 1

        dst.sortItems()
        return moved, skipped

    def on_move_right(self):
        """Move selected items to consolidate list."""
        rows = [self.file_list.row(it) for it in self.file_list.selectedItems()]
        moved, skipped = self._move_items(self.file_list, self.consolidate_list, rows)

    def on_move_all_right(self):
        """Move all visible items to consolidate list."""
        rows = [i for i in range(self.file_list.count())
                if not self.file_list.item(i).isHidden()]
        moved, skipped = self._move_items(self.file_list, self.consolidate_list, rows)

    def on_move_left(self):
        """Move selected items back to file list."""
        rows = [self.consolidate_list.row(it) for it in self.consolidate_list.selectedItems()]
        moved, skipped = self._move_items(self.consolidate_list, self.file_list, rows)

    def on_move_all_left(self):
        """Move all items back to file list."""
        rows = list(range(self.consolidate_list.count()))
        moved, skipped = self._move_items(self.consolidate_list, self.file_list, rows)

    def on_consolidate(self):
        """Consolidate selected NPZ files into a single dataset."""
        if self.consolidate_list.count() == 0:
            return

        # Gather all NPZ file paths
        npz_paths = []
        for i in range(self.consolidate_list.count()):
            item = self.consolidate_list.item(i)
            if item:
                npz_paths.append(self._get_item_path(item))

        # Generate auto-suggested filename based on filter criteria
        from core.consolidator import generate_consolidated_filename
        suggested_name = generate_consolidated_filename(
            filter_criteria=self._filter_criteria,
            n_animals=len(npz_paths)
        )

        # Determine default save directory - create 'consolidated' subfolder
        if self._scan_dir:
            default_dir = Path(self._scan_dir) / "consolidated"
        else:
            settings = QSettings("PhysioMetrics", "CageMetrics")
            default_dir = Path(settings.value("consolidation_save_dir", str(Path.home()))) / "consolidated"

        # Create the consolidated subfolder if it doesn't exist
        default_dir.mkdir(parents=True, exist_ok=True)

        # Ask for output file (user can modify the suggested name)
        # Note: .npz, .xlsx, and .pdf will all be created with the same base name
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Consolidated Data (will create .npz, .xlsx, and .pdf)",
            str(default_dir / f"{suggested_name}.npz"),
            "NPZ Files (*.npz);;All Files (*)"
        )

        if not save_path:
            return

        # Save the parent directory for next time
        settings = QSettings("PhysioMetrics", "CageMetrics")
        settings.setValue("consolidation_save_dir", str(Path(save_path).parent))

        try:
            from core.consolidator import Consolidator
            consolidator = Consolidator()
            result = consolidator.consolidate(
                npz_paths, save_path,
                filter_criteria=self._filter_criteria,
                save_npz=True,
                save_pdf=True
            )

            # Build output message
            output_files = result.get('output_paths', [save_path])
            files_str = '\n'.join(f"  • {Path(f).name}" for f in output_files)

            QMessageBox.information(
                self, "Consolidation Complete",
                f"Successfully consolidated {len(npz_paths)} animals.\n\n"
                f"Output files saved to:\n{Path(save_path).parent}\n\n"
                f"Files created:\n{files_str}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Consolidation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # === Filter Panel Methods ===

    def _setup_filter_panel(self, parent_layout):
        """Set up the filters panel with dynamic filter discovery."""
        self.filter_group = QGroupBox("Filters")
        self.filter_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        self.filter_main_layout = QVBoxLayout(self.filter_group)
        self.filter_main_layout.setContentsMargins(10, 15, 10, 10)
        self.filter_main_layout.setSpacing(5)

        # === ID Filters Section (Animal ID and Cagemate ID side by side) ===
        id_filters_container = QWidget()
        id_filters_main_layout = QVBoxLayout(id_filters_container)
        id_filters_main_layout.setContentsMargins(0, 0, 0, 10)
        id_filters_main_layout.setSpacing(5)

        # Horizontal layout for the two filter boxes
        id_filters_row = QHBoxLayout()
        id_filters_row.setSpacing(15)

        # === Left: Animal ID Filter ===
        animal_id_widget = QWidget()
        animal_id_layout = QVBoxLayout(animal_id_widget)
        animal_id_layout.setContentsMargins(0, 0, 0, 0)
        animal_id_layout.setSpacing(3)

        # Header with mode selector and help button
        id_header_layout = QHBoxLayout()
        id_header_layout.setSpacing(5)
        id_label = QLabel("Animal ID Filter:")
        id_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        id_header_layout.addWidget(id_label)

        # Help button
        id_help_btn = QPushButton("?")
        id_help_btn.setFixedSize(18, 18)
        id_help_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #666666;
                border-radius: 9px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        id_help_btn.setToolTip("Click for filter syntax help")
        id_help_btn.clicked.connect(self._show_filter_help)
        id_header_layout.addWidget(id_help_btn)

        self.animal_id_mode = QComboBox()
        self.animal_id_mode.addItems(["OR (any match)", "AND (all match)"])
        self.animal_id_mode.setFixedWidth(110)
        self.animal_id_mode.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 5px;
                font-size: 9pt;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border: none; }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
        """)
        self.animal_id_mode.setToolTip("OR: show if ANY term matches\nAND: show only if ALL terms match")
        self.animal_id_mode.currentIndexChanged.connect(self._on_animal_id_input_changed)
        id_header_layout.addWidget(self.animal_id_mode)
        id_header_layout.addStretch()
        animal_id_layout.addLayout(id_header_layout)

        # Text input
        self.animal_id_input = QPlainTextEdit()
        self.animal_id_input.setMaximumHeight(55)
        self.animal_id_input.setPlaceholderText("Comma or newline separated terms...")
        self.animal_id_input.setToolTip(
            "Filter by Animal ID - supports partial matching, wildcards, and anchors\n\n"
            "Syntax:\n"
            "  • _wt         → substring: matches 'mouse_wt_01', 'cage_wt'\n"
            "  • _wt$        → ends with: matches 'mouse_wt' but NOT 'mouse_wt_01'\n"
            "  • ^mouse      → starts with: matches 'mouse_01' but NOT 'test_mouse'\n"
            "  • \"mouse_wt\"  → exact: matches ONLY 'mouse_wt'\n"
            "  • _*wt        → wildcard: matches '_anywt', '_xyzwt'\n"
            "  • *_wt$       → wildcard + ends with: anything ending in '_wt'\n\n"
            "Multiple terms: separate with commas or newlines\n"
            "  • _wt, _ko    → OR mode: matches _wt OR _ko\n\n"
            "Use dropdown to switch between OR (any) and AND (all) mode."
        )
        self.animal_id_input.setStyleSheet("""
            QPlainTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px;
                font-size: 10pt;
            }
            QPlainTextEdit:focus { border-color: #0078d4; }
        """)
        self.animal_id_input.textChanged.connect(self._on_animal_id_input_changed)
        animal_id_layout.addWidget(self.animal_id_input)

        # Clear button
        self.clear_id_filter_btn = QPushButton("Clear")
        self.clear_id_filter_btn.setFixedWidth(60)
        self.clear_id_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QPushButton:hover { background-color: #505050; }
        """)
        self.clear_id_filter_btn.clicked.connect(self._clear_animal_id_filter)
        animal_id_layout.addWidget(self.clear_id_filter_btn)

        # Feedback label
        self.id_filter_feedback = QLabel("")
        self.id_filter_feedback.setStyleSheet("color: #f0a000; font-size: 8pt;")
        self.id_filter_feedback.setWordWrap(True)
        self.id_filter_feedback.setVisible(False)
        animal_id_layout.addWidget(self.id_filter_feedback)

        id_filters_row.addWidget(animal_id_widget, stretch=1)

        # Debounce timer for Animal ID filter (300ms delay)
        self._animal_id_debounce_timer = QTimer()
        self._animal_id_debounce_timer.setSingleShot(True)
        self._animal_id_debounce_timer.setInterval(300)
        self._animal_id_debounce_timer.timeout.connect(self._apply_animal_id_filter)

        # === Right: Cagemate ID Filter ===
        cagemate_id_widget = QWidget()
        cagemate_id_layout = QVBoxLayout(cagemate_id_widget)
        cagemate_id_layout.setContentsMargins(0, 0, 0, 0)
        cagemate_id_layout.setSpacing(3)

        # Header with mode selector
        cagemate_header_layout = QHBoxLayout()
        cagemate_header_layout.setSpacing(5)
        cagemate_label = QLabel("Cagemate ID Filter:")
        cagemate_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        cagemate_header_layout.addWidget(cagemate_label)

        self.cagemate_id_mode = QComboBox()
        self.cagemate_id_mode.addItems(["OR (any match)", "AND (all match)"])
        self.cagemate_id_mode.setFixedWidth(110)
        self.cagemate_id_mode.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 5px;
                font-size: 9pt;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border: none; }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
        """)
        self.cagemate_id_mode.setToolTip("OR: show if ANY term matches\nAND: show only if ALL terms match")
        self.cagemate_id_mode.currentIndexChanged.connect(self._on_cagemate_id_input_changed)
        cagemate_header_layout.addWidget(self.cagemate_id_mode)
        cagemate_header_layout.addStretch()
        cagemate_id_layout.addLayout(cagemate_header_layout)

        # Text input
        self.cagemate_id_input = QPlainTextEdit()
        self.cagemate_id_input.setMaximumHeight(55)
        self.cagemate_id_input.setPlaceholderText("Comma or newline separated terms...")
        self.cagemate_id_input.setToolTip(
            "Filter by Cagemate ID - supports partial matching, wildcards, and anchors\n\n"
            "Syntax:\n"
            "  • _wt         → substring: matches cagemates containing '_wt'\n"
            "  • _wt$        → ends with: matches 'mouse_wt' but NOT 'mouse_wt_01'\n"
            "  • ^mouse      → starts with: matches 'mouse_01' but NOT 'test_mouse'\n"
            "  • \"mouse_wt\"  → exact: matches ONLY 'mouse_wt'\n"
            "  • _*wt        → wildcard: matches '_anywt', '_xyzwt'\n"
            "  • *_wt$       → wildcard + ends with: anything ending in '_wt'\n\n"
            "Multiple terms: separate with commas or newlines\n"
            "  • _wt, _ko    → OR mode: matches _wt OR _ko\n\n"
            "Use dropdown to switch between OR (any) and AND (all) mode."
        )
        self.cagemate_id_input.setStyleSheet("""
            QPlainTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px;
                font-size: 10pt;
            }
            QPlainTextEdit:focus { border-color: #0078d4; }
        """)
        self.cagemate_id_input.textChanged.connect(self._on_cagemate_id_input_changed)
        cagemate_id_layout.addWidget(self.cagemate_id_input)

        # Clear button
        self.clear_cagemate_filter_btn = QPushButton("Clear")
        self.clear_cagemate_filter_btn.setFixedWidth(60)
        self.clear_cagemate_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QPushButton:hover { background-color: #505050; }
        """)
        self.clear_cagemate_filter_btn.clicked.connect(self._clear_cagemate_id_filter)
        cagemate_id_layout.addWidget(self.clear_cagemate_filter_btn)

        # Feedback label
        self.cagemate_filter_feedback = QLabel("")
        self.cagemate_filter_feedback.setStyleSheet("color: #f0a000; font-size: 8pt;")
        self.cagemate_filter_feedback.setWordWrap(True)
        self.cagemate_filter_feedback.setVisible(False)
        cagemate_id_layout.addWidget(self.cagemate_filter_feedback)

        id_filters_row.addWidget(cagemate_id_widget, stretch=1)

        # Debounce timer for Cagemate ID filter (300ms delay)
        self._cagemate_id_debounce_timer = QTimer()
        self._cagemate_id_debounce_timer.setSingleShot(True)
        self._cagemate_id_debounce_timer.setInterval(300)
        self._cagemate_id_debounce_timer.timeout.connect(self._apply_cagemate_id_filter)

        id_filters_main_layout.addLayout(id_filters_row)
        self.filter_main_layout.addWidget(id_filters_container)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #3d3d3d;")
        self.filter_main_layout.addWidget(separator)

        # === Dynamic Filters Container (populated after scan) ===
        self.filters_container = QWidget()
        self.filters_layout = QVBoxLayout(self.filters_container)
        self.filters_layout.setContentsMargins(0, 0, 0, 0)
        self.filters_layout.setSpacing(5)
        self.filter_main_layout.addWidget(self.filters_container)

        # Placeholder message (shown before folder selection)
        self.filter_placeholder = QLabel("Select a folder to see available filters")
        self.filter_placeholder.setStyleSheet("color: #888888; font-style: italic; padding: 10px;")
        self.filter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filters_layout.addWidget(self.filter_placeholder)

        # Clear and Reset buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.clear_filter_btn = QPushButton("Clear Filters")
        self.clear_filter_btn.setFixedWidth(100)
        self.clear_filter_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_filter_btn)

        self.reset_all_btn = QPushButton("Reset All")
        self.reset_all_btn.setFixedWidth(80)
        self.reset_all_btn.setToolTip("Clear all filters AND empty the consolidate list")
        self.reset_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b4513;
                color: #ffffff;
                border: 1px solid #a0522d;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover { background-color: #a0522d; }
        """)
        btn_layout.addWidget(self.reset_all_btn)

        self.filter_main_layout.addLayout(btn_layout)

        # Connect filter signals
        self.clear_filter_btn.clicked.connect(self._clear_advanced_filters)
        self.reset_all_btn.clicked.connect(self._reset_all)

        parent_layout.addWidget(self.filter_group)

    def _populate_filter_checkboxes(self):
        """Populate filter checkboxes based on discovered metadata values."""
        from PyQt6.QtWidgets import QCheckBox
        from core.consolidation_filters import MetadataDiscovery

        if not self._metadata_discovery:
            return

        # Clear existing checkboxes
        self._filter_checkboxes.clear()

        # Clear filters layout (except placeholder)
        while self.filters_layout.count() > 0:
            child = self.filters_layout.takeAt(0)
            if child.widget() and child.widget() != self.filter_placeholder:
                child.widget().deleteLater()
            elif child.widget() == self.filter_placeholder:
                child.widget().setVisible(False)

        has_filters = False

        # Add primary filter fields first
        for field_name in self._metadata_discovery.get_primary_fields():
            values = self._metadata_discovery.get_values(field_name)
            if len(values) > 1:  # Only show if multiple values exist
                self._add_filter_row(
                    self.filters_layout,
                    field_name,
                    self._metadata_discovery.get_field_label(field_name),
                    values
                )
                has_filters = True

        # Add secondary filter fields
        for field_name in self._metadata_discovery.get_secondary_fields():
            values = self._metadata_discovery.get_values(field_name)
            if len(values) > 1:  # Only show if multiple values exist
                self._add_filter_row(
                    self.filters_layout,
                    field_name,
                    self._metadata_discovery.get_field_label(field_name),
                    values
                )
                has_filters = True

        # Show/hide placeholder
        self.filter_placeholder.setVisible(not has_filters)

        # Enable clear button if we have filters
        self.clear_filter_btn.setEnabled(has_filters)

    def _add_filter_row(self, layout, field_name: str, label: str, values: list):
        """Add a filter row with checkboxes for each value."""
        from PyQt6.QtWidgets import QCheckBox

        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        # Label
        field_label = QLabel(f"{label}:")
        field_label.setFixedWidth(130)
        field_label.setStyleSheet("color: #cccccc;")
        row_layout.addWidget(field_label)

        # Checkboxes for each value
        self._filter_checkboxes[field_name] = {}
        for value in values:
            cb = QCheckBox(str(value))
            cb.setStyleSheet("color: #ffffff;")
            # Auto-apply filters when checkbox is toggled
            cb.toggled.connect(self._on_filter_checkbox_changed)
            self._filter_checkboxes[field_name][value] = cb
            row_layout.addWidget(cb)

        row_layout.addStretch()
        layout.addLayout(row_layout)

    def _on_filter_checkbox_changed(self):
        """Called when any filter checkbox is toggled - auto-apply filters."""
        self._apply_advanced_filters()

    def _init_filter_criteria(self):
        """Initialize filter criteria and metadata discovery objects."""
        from core.consolidation_filters import FilterCriteria, MetadataDiscovery
        self._filter_criteria = FilterCriteria()
        self._metadata_discovery = MetadataDiscovery()

    def _apply_advanced_filters(self):
        """Apply advanced filter checkboxes to file list."""
        from core.consolidation_filters import FilterCriteria

        # Build filter criteria from dynamic checkboxes
        criteria = FilterCriteria()

        for field_name, value_checkboxes in self._filter_checkboxes.items():
            selected_values = set()
            for value, cb in value_checkboxes.items():
                if cb.isChecked():
                    selected_values.add(value)
            if selected_values:
                criteria.set_filter(field_name, selected_values)

        self._filter_criteria = criteria

        # Apply to file list
        self._apply_filters_to_list()

    def _clear_advanced_filters(self):
        """Clear all advanced filter checkboxes."""
        # Block signals temporarily to avoid multiple filter applications
        for field_name, value_checkboxes in self._filter_checkboxes.items():
            for value, cb in value_checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)

        # Clear criteria
        if self._filter_criteria:
            self._filter_criteria.clear()

        # Also clear Animal ID filter (without reapplying filters yet)
        self.animal_id_input.blockSignals(True)
        self.animal_id_input.clear()
        self.animal_id_input.blockSignals(False)
        self._animal_id_debounce_timer.stop()
        self.id_filter_feedback.setVisible(False)
        self._animal_id_filter_terms = []

        # Also clear Cagemate ID filter
        self.cagemate_id_input.blockSignals(True)
        self.cagemate_id_input.clear()
        self.cagemate_id_input.blockSignals(False)
        self._cagemate_id_debounce_timer.stop()
        self.cagemate_filter_feedback.setVisible(False)
        self._cagemate_id_filter_terms = []

        # Apply filters once (will show all since all filters are cleared)
        self._apply_filters_to_list()

    def _reset_all(self):
        """Reset everything: clear all filters AND move all items from consolidate list back."""
        # First, move all items from consolidate list back to file list
        self.on_move_all_left()

        # Then clear all filters (this will also reapply and show all files)
        self._clear_advanced_filters()

        # Update UI
        self._update_consolidate_list_header()

    def _on_animal_id_input_changed(self):
        """Called when Animal ID input text changes - restarts debounce timer."""
        # Restart the debounce timer (will fire after 300ms of no typing)
        self._animal_id_debounce_timer.start()

    def _apply_animal_id_filter(self):
        """Apply Animal ID filter from text input."""
        input_text = self.animal_id_input.toPlainText().strip()

        if not input_text:
            # Clear filter terms and hide feedback (but don't clear the text input)
            self._animal_id_filter_terms = []
            self.id_filter_feedback.setVisible(False)
            self._apply_filters_to_list()
            return

        # Parse input - support comma, newline, and space-separated IDs
        # First replace newlines with commas, then split
        normalized = input_text.replace('\n', ',').replace('\r', ',')
        search_terms = [term.strip() for term in normalized.split(',') if term.strip()]

        if not search_terms:
            # Clear filter terms and hide feedback (but don't clear the text input)
            self._animal_id_filter_terms = []
            self.id_filter_feedback.setVisible(False)
            self._apply_filters_to_list()
            return

        # Store the current ID filter terms for use in combined filtering
        self._animal_id_filter_terms = search_terms

        # Track which search terms were found
        found_terms = set()
        not_found_terms = []

        # Get all animal IDs from the file list
        all_animal_ids = {}
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item:
                item_data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(item_data, dict):
                    animal_id = item_data.get('metadata', {}).get('animal_id', '')
                    all_animal_ids[i] = animal_id.lower()

        # Determine matches - partial matching (search term is contained in animal ID)
        visible_count = 0
        total_count = self.file_list.count()

        for i in range(total_count):
            item = self.file_list.item(i)
            if not item:
                continue

            item_data = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(item_data, dict):
                continue

            animal_id = item_data.get('metadata', {}).get('animal_id', '').lower()

            # Check search terms based on mode (OR = index 0, AND = index 1)
            use_and_mode = self.animal_id_mode.currentIndex() == 1

            if use_and_mode:
                # AND mode: ALL terms must match
                matches = True
                for term in search_terms:
                    if self._match_wildcard(term, animal_id):
                        found_terms.add(term)
                    else:
                        matches = False
                        # Don't break - we still want to track which terms were found
            else:
                # OR mode: ANY term matches
                matches = False
                for term in search_terms:
                    if self._match_wildcard(term, animal_id):
                        matches = True
                        found_terms.add(term)
                        break

            # Also check if item is hidden by other filters
            metadata = item_data.get('metadata', {})
            cagemate_geno = item_data.get('cagemate_genotype')

            if self._filter_criteria and not self._filter_criteria.is_empty():
                if not self._filter_criteria.matches(metadata, cagemate_geno):
                    matches = False

            # Check cagemate ID filter (respecting AND/OR mode)
            if matches and hasattr(self, '_cagemate_id_filter_terms') and self._cagemate_id_filter_terms:
                companion = metadata.get('companion', '')
                if isinstance(companion, list):
                    cagemate_ids = [str(c).lower() for c in companion]
                else:
                    cagemate_ids = [str(companion).lower()] if companion else []

                cagemate_and_mode = hasattr(self, 'cagemate_id_mode') and self.cagemate_id_mode.currentIndex() == 1

                if cagemate_and_mode:
                    # AND mode: ALL terms must match
                    cagemate_matches = True
                    for term in self._cagemate_id_filter_terms:
                        term_found = any(self._match_wildcard(term, cid) for cid in cagemate_ids)
                        if not term_found:
                            cagemate_matches = False
                            break
                else:
                    # OR mode: ANY term matches
                    cagemate_matches = False
                    for term in self._cagemate_id_filter_terms:
                        if any(self._match_wildcard(term, cid) for cid in cagemate_ids):
                            cagemate_matches = True
                            break
                if not cagemate_matches:
                    matches = False

            item.setHidden(not matches)
            if matches:
                visible_count += 1

        # Identify which terms weren't found
        for term in search_terms:
            if term not in found_terms:
                # Check if term matches any animal ID using wildcard
                term_found = any(self._match_wildcard(term, aid) for aid in all_animal_ids.values())
                if not term_found:
                    not_found_terms.append(term)

        # Show feedback
        if not_found_terms:
            if len(not_found_terms) <= 5:
                feedback_text = f"Not found: {', '.join(not_found_terms)}"
            else:
                feedback_text = f"Not found: {', '.join(not_found_terms[:5])} ... and {len(not_found_terms) - 5} more"
            self.id_filter_feedback.setText(feedback_text)
            self.id_filter_feedback.setVisible(True)
        else:
            self.id_filter_feedback.setVisible(False)

        # Update header
        self._update_file_list_header(total_count, visible_count)

    def _clear_animal_id_filter(self):
        """Clear the Animal ID filter."""
        # Block signals to avoid triggering textChanged during clear
        self.animal_id_input.blockSignals(True)
        self.animal_id_input.clear()
        self.animal_id_input.blockSignals(False)

        # Stop any pending debounce timer
        self._animal_id_debounce_timer.stop()

        self.id_filter_feedback.setVisible(False)
        self._animal_id_filter_terms = []

        # Reapply other filters (or show all if no other filters)
        self._apply_filters_to_list()

    def _on_cagemate_id_input_changed(self):
        """Called when Cagemate ID input text changes - restarts debounce timer."""
        self._cagemate_id_debounce_timer.start()

    def _apply_cagemate_id_filter(self):
        """Apply Cagemate ID filter from text input."""
        input_text = self.cagemate_id_input.toPlainText().strip()

        if not input_text:
            # Clear filter terms and hide feedback
            self._cagemate_id_filter_terms = []
            self.cagemate_filter_feedback.setVisible(False)
            self._apply_filters_to_list()
            return

        # Parse input - support comma, newline, and space-separated IDs
        normalized = input_text.replace('\n', ',').replace('\r', ',')
        search_terms = [term.strip() for term in normalized.split(',') if term.strip()]

        if not search_terms:
            self._cagemate_id_filter_terms = []
            self.cagemate_filter_feedback.setVisible(False)
            self._apply_filters_to_list()
            return

        # Store the current cagemate ID filter terms
        self._cagemate_id_filter_terms = search_terms

        # Track which search terms were found
        found_terms = set()
        not_found_terms = []

        # Get all cagemate IDs from the file list
        all_cagemate_ids = {}
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item:
                item_data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(item_data, dict):
                    companion = item_data.get('metadata', {}).get('companion', '')
                    # Companion can be a list or string
                    if isinstance(companion, list):
                        cagemate_ids = [str(c).lower() for c in companion]
                    else:
                        cagemate_ids = [str(companion).lower()] if companion else []
                    all_cagemate_ids[i] = cagemate_ids

        # Determine matches
        visible_count = 0
        total_count = self.file_list.count()

        for i in range(total_count):
            item = self.file_list.item(i)
            if not item:
                continue

            item_data = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(item_data, dict):
                continue

            companion = item_data.get('metadata', {}).get('companion', '')
            if isinstance(companion, list):
                cagemate_ids = [str(c).lower() for c in companion]
            else:
                cagemate_ids = [str(companion).lower()] if companion else []

            # Check search terms based on mode (OR = index 0, AND = index 1)
            use_and_mode = self.cagemate_id_mode.currentIndex() == 1

            if use_and_mode:
                # AND mode: ALL terms must match
                matches = True
                for term in search_terms:
                    term_found = False
                    for cagemate_id in cagemate_ids:
                        if self._match_wildcard(term, cagemate_id):
                            term_found = True
                            found_terms.add(term)
                            break
                    if not term_found:
                        matches = False
                        # Don't break - still track found terms
            else:
                # OR mode: ANY term matches
                matches = False
                for term in search_terms:
                    for cagemate_id in cagemate_ids:
                        if self._match_wildcard(term, cagemate_id):
                            matches = True
                            found_terms.add(term)
                            break
                    if matches:
                        break

            # Also check other filters (checkbox filters and animal ID filter)
            metadata = item_data.get('metadata', {})
            cagemate_geno = item_data.get('cagemate_genotype')

            if self._filter_criteria and not self._filter_criteria.is_empty():
                if not self._filter_criteria.matches(metadata, cagemate_geno):
                    matches = False

            # Check animal ID filter (respecting AND/OR mode)
            if matches and hasattr(self, '_animal_id_filter_terms') and self._animal_id_filter_terms:
                animal_id = metadata.get('animal_id', '').lower()
                animal_and_mode = hasattr(self, 'animal_id_mode') and self.animal_id_mode.currentIndex() == 1

                if animal_and_mode:
                    # AND mode: ALL terms must match
                    animal_id_matches = True
                    for term in self._animal_id_filter_terms:
                        if not self._match_wildcard(term, animal_id):
                            animal_id_matches = False
                            break
                else:
                    # OR mode: ANY term matches
                    animal_id_matches = False
                    for term in self._animal_id_filter_terms:
                        if self._match_wildcard(term, animal_id):
                            animal_id_matches = True
                            break
                if not animal_id_matches:
                    matches = False

            item.setHidden(not matches)
            if matches:
                visible_count += 1

        # Identify which terms weren't found
        for term in search_terms:
            if term not in found_terms:
                term_found = False
                for cagemate_ids in all_cagemate_ids.values():
                    for cagemate_id in cagemate_ids:
                        if self._match_wildcard(term, cagemate_id):
                            term_found = True
                            break
                    if term_found:
                        break
                if not term_found:
                    not_found_terms.append(term)

        # Show feedback
        if not_found_terms:
            if len(not_found_terms) <= 5:
                feedback_text = f"Not found: {', '.join(not_found_terms)}"
            else:
                feedback_text = f"Not found: {', '.join(not_found_terms[:5])} ... and {len(not_found_terms) - 5} more"
            self.cagemate_filter_feedback.setText(feedback_text)
            self.cagemate_filter_feedback.setVisible(True)
        else:
            self.cagemate_filter_feedback.setVisible(False)

        # Update header
        self._update_file_list_header(total_count, visible_count)

    def _clear_cagemate_id_filter(self):
        """Clear the Cagemate ID filter."""
        self.cagemate_id_input.blockSignals(True)
        self.cagemate_id_input.clear()
        self.cagemate_id_input.blockSignals(False)

        self._cagemate_id_debounce_timer.stop()
        self.cagemate_filter_feedback.setVisible(False)
        self._cagemate_id_filter_terms = []

        # Reapply other filters
        self._apply_filters_to_list()

    def _apply_filters_to_list(self):
        """Apply current filter criteria to file list visibility."""
        total_count = self.file_list.count()
        visible_count = 0

        has_checkbox_filters = self._filter_criteria and not self._filter_criteria.is_empty()
        has_animal_id_filters = bool(getattr(self, '_animal_id_filter_terms', []))
        has_cagemate_id_filters = bool(getattr(self, '_cagemate_id_filter_terms', []))

        for i in range(total_count):
            item = self.file_list.item(i)
            if not item:
                continue

            item_data = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(item_data, dict):
                item.setHidden(False)
                visible_count += 1
                continue

            matches = True
            metadata = item_data.get('metadata', {})

            # Apply checkbox filters
            if has_checkbox_filters:
                cagemate_geno = item_data.get('cagemate_genotype')
                if not self._filter_criteria.matches(metadata, cagemate_geno):
                    matches = False

            # Apply Animal ID filter (respecting AND/OR mode, with wildcard support)
            if matches and has_animal_id_filters:
                animal_id = metadata.get('animal_id', '').lower()
                use_and_mode = hasattr(self, 'animal_id_mode') and self.animal_id_mode.currentIndex() == 1

                if use_and_mode:
                    # AND mode: ALL terms must match
                    id_matches = True
                    for term in self._animal_id_filter_terms:
                        if not self._match_wildcard(term, animal_id):
                            id_matches = False
                            break
                else:
                    # OR mode: ANY term matches
                    id_matches = False
                    for term in self._animal_id_filter_terms:
                        if self._match_wildcard(term, animal_id):
                            id_matches = True
                            break
                if not id_matches:
                    matches = False

            # Apply Cagemate ID filter (respecting AND/OR mode, with wildcard support)
            if matches and has_cagemate_id_filters:
                companion = metadata.get('companion', '')
                if isinstance(companion, list):
                    cagemate_ids = [str(c).lower() for c in companion]
                else:
                    cagemate_ids = [str(companion).lower()] if companion else []

                use_and_mode = hasattr(self, 'cagemate_id_mode') and self.cagemate_id_mode.currentIndex() == 1

                if use_and_mode:
                    # AND mode: ALL terms must match
                    cagemate_matches = True
                    for term in self._cagemate_id_filter_terms:
                        term_found = any(self._match_wildcard(term, cid) for cid in cagemate_ids)
                        if not term_found:
                            cagemate_matches = False
                            break
                else:
                    # OR mode: ANY term matches
                    cagemate_matches = False
                    for term in self._cagemate_id_filter_terms:
                        if any(self._match_wildcard(term, cid) for cid in cagemate_ids):
                            cagemate_matches = True
                            break
                if not cagemate_matches:
                    matches = False

            item.setHidden(not matches)
            if matches:
                visible_count += 1

        # Update header to show counts
        self._update_file_list_header(total_count, visible_count)

    def _update_file_list_header(self, total_count: int, visible_count: int):
        """Update the NPZ Files Detected header with count information."""
        if total_count == 0:
            self.left_label.setText("NPZ Files Detected")
        elif visible_count == total_count:
            self.left_label.setText(f"NPZ Files Detected (n={total_count})")
        else:
            self.left_label.setText(f"NPZ Files Detected (n={total_count}, showing {visible_count})")

    # === Preview Panel Methods ===

    def _setup_preview_panel_widget(self):
        """Set up the preview panel widget (for use in splitter)."""
        # Create the preview widget container
        self.preview_widget = QWidget()
        self.preview_widget.setStyleSheet("background-color: #2d2d2d;")
        preview_layout = QVBoxLayout(self.preview_widget)
        preview_layout.setContentsMargins(0, 5, 0, 5)
        preview_layout.setSpacing(5)

        # Preview section header
        preview_header = QLabel("Preview Figures")
        preview_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        preview_layout.addWidget(preview_header)

        # Preview container for figures (no separate scroll - main_scroll handles it)
        self.preview_container = QWidget()
        self.preview_container.setStyleSheet("background-color: #2d2d2d;")
        self.preview_container_layout = QVBoxLayout(self.preview_container)
        self.preview_container_layout.setContentsMargins(0, 5, 0, 5)
        self.preview_container_layout.setSpacing(15)

        # Placeholder content
        placeholder = QLabel("Click 'Generate Preview' to see consolidated figures")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #8d8d8d; font-size: 14px; padding: 50px;")
        self.preview_container_layout.addWidget(placeholder)

        preview_layout.addWidget(self.preview_container, stretch=1)

    def on_edit_all_metadata(self):
        """Open the metadata editor for all detected NPZ files."""
        # Collect file paths from file_list (all detected files)
        npz_paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item:
                path = self._get_item_path(item)
                if path:
                    npz_paths.append(path)

        if not npz_paths:
            QMessageBox.information(self, "No Files", "No NPZ files to edit.")
            return

        # Open the NPZ metadata editor dialog
        from dialogs.npz_metadata_editor import NpzMetadataEditorDialog
        dialog = NpzMetadataEditorDialog(npz_paths, self)
        result = dialog.exec()

        # If files were modified, rescan to update displayed metadata
        if result == QDialog.DialogCode.Accepted:
            # Refresh the display by rescanning the current folder
            if self._scan_dir:
                self._scan_for_npz_files()

    def on_generate_preview(self):
        """Generate preview figures for selected experiments."""
        if self.consolidate_list.count() == 0:
            return

        try:
            # Load full NPZ data for all selected files
            experiments = []
            for i in range(self.consolidate_list.count()):
                item = self.consolidate_list.item(i)
                if item:
                    path = self._get_item_path(item)
                    exp_data = self._load_npz_full(path)
                    if exp_data:
                        experiments.append(exp_data)

            if not experiments:
                QMessageBox.warning(self, "Error", "No valid experiments to preview")
                return

            # Generate figures
            from core.consolidation_figure_generator import ConsolidationFigureGenerator
            generator = ConsolidationFigureGenerator()

            filter_desc = self._filter_criteria.to_description() if self._filter_criteria else "No filters"
            self._preview_figures = generator.generate_all_pages(experiments, filter_desc)

            # Display in preview panel
            self._display_preview_figures()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preview generation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _load_npz_full(self, npz_path: str) -> dict:
        """Load complete NPZ data including metrics."""
        import json
        import numpy as np

        try:
            data = np.load(npz_path, allow_pickle=True)

            result = {
                'path': npz_path,
                'metadata': {},
                'quality': {},
                'metrics': {},
                'metric_names': []
            }

            # Load metadata
            if 'metadata_json' in data:
                result['metadata'] = json.loads(str(data['metadata_json']))

            # Load quality
            if 'quality_json' in data:
                result['quality'] = json.loads(str(data['quality_json']))

            # Load metric names
            if 'metric_names' in data:
                result['metric_names'] = list(data['metric_names'])

            # Load per-metric data
            for metric_name in result['metric_names']:
                key_base = metric_name.replace(' ', '_').replace('/', '_')\
                                     .replace('(', '').replace(')', '')\
                                     .replace('%', 'pct')

                metric_data = {}

                cta_key = f'{key_base}_cta'
                if cta_key in data:
                    metric_data['cta'] = np.array(data[cta_key])

                if f'{key_base}_cta_sem' in data:
                    metric_data['cta_sem'] = np.array(data[f'{key_base}_cta_sem'])
                if f'{key_base}_daily' in data:
                    metric_data['daily_data'] = np.array(data[f'{key_base}_daily'])
                if f'{key_base}_dark_mean' in data:
                    metric_data['dark_mean'] = float(data[f'{key_base}_dark_mean'])
                if f'{key_base}_light_mean' in data:
                    metric_data['light_mean'] = float(data[f'{key_base}_light_mean'])
                if f'{key_base}_overall_mean' in data:
                    metric_data['overall_mean'] = float(data[f'{key_base}_overall_mean'])

                result['metrics'][metric_name] = metric_data

            # Load sleep analysis data if present
            if 'sleep_stats_json' in data:
                from core.sleep_analysis import SleepBout

                sleep_analysis = {
                    'parameters': {
                        'threshold': float(data.get('sleep_threshold', 0.5)),
                        'bin_width': float(data.get('sleep_bin_width', 5.0)),
                    },
                    'n_days': int(data.get('sleep_n_days', 0)),
                }

                # Load stats from JSON
                stats = json.loads(str(data['sleep_stats_json']))
                sleep_analysis['light_stats'] = stats.get('light', {})
                sleep_analysis['dark_stats'] = stats.get('dark', {})
                sleep_analysis['total_stats'] = stats.get('total', {})
                sleep_analysis['per_day_stats'] = stats.get('per_day', [])

                # Reconstruct bouts from structured array
                if 'sleep_bouts' in data:
                    bout_array = data['sleep_bouts']
                    bouts = []
                    for row in bout_array:
                        bouts.append(SleepBout(
                            day=int(row['day']),
                            bout_num=int(row['bout_num']),
                            start_minute=int(row['start_minute']),
                            end_minute=int(row['end_minute']),
                            duration=float(row['duration']),
                            phase='light' if row['phase'] == 0 else 'dark'
                        ))
                    sleep_analysis['bouts'] = bouts
                else:
                    sleep_analysis['bouts'] = []

                # Load histograms
                if 'sleep_hist_light_edges' in data:
                    sleep_analysis['histogram_light'] = (
                        np.array(data['sleep_hist_light_edges']),
                        np.array(data['sleep_hist_light_counts'])
                    )
                if 'sleep_hist_dark_edges' in data:
                    sleep_analysis['histogram_dark'] = (
                        np.array(data['sleep_hist_dark_edges']),
                        np.array(data['sleep_hist_dark_counts'])
                    )

                result['sleep_analysis'] = sleep_analysis

            return result

        except Exception as e:
            print(f"Error loading NPZ {npz_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _display_preview_figures(self):
        """Display generated preview figures in the preview panel."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

        # Clear existing content
        while self.preview_container_layout.count():
            child = self.preview_container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add each figure
        for title, fig in self._preview_figures:
            fig_frame = QFrame()
            fig_frame.setStyleSheet("QFrame { background-color: #252525; border: 1px solid #3d3d3d; }")
            fig_layout = QVBoxLayout(fig_frame)
            fig_layout.setContentsMargins(5, 5, 5, 5)
            fig_layout.setSpacing(2)

            # Title
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fig_layout.addWidget(title_label)

            # Canvas
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(600)

            # Toolbar
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setStyleSheet("background-color: #2d2d2d;")

            fig_layout.addWidget(toolbar)
            fig_layout.addWidget(canvas)

            self.preview_container_layout.addWidget(fig_frame)

        # Re-apply scroll event filters on the main scroll area
        self.main_scroll.install_filter_on_new_widgets()


class ComparisonTab(QWidget):
    """Tab for comparing multiple consolidated datasets."""

    # Dark theme stylesheets (same as ConsolidationTab)
    LIST_STYLE = """
        QListWidget {
            background-color: #2b2b2b;
            border: 2px solid #555555;
            border-radius: 6px;
            color: #ffffff;
            padding: 4px;
            font-size: 12px;
            outline: none;
        }
        QListWidget::item {
            background-color: transparent;
            padding: 6px 8px;
            margin: 1px 0px;
            border-radius: 3px;
            color: #ffffff;
        }
        QListWidget::item:hover {
            background-color: #404040;
        }
        QListWidget::item:selected {
            background-color: #0078d4;
            color: #ffffff;
        }
        QListWidget:focus {
            border-color: #0078d4;
        }
    """

    BUTTON_STYLE = """
        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: 1px solid #0b5ed7;
            border-radius: 6px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #0b5ed7;
        }
        QPushButton:pressed {
            background-color: #0a58ca;
        }
        QPushButton:disabled {
            background-color: #9ec5fe;
            border-color: #9ec5fe;
            color: #eef4ff;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets = []  # List of loaded consolidated NPZ data
        self._comparison_figures = []
        self._dataset_colors = {}  # dataset index -> color
        self._smoothing_window = 60  # Default smoothing window (minutes)
        self._bar_grouping = 'phase'  # 'dataset' or 'phase' (light/dark) - default to phase
        self._light_mode = False  # Dark mode by default
        self._show_statistics = True  # Show stats on bar charts by default
        self._statistics_results = []  # Store stats results for export
        self.setup_ui()

    def setup_ui(self):
        """Set up the comparison tab UI with full-tab scrolling and splitter."""
        from PyQt6.QtWidgets import (QSpinBox, QComboBox, QCheckBox, QTableWidget,
                                      QTableWidgetItem, QHeaderView, QColorDialog)
        from PyQt6.QtGui import QColor

        # Default colors for datasets
        self.DEFAULT_COLORS = [
            '#3daee9',  # Blue
            '#e74c3c',  # Red
            '#2ecc71',  # Green
            '#f39c12',  # Orange
            '#9b59b6',  # Purple
            '#1abc9c',  # Teal
            '#e91e63',  # Pink
            '#00bcd4',  # Cyan
        ]

        # Root layout for the tab - just holds the scroll area
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Main scroll area for entire tab content
        self.main_scroll = WheelScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2d2d2d;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 4px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        # Container widget for all tab content
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #2d2d2d;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        # Header
        header = QLabel("Compare Consolidated Datasets")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        content_layout.addWidget(header)

        desc = QLabel(
            "Load consolidated NPZ files (created in the Consolidation tab) to compare "
            "CTAs and statistics across different experimental groups."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaaaaa; margin-bottom: 5px;")
        content_layout.addWidget(desc)

        # Create top container widget for the splitter
        top_container = QWidget()
        top_container_layout = QHBoxLayout(top_container)
        top_container_layout.setContentsMargins(0, 0, 0, 0)
        top_container_layout.setSpacing(10)

        # Dataset table group
        list_group = QGroupBox("Loaded Datasets")
        list_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(8, 15, 8, 8)
        list_layout.setSpacing(5)

        # Dataset table with columns: Enabled, Order, Filename, Display Name, Color, Show n=
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(6)
        self.dataset_table.setHorizontalHeaderLabels(['✓', '↕', 'Filename', 'Display Name', 'Color', 'n='])
        self.dataset_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                gridline-color: #3d3d3d;
                selection-background-color: #0078d4;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #4d4d4d;
                font-weight: bold;
            }
            QTableWidget QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: none;
                padding: 4px 8px;
                selection-background-color: #0078d4;
            }
        """)
        self.dataset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dataset_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.dataset_table.setMinimumHeight(100)
        self.dataset_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed
        )

        # Enable drag and drop for reordering
        self.dataset_table.setDragEnabled(True)
        self.dataset_table.setAcceptDrops(True)
        self.dataset_table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.dataset_table.setDropIndicatorShown(True)

        # Set column widths - all interactive (user resizable) except checkbox
        header_view = self.dataset_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Enabled checkbox
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)  # Order buttons
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)  # Filename
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Display Name - stretches to fill
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)  # Color
        header_view.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)  # Show n=
        self.dataset_table.setColumnWidth(0, 35)   # Enabled checkbox
        self.dataset_table.setColumnWidth(1, 50)   # Order buttons
        self.dataset_table.setColumnWidth(2, 180)  # Filename
        self.dataset_table.setColumnWidth(4, 45)   # Color
        self.dataset_table.setColumnWidth(5, 35)   # n= checkbox

        # Connect cell change signal
        self.dataset_table.cellChanged.connect(self._on_table_cell_changed)

        list_layout.addWidget(self.dataset_table, stretch=1)

        # Buttons for list management
        btn_row = QHBoxLayout()
        btn_row.setSpacing(5)

        self.add_btn = QPushButton("Add Datasets...")
        self.add_btn.setStyleSheet(self.BUTTON_STYLE)
        self.add_btn.setMinimumWidth(110)
        self.add_btn.clicked.connect(self.on_add_dataset)
        btn_row.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: 1px solid #b02a37;
                border-radius: 6px;
                padding: 5px 10px;
            }
            QPushButton:hover { background-color: #bb2d3b; }
            QPushButton:pressed { background-color: #a52834; }
            QPushButton:disabled { background-color: #f1aeb5; }
        """)
        self.remove_btn.clicked.connect(self.on_remove_selected)
        self.remove_btn.setEnabled(False)
        btn_row.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: 1px solid #5c636a;
                border-radius: 6px;
                padding: 5px 10px;
            }
            QPushButton:hover { background-color: #5c636a; }
            QPushButton:pressed { background-color: #4d5459; }
            QPushButton:disabled { background-color: #c6cace; }
        """)
        self.clear_btn.clicked.connect(self.on_clear_all)
        self.clear_btn.setEnabled(False)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()
        list_layout.addLayout(btn_row)

        top_container_layout.addWidget(list_group, stretch=3)  # Wider dataset list

        # Right side: Options and Generate button (narrower)
        action_group = QGroupBox("Options")
        action_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(8, 15, 8, 8)
        action_layout.setSpacing(8)

        # Smoothing option
        smooth_row = QHBoxLayout()
        smooth_label = QLabel("Smoothing (min):")
        smooth_label.setStyleSheet("color: #cccccc;")
        smooth_row.addWidget(smooth_label)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 120)
        self.smoothing_spin.setValue(self._smoothing_window)
        self.smoothing_spin.setToolTip("Rolling average window size in minutes")
        self.smoothing_spin.setStyleSheet("""
            QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 3px;
                color: #ffffff;
            }
        """)
        self.smoothing_spin.valueChanged.connect(self._on_smoothing_changed)
        smooth_row.addWidget(self.smoothing_spin)
        smooth_row.addStretch()
        action_layout.addLayout(smooth_row)

        # Bar chart grouping option
        group_row = QHBoxLayout()
        group_label = QLabel("Bar grouping:")
        group_label.setStyleSheet("color: #cccccc;")
        group_row.addWidget(group_label)
        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems(["By Dataset", "By Light/Dark"])
        self.grouping_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 3px 8px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
        """)
        self.grouping_combo.setCurrentIndex(1)  # Default to "By Light/Dark"
        self.grouping_combo.currentIndexChanged.connect(self._on_grouping_changed)
        group_row.addWidget(self.grouping_combo)
        group_row.addStretch()
        action_layout.addLayout(group_row)

        # Light/Dark mode toggle
        self.light_mode_cb = QCheckBox("Light mode figures")
        self.light_mode_cb.setStyleSheet("color: #cccccc;")
        self.light_mode_cb.setToolTip("Use light background for figures (better for publications)")
        self.light_mode_cb.toggled.connect(self._on_light_mode_changed)
        action_layout.addWidget(self.light_mode_cb)

        # Show statistics checkbox
        self.show_stats_cb = QCheckBox("Show statistics")
        self.show_stats_cb.setStyleSheet("color: #cccccc;")
        self.show_stats_cb.setToolTip("Show significance tests on bar charts (t-test for 2 groups, ANOVA for 3+)")
        self.show_stats_cb.setChecked(True)  # Default to showing stats
        self.show_stats_cb.toggled.connect(self._on_show_stats_changed)
        action_layout.addWidget(self.show_stats_cb)

        action_layout.addStretch()

        # Generate button
        self.compare_btn = QPushButton("Generate Comparison")
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #198754;
                color: white;
                border: 1px solid #157347;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #157347; }
            QPushButton:pressed { background-color: #146c43; }
            QPushButton:disabled { background-color: #a3cfbb; color: #eef8f3; }
        """)
        self.compare_btn.clicked.connect(self.on_generate_comparison)
        self.compare_btn.setEnabled(False)
        action_layout.addWidget(self.compare_btn)

        # Save button
        self.save_btn = QPushButton("Save Figures...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: 1px solid #0a58ca;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:pressed { background-color: #0a58ca; }
            QPushButton:disabled { background-color: #6ea8fe; color: #e7f1ff; }
        """)
        self.save_btn.clicked.connect(self.on_save_comparison)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)

        top_container_layout.addWidget(action_group, stretch=1)

        # Use a vertical splitter between top section and figures
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555555;
                height: 8px;
                border-radius: 2px;
                margin: 2px 50px;
            }
            QSplitter::handle:hover {
                background-color: #0078d4;
            }
        """)
        self.main_splitter.setHandleWidth(8)

        # Add top container to splitter
        self.main_splitter.addWidget(top_container)
        top_container.setMinimumHeight(150)
        top_container.setMaximumHeight(350)  # Prevent expansion beyond reasonable height

        # Set size policy to prevent top container from expanding when figures are added
        from PyQt6.QtWidgets import QSizePolicy
        top_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # Create figures container widget for splitter
        figures_widget = QWidget()
        figures_widget.setStyleSheet("background-color: #2d2d2d;")
        figures_layout = QVBoxLayout(figures_widget)
        figures_layout.setContentsMargins(0, 5, 0, 5)
        figures_layout.setSpacing(5)

        # Figures section header
        figures_header = QLabel("Comparison Figures")
        figures_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        figures_layout.addWidget(figures_header)

        # Figures container
        self.figures_container = QWidget()
        self.figures_container_layout = QVBoxLayout(self.figures_container)
        self.figures_container_layout.setContentsMargins(0, 5, 0, 5)
        self.figures_container_layout.setSpacing(15)

        # Placeholder
        placeholder = QLabel("No comparison generated yet. Add datasets and click 'Generate Comparison'.")
        placeholder.setStyleSheet("color: #888888; font-size: 12px; padding: 40px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.figures_container_layout.addWidget(placeholder)

        figures_layout.addWidget(self.figures_container, stretch=1)

        # Add figures widget to splitter
        self.main_splitter.addWidget(figures_widget)
        figures_widget.setMinimumHeight(100)

        # Set initial splitter sizes (top section, figures get more space)
        self.main_splitter.setSizes([220, 500])

        content_layout.addWidget(self.main_splitter, stretch=1)

        # Set scroll content
        self.main_scroll.setWidget(scroll_content)
        root_layout.addWidget(self.main_scroll)

        # Connect table selection changes
        self.dataset_table.itemSelectionChanged.connect(self._update_buttons)

    def _on_smoothing_changed(self, value):
        """Handle smoothing spinbox value change."""
        self._smoothing_window = value

    def _on_grouping_changed(self, index):
        """Handle bar grouping combo change."""
        self._bar_grouping = 'dataset' if index == 0 else 'phase'

    def _on_light_mode_changed(self, checked):
        """Handle light mode checkbox change."""
        self._light_mode = checked

    def _on_show_stats_changed(self, checked):
        """Handle show statistics checkbox change."""
        self._show_statistics = checked

    def _on_table_cell_changed(self, row, column):
        """Handle changes to table cells."""
        if row >= len(self._datasets):
            return

        # Column 3 is Display Name - update the dataset metadata
        if column == 3:
            new_name = self.dataset_table.item(row, column).text()
            if 'consolidation_metadata' not in self._datasets[row]:
                self._datasets[row]['consolidation_metadata'] = {}
            self._datasets[row]['consolidation_metadata']['display_name'] = new_name

    def _update_buttons(self):
        """Update button enabled states based on current state."""
        n_datasets = len(self._datasets)
        n_selected = len(self.dataset_table.selectedItems()) > 0

        self.remove_btn.setEnabled(n_selected)
        self.clear_btn.setEnabled(n_datasets > 0)

        # Count enabled datasets
        enabled_count = self._count_enabled_datasets()
        self.compare_btn.setEnabled(enabled_count >= 2)

    def _count_enabled_datasets(self):
        """Count datasets with enabled checkbox checked."""
        from PyQt6.QtWidgets import QCheckBox
        count = 0
        for row in range(self.dataset_table.rowCount()):
            checkbox_widget = self.dataset_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    count += 1
        return count

    def _on_enabled_changed(self, row, state):
        """Handle enabled checkbox state change - store in dataset."""
        if row < len(self._datasets):
            self._datasets[row]['_enabled'] = (state == 2)  # Qt.CheckState.Checked = 2
        self._update_buttons()

    def _on_show_n_changed(self, row, state):
        """Handle show n= checkbox state change - store in dataset."""
        if row < len(self._datasets):
            self._datasets[row]['_show_n'] = (state == 2)  # Qt.CheckState.Checked = 2

    def _get_default_display_name(self, dataset):
        """Generate default display name from dataset metadata."""
        meta = dataset.get('consolidation_metadata', {})
        n_animals = meta.get('n_animals', '?')

        # Use filter description if available
        filter_desc = meta.get('filter_description', '')
        if filter_desc and filter_desc != 'No filters applied':
            return f"{filter_desc} (n={n_animals})"

        # Fall back to filename
        filename = dataset.get('filename', 'Dataset')
        # Remove extension and prefix
        name = filename.replace('.npz', '').replace('CageMetrics_Consolidated_', '')
        return f"{name} (n={n_animals})"

    def _create_order_buttons(self, row):
        """Create up/down buttons for a row."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)

        up_btn = QPushButton("▲")
        up_btn.setFixedSize(20, 20)
        up_btn.setStyleSheet("QPushButton { padding: 0; font-size: 10px; }")
        up_btn.clicked.connect(lambda checked, r=row: self._move_row_up(r))

        down_btn = QPushButton("▼")
        down_btn.setFixedSize(20, 20)
        down_btn.setStyleSheet("QPushButton { padding: 0; font-size: 10px; }")
        down_btn.clicked.connect(lambda checked, r=row: self._move_row_down(r))

        layout.addWidget(up_btn)
        layout.addWidget(down_btn)
        return widget

    def _create_color_button(self, row, color):
        """Create a color picker button for a row."""
        btn = QPushButton()
        btn.setFixedSize(40, 22)
        btn.setStyleSheet(f"background-color: {color}; border: 1px solid #555555; border-radius: 3px;")
        btn.clicked.connect(lambda checked, r=row: self._pick_color(r))
        return btn

    def _pick_color(self, row):
        """Open color picker for the specified row."""
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor

        # Get current color
        current_color = self._dataset_colors.get(row, self.DEFAULT_COLORS[row % len(self.DEFAULT_COLORS)])
        color = QColorDialog.getColor(QColor(current_color), self, "Select Dataset Color")

        if color.isValid():
            self._dataset_colors[row] = color.name()
            # Update button color
            color_btn = self.dataset_table.cellWidget(row, 4)
            if color_btn:
                color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #555555; border-radius: 3px;")

    def _move_row_up(self, row):
        """Move the specified row up."""
        if row <= 0:
            return
        self._swap_rows(row, row - 1)

    def _move_row_down(self, row):
        """Move the specified row down."""
        if row >= self.dataset_table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)

    def _swap_rows(self, row1, row2):
        """Swap two rows in the table and datasets list."""
        # Swap in data list
        self._datasets[row1], self._datasets[row2] = self._datasets[row2], self._datasets[row1]

        # Swap colors
        color1 = self._dataset_colors.get(row1, self.DEFAULT_COLORS[row1 % len(self.DEFAULT_COLORS)])
        color2 = self._dataset_colors.get(row2, self.DEFAULT_COLORS[row2 % len(self.DEFAULT_COLORS)])
        self._dataset_colors[row1] = color2
        self._dataset_colors[row2] = color1

        # Rebuild the table
        self._rebuild_table()

    def _add_dataset_row(self, row, dataset):
        """Add a row to the dataset table for the given dataset."""
        from PyQt6.QtWidgets import QTableWidgetItem, QCheckBox

        self.dataset_table.blockSignals(True)

        meta = dataset.get('consolidation_metadata', {})
        n_animals = meta.get('n_animals', '?')

        # Checkbox style for better visibility
        checkbox_style = """
            QCheckBox {
                spacing: 0px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #666666;
                background-color: #2d2d2d;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                background-color: #0078d4;
                border-radius: 3px;
            }
        """

        # Column 0: Enabled checkbox
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        enabled_cb = QCheckBox()
        # Restore enabled state from dataset, default to True
        enabled_cb.setChecked(dataset.get('_enabled', True))
        enabled_cb.setStyleSheet(checkbox_style)
        enabled_cb.setToolTip("Include in comparison")
        enabled_cb.stateChanged.connect(lambda state, r=row: self._on_enabled_changed(r, state))
        checkbox_layout.addWidget(enabled_cb)
        self.dataset_table.setCellWidget(row, 0, checkbox_widget)

        # Column 1: Order buttons
        self.dataset_table.setCellWidget(row, 1, self._create_order_buttons(row))

        # Column 2: Filename (read-only)
        filename_item = QTableWidgetItem(dataset.get('filename', 'Unknown'))
        filename_item.setFlags(filename_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.dataset_table.setItem(row, 2, filename_item)

        # Column 3: Display Name (editable) - auto-fill with default
        display_name = meta.get('display_name', '')
        if not display_name:
            display_name = self._get_default_display_name(dataset)
        display_item = QTableWidgetItem(display_name)
        self.dataset_table.setItem(row, 3, display_item)

        # Column 4: Color button
        color = self._dataset_colors.get(row, self.DEFAULT_COLORS[row % len(self.DEFAULT_COLORS)])
        self._dataset_colors[row] = color
        self.dataset_table.setCellWidget(row, 4, self._create_color_button(row, color))

        # Column 5: Show n= checkbox
        n_widget = QWidget()
        n_layout = QHBoxLayout(n_widget)
        n_layout.setContentsMargins(0, 0, 0, 0)
        n_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        n_cb = QCheckBox()
        # Restore show_n state from dataset, default to True
        n_cb.setChecked(dataset.get('_show_n', True))
        n_cb.setStyleSheet(checkbox_style)
        n_cb.setToolTip(f"Show (n={n_animals}) in legend")
        n_cb.stateChanged.connect(lambda state, r=row: self._on_show_n_changed(r, state))
        n_layout.addWidget(n_cb)
        self.dataset_table.setCellWidget(row, 5, n_widget)

        self.dataset_table.blockSignals(False)

    def _rebuild_table(self):
        """Rebuild the entire table from the datasets list."""
        self.dataset_table.setRowCount(0)
        self.dataset_table.setRowCount(len(self._datasets))

        for row, dataset in enumerate(self._datasets):
            self._add_dataset_row(row, dataset)

        self._update_buttons()

    def on_add_dataset(self):
        """Add consolidated NPZ files to compare."""
        settings = QSettings("PhysioMetrics", "CageMetrics")

        # Try to default to a 'consolidated' subfolder if one exists
        last_dir = settings.value("comparison_load_dir", "")
        if not last_dir:
            # Check if consolidation_save_dir has a consolidated folder
            consolidation_dir = settings.value("consolidation_save_dir", str(Path.home()))
            consolidated_path = Path(consolidation_dir)
            if consolidated_path.exists():
                last_dir = str(consolidated_path)
            else:
                last_dir = str(Path.home())

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Consolidated NPZ Files",
            last_dir,
            "CageMetrics Consolidated (CageMetrics_Consolidated*.npz);;All NPZ Files (*.npz);;All Files (*)"
        )

        if not file_paths:
            return

        settings.setValue("comparison_load_dir", str(Path(file_paths[0]).parent))

        from core.comparison_figure_generator import load_consolidated_npz

        added = 0
        for path in file_paths:
            # Check if already loaded
            existing_paths = [d.get('path', '') for d in self._datasets]
            if path in existing_paths:
                continue

            # Load the dataset
            data = load_consolidated_npz(path)
            if data is None:
                QMessageBox.warning(
                    self, "Invalid File",
                    f"'{Path(path).name}' is not a valid consolidated NPZ file.\n\n"
                    "Only NPZ files created by the Consolidation tab can be compared."
                )
                continue

            data['path'] = path
            self._datasets.append(data)

            # Add row to table
            row = self.dataset_table.rowCount()
            self.dataset_table.setRowCount(row + 1)
            self._add_dataset_row(row, data)
            added += 1

        if added > 0:
            self._update_buttons()

    def on_remove_selected(self):
        """Remove selected datasets from the table."""
        selected_rows = sorted(set(item.row() for item in self.dataset_table.selectedItems()),
                              reverse=True)

        for row in selected_rows:
            self.dataset_table.removeRow(row)
            del self._datasets[row]
            # Also remove color
            if row in self._dataset_colors:
                del self._dataset_colors[row]

        # Rebuild to fix row indices for buttons
        self._rebuild_table()

    def on_clear_all(self):
        """Clear all loaded datasets."""
        if not self._datasets:
            return

        reply = QMessageBox.question(
            self, "Clear All",
            "Remove all loaded datasets?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.dataset_table.setRowCount(0)
            self._datasets.clear()
            self._dataset_colors.clear()
            self._update_buttons()

    def _get_enabled_datasets(self):
        """Get list of datasets that are enabled (checked) in the table, with their colors and display names."""
        from PyQt6.QtWidgets import QCheckBox
        enabled = []
        for row in range(self.dataset_table.rowCount()):
            # Check enabled checkbox
            checkbox_widget = self.dataset_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    dataset = self._datasets[row].copy()

                    # Get display name from table
                    display_item = self.dataset_table.item(row, 3)
                    if display_item:
                        display_name = display_item.text()
                        # Check if show n= is checked
                        n_widget = self.dataset_table.cellWidget(row, 5)
                        if n_widget:
                            n_cb = n_widget.findChild(QCheckBox)
                            if n_cb and not n_cb.isChecked():
                                # Remove (n=X) from display name if checkbox unchecked
                                import re
                                display_name = re.sub(r'\s*\(n=\d+\)', '', display_name)

                        if 'consolidation_metadata' not in dataset:
                            dataset['consolidation_metadata'] = {}
                        dataset['consolidation_metadata']['display_name'] = display_name

                    # Get color
                    color = self._dataset_colors.get(row, self.DEFAULT_COLORS[row % len(self.DEFAULT_COLORS)])
                    dataset['color'] = color

                    enabled.append(dataset)
        return enabled

    def on_generate_comparison(self):
        """Generate comparison figures for loaded datasets."""
        enabled_datasets = self._get_enabled_datasets()

        if len(enabled_datasets) < 2:
            QMessageBox.information(
                self, "Not Enough Datasets",
                "Please enable at least 2 datasets to compare."
            )
            return

        try:
            from core.comparison_figure_generator import ComparisonFigureGenerator

            # Extract colors from enabled datasets
            dataset_colors = [ds.get('color', self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)])
                            for i, ds in enumerate(enabled_datasets)]

            generator = ComparisonFigureGenerator(
                smoothing_window=self._smoothing_window,
                bar_grouping=self._bar_grouping,
                light_mode=self._light_mode,
                dataset_colors=dataset_colors,
                show_statistics=self._show_statistics
            )
            self._comparison_figures = generator.generate_all_pages(enabled_datasets)
            self._enabled_datasets_for_save = enabled_datasets  # Store for saving
            self._statistics_results = generator.statistics_results  # Store statistics for export

            self._display_comparison_figures()

            # Enable save button now that figures exist
            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison generation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def on_save_comparison(self):
        """Save comparison figures to PDF and dataset configuration to JSON."""
        if not self._comparison_figures:
            QMessageBox.information(self, "No Figures", "Generate a comparison first before saving.")
            return

        settings = QSettings("PhysioMetrics", "CageMetrics")
        last_dir = settings.value("comparison_save_dir", str(Path.home()))

        # Create comparisons subfolder
        comparisons_dir = Path(last_dir) / "comparisons"
        comparisons_dir.mkdir(exist_ok=True)

        # Ask for save location (default to comparisons subfolder)
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Comparison",
            str(comparisons_dir / "Comparison"),
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not file_path:
            return

        # Save the parent of the comparisons folder as the base dir
        save_path = Path(file_path)
        if save_path.parent.name == "comparisons":
            settings.setValue("comparison_save_dir", str(save_path.parent.parent))
        else:
            settings.setValue("comparison_save_dir", str(save_path.parent))

        # Ensure .pdf extension
        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'

        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import json

            # Save figures to PDF
            with PdfPages(file_path) as pdf:
                for title, fig in self._comparison_figures:
                    pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')

            # Also save dataset configuration as JSON alongside PDF
            json_path = file_path.replace('.pdf', '_config.json')
            config = {
                'smoothing_window': self._smoothing_window,
                'bar_grouping': self._bar_grouping,
                'light_mode': self._light_mode,
                'datasets': []
            }

            for ds in self._enabled_datasets_for_save:
                meta = ds.get('consolidation_metadata', {})
                config['datasets'].append({
                    'filename': ds.get('filename', ''),
                    'path': ds.get('path', ''),
                    'display_name': meta.get('display_name', ''),
                    'color': ds.get('color', ''),
                    'n_animals': meta.get('n_animals', 0),
                    'filter_description': meta.get('filter_description', '')
                })

            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Save statistics to CSV if available
            saved_files = [Path(file_path).name, Path(json_path).name]
            if self._statistics_results:
                import pandas as pd
                stats_path = file_path.replace('.pdf', '_statistics.csv')
                stats_df = pd.DataFrame(self._statistics_results)
                # Reorder columns for clarity
                column_order = ['metric', 'phase', 'comparison', 'group1', 'group1_mean', 'group1_sem',
                               'group2', 'group2_mean', 'group2_sem', 'test', 'p_value', 'significance', 'anova_p']
                stats_df = stats_df[[c for c in column_order if c in stats_df.columns]]
                stats_df.to_csv(stats_path, index=False)
                saved_files.append(Path(stats_path).name)

            QMessageBox.information(
                self, "Saved",
                f"Comparison saved:\n" + "\n".join(f"• {f}" for f in saved_files)
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save comparison:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _display_comparison_figures(self):
        """Display generated comparison figures in the scroll area."""
        # Save current splitter sizes to restore after adding figures
        saved_sizes = self.main_splitter.sizes()

        # Clear existing figures
        while self.figures_container_layout.count() > 0:
            item = self.figures_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._comparison_figures:
            placeholder = QLabel("No figures generated.")
            placeholder.setStyleSheet("color: #888888; padding: 40px;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.figures_container_layout.addWidget(placeholder)
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        for title, fig in self._comparison_figures:
            # Create frame for each figure
            fig_frame = QFrame()
            fig_frame.setStyleSheet("""
                QFrame {
                    background-color: #363636;
                    border: 1px solid #4d4d4d;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            frame_layout = QVBoxLayout(fig_frame)
            frame_layout.setContentsMargins(8, 8, 8, 8)

            # Title
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
            frame_layout.addWidget(title_label)

            # Canvas
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(500)
            frame_layout.addWidget(canvas)

            # Toolbar
            toolbar = NavigationToolbar2QT(canvas, fig_frame)
            toolbar.setStyleSheet("background-color: #2d2d2d;")
            frame_layout.addWidget(toolbar)

            self.figures_container_layout.addWidget(fig_frame)

        # Re-apply scroll event filters on the main scroll area
        self.main_scroll.install_filter_on_new_widgets()

        # Restore splitter sizes to prevent top section from expanding
        if saved_sizes and saved_sizes[0] > 0:
            self.main_splitter.setSizes(saved_sizes)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CageMetrics v{VERSION_STRING} - Behavioral Analysis")
        self.setMinimumSize(1200, 800)

        # Cache for update info (checked in background)
        self._update_info = None

        settings = QSettings("PhysioMetrics", "CageMetrics")
        geometry = settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1400, 900)

        self.setup_ui()
        self._setup_shortcuts()
        self._check_for_updates_async()

    def setup_ui(self):
        """Set up the main window UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header row with version and help button
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #252525;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)

        # Version label on left
        version_label = QLabel(f"v{VERSION_STRING}")
        version_label.setStyleSheet("color: #888888; font-size: 9pt;")
        header_layout.addWidget(version_label)

        header_layout.addStretch()

        # Update available label (hidden by default)
        self.update_available_label = QLabel()
        self.update_available_label.setStyleSheet("""
            color: #4CAF50;
            font-size: 9pt;
            font-weight: bold;
        """)
        self.update_available_label.hide()
        header_layout.addWidget(self.update_available_label)

        # Help button on right (link style)
        help_btn = QPushButton("Help")
        help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #3daee9;
                border: none;
                font-size: 10pt;
                text-decoration: underline;
                padding: 5px 10px;
            }
            QPushButton:hover {
                color: #5bc0ff;
            }
        """)
        help_btn.clicked.connect(self.show_help_dialog)
        header_layout.addWidget(help_btn)

        main_layout.addWidget(header_widget)

        # Main tab widget
        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)

        # Analysis tab
        self.analysis_tab = AnalysisTab(self)
        self.main_tabs.addTab(self.analysis_tab, "Analysis")

        # Consolidation tab
        self.consolidation_tab = ConsolidationTab(self)
        self.main_tabs.addTab(self.consolidation_tab, "Consolidation")

        # Comparison tab
        self.comparison_tab = ComparisonTab(self)
        self.main_tabs.addTab(self.comparison_tab, "Comparison")

        main_layout.addWidget(self.main_tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # F1 for help
        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self.show_help_dialog)

    def _check_for_updates_async(self):
        """Check for updates in background thread."""
        class UpdateChecker(QThread):
            update_checked = pyqtSignal(object)

            def run(self):
                info = update_checker.check_for_updates()
                self.update_checked.emit(info)

        def on_update_checked(update_info):
            if update_info:
                self._update_info = update_info
                # Show update available indicator
                version = update_info.get('version', '')
                self.update_available_label.setText(f"New version available (v{version})")
                self.update_available_label.show()

        self._update_thread = UpdateChecker()
        self._update_thread.update_checked.connect(on_update_checked)
        self._update_thread.start()

    def show_help_dialog(self):
        """Show the help dialog."""
        from dialogs.help_dialog import HelpDialog
        dialog = HelpDialog(self, update_info=self._update_info)
        dialog.exec()

    def closeEvent(self, event):
        """Save window state on close and log session end."""
        settings = QSettings("PhysioMetrics", "CageMetrics")
        settings.setValue("window_geometry", self.saveGeometry())

        # Log session end for telemetry
        telemetry.log_session_end()

        event.accept()


def main():
    """Main entry point."""
    plt.style.use('dark_background')

    app = QApplication(sys.argv)
    app.setApplicationName("CageMetrics")
    app.setOrganizationName("PhysioMetrics")

    app.setStyleSheet(DARK_STYLESHEET)

    # Show first launch dialog if needed
    if app_config.is_first_launch():
        from dialogs.first_launch_dialog import FirstLaunchDialog
        dialog = FirstLaunchDialog()
        dialog.exec()

        # Save user preferences
        app_config.set_telemetry_enabled(dialog.get_telemetry_enabled())
        app_config.set_crash_reports_enabled(dialog.get_crash_reports_enabled())
        app_config.set_first_launch_completed()

    # Initialize telemetry (respects user preference)
    telemetry.init_telemetry()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
