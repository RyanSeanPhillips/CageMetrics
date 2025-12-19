"""
CageMetrics Help Dialog

Multi-tabbed help dialog with shortcuts, features, and About information.
"""

import sys

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QLabel,
    QDialogButtonBox, QTabWidget, QWidget, QPushButton,
    QGroupBox, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from version_info import VERSION_STRING, DOI


class HelpDialog(QDialog):
    """Help dialog with shortcuts, features, and About tabs."""

    def __init__(self, parent=None, update_info=None):
        super().__init__(parent)
        self.setWindowTitle("CageMetrics - Help")
        self.resize(800, 650)
        self.update_info = update_info
        self.update_thread = None

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()

    def _setup_ui(self):
        """Create the help dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_shortcuts_tab(), "Shortcuts")
        tabs.addTab(self._create_features_tab(), "Features")
        tabs.addTab(self._create_about_tab(), "About")
        layout.addWidget(tabs)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

    def _create_shortcuts_tab(self):
        """Create keyboard shortcuts tab."""
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml("""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; padding: 15px;">

            <h2 style="color: #3daee9; margin-top: 0;">Keyboard Shortcuts</h2>

            <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; margin-top: 10px;">
                <tr style="background-color: #2a2a2a;">
                    <th style="width: 150px; text-align: left;">Shortcut</th>
                    <th style="text-align: left;">Action</th>
                </tr>
                <tr>
                    <td><b>F1</b></td>
                    <td>Open this Help dialog</td>
                </tr>
                <tr>
                    <td><b>Ctrl+O</b></td>
                    <td>Open/Load files</td>
                </tr>
                <tr>
                    <td><b>Ctrl+S</b></td>
                    <td>Save/Export data</td>
                </tr>
            </table>

            <h2 style="color: #3daee9; margin-top: 25px;">Quick Start Guide</h2>

            <h3 style="color: #888888; margin-bottom: 5px;">Analysis Tab</h3>
            <ol>
                <li>Click <b>"Load NPZ Files"</b> to select animal data files</li>
                <li>View circadian time averages (CTAs) for each metric</li>
                <li>Click <b>"Export"</b> to save figures and data</li>
            </ol>

            <h3 style="color: #888888; margin-bottom: 5px;">Consolidation Tab</h3>
            <ol>
                <li>Click <b>"Scan Directory"</b> to find all NPZ files in a folder</li>
                <li>Use filters to select animals by genotype, sex, treatment</li>
                <li>Click <b>"Generate Preview"</b> to see consolidated data</li>
                <li>Click <b>"Export"</b> to save consolidated results</li>
            </ol>

            <h3 style="color: #888888; margin-bottom: 5px;">Comparison Tab</h3>
            <ol>
                <li>Click <b>"Add Dataset"</b> to load consolidated NPZ files</li>
                <li>Enable <b>"Show Statistics"</b> for automatic statistical analysis</li>
                <li>Click <b>"Generate Comparison"</b> to create comparison figures</li>
                <li>Click <b>"Export"</b> to save comparison results with statistics</li>
            </ol>

        </body>
        </html>
        """)
        return browser

    def _create_features_tab(self):
        """Create features documentation tab."""
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml("""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; padding: 15px;">

            <h2 style="color: #3daee9; margin-top: 0;">Features Overview</h2>

            <h3 style="color: #3daee9;">Consolidation</h3>
            <ul>
                <li><b>Batch Processing:</b> Load multiple NPZ files at once</li>
                <li><b>Smart Filtering:</b> Filter by genotype, cagemate genotype, sex, treatment</li>
                <li><b>Metadata Editing:</b> Update animal metadata across multiple files</li>
                <li><b>Population Analysis:</b> Calculate grand means and SEMs across animals</li>
            </ul>

            <h3 style="color: #3daee9;">Comparison</h3>
            <ul>
                <li><b>Multi-Dataset:</b> Compare 2 or more consolidated datasets</li>
                <li><b>Automatic Statistics:</b>
                    <ul>
                        <li>2 groups: Welch's t-test</li>
                        <li>3+ groups: ANOVA with Bonferroni-corrected pairwise comparisons</li>
                    </ul>
                </li>
                <li><b>Significance Brackets:</b> Visual indicators (*, **, ***) on figures</li>
                <li><b>Statistics Export:</b> CSV file with all statistical results</li>
            </ul>

            <h3 style="color: #3daee9;">Available Metrics</h3>
            <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; margin-top: 5px;">
                <tr style="background-color: #2a2a2a;">
                    <th>Metric</th>
                    <th>Unit</th>
                </tr>
                <tr><td>Distance</td><td>meters</td></tr>
                <tr><td>Active Time</td><td>%</td></tr>
                <tr><td>Rearing Time</td><td>%</td></tr>
                <tr><td>Food Zone Time</td><td>%</td></tr>
                <tr><td>Water Zone Time</td><td>%</td></tr>
                <tr><td>Nest Zone Time</td><td>%</td></tr>
                <tr><td>Wheel Rotations</td><td>count</td></tr>
                <tr><td>Food Consumption</td><td>grams</td></tr>
                <tr><td>Water Consumption</td><td>mL</td></tr>
                <tr><td>Body Weight</td><td>grams</td></tr>
                <tr><td>Temperature</td><td>°C</td></tr>
            </table>

            <h3 style="color: #3daee9; margin-top: 20px;">File Formats</h3>
            <ul>
                <li><b>.npz</b> - Input: Individual animal data from PhysioMetrics</li>
                <li><b>_consolidated.npz</b> - Output: Consolidated population data</li>
                <li><b>.xlsx</b> - Output: Excel workbook with all metrics</li>
                <li><b>.pdf</b> - Output: Publication-ready figures</li>
                <li><b>.csv</b> - Output: Statistics results (comparison tab)</li>
            </ul>

        </body>
        </html>
        """)
        return browser

    def _create_about_tab(self):
        """Create About tab with version, logo placeholder, updates, and citation."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Top section: Logo placeholder and version
        top_section = QHBoxLayout()
        top_section.setSpacing(20)

        # Logo placeholder (left side)
        logo_frame = QFrame()
        logo_frame.setFixedSize(120, 120)
        logo_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 2px dashed #3a3a3a;
                border-radius: 10px;
            }
        """)
        logo_layout = QVBoxLayout(logo_frame)
        logo_label = QLabel("Logo\n(Coming Soon)")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #666666; font-size: 10pt; border: none;")
        logo_layout.addWidget(logo_label)
        top_section.addWidget(logo_frame)

        # Version and app info (right side)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)

        app_name = QLabel("CageMetrics")
        app_name.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        app_name.setStyleSheet("color: #3daee9;")
        info_layout.addWidget(app_name)

        version_label = QLabel(f"Version {VERSION_STRING}")
        version_label.setStyleSheet("color: #cccccc; font-size: 12pt;")
        info_layout.addWidget(version_label)

        desc_label = QLabel("Behavioral analysis for Allentown cage monitoring systems")
        desc_label.setStyleSheet("color: #888888; font-size: 10pt;")
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)

        info_layout.addStretch()
        top_section.addLayout(info_layout)
        top_section.addStretch()

        main_layout.addLayout(top_section)

        # Update section
        update_frame = QFrame()
        update_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        update_layout = QHBoxLayout(update_frame)
        update_layout.setContentsMargins(15, 10, 15, 10)

        self.update_label = QLabel()
        self.update_label.setOpenExternalLinks(True)
        self.update_label.setTextFormat(Qt.TextFormat.RichText)
        self.update_label.setWordWrap(True)
        self.update_label.setStyleSheet("border: none;")
        update_layout.addWidget(self.update_label, stretch=1)

        check_update_btn = QPushButton("Check for Updates")
        check_update_btn.setFixedWidth(140)
        check_update_btn.clicked.connect(self._on_check_updates_clicked)
        update_layout.addWidget(check_update_btn)

        main_layout.addWidget(update_frame)

        # Check for updates on dialog open
        self._check_for_updates_async()

        # Citation section
        citation_frame = QFrame()
        citation_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        citation_layout = QVBoxLayout(citation_frame)
        citation_layout.setContentsMargins(15, 15, 15, 15)
        citation_layout.setSpacing(10)

        citation_header = QLabel("Citation")
        citation_header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        citation_header.setStyleSheet("color: #3daee9; border: none;")
        citation_layout.addWidget(citation_header)

        citation_text = QLabel(
            f"Phillips, R.S. (2025). CageMetrics v{VERSION_STRING}.\n"
            f"DOI: {DOI}"
        )
        citation_text.setStyleSheet("color: #cccccc; font-family: monospace; font-size: 10pt; border: none;")
        citation_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        citation_layout.addWidget(citation_text)

        doi_link = QLabel(f'<a href="https://doi.org/{DOI}" style="color: #3daee9;">https://doi.org/{DOI}</a>')
        doi_link.setOpenExternalLinks(True)
        doi_link.setStyleSheet("border: none;")
        citation_layout.addWidget(doi_link)

        main_layout.addWidget(citation_frame)

        # Developer and links section
        dev_layout = QHBoxLayout()
        dev_layout.setSpacing(30)

        # Developer info
        dev_info = QLabel(
            "<b style='color: #3daee9;'>Developer</b><br>"
            "Ryan Sean Phillips<br>"
            "<span style='color: #888888;'>Seattle Children's Research Institute</span><br>"
            "<a href='mailto:ryan.phillips@seattlechildrens.org' style='color: #3daee9;'>ryan.phillips@seattlechildrens.org</a><br>"
            "<a href='https://orcid.org/0000-0002-8570-2348' style='color: #3daee9;'>ORCID: 0000-0002-8570-2348</a>"
        )
        dev_info.setOpenExternalLinks(True)
        dev_info.setStyleSheet("font-size: 9pt;")
        dev_layout.addWidget(dev_info)

        # Links
        links_info = QLabel(
            "<b style='color: #3daee9;'>Links</b><br>"
            "<a href='https://github.com/RyanSeanPhillips/CageMetrics' style='color: #3daee9;'>GitHub Repository</a><br>"
            "<a href='https://github.com/RyanSeanPhillips/CageMetrics/releases' style='color: #3daee9;'>Download Releases</a><br>"
            "<a href='https://github.com/RyanSeanPhillips/CageMetrics/issues' style='color: #3daee9;'>Report an Issue</a>"
        )
        links_info.setOpenExternalLinks(True)
        links_info.setStyleSheet("font-size: 9pt;")
        dev_layout.addWidget(links_info)

        dev_layout.addStretch()
        main_layout.addLayout(dev_layout)

        # Telemetry settings
        telemetry_group = QGroupBox("Privacy Settings")
        telemetry_layout = QVBoxLayout()
        telemetry_layout.setSpacing(5)

        from core import config as app_config

        self.telemetry_checkbox = QCheckBox("Share anonymous usage statistics")
        self.telemetry_checkbox.setChecked(app_config.is_telemetry_enabled())
        self.telemetry_checkbox.toggled.connect(self._on_telemetry_toggled)
        telemetry_layout.addWidget(self.telemetry_checkbox)

        self.crash_reports_checkbox = QCheckBox("Send crash reports")
        self.crash_reports_checkbox.setChecked(app_config.is_crash_reports_enabled())
        self.crash_reports_checkbox.toggled.connect(self._on_crash_reports_toggled)
        telemetry_layout.addWidget(self.crash_reports_checkbox)

        privacy_note = QLabel(
            "<span style='color: #666666; font-size: 8pt;'>"
            "No personal data, file names, or experimental data is ever collected."
            "</span>"
        )
        privacy_note.setWordWrap(True)
        telemetry_layout.addWidget(privacy_note)

        telemetry_group.setLayout(telemetry_layout)
        main_layout.addWidget(telemetry_group)

        main_layout.addStretch()

        return widget

    def _on_telemetry_toggled(self, checked):
        """Handle telemetry checkbox toggle."""
        from core import config as app_config
        app_config.set_telemetry_enabled(checked)

    def _on_crash_reports_toggled(self, checked):
        """Handle crash reports checkbox toggle."""
        from core import config as app_config
        app_config.set_crash_reports_enabled(checked)

    def _on_check_updates_clicked(self):
        """Handle manual check for updates."""
        self.update_label.setText('<span style="color: #888;">Checking for updates...</span>')
        self._check_for_updates_async(force_check=True)

    def _check_for_updates_async(self, force_check=False):
        """Check for updates in background thread."""
        from core import update_checker

        if self.update_info is not None and not force_check:
            self._show_update_result(self.update_info)
            return

        class UpdateChecker(QThread):
            update_checked = pyqtSignal(object)

            def run(self):
                info = update_checker.check_for_updates()
                self.update_checked.emit(info)

        def on_update_checked(update_info):
            self._show_update_result(update_info, force_check)
            if update_info:
                self.update_info = update_info

        self.update_thread = UpdateChecker()
        self.update_thread.update_checked.connect(on_update_checked)
        self.update_thread.start()

    def _show_update_result(self, update_info, was_manual_check=False):
        """Display update check result."""
        if update_info:
            version = update_info.get('version', 'Unknown')
            url = update_info.get('url', '')
            self.update_label.setText(
                f'<span style="color: #4CAF50; font-weight: bold;">Update available: v{version}</span> '
                f'<a href="{url}" style="color: #3daee9;">Download</a>'
            )
        elif was_manual_check:
            self.update_label.setText(
                f'<span style="color: #4CAF50;">You\'re up to date! (v{VERSION_STRING})</span>'
            )
        else:
            self.update_label.setText(
                f'<span style="color: #888888;">Current version: {VERSION_STRING}</span>'
            )

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QTextBrowser {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                padding: 10px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #4a4a4a;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 8px 20px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                border: 2px solid #3daee9;
                border-bottom: none;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background-color: #1e1e1e;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #3daee9;
            }
            QCheckBox {
                color: #e0e0e0;
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
