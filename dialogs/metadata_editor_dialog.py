"""
Metadata Editor Dialog for CageMetrics.

Provides a table-based editor for viewing and modifying experiment metadata,
adding custom fields, and saving data with updated metadata.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QLineEdit, QComboBox,
    QMessageBox, QGroupBox, QInputDialog, QAbstractItemView,
    QCheckBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metadata import (
    ALL_FIELDS, EXTENDED_FIELDS, STANDARD_FIELDS,
    get_display_columns, get_editable_fields, metadata_manager,
    copy_cagemate_metadata
)


class MetadataEditorDialog(QDialog):
    """Dialog for editing experiment metadata in a table format."""

    # Signal emitted when user clicks Save, returns True if save requested
    save_requested = pyqtSignal()

    # Dark theme colors
    BG_COLOR = '#2d2d2d'
    TEXT_COLOR = '#ffffff'
    HEADER_BG = '#3d3d3d'
    EDITABLE_BG = '#363636'
    READONLY_BG = '#2a2a2a'
    SELECTION_BG = '#0078d4'

    STYLE = """
        QDialog {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        QTableWidget {
            background-color: #2d2d2d;
            color: #ffffff;
            gridline-color: #4d4d4d;
            border: 1px solid #4d4d4d;
            selection-background-color: #0078d4;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget QHeaderView::section {
            background-color: #3d3d3d;
            color: #ffffff;
            padding: 6px;
            border: 1px solid #4d4d4d;
            font-weight: bold;
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
            background-color: #555555;
            color: #888888;
        }
        QLabel {
            color: #ffffff;
        }
        QLineEdit {
            background-color: #363636;
            color: #ffffff;
            border: 1px solid #4d4d4d;
            padding: 4px;
            border-radius: 3px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #4d4d4d;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """

    def __init__(self, analyzed_data: Dict[str, Dict], parent=None):
        super().__init__(parent)
        self.analyzed_data = analyzed_data
        self._modified = False
        self._custom_columns = []  # Track custom columns added

        self.setWindowTitle("Metadata Editor")
        self.setMinimumSize(1200, 600)
        self.setStyleSheet(self.STYLE)

        self._setup_ui()
        self._populate_table()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header with instructions
        header_label = QLabel(
            "Edit metadata for all experiments. Gray cells are auto-generated and read-only. "
            "White cells can be edited. Changes are applied when you click 'Apply Changes' or 'Save Data'."
        )
        header_label.setWordWrap(True)
        header_label.setStyleSheet("color: #aaaaaa; padding: 5px;")
        layout.addWidget(header_label)

        # Main table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)

        # Bottom section with custom field and buttons
        bottom_layout = QHBoxLayout()

        # Add Custom Field section
        custom_group = QGroupBox("Add Custom Field")
        custom_layout = QHBoxLayout(custom_group)
        custom_layout.setContentsMargins(10, 15, 10, 10)

        self.custom_field_edit = QLineEdit()
        self.custom_field_edit.setPlaceholderText("Field name (e.g., injection_site)")
        self.custom_field_edit.setFixedWidth(200)
        custom_layout.addWidget(self.custom_field_edit)

        self.add_field_btn = QPushButton("Add Column")
        self.add_field_btn.setFixedWidth(100)
        self.add_field_btn.clicked.connect(self._add_custom_field)
        custom_layout.addWidget(self.add_field_btn)

        custom_layout.addStretch()
        bottom_layout.addWidget(custom_group)

        bottom_layout.addStretch()

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setFixedWidth(120)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_changes)
        btn_layout.addWidget(self.apply_btn)

        self.save_btn = QPushButton("Save Data...")
        self.save_btn.setFixedWidth(120)
        self.save_btn.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(self.save_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.setFixedWidth(80)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.close_btn.clicked.connect(self._on_close)
        btn_layout.addWidget(self.close_btn)

        bottom_layout.addLayout(btn_layout)
        layout.addLayout(bottom_layout)

    def _get_columns(self) -> List[str]:
        """Get list of columns to display."""
        columns = get_display_columns()

        # Add any custom columns
        for col in self._custom_columns:
            if col not in columns:
                columns.append(col)

        # Check for any fields in analyzed_data that aren't in our list
        for animal_id, data in self.analyzed_data.items():
            metadata = data.get('metadata', {})
            for key in metadata.keys():
                if key not in columns and not key.startswith('_'):
                    columns.append(key)

        return columns

    def _populate_table(self):
        """Populate the table with experiment metadata."""
        self.table.blockSignals(True)

        animal_ids = list(self.analyzed_data.keys())
        columns = self._get_columns()

        self.table.setRowCount(len(animal_ids))
        self.table.setColumnCount(len(columns))

        # Set headers
        header_labels = []
        for col in columns:
            if col in ALL_FIELDS:
                header_labels.append(ALL_FIELDS[col]['label'])
            else:
                # Custom field - format nicely
                header_labels.append(col.replace('_', ' ').title())
        self.table.setHorizontalHeaderLabels(header_labels)

        # Populate rows
        editable_fields = get_editable_fields()

        for row, animal_id in enumerate(animal_ids):
            metadata = self.analyzed_data[animal_id].get('metadata', {})

            for col_idx, col_name in enumerate(columns):
                value = metadata.get(col_name, '')

                # Handle list values
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                else:
                    value = str(value) if value is not None else ''

                item = QTableWidgetItem(value)

                # Determine if editable
                is_editable = (col_name in editable_fields or
                               col_name in self._custom_columns or
                               col_name not in ALL_FIELDS)

                if is_editable:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(QColor(self.EDITABLE_BG))
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(QColor(self.READONLY_BG))
                    item.setForeground(QColor('#888888'))

                self.table.setItem(row, col_idx, item)

        # Resize columns
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Store column mapping for later use
        self._column_names = columns
        self._animal_ids = animal_ids

        self.table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int):
        """Handle cell value changes."""
        self._modified = True
        self.apply_btn.setEnabled(True)

    def _add_custom_field(self):
        """Add a custom metadata field column."""
        field_name = self.custom_field_edit.text().strip()

        if not field_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a field name.")
            return

        # Clean the name
        clean_name = field_name.lower().replace(' ', '_')

        # Check if already exists
        if clean_name in self._get_columns():
            QMessageBox.warning(self, "Field Exists",
                                f"A field named '{clean_name}' already exists.")
            return

        # Add to custom columns
        self._custom_columns.append(clean_name)
        metadata_manager.add_custom_field(clean_name)

        # Refresh table
        self._save_current_edits()
        self._populate_table()

        self.custom_field_edit.clear()
        self._modified = True
        self.apply_btn.setEnabled(True)

        QMessageBox.information(self, "Field Added",
                                f"Added custom field: {clean_name}")

    def _save_current_edits(self):
        """Save current table edits to analyzed_data."""
        for row, animal_id in enumerate(self._animal_ids):
            if animal_id not in self.analyzed_data:
                continue

            metadata = self.analyzed_data[animal_id].get('metadata', {})

            for col_idx, col_name in enumerate(self._column_names):
                item = self.table.item(row, col_idx)
                if item and (item.flags() & Qt.ItemFlag.ItemIsEditable):
                    value = item.text().strip()
                    if value:
                        metadata[col_name] = value
                    elif col_name in metadata:
                        # Don't delete standard fields, just leave empty
                        if col_name not in STANDARD_FIELDS:
                            del metadata[col_name]

            self.analyzed_data[animal_id]['metadata'] = metadata

    def _apply_changes(self):
        """Apply all changes and update cagemate metadata."""
        self._save_current_edits()

        # Propagate cagemate metadata
        metadata_manager.propagate_cagemate_metadata(self.analyzed_data)

        # Refresh table to show cagemate updates
        self._populate_table()

        self._modified = False
        self.apply_btn.setEnabled(False)

        QMessageBox.information(self, "Changes Applied",
                                "Metadata changes have been applied.\n"
                                "Cagemate metadata has been synchronized.")

    def _on_save_clicked(self):
        """Handle save button click."""
        if self._modified:
            # Apply changes first
            self._save_current_edits()
            metadata_manager.propagate_cagemate_metadata(self.analyzed_data)
            self._modified = False

        self.save_requested.emit()
        # Don't close - let parent handle the save dialog

    def _on_close(self):
        """Handle close button click."""
        if self._modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Apply them before closing?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._apply_changes()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.accept()

    def get_analyzed_data(self) -> Dict[str, Dict]:
        """Get the modified analyzed_data."""
        return self.analyzed_data


class QuickMetadataDialog(QDialog):
    """Simplified metadata dialog for quick edits to common fields."""

    STYLE = """
        QDialog {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        QLabel {
            color: #ffffff;
        }
        QLineEdit, QComboBox {
            background-color: #363636;
            color: #ffffff;
            border: 1px solid #4d4d4d;
            padding: 6px;
            border-radius: 3px;
            min-width: 150px;
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
    """

    def __init__(self, analyzed_data: Dict[str, Dict], parent=None):
        super().__init__(parent)
        self.analyzed_data = analyzed_data

        self.setWindowTitle("Quick Metadata Edit")
        self.setMinimumWidth(400)
        self.setStyleSheet(self.STYLE)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the simplified UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Get unique values for dropdowns
        treatments = metadata_manager.get_unique_values(self.analyzed_data, 'treatment')
        doses = metadata_manager.get_unique_values(self.analyzed_data, 'dose')

        # Batch edit fields
        batch_group = QGroupBox("Set for ALL experiments")
        batch_layout = QVBoxLayout(batch_group)

        # Treatment
        treat_layout = QHBoxLayout()
        treat_layout.addWidget(QLabel("Treatment:"))
        self.treatment_edit = QComboBox()
        self.treatment_edit.setEditable(True)
        self.treatment_edit.addItems([''] + sorted(treatments))
        treat_layout.addWidget(self.treatment_edit)
        batch_layout.addLayout(treat_layout)

        # Dose
        dose_layout = QHBoxLayout()
        dose_layout.addWidget(QLabel("Dose:"))
        self.dose_edit = QComboBox()
        self.dose_edit.setEditable(True)
        self.dose_edit.addItems([''] + sorted(doses))
        dose_layout.addWidget(self.dose_edit)
        batch_layout.addLayout(dose_layout)

        # Experiment date
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Experiment Date:"))
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("YYYY-MM-DD")
        date_layout.addWidget(self.date_edit)
        batch_layout.addLayout(date_layout)

        # Experimenter
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Experimenter:"))
        self.experimenter_edit = QLineEdit()
        exp_layout.addWidget(self.experimenter_edit)
        batch_layout.addLayout(exp_layout)

        layout.addWidget(batch_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        apply_btn = QPushButton("Apply to All")
        apply_btn.clicked.connect(self._apply_batch)
        btn_layout.addWidget(apply_btn)

        full_editor_btn = QPushButton("Full Editor...")
        full_editor_btn.clicked.connect(self._open_full_editor)
        btn_layout.addWidget(full_editor_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #555555;")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _apply_batch(self):
        """Apply batch edits to all experiments."""
        treatment = self.treatment_edit.currentText().strip()
        dose = self.dose_edit.currentText().strip()
        date = self.date_edit.text().strip()
        experimenter = self.experimenter_edit.text().strip()

        count = 0
        for animal_id, data in self.analyzed_data.items():
            metadata = data.get('metadata', {})

            if treatment:
                metadata['treatment'] = treatment
            if dose:
                metadata['dose'] = dose
            if date:
                metadata['experiment_date'] = date
            if experimenter:
                metadata['experimenter'] = experimenter

            data['metadata'] = metadata
            count += 1

        # Propagate cagemate metadata
        metadata_manager.propagate_cagemate_metadata(self.analyzed_data)

        QMessageBox.information(self, "Applied",
                                f"Updated metadata for {count} experiments.")
        self.accept()

    def _open_full_editor(self):
        """Open the full metadata editor."""
        self.done(2)  # Custom return code to signal full editor request
