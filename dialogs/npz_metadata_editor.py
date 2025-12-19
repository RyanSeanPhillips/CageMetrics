"""
NPZ Metadata Editor Dialog for CageMetrics Consolidation Tab.

Provides a table-based editor for viewing and modifying metadata in NPZ files,
with the ability to save changes back to the original files.
"""

import json
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QLineEdit, QComboBox,
    QMessageBox, QGroupBox, QAbstractItemView, QProgressDialog,
    QApplication, QMenu
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QAction
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metadata import (
    ALL_FIELDS, EXTENDED_FIELDS, STANDARD_FIELDS,
    get_editable_fields, metadata_manager
)


class NpzMetadataEditorDialog(QDialog):
    """Dialog for editing metadata in NPZ files from the consolidation tab."""

    # Dark theme colors
    BG_COLOR = '#2d2d2d'
    TEXT_COLOR = '#ffffff'
    HEADER_BG = '#3d3d3d'
    EDITABLE_BG = '#363636'
    READONLY_BG = '#2a2a2a'
    MODIFIED_BG = '#3d4a3d'  # Greenish tint for modified cells

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

    def __init__(self, npz_file_paths: List[str], parent=None):
        super().__init__(parent)
        self.npz_file_paths = npz_file_paths
        self._npz_data = {}  # path -> full npz data dict
        self._original_metadata = {}  # path -> original metadata (for change detection)
        self._modified_cells = set()  # (row, col) tuples of modified cells
        self._custom_columns = []

        self.setWindowTitle(f"Edit Metadata ({len(npz_file_paths)} files)")
        self.setMinimumSize(1200, 600)
        self.setStyleSheet(self.STYLE)

        self._setup_ui()
        self._load_npz_files()
        self._populate_table()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header with instructions
        header_label = QLabel(
            "Edit metadata for NPZ files. Gray cells are read-only. "
            "Modified cells are highlighted in green. Click 'Save Changes' to update the NPZ files."
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

        # Enable context menu on header for column deletion
        self.table.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._show_header_context_menu)

        layout.addWidget(self.table)

        # Bottom section
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

        self.delete_field_btn = QPushButton("Delete Column")
        self.delete_field_btn.setFixedWidth(110)
        self.delete_field_btn.setToolTip("Delete selected column (only if all cells are empty)")
        self.delete_field_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6b6b, stop:1 #ee5a5a);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff8080, stop:1 #ff6b6b);
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        self.delete_field_btn.clicked.connect(self._delete_selected_column)
        custom_layout.addWidget(self.delete_field_btn)

        custom_layout.addStretch()
        bottom_layout.addWidget(custom_group)

        bottom_layout.addStretch()

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaaaaa;")
        bottom_layout.addWidget(self.status_label)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.save_btn = QPushButton("Save Changes")
        self.save_btn.setFixedWidth(120)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_changes)
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

    def _load_npz_files(self):
        """Load metadata from all NPZ files."""
        for npz_path in self.npz_file_paths:
            try:
                data = np.load(npz_path, allow_pickle=True)

                # Extract metadata
                metadata = {}
                if 'metadata_json' in data:
                    metadata = json.loads(str(data['metadata_json']))

                self._npz_data[npz_path] = {
                    'metadata': metadata,
                    'npz_keys': list(data.keys())  # Store keys for later save
                }

                # Store original for change detection
                self._original_metadata[npz_path] = metadata.copy()

            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                self._npz_data[npz_path] = {'metadata': {}, 'npz_keys': []}
                self._original_metadata[npz_path] = {}

    def _get_columns(self) -> List[str]:
        """Get list of columns to display."""
        # Priority columns
        priority = [
            'animal_id', 'genotype', 'sex', 'cohort', 'cage_id',
            'treatment', 'dose', 'age', 'strain',
            'companion', 'cagemate_genotype',
            'experiment_date', 'experimenter', 'notes'
        ]

        columns = []
        for name in priority:
            columns.append(name)

        # Add custom columns
        for col in self._custom_columns:
            if col not in columns:
                columns.append(col)

        # Check for any additional fields in the NPZ files
        for npz_path, data in self._npz_data.items():
            metadata = data.get('metadata', {})
            for key in metadata.keys():
                if key not in columns and not key.startswith('_'):
                    columns.append(key)

        return columns

    def _populate_table(self):
        """Populate the table with NPZ metadata."""
        self.table.blockSignals(True)

        npz_paths = list(self._npz_data.keys())
        columns = self._get_columns()

        self.table.setRowCount(len(npz_paths))
        self.table.setColumnCount(len(columns))

        # Set headers
        header_labels = []
        for col in columns:
            if col in ALL_FIELDS:
                header_labels.append(ALL_FIELDS[col]['label'])
            else:
                header_labels.append(col.replace('_', ' ').title())
        self.table.setHorizontalHeaderLabels(header_labels)

        # Get editable fields
        editable_fields = get_editable_fields()

        for row, npz_path in enumerate(npz_paths):
            metadata = self._npz_data[npz_path].get('metadata', {})

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

        # Store mappings
        self._column_names = columns
        self._npz_paths_list = npz_paths

        self._update_status()
        self.table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int):
        """Handle cell value changes."""
        self._modified_cells.add((row, col))

        # Highlight modified cell
        item = self.table.item(row, col)
        if item:
            item.setBackground(QColor(self.MODIFIED_BG))

        self.save_btn.setEnabled(True)
        self._update_status()

    def _update_status(self):
        """Update the status label."""
        if self._modified_cells:
            self.status_label.setText(f"{len(self._modified_cells)} cell(s) modified")
        else:
            self.status_label.setText(f"{len(self._npz_data)} files loaded")

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
        self._populate_table()

        self.custom_field_edit.clear()
        QMessageBox.information(self, "Field Added",
                                f"Added custom field: {clean_name}")

    def _show_header_context_menu(self, pos):
        """Show context menu on column header right-click."""
        col_idx = self.table.horizontalHeader().logicalIndexAt(pos)
        if col_idx < 0 or col_idx >= len(self._column_names):
            return

        col_name = self._column_names[col_idx]

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #4d4d4d;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
            QMenu::item:disabled {
                color: #888888;
            }
        """)

        # Check if column can be deleted
        can_delete, reason = self._can_delete_column(col_idx)

        delete_action = menu.addAction(f"Delete '{col_name}' column")
        delete_action.setEnabled(can_delete)
        if not can_delete:
            delete_action.setToolTip(reason)

        action = menu.exec(self.table.horizontalHeader().mapToGlobal(pos))
        if action == delete_action and can_delete:
            self._delete_column(col_idx)

    def _delete_selected_column(self):
        """Delete the currently selected column."""
        # Get selected column from current selection
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection",
                                    "Please select a cell in the column you want to delete.")
            return

        col_idx = selected_items[0].column()
        can_delete, reason = self._can_delete_column(col_idx)

        if not can_delete:
            QMessageBox.warning(self, "Cannot Delete Column", reason)
            return

        col_name = self._column_names[col_idx]
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the '{col_name}' column?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._delete_column(col_idx)

    def _can_delete_column(self, col_idx: int) -> tuple:
        """
        Check if a column can be deleted.

        Returns:
            (can_delete: bool, reason: str)
        """
        if col_idx < 0 or col_idx >= len(self._column_names):
            return False, "Invalid column index."

        col_name = self._column_names[col_idx]

        # Standard fields cannot be deleted
        if col_name in STANDARD_FIELDS:
            return False, f"'{col_name}' is a standard metadata field and cannot be deleted."

        # Priority/important fields cannot be deleted
        protected_fields = ['animal_id', 'genotype', 'sex', 'cohort', 'cage_id',
                           'companion', 'cagemate_genotype']
        if col_name in protected_fields:
            return False, f"'{col_name}' is a core metadata field and cannot be deleted."

        # Check if all cells in this column are empty
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col_idx)
            if item and item.text().strip():
                return False, f"Cannot delete '{col_name}': column contains data.\n\nClear all values in the column first."

        return True, ""

    def _delete_column(self, col_idx: int):
        """Delete a column from the table and metadata."""
        col_name = self._column_names[col_idx]

        # Remove from custom columns if present
        if col_name in self._custom_columns:
            self._custom_columns.remove(col_name)

        # Remove from all metadata
        for npz_path in self._npz_data:
            metadata = self._npz_data[npz_path].get('metadata', {})
            if col_name in metadata:
                del metadata[col_name]

        # Mark all files as modified so the deletion gets saved
        for row in range(len(self._npz_paths_list)):
            self._modified_cells.add((row, 0))  # Mark row as modified

        self.save_btn.setEnabled(True)

        # Refresh table
        self._populate_table()

        QMessageBox.information(self, "Column Deleted",
                                f"Deleted column '{col_name}'.\n\n"
                                "Click 'Save Changes' to update the NPZ files.")

    def _save_changes(self):
        """Save all changes back to the NPZ files."""
        if not self._modified_cells:
            return

        # Collect changes per file
        files_to_update = set()
        for row, col in self._modified_cells:
            files_to_update.add(row)

        # Update metadata in memory first
        for row in files_to_update:
            npz_path = self._npz_paths_list[row]
            metadata = self._npz_data[npz_path].get('metadata', {})

            for col_idx, col_name in enumerate(self._column_names):
                item = self.table.item(row, col_idx)
                if item and (item.flags() & Qt.ItemFlag.ItemIsEditable):
                    value = item.text().strip()
                    if value:
                        metadata[col_name] = value
                    elif col_name in metadata and col_name not in STANDARD_FIELDS:
                        # Only remove non-standard fields if empty
                        del metadata[col_name]

            self._npz_data[npz_path]['metadata'] = metadata

        # Show progress dialog
        progress = QProgressDialog("Saving changes to NPZ files...", "Cancel", 0, len(files_to_update), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        # Save to files
        saved_count = 0
        errors = []

        for idx, row in enumerate(files_to_update):
            if progress.wasCanceled():
                break

            npz_path = self._npz_paths_list[row]
            progress.setValue(idx)
            progress.setLabelText(f"Saving {Path(npz_path).name}...")
            QApplication.processEvents()

            try:
                self._update_npz_file(npz_path)
                saved_count += 1
            except Exception as e:
                errors.append(f"{Path(npz_path).name}: {str(e)}")

        progress.setValue(len(files_to_update))

        # Clear modified state
        self._modified_cells.clear()

        # Reset cell backgrounds
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    col_name = self._column_names[col]
                    editable_fields = get_editable_fields()
                    is_editable = (col_name in editable_fields or
                                   col_name in self._custom_columns or
                                   col_name not in ALL_FIELDS)
                    if is_editable:
                        item.setBackground(QColor(self.EDITABLE_BG))
                    else:
                        item.setBackground(QColor(self.READONLY_BG))

        self.save_btn.setEnabled(False)
        self._update_status()

        # Show result
        if errors:
            QMessageBox.warning(self, "Save Complete with Errors",
                                f"Saved {saved_count} file(s).\n\nErrors:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Save Complete",
                                    f"Successfully saved changes to {saved_count} file(s).")

    def _update_npz_file(self, npz_path: str):
        """Update a single NPZ file with new metadata."""
        # Load all existing data
        original_data = np.load(npz_path, allow_pickle=True)

        # Build new data dict with all original arrays
        new_data = {}
        for key in original_data.keys():
            if key != 'metadata_json':  # We'll update this
                new_data[key] = original_data[key]

        # Update metadata
        metadata = self._npz_data[npz_path].get('metadata', {})
        new_data['metadata_json'] = json.dumps(self._convert_to_serializable(metadata))

        # Close original file before overwriting
        original_data.close()

        # Save back to same file
        np.savez_compressed(npz_path, **new_data)

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _on_close(self):
        """Handle close button click."""
        if self._modified_cells:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save them before closing?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_changes()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.accept()

    def get_modified_paths(self) -> List[str]:
        """Get list of NPZ paths that were modified."""
        return [self._npz_paths_list[row] for row, col in self._modified_cells]
