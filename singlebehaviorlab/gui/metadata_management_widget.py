"""
Metadata Management Widget for SingleBehavior Lab.
Allows adding columns, managing classes, and assigning values to videos/objects.
"""

import os
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QMessageBox, QGroupBox, QFileDialog,
    QHeaderView, QDialog, QDialogButtonBox, QFormLayout, QSpinBox,
    QTextEdit, QListWidget, QListWidgetItem, QSplitter, QScrollArea,
    QTabWidget, QFrame, QSizePolicy, QTableView, QAbstractItemView,
    QProgressBar, QApplication, QTableWidget, QTableWidgetItem, QCheckBox
)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from PyQt6.QtGui import QFont


class AddColumnDialog(QDialog):
    """Dialog for adding a new categorical column to metadata."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add new category column")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        info = QLabel("Add a new categorical column. Define the allowed options (classes) for this column.")
        info.setStyleSheet("color: gray; font-style: italic; margin-bottom: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        self.column_name_edit = QLineEdit()
        self.column_name_edit.setPlaceholderText("e.g., Treatment, Condition, Genotype")
        form.addRow("Column Name:", self.column_name_edit)
        
        self.categories_edit = QLineEdit()
        self.categories_edit.setPlaceholderText("Comma-separated options (e.g., Control, Drug, Placebo)")
        form.addRow("Options (Classes):", self.categories_edit)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_column_info(self):
        """Get column information from dialog."""
        name = self.column_name_edit.text().strip()
        categories_text = self.categories_edit.text().strip()
        categories = [c.strip() for c in categories_text.split(",") if c.strip()]
        
        return name, categories


class PandasTableModel(QAbstractTableModel):
    """Lightweight pandas-backed table model with paging."""

    def __init__(self, df: pd.DataFrame, start: int = 0, end: int = 0, parent=None):
        super().__init__(parent)
        self.df = df
        self.start = start
        self.end = end if end else len(df)

    def update_range(self, start: int, end: int):
        self.start = start
        self.end = min(end, len(self.df))
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()) -> int:
        return max(0, self.end - self.start)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if self.df is None else self.df.shape[1]

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.df is None:
            return QVariant()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            try:
                value = self.df.iat[self.start + index.row(), index.column()]
                return "" if pd.isna(value) else str(value)
            except Exception:
                return QVariant()
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or self.df is None:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            try:
                return self.df.columns[section]
            except Exception:
                return QVariant()
        else:
            return section + self.start + 1

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or self.df is None or not index.isValid():
            return False
        r = self.start + index.row()
        c = index.column()
        col_name = self.df.columns[c]

        new_val = value
        # Enforce category constraints
        if isinstance(self.df[col_name].dtype, pd.CategoricalDtype):
            cats = list(self.df[col_name].dtype.categories)
            if new_val not in cats:
                return False
        else:
            # Try to cast numeric columns
            if self.df[col_name].dtype in [np.int64, np.float64]:
                try:
                    new_val = float(value) if value != "" else np.nan
                except Exception:
                    return False

        try:
            self.df.iat[r, c] = new_val
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            return True
        except Exception:
            return False


class EditColumnDialog(QDialog):
    """Dialog for editing column name and renaming values within a column."""
    
    def __init__(self, column_name: str, metadata: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.column_name = column_name
        self.metadata = metadata
        self.setWindowTitle(f"Edit column: {column_name}")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout(self)
        
        # Column name section
        name_group = QGroupBox("Column name")
        name_layout = QVBoxLayout()
        self.column_name_edit = QLineEdit(column_name)
        name_layout.addWidget(QLabel("Rename column to:"))
        name_layout.addWidget(self.column_name_edit)
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)
        
        # Value renaming section
        values_group = QGroupBox("Rename values (Classes)")
        values_layout = QVBoxLayout()
        
        info_label = QLabel("Rename unique values in this column. Leave unchanged to keep original value.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-style: italic;")
        values_layout.addWidget(info_label)
        
        # Get unique values
        unique_vals = sorted(self.metadata[column_name].dropna().unique().astype(str))
        
        # Scroll area for value edits
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(5)
        
        self.value_edits = {}
        for val in unique_vals:
            row = QHBoxLayout()
            old_label = QLabel(f"{val} →")
            old_label.setMinimumWidth(150)
            old_label.setStyleSheet("font-weight: bold;")
            new_edit = QLineEdit(val)
            new_edit.setPlaceholderText("New name (leave unchanged to keep)")
            self.value_edits[val] = new_edit
            row.addWidget(old_label)
            row.addWidget(new_edit)
            scroll_layout.addLayout(row)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        values_layout.addWidget(scroll)
        
        values_group.setLayout(values_layout)
        layout.addWidget(values_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_changes(self):
        """Get the changes: (new_column_name, value_mapping_dict)."""
        new_name = self.column_name_edit.text().strip()
        if not new_name:
            new_name = self.column_name
            
        # Build value mapping
        value_map = {}
        for old_val, edit in self.value_edits.items():
            new_val = edit.text().strip()
            if new_val and new_val != old_val:
                value_map[old_val] = new_val
                
        return new_name, value_map


class MetadataManagementDialog(QDialog):
    """Dialog for managing metadata columns and assignments."""
    
    def __init__(self, metadata: pd.DataFrame, metadata_file_path: str, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.metadata = metadata.copy()
        self.metadata_file_path = metadata_file_path
        
        self.setWindowTitle("Metadata management")
        self.resize(1200, 800)
        
        self._setup_ui()
        
    def get_metadata(self):
        return self.metadata
        
    def get_metadata_path(self):
        return self.metadata_file_path
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 6)  # Minimal margins
        layout.setSpacing(2)  # Tight vertical spacing
        
        # -- Header --
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        title = QLabel("Metadata Editor")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setContentsMargins(0, 0, 0, 0)
        title.setStyleSheet("margin: 0px; padding: 0px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Load status
        self.status_label = QLabel()
        if self.metadata_file_path:
             self.status_label.setText(f"Editing: {os.path.basename(self.metadata_file_path)}")
        else:
             self.status_label.setText("Editing: New/Unsaved Metadata")
        self.status_label.setStyleSheet("color: gray; margin: 0px; padding: 0px;")
        header_layout.addWidget(self.status_label)
        
        layout.addLayout(header_layout)
        
        # -- Main Content (Splitter) --
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(0, 0, 0, 0)
        
        # 1. Left Panel: Controls (Tabbed)
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        self.tabs = QTabWidget()
        
        # Tab 1: Manage Structure (Columns) - FIRST
        self.structure_tab = QWidget()
        self._setup_structure_tab()
        self.tabs.addTab(self.structure_tab, "Manage Columns")
        
        # Tab 2: Edit Data (Bulk & Quick Actions) - SECOND
        self.edit_tab = QWidget()
        self._setup_edit_tab()
        self.tabs.addTab(self.edit_tab, "Edit Values")
        
        controls_layout.addWidget(self.tabs)
        
        # Save/Load buttons at bottom of controls
        file_group = QGroupBox("File operations")
        file_layout = QVBoxLayout()
        
        hbox_load = QHBoxLayout()
        self.load_btn = QPushButton("Load experiment data")
        self.load_btn.clicked.connect(self.load_metadata)
        self.load_file_btn = QPushButton("Load CSV...")
        self.load_file_btn.clicked.connect(self.load_external_metadata)
        hbox_load.addWidget(self.load_btn)
        hbox_load.addWidget(self.load_file_btn)
        file_layout.addLayout(hbox_load)
        
        self.save_btn = QPushButton("Save changes to file")
        self.save_btn.setStyleSheet("font-weight: bold;")
        self.save_btn.clicked.connect(self.save_metadata)
        file_layout.addWidget(self.save_btn)
        
        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)
        
        controls_panel.setMinimumWidth(350)
        controls_panel.setMaximumWidth(400)
        splitter.addWidget(controls_panel)
        
        # 2. Right Panel: Data Table
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)  # No spacing
        table_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        preview_label = QLabel("<b>Data preview</b>")
        preview_label.setContentsMargins(0, 0, 0, 0)
        preview_label.setStyleSheet("margin: 0px; padding: 0px 0px 2px 0px;")
        preview_label.setMaximumHeight(18)
        table_layout.addWidget(preview_label)
        
        # Paging controls
        paging_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton("Prev")
        self.next_page_btn = QPushButton("Next")
        self.page_info_label = QLabel("")
        self.page_size_spin = QSpinBox()
        self.page_size_spin.setRange(100, 20000)
        self.page_size_spin.setSingleStep(500)
        self.page_size_spin.setValue(5000)
        paging_layout.addWidget(self.prev_page_btn)
        paging_layout.addWidget(self.next_page_btn)
        paging_layout.addWidget(self.page_info_label, 1)
        paging_layout.addWidget(QLabel("Page size:"))
        paging_layout.addWidget(self.page_size_spin)
        table_layout.addLayout(paging_layout)

        # Table view with model
        self.table_model = PandasTableModel(self.metadata if hasattr(self, "metadata") else pd.DataFrame())
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.SelectedClicked)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        table_layout.addWidget(self.table, 1)  # Give table stretch factor

        # Progress bar for heavy operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        table_layout.addWidget(self.progress_bar)
        
        table_panel.setLayout(table_layout)
        splitter.addWidget(table_panel)
        
        splitter.setStretchFactor(1, 1) # Give table more space
        splitter.setCollapsible(0, False)
        layout.addWidget(splitter, 1)
        
        # -- Footer Buttons --
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initial Load
        self.page_size = self.page_size_spin.value()
        self.current_page = 0
        self.prev_page_btn.clicked.connect(self._prev_page)
        self.next_page_btn.clicked.connect(self._next_page)
        self.page_size_spin.valueChanged.connect(self._on_page_size_changed)
        if self.metadata is not None:
            self._update_table()
            self._update_combos()
            self._update_paging()
            
    def _setup_edit_tab(self):
        layout = QVBoxLayout(self.edit_tab)
        layout.setSpacing(15)
        
        # Initialize filter rows list
        self.filter_rows = []
        
        # --- Bulk Assignment ---
        bulk_group = QGroupBox("Bulk edit rule")
        bulk_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        bulk_layout = QVBoxLayout()
        
        # Sentence builder style
        
        # Row 1: "Set [Target Column] to [Value]"
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Set column"))
        self.target_column_combo = QComboBox()
        row1.addWidget(self.target_column_combo, 1)
        row1.addWidget(QLabel("to value"))
        bulk_layout.addLayout(row1)
        
        self.target_value_combo = QComboBox()
        self.target_value_combo.setEditable(True)  # allow free text but suggest classes
        self.target_value_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.target_value_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.target_value_combo.setMinimumHeight(26)
        bulk_layout.addWidget(self.target_value_combo)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        bulk_layout.addWidget(line)
        
        # Conditions Header
        bulk_layout.addWidget(QLabel("Conditions (Match ALL):"))
        
        # Scroll area for conditions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(150)
        
        self.conditions_container = QWidget()
        self.conditions_layout = QVBoxLayout(self.conditions_container)
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)
        self.conditions_layout.setSpacing(5)
        self.conditions_layout.addStretch()  # Push items up
        
        scroll.setWidget(self.conditions_container)
        bulk_layout.addWidget(scroll)
        
        # Add Condition Button
        add_cond_btn = QPushButton("+ Add Condition")
        add_cond_btn.clicked.connect(self.add_condition_row)
        bulk_layout.addWidget(add_cond_btn)
        
        # Apply Button
        self.apply_bulk_btn = QPushButton("Apply rule")
        self.apply_bulk_btn.setStyleSheet("background-color: #e1f5fe; color: #0277bd; border: 1px solid #0277bd; padding: 5px; font-weight: bold;")
        self.apply_bulk_btn.clicked.connect(self.apply_bulk_assignment)
        bulk_layout.addWidget(self.apply_bulk_btn)
        
        bulk_group.setLayout(bulk_layout)
        layout.addWidget(bulk_group)
        
        # --- Quick Helpers ---
        quick_group = QGroupBox("Quick helpers")
        quick_layout = QVBoxLayout()
        
        self.assign_by_video_btn = QPushButton("Set values per video...")
        self.assign_by_video_btn.setToolTip("Assign a specific value to a column for all rows belonging to a specific video.")
        self.assign_by_video_btn.clicked.connect(self.assign_by_video)
        quick_layout.addWidget(self.assign_by_video_btn)
        
        self.assign_by_object_btn = QPushButton("Set values per object...")
        self.assign_by_object_btn.setToolTip("Assign a specific value to a column for all rows belonging to a specific object ID.")
        self.assign_by_object_btn.clicked.connect(self.assign_by_object)
        quick_layout.addWidget(self.assign_by_object_btn)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        layout.addStretch()
        
    def _setup_structure_tab(self):
        layout = QVBoxLayout(self.structure_tab)
        layout.setSpacing(10)
        
        info = QLabel("Manage the columns in your metadata table.")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.columns_list = QListWidget()
        self.columns_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.columns_list)
        
        btn_layout = QHBoxLayout()
        self.add_col_btn = QPushButton("Add")
        self.add_col_btn.clicked.connect(self.add_column)
        self.edit_col_btn = QPushButton("Edit")
        self.edit_col_btn.clicked.connect(self.edit_column)
        self.remove_col_btn = QPushButton("Delete")
        self.remove_col_btn.clicked.connect(self.remove_column)
        
        btn_layout.addWidget(self.add_col_btn)
        btn_layout.addWidget(self.edit_col_btn)
        btn_layout.addWidget(self.remove_col_btn)
        layout.addLayout(btn_layout)
        
    def load_metadata(self):
        """Load metadata from experiment folder."""
        experiment_path = self.config.get("experiment_path")
        if not experiment_path:
            QMessageBox.warning(self, "No Experiment", "Please create or load an experiment first.")
            return
            
        registered_clips_dir = os.path.join(experiment_path, "registered_clips")
        if not os.path.exists(registered_clips_dir):
            QMessageBox.warning(self, "No Data", "No registered clips directory found.")
            return
            
        # Look for behaviorome metadata CSV
        csv_files = [f for f in os.listdir(registered_clips_dir) 
                    if f.startswith("behaviorome_") and f.endswith("_metadata.csv")]
        
        if not csv_files:
            QMessageBox.warning(self, "No Data", "No metadata files found.")
            return
            
        csv_files.sort(reverse=True)
        metadata_file = os.path.join(registered_clips_dir, csv_files[0])
        self._load_metadata_file(metadata_file)
        
    def load_external_metadata(self):
        """Load external metadata CSV."""
        metadata_path, _ = QFileDialog.getOpenFileName(self, "Open Metadata CSV", "", "CSV Files (*.csv)")
        if metadata_path:
            self._load_metadata_file(metadata_path)
    
    def accept(self):
        """Override accept to save metadata before closing."""
        if self.metadata_file_path:
            try:
                self._save_metadata_to_file(self.metadata, self.metadata_file_path)
            except Exception as e:
                QMessageBox.warning(self, "Save Warning", f"Could not save metadata: {e}")
        
        super().accept()
            
    def _load_metadata_file(self, file_path):
        try:
            self._set_busy(True, "Loading metadata...")
            self.metadata = pd.read_csv(file_path)
            self.metadata_file_path = file_path
            
            # Restore categorical columns if they exist
            # Check if any columns are already categorical
            for col in self.metadata.columns:
                if isinstance(self.metadata[col].dtype, pd.CategoricalDtype):
                    # Categories are already preserved in Categorical dtype
                    pass
            
            self._update_table()
            self._update_combos()
            self._update_paging()
            self.status_label.setText(f"Editing: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load metadata: {e}")
        finally:
            self._set_busy(False)
            
    def _update_table(self):
        if self.metadata is None:
            return
        self._update_paging()
                
    def _update_combos(self):
        if self.metadata is None:
            return
            
        columns = self.metadata.columns.tolist()
        
        self.target_column_combo.blockSignals(True)
        self.target_column_combo.clear()
        self.target_column_combo.addItems(columns)
        self.target_column_combo.blockSignals(False)
        
        # Columns list
        self.columns_list.clear()
        self.columns_list.addItems(columns)
        
        # Update existing filter rows
        for row in self.filter_rows:
            curr = row['combo'].currentText()
            row['combo'].blockSignals(True)
            row['combo'].clear()
            row['combo'].addItems(columns)
            if curr in columns:
                row['combo'].setCurrentText(curr)
            row['combo'].blockSignals(False)
        
        self._update_target_values()
        self.target_column_combo.currentTextChanged.connect(self._update_target_values)
        
    def add_condition_row(self):
        """Add a new filter condition row."""
        if self.metadata is None:
            return
            
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        # Column combo
        combo = QComboBox()
        combo.addItems(self.metadata.columns.tolist())
        row_layout.addWidget(combo, 1)
        
        # Values button
        val_btn = QPushButton("Select Values...")
        row_layout.addWidget(val_btn, 2)
        
        # Remove button
        remove_btn = QPushButton("X")
        remove_btn.setMaximumWidth(30)
        remove_btn.setStyleSheet("color: red; font-weight: bold;")
        row_layout.addWidget(remove_btn)
        
        # Row data structure
        row_data = {
            'widget': row_widget,
            'combo': combo,
            'val_btn': val_btn,
            'values': set()  # Selected values
        }
        
        # Connect signals
        val_btn.clicked.connect(lambda: self.open_values_dialog(row_data))
        remove_btn.clicked.connect(lambda: self.remove_condition_row(row_data))
        combo.currentTextChanged.connect(lambda: self.reset_row_values(row_data))
        
        # Insert before the stretch item
        self.conditions_layout.insertWidget(self.conditions_layout.count() - 1, row_widget)
        self.filter_rows.append(row_data)
        
    def remove_condition_row(self, row_data):
        """Remove a filter row."""
        if row_data in self.filter_rows:
            self.filter_rows.remove(row_data)
            row_data['widget'].deleteLater()
            
    def reset_row_values(self, row_data):
        """Reset values when column changes."""
        row_data['values'] = set()
        row_data['val_btn'].setText("Select Values...")
        
    def open_values_dialog(self, row_data):
        """Open dialog with checkboxes to select values."""
        col = row_data['combo'].currentText()
        if not col or col not in self.metadata.columns:
            return
            
            unique_vals = sorted(self.metadata[col].dropna().unique().astype(str))
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select values for '{col}'")
        dialog.setMinimumWidth(300)
        dialog.setMinimumHeight(400)
        layout = QVBoxLayout(dialog)
        
        # Search/Filter
        search_edit = QLineEdit()
        search_edit.setPlaceholderText("Search...")
        layout.addWidget(search_edit)
        
        # Checkboxes area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        chk_layout = QVBoxLayout(container)
        chk_layout.setSpacing(2)
        
        checkboxes = []
        for val in unique_vals:
            chk = QCheckBox(val)
            if val in row_data['values']:
                chk.setChecked(True)
            chk_layout.addWidget(chk)
            checkboxes.append(chk)

        chk_layout.addStretch()
        container.setLayout(chk_layout)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Filter logic
        def filter_items(text):
            text = text.lower()
            for chk in checkboxes:
                chk.setVisible(text in chk.text().lower())
        search_edit.textChanged.connect(filter_items)
        
        # Select All / None
        btn_row = QHBoxLayout()
        all_btn = QPushButton("All")
        none_btn = QPushButton("None")
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        layout.addLayout(btn_row)
        
        def select_all():
            for chk in checkboxes:
                if chk.isVisible(): chk.setChecked(True)
        def select_none():
            for chk in checkboxes:
                if chk.isVisible(): chk.setChecked(False)
                
        all_btn.clicked.connect(select_all)
        none_btn.clicked.connect(select_none)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_selection = set()
            for chk in checkboxes:
                if chk.isChecked():
                    new_selection.add(chk.text())
            
            row_data['values'] = new_selection
            
            # Update button text
            if not new_selection:
                row_data['val_btn'].setText("Select Values...")
            elif len(new_selection) <= 3:
                row_data['val_btn'].setText(", ".join(sorted(new_selection)))
            else:
                row_data['val_btn'].setText(f"{len(new_selection)} selected")

    def _update_target_values(self):
        """Populate target value combo with existing values when column is categorical."""
        if self.metadata is None:
            return
        col = self.target_column_combo.currentText()
        if col and col in self.metadata.columns:
            # Check if column is categorical - use categories if available
            if isinstance(self.metadata[col].dtype, pd.CategoricalDtype):
                # Use the defined categories
                categories = list(self.metadata[col].dtype.categories)
                self.target_value_combo.blockSignals(True)
                self.target_value_combo.clear()
                self.target_value_combo.addItems(categories)
                self.target_value_combo.setEditText("")  # allow user to type new value
                self.target_value_combo.blockSignals(False)
            else:
                # Regular column - use unique values
                unique_vals = sorted(self.metadata[col].dropna().unique().astype(str))
                self.target_value_combo.blockSignals(True)
                self.target_value_combo.clear()
                self.target_value_combo.addItems(unique_vals)
                self.target_value_combo.setEditText("")  # allow user to type new value
                self.target_value_combo.blockSignals(False)

    def _update_target_values(self):
        """Populate target value combo with existing values when column is categorical."""
        if self.metadata is None:
            return
        col = self.target_column_combo.currentText()
        if col and col in self.metadata.columns:
            # Check if column is categorical - use categories if available
            if isinstance(self.metadata[col].dtype, pd.CategoricalDtype):
                # Use the defined categories
                categories = list(self.metadata[col].dtype.categories)
                self.target_value_combo.blockSignals(True)
                self.target_value_combo.clear()
                self.target_value_combo.addItems(categories)
                self.target_value_combo.setEditText("")  # allow user to type new value
                self.target_value_combo.blockSignals(False)
            else:
                # Regular column - use unique values
                unique_vals = sorted(self.metadata[col].dropna().unique().astype(str))
                self.target_value_combo.blockSignals(True)
                self.target_value_combo.clear()
                self.target_value_combo.addItems(unique_vals)
                self.target_value_combo.setEditText("")  # allow user to type new value
                self.target_value_combo.blockSignals(False)

    def _open_filter_value_dialog(self, item: QTableWidgetItem):
        """Open dialog to select filter values for a column."""
        if item.column() != 1:  # Only handle the values column
            return
        
        col = item.data(Qt.ItemDataRole.UserRole)
        if not col or col not in self.metadata.columns:
            return
        
        # Get unique values for this column
        unique_vals = sorted(self.metadata[col].dropna().unique().astype(str))
        
        # Create dialog with multi-select list
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select values for '{col}'")
        dialog.setMinimumWidth(300)
        dialog.setMinimumHeight(400)
        layout = QVBoxLayout(dialog)
        
        info = QLabel("Select one or more values (Ctrl+Click for multiple):")
        layout.addWidget(info)
        
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for val in unique_vals:
            list_widget.addItem(val)
        
        # Pre-select previously selected values
        if not hasattr(self, 'filter_values_dict'):
            self.filter_values_dict = {}
        if col in self.filter_values_dict:
            for i in range(list_widget.count()):
                item_widget = list_widget.item(i)
                if item_widget.text() in self.filter_values_dict[col]:
                    item_widget.setSelected(True)
        
        layout.addWidget(list_widget)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_items = list_widget.selectedItems()
            selected_vals = [item.text() for item in selected_items]
            self.filter_values_dict[col] = selected_vals
            # Update the table display
            val_text = ", ".join(selected_vals) if selected_vals else "(click to select)"
            item.setText(val_text)

    def _update_paging(self):
        """Refresh table model to current page."""
        total = len(self.metadata) if self.metadata is not None else 0
        if total == 0:
            self.table_model.update_range(0, 0)
            self._update_paging_label(total, 0, 0)
            return
        self.page_size = self.page_size_spin.value()
        max_page = max(0, (total - 1) // self.page_size)
        if self.current_page > max_page:
            self.current_page = max_page
        start = self.current_page * self.page_size
        end = min(start + self.page_size, total)
        self.table_model.df = self.metadata
        self.table_model.update_range(start, end)
        self._update_paging_label(total, start, end)

    def _update_paging_label(self, total: int, start: int, end: int):
        if total == 0:
            self.page_info_label.setText("No rows")
        else:
            page_num = self.current_page + 1
            page_count = max(1, (total + self.page_size - 1) // self.page_size)
            self.page_info_label.setText(f"Page {page_num}/{page_count}  rows {start+1}–{end} of {total}")
        self.prev_page_btn.setEnabled(self.current_page > 0)
        self.next_page_btn.setEnabled(end < total)

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._update_paging()

    def _next_page(self):
        total = len(self.metadata) if self.metadata is not None else 0
        if (self.current_page + 1) * self.page_size < total:
            self.current_page += 1
            self._update_paging()

    def _on_page_size_changed(self, val: int):
        self.page_size = val
        self.current_page = 0
        self._update_paging()

    def _set_busy(self, busy: bool, message: str = "Working..."):
        """Show/hide the progress bar for long operations."""
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setFormat(message)
        QApplication.processEvents()
            
    def add_column(self):
        if self.metadata is None:
            return
            
        dialog = AddColumnDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, categories = dialog.get_column_info()
            
            if not name or name in self.metadata.columns:
                QMessageBox.warning(self, "Error", "Invalid or duplicate column name.")
                return
                
            if not categories:
                QMessageBox.warning(self, "Error", "Please provide at least one option (class) for the category column.")
                return
            
            # Create column with Categorical dtype to preserve categories
            self.metadata[name] = pd.Categorical([categories[0]] * len(self.metadata), categories=categories)

            # Persist categories in DataFrame.attrs (pandas 1.5+).
            if not hasattr(self.metadata, 'attrs'):
                pass
            else:
                if '_column_categories' not in self.metadata.attrs:
                    self.metadata.attrs['_column_categories'] = {}
                self.metadata.attrs['_column_categories'][name] = categories
                 
            self._update_table()
            self._update_combos()
            
            QMessageBox.information(self, "Success", f"Category column '{name}' created with {len(categories)} options: {', '.join(categories)}")
            
    def edit_column(self):
        """Edit column name and rename values within the column."""
        if self.metadata is None:
            QMessageBox.warning(self, "No Data", "Please load metadata first.")
            return
            
        item = self.columns_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Selection", "Please select a column from the list to edit.")
            return
            
        col = item.text()
        
        # Open edit dialog
        dialog = EditColumnDialog(col, self.metadata, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name, value_map = dialog.get_changes()
            
            # Apply value renamings first (before column rename)
            if value_map:
                self.metadata[col] = self.metadata[col].astype(str).replace(value_map)
                QMessageBox.information(self, "Values Updated", f"Renamed {len(value_map)} value(s) in column '{col}'.")
            
            # Rename column if changed
            if new_name != col:
                if new_name in self.metadata.columns:
                    QMessageBox.warning(self, "Duplicate", f"Column '{new_name}' already exists.")
                    return
                self.metadata.rename(columns={col: new_name}, inplace=True)
                QMessageBox.information(self, "Column Renamed", f"Column '{col}' renamed to '{new_name}'.")
            
            # Refresh UI
            self._update_table()
            self._update_combos()
            
    def remove_column(self):
        if self.metadata is None:
            return
            
        item = self.columns_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Selection", "Please select a column from the list to remove.")
            return
            
        col = item.text()
        if col in ['snippet', 'span_id', 'video_id', 'object_id', 'clip_index']:  # span_id for backward compatibility
            QMessageBox.warning(self, "Protected", f"Cannot remove essential column '{col}'.")
            return
            
        reply = QMessageBox.question(self, "Confirm", f"Remove column '{col}'?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.metadata = self.metadata.drop(columns=[col])
            self._update_table()
            self._update_combos()
            
    def apply_bulk_assignment(self):
        if self.metadata is None:
            return
            
        target_col = self.target_column_combo.currentText()
        target_val = self.target_value_combo.currentText().strip()
        
        if not self.filter_rows:
            QMessageBox.information(self, "No Filter", "Please add at least one condition.")
            return

        if not target_col:
            QMessageBox.information(self, "No Target", "Please select a target column.")
            return

        # Check for missing values in rows
        missing_cols = []
        for row in self.filter_rows:
            if not row['values']:
                missing_cols.append(row['combo'].currentText())
        
        if missing_cols:
            QMessageBox.information(
                self, 
                "Missing Values", 
                f"Please select values for the following columns:\n{', '.join(missing_cols)}"
            )
            return

        # Build combined mask: all conditions must match (AND logic)
        mask = pd.Series([True] * len(self.metadata), index=self.metadata.index)
        for row in self.filter_rows:
            col = row['combo'].currentText()
            filter_vals = list(row['values'])
            col_mask = self.metadata[col].astype(str).isin(filter_vals)
            mask = mask & col_mask
        
        count = mask.sum()
        
        if count == 0:
            QMessageBox.information(self, "No Matches", "No rows matched all the criteria.")
            return
        
        # Validate target value for categorical columns
        if isinstance(self.metadata[target_col].dtype, pd.CategoricalDtype):
            if target_val not in self.metadata[target_col].dtype.categories:
                QMessageBox.warning(self, "Invalid Category", 
                                   f"Value '{target_val}' is not a valid option for column '{target_col}'.\n"
                                   f"Valid options are: {', '.join(self.metadata[target_col].dtype.categories)}")
                return
        # Numeric conversion
        elif self.metadata[target_col].dtype in [np.int64, np.float64]:
            try:
                target_val = float(target_val)
            except:
                QMessageBox.warning(self, "Invalid", "Target column is numeric.")
                return
                
        self._set_busy(True, "Applying rule...")
        try:
            self.metadata.loc[mask, target_col] = target_val
            self._update_table()
            QMessageBox.information(self, "Success", f"Updated {count} rows.")
        finally:
            self._set_busy(False)
        
    def assign_by_video(self):
        """Assign by video helper."""
        if self.metadata is None or 'video_id' not in self.metadata.columns:
            return
            
        from PyQt6.QtWidgets import QInputDialog
        target_col, ok = QInputDialog.getItem(self, "Select Column", "Target column to set:", 
                                            self.metadata.columns.tolist(), 0, False)
        if not ok: return
        
        unique_videos = sorted(self.metadata['video_id'].dropna().unique())
        for video_id in unique_videos:
            count = (self.metadata['video_id'] == video_id).sum()
            val, ok = QInputDialog.getText(self, f"Assign: {video_id}", f"Value for '{target_col}' ({count} rows):")
            if ok and val:
                self.metadata.loc[self.metadata['video_id'] == video_id, target_col] = val
        self._update_table()
        
    def assign_by_object(self):
        """Assign by object helper."""
        if self.metadata is None or 'object_id' not in self.metadata.columns:
            return
            
        from PyQt6.QtWidgets import QInputDialog
        target_col, ok = QInputDialog.getItem(self, "Select Column", "Target column to set:", 
                                            self.metadata.columns.tolist(), 0, False)
        if not ok: return
        
        unique_objs = sorted(self.metadata['object_id'].dropna().unique())
        for obj in unique_objs:
            count = (self.metadata['object_id'].astype(str) == str(obj)).sum()
            val, ok = QInputDialog.getText(self, f"Assign: Object {obj}", f"Value for '{target_col}' ({count} rows):")
            if ok and val:
                self.metadata.loc[self.metadata['object_id'].astype(str) == str(obj), target_col] = val
        self._update_table()
        
    def _sync_table_to_metadata(self):
        # No-op because edits are committed directly via the model
        return

    def _save_metadata_to_file(self, df: pd.DataFrame, path: str):
        """Save metadata DataFrame to file, respecting the file format based on extension."""
        if path.endswith(".npz"):
            # Save as NPZ format
            np.savez_compressed(
                path,
                metadata=df.values,
                columns=np.array(df.columns, dtype=object),
            )
        elif path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            # Default to CSV
            df.to_csv(path, index=False)
    
    def save_metadata(self):
        # Edits are already in the model/DataFrame
        if self.metadata_file_path:
            # Save categorical columns properly - they will be saved as strings in CSV
            # but we preserve the categorical structure
            self._save_metadata_to_file(self.metadata, self.metadata_file_path)
            QMessageBox.information(self, "Saved", f"Saved to {os.path.basename(self.metadata_file_path)}")
        else:
             path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "CSV (*.csv);;NPZ (*.npz);;Parquet (*.parquet)")
             if path:
                 self._save_metadata_to_file(self.metadata, path)
                 self.metadata_file_path = path


# Keep the old widget class for backward compatibility
class MetadataManagementWidget(QWidget):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
    def update_config(self, config: dict): self.config = config
    def load_metadata(self): pass
    def _sync_table_to_metadata(self): pass
