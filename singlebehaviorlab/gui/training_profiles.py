from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QInputDialog, QMessageBox, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt
import json
import logging
import os
import copy
import yaml

logger = logging.getLogger(__name__)

class TrainingProfileDialog(QDialog):
    """Dialog to manage training profiles and select them for batch training."""
    
    def __init__(self, parent=None, profiles_file="training_profiles.json"):
        super().__init__(parent)
        self.setWindowTitle("Training Profiles")
        self.resize(450, 600)
        self.parent_widget = parent
        self.profiles_file = profiles_file
        self.profiles = self._load_profiles()
        self._setup_ui()

    def _load_profiles(self):
        """Load profiles from JSON file."""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load training profiles from %s: %s", self.profiles_file, e)
                return {}
        return {}

    def _save_profiles(self):
        """Save profiles to JSON file."""
        try:
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save profiles: {e}")

    def reload_profiles(self, profiles_file=None):
        """Reload profiles from disk and refresh the visible list."""
        if profiles_file:
            self.profiles_file = profiles_file
        self.profiles = self._load_profiles()
        self._refresh_list()

    def showEvent(self, event):
        self.reload_profiles()
        super().showEvent(event)

    def _setup_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Manage Training Profiles</b>"))
        layout.addWidget(QLabel("Check multiple profiles to run them as a batch sequence."))
        
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)
        
        btn_layout = QVBoxLayout()
        
        # --- Group 1: Create/Update ---
        create_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save New Profile")
        save_btn.clicked.connect(self._save_new)
        save_btn.setStyleSheet("font-weight: bold;")
        save_btn.setToolTip("Save current UI settings as a new profile (prompts for name)")
        create_layout.addWidget(save_btn)
        
        self.update_btn = QPushButton("Update Selected")
        self.update_btn.clicked.connect(self._update_selected)
        self.update_btn.setEnabled(False)
        self.update_btn.setToolTip("Overwrite the selected profile with current UI settings")
        create_layout.addWidget(self.update_btn)

        self.import_btn = QPushButton("Import From Experiment...")
        self.import_btn.clicked.connect(self._import_profiles)
        self.import_btn.setToolTip("Import training profiles from another experiment's config.yaml or training_profiles.json")
        create_layout.addWidget(self.import_btn)
        
        btn_layout.addLayout(create_layout)
        
        # --- Group 2: Manage Selected ---
        manage_layout = QHBoxLayout()
        
        self.duplicate_btn = QPushButton("Duplicate")
        self.duplicate_btn.clicked.connect(self._duplicate_selected)
        self.duplicate_btn.setEnabled(False)
        self.duplicate_btn.setToolTip("Create a copy of the selected profile")
        manage_layout.addWidget(self.duplicate_btn)
        
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_selected)
        self.rename_btn.setEnabled(False)
        self.rename_btn.setToolTip("Rename the selected profile")
        manage_layout.addWidget(self.rename_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("color: red;")
        manage_layout.addWidget(self.delete_btn)
        
        btn_layout.addLayout(manage_layout)
        
        # --- Group 3: Load to UI ---
        self.load_btn = QPushButton("Load Profile")
        self.load_btn.clicked.connect(self._load_selected)
        self.load_btn.setEnabled(False)
        self.load_btn.setToolTip("Apply the selected profile settings to the main Training tab")
        btn_layout.addWidget(self.load_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addSpacing(10)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Refresh list now that UI elements are created
        self._refresh_list()

    def _refresh_list(self):
        """Refresh the list widget from self.profiles."""
        current_item_text = None
        if self.list_widget.currentItem():
            current_item_text = self.list_widget.currentItem().text()
        checked_names = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_names.add(item.text())
            
        self.list_widget.clear()
        for name in sorted(self.profiles.keys()):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in checked_names else Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
            if name == current_item_text:
                self.list_widget.setCurrentItem(item)
                item.setSelected(True)
        
        # Re-trigger selection logic to update buttons
        if self.list_widget.currentItem():
            self._on_item_clicked(self.list_widget.currentItem())
        else:
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.update_btn.setEnabled(False)
            self.duplicate_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)

    def _on_item_clicked(self, item):
        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.update_btn.setEnabled(True)
        self.duplicate_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)

    def _save_new(self):
        """Save current main UI settings as a new profile."""
        if not self.parent_widget:
            return
            
        name, ok = QInputDialog.getText(self, "Save New Profile", "Profile Name:")
        if ok and name:
            name = name.strip()
            if not name: return
            
            if name in self.profiles:
                QMessageBox.warning(self, "Error", f"Profile '{name}' already exists.\nUse 'Update Selected' or choose a different name.")
                return

            config = self.parent_widget.get_training_config()
            self.profiles[name] = config
            self._save_profiles()
            self._refresh_list()
            
            # Select the new item
            items = self.list_widget.findItems(name, Qt.MatchFlag.MatchExactly)
            if items:
                self.list_widget.setCurrentItem(items[0])
                self._on_item_clicked(items[0])
            
            QMessageBox.information(self, "Saved", f"Profile '{name}' saved.")

    def _default_import_dir(self):
        if self.parent_widget and hasattr(self.parent_widget, "config"):
            cfg = self.parent_widget.config or {}
            for key in ("experiments_dir", "experiment_path", "config_path"):
                path = cfg.get(key)
                if path:
                    if os.path.isfile(path):
                        return os.path.dirname(path)
                    return path
        return os.path.dirname(self.profiles_file) if self.profiles_file else os.getcwd()

    def _resolve_import_profiles_path(self, selected_path):
        """Resolve a user-chosen experiment/config file to training_profiles.json."""
        if not selected_path:
            return None
        selected_path = os.path.abspath(selected_path)
        if os.path.isdir(selected_path):
            candidate = os.path.join(selected_path, "training_profiles.json")
            return candidate if os.path.exists(candidate) else None
        if os.path.basename(selected_path) == "training_profiles.json":
            return selected_path
        if selected_path.lower().endswith((".yaml", ".yml")):
            try:
                with open(selected_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                profiles_path = cfg.get("training_profiles_path")
                if profiles_path:
                    if not os.path.isabs(profiles_path):
                        profiles_path = os.path.join(os.path.dirname(selected_path), profiles_path)
                    if os.path.exists(profiles_path):
                        return os.path.abspath(profiles_path)
                candidate = os.path.join(os.path.dirname(selected_path), "training_profiles.json")
                return candidate if os.path.exists(candidate) else None
            except Exception:
                return None
        return selected_path if os.path.exists(selected_path) else None

    def _load_external_profiles(self, profiles_path):
        """Load and validate external training profiles."""
        with open(profiles_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("profiles file must contain a JSON object")
        valid = {str(name): cfg for name, cfg in loaded.items() if isinstance(cfg, dict)}
        if not valid:
            raise ValueError("no valid profile entries found")
        return valid

    def _import_profiles(self):
        """Import profiles from another experiment into the current experiment."""
        start_dir = self._default_import_dir()
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Profiles From Experiment",
            start_dir,
            "Experiment/Profile Files (*.yaml *.yml *.json);;All Files (*)",
        )
        if not selected_path:
            return

        profiles_path = self._resolve_import_profiles_path(selected_path)
        if not profiles_path or not os.path.exists(profiles_path):
            QMessageBox.warning(
                self,
                "Profiles Not Found",
                "Could not locate a valid 'training_profiles.json' from the selected experiment/file.",
            )
            return

        try:
            external_profiles = self._load_external_profiles(profiles_path)
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", f"Failed to load profiles:\n{e}")
            return

        duplicate_names = sorted(name for name in external_profiles if name in self.profiles)
        if duplicate_names:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Duplicate Profile Names")
            preview = ", ".join(duplicate_names[:6])
            if len(duplicate_names) > 6:
                preview += ", ..."
            msg.setText(
                "Some imported profile names already exist in this experiment.\n\n"
                f"Duplicates: {preview}"
            )
            overwrite_btn = msg.addButton("Overwrite Duplicates", QMessageBox.ButtonRole.AcceptRole)
            rename_btn = msg.addButton("Keep Both (rename imported)", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.exec()
            if msg.clickedButton() == overwrite_btn:
                duplicate_mode = "overwrite"
            elif msg.clickedButton() == rename_btn:
                duplicate_mode = "rename"
            else:
                return
        else:
            duplicate_mode = "overwrite"

        imported_names = []
        source_tag = os.path.splitext(os.path.basename(os.path.dirname(profiles_path) or profiles_path))[0] or "imported"
        for name, cfg in external_profiles.items():
            target_name = name
            if duplicate_mode == "rename" and target_name in self.profiles:
                suffix = 1
                while True:
                    candidate = f"{name} ({source_tag})" if suffix == 1 else f"{name} ({source_tag} {suffix})"
                    if candidate not in self.profiles:
                        target_name = candidate
                        break
                    suffix += 1
            self.profiles[target_name] = copy.deepcopy(cfg)
            imported_names.append(target_name)

        self._save_profiles()
        self._refresh_list()

        if imported_names:
            items = self.list_widget.findItems(imported_names[-1], Qt.MatchFlag.MatchExactly)
            if items:
                self.list_widget.setCurrentItem(items[0])
                self._on_item_clicked(items[0])

        QMessageBox.information(
            self,
            "Profiles Imported",
            f"Imported {len(imported_names)} profile(s) from:\n{profiles_path}",
        )

    def _update_selected(self):
        """Update the selected profile with current UI settings."""
        item = self.list_widget.currentItem()
        if not item: return
        name = item.text()
        
        if QMessageBox.question(self, "Update Profile", f"Overwrite profile '{name}' with current settings from the UI?") == QMessageBox.StandardButton.Yes:
            config = self.parent_widget.get_training_config()
            self.profiles[name] = config
            self._save_profiles()
            QMessageBox.information(self, "Updated", f"Profile '{name}' updated.")

    def _duplicate_selected(self):
        """Duplicate the selected profile."""
        item = self.list_widget.currentItem()
        if not item: return
        name = item.text()
        
        new_name, ok = QInputDialog.getText(self, "Duplicate Profile", "New Profile Name:", text=f"Copy of {name}")
        if ok and new_name:
            new_name = new_name.strip()
            if not new_name: return
            if new_name in self.profiles:
                QMessageBox.warning(self, "Error", f"Profile '{new_name}' already exists.")
                return
            
            self.profiles[new_name] = self.profiles[name].copy()
            self._save_profiles()
            self._refresh_list()

    def _rename_selected(self):
        """Rename the selected profile."""
        item = self.list_widget.currentItem()
        if not item: return
        old_name = item.text()
        
        new_name, ok = QInputDialog.getText(self, "Rename Profile", "New Name:", text=old_name)
        if ok and new_name:
            new_name = new_name.strip()
            if not new_name or new_name == old_name: return
            
            if new_name in self.profiles:
                QMessageBox.warning(self, "Error", f"Profile '{new_name}' already exists.")
                return
            
            # Preserve order/data by popping and setting
            config = self.profiles.pop(old_name)
            self.profiles[new_name] = config
            self._save_profiles()
            self._refresh_list()
            
            # Select the renamed item
            items = self.list_widget.findItems(new_name, Qt.MatchFlag.MatchExactly)
            if items:
                self.list_widget.setCurrentItem(items[0])
                self._on_item_clicked(items[0])

    def _load_selected(self):
        """Load selected profile settings into main UI."""
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text()
        if name in self.profiles:
            try:
                self.parent_widget.apply_training_config(self.profiles[name])
                QMessageBox.information(self, "Loaded", f"Loaded profile: {name}\nSettings applied to UI.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to apply profile: {e}")

    def _delete_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text()
        if QMessageBox.question(self, "Delete", f"Delete profile '{name}'?") == QMessageBox.StandardButton.Yes:
            del self.profiles[name]
            self._save_profiles()
            self._refresh_list()

    def get_selected_profiles_for_batch(self):
        """Return list of (name, config) for checked items."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                name = item.text()
                if name in self.profiles:
                    selected.append((name, self.profiles[name]))
        return selected

