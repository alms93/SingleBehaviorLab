#!/usr/bin/env python3
"""
Main entry point for SingleBehaviorLab.
Runs when invoked as:
  python -m singlebehaviorlab
  singlebehaviorlab          (pip entry point)
"""

import logging
import sys
import os

# Let JAX grow GPU memory on demand and leave headroom for PyTorch.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
# Fall back to driver JIT compilation when ptxas/nvlink is unavailable.
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"

import yaml
from singlebehaviorlab._paths import get_default_config_path, get_experiments_dir
from singlebehaviorlab.gui.main_window import MainWindow


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = str(get_default_config_path())

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # For path keys that are missing or blank, resolve relative to the
    # experiments directory (pip install) or the package parent (source install).
    from singlebehaviorlab._paths import get_package_dir
    base_dir = str(get_package_dir().parent)

    defaults = {
        "data_dir":        os.path.join(base_dir, "data"),
        "raw_videos_dir":  os.path.join(base_dir, "data", "raw_videos"),
        "clips_dir":       os.path.join(base_dir, "data", "clips"),
        "annotations_dir": os.path.join(base_dir, "data", "annotations"),
        "models_dir":      os.path.join(base_dir, "models", "behavior_heads"),
        "backbone_dir":    os.path.join(base_dir, "models", "videoprism_backbone"),
        "annotation_file": os.path.join(base_dir, "data", "annotations", "annotations.json"),
    }

    for key, value in defaults.items():
        if not config.get(key):
            config[key] = value
        elif not os.path.isabs(config[key]):
            config[key] = os.path.join(base_dir, config[key])

    experiments_dir = str(get_experiments_dir())
    if not config.get("experiments_dir"):
        config["experiments_dir"] = experiments_dir
    os.makedirs(config["experiments_dir"], exist_ok=True)

    config["config_path"] = config_path
    config.setdefault("experiment_name", None)
    config.setdefault("experiment_path", None)

    return config


def main():
    """Application entry point."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QPushButton, QLabel,
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SingleBehaviorLab")

    config = load_config()

    startup_dialog = QDialog()
    startup_dialog.setWindowTitle("Welcome - Experiment Management")
    startup_dialog.setMinimumSize(400, 200)
    startup_dialog.setModal(True)

    layout = QVBoxLayout()

    welcome_label = QLabel(
        "<h2>Welcome to SingleBehaviorLab</h2>"
        "<p>Please choose an option to get started:</p>"
    )
    welcome_label.setWordWrap(True)
    layout.addWidget(welcome_label)

    create_btn = QPushButton("Create New Experiment")
    create_btn.setMinimumHeight(40)
    create_btn.setStyleSheet("font-size: 12px; font-weight: bold;")
    create_btn.clicked.connect(startup_dialog.accept)
    layout.addWidget(create_btn)

    load_btn = QPushButton("Load Existing Experiment")
    load_btn.setMinimumHeight(40)
    load_btn.setStyleSheet("font-size: 12px; font-weight: bold;")
    load_btn.clicked.connect(startup_dialog.reject)
    layout.addWidget(load_btn)

    startup_dialog.setLayout(layout)

    result = startup_dialog.exec()

    window = MainWindow(config)
    window.show()

    if result == QDialog.DialogCode.Accepted:
        window._create_experiment()
    elif result == QDialog.DialogCode.Rejected:
        window._load_experiment()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
