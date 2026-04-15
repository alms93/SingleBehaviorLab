#!/usr/bin/env python3
"""
Main entry point for SingleBehaviorLab.
Runs when invoked as:
  python -m singlebehaviorlab
  singlebehaviorlab          (pip entry point)
"""

import logging
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
os.environ["XLA_FLAGS"] = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"

from singlebehaviorlab.config import load_config  # noqa: E402  (re-exported for backward compat)


def run_gui_app():
    """Launch the PyQt6 graphical interface."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QPushButton, QLabel,
    )
    from singlebehaviorlab.gui.main_window import MainWindow

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


def main():
    from singlebehaviorlab.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
