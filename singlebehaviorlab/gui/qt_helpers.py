from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QFont


def create_section(title: str, layout=None):
    """Create a section header."""
    label = QLabel(f"<h3>{title}</h3>")
    label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    if layout:
        layout.addWidget(label)
    return label


def create_status_label(text: str = "Ready", layout=None):
    """Create a status label."""
    label = QLabel(text)
    label.setStyleSheet("color: blue; font-weight: bold;")
    if layout:
        layout.addWidget(label)
    return label


def update_status_label(label: QLabel, text: str, success: bool = True):
    """Update status label with color coding."""
    label.setText(text)
    if success:
        label.setStyleSheet("color: green; font-weight: bold;")
    else:
        label.setStyleSheet("color: red; font-weight: bold;")
