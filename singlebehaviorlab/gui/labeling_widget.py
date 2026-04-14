from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QMessageBox, QFileDialog, QDialog,
    QDialogButtonBox, QCheckBox, QScrollArea, QSlider, QAbstractItemView,
    QFormLayout, QLineEdit, QSizePolicy, QSpinBox, QProgressBar,
    QInputDialog, QApplication,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QRect, QRectF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
import cv2
import os
import random
import json
import shutil
from singlebehaviorlab.backend.data_store import AnnotationManager


class GridOverlayLabel(QLabel):
    """QLabel for video frames with bbox drawing support."""
    bbox_moved = pyqtSignal()  # emitted when user finishes moving an existing bbox

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bbox_enabled = False
        self.bbox_norm = None  # (x1, y1, x2, y2) normalized within frame rect
        self._bbox_drag_start = None
        self._bbox_drag_current = None
        self._bbox_moving = False
        self._bbox_move_start = None
        self._frame_rect = QRect()  # Region where the scaled frame is drawn

    def set_bbox_enabled(self, enabled: bool):
        self.bbox_enabled = enabled
        if not enabled:
            self._bbox_drag_start = None
            self._bbox_drag_current = None
        if self.bbox_enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_bbox_norm(self, bbox_norm):
        """Set normalized bbox as (x1, y1, x2, y2) in [0,1] within frame rect."""
        if not bbox_norm or len(bbox_norm) != 4:
            self.bbox_norm = None
            self.update()
            return
        x1, y1, x2, y2 = [float(v) for v in bbox_norm]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            self.bbox_norm = None
        else:
            self.bbox_norm = (x1, y1, x2, y2)
        self.update()

    def get_bbox_norm(self):
        return list(self.bbox_norm) if self.bbox_norm is not None else None

    def clear_bbox(self):
        self.bbox_norm = None
        self._bbox_drag_start = None
        self._bbox_drag_current = None
        self.update()

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        # Compute where the scaled pixmap is drawn (centered in the label)
        if pixmap and not pixmap.isNull():
            label_w, label_h = self.width(), self.height()
            pm_w, pm_h = pixmap.width(), pixmap.height()
            x = (label_w - pm_w) // 2
            y = (label_h - pm_h) // 2
            self._frame_rect = QRect(x, y, pm_w, pm_h)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame_rect.isEmpty():
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        r = self._frame_rect

        # Draw saved bbox if present
        if self.bbox_enabled and self.bbox_norm is not None:
            bx1, by1, bx2, by2 = self.bbox_norm
            x = r.x() + bx1 * r.width()
            y = r.y() + by1 * r.height()
            w = (bx2 - bx1) * r.width()
            h = (by2 - by1) * r.height()
            bbox_fill = QColor(255, 140, 0, 55)
            bbox_pen = QPen(QColor(255, 140, 0, 220))
            bbox_pen.setWidth(2)
            painter.fillRect(QRectF(x, y, w, h), bbox_fill)
            painter.setPen(bbox_pen)
            painter.drawRect(QRectF(x, y, w, h))
            # Red center dot
            cx = x + w / 2.0
            cy = y + h / 2.0
            painter.setBrush(QColor(255, 0, 0, 220))
            painter.setPen(QPen(QColor(180, 0, 0, 255), 1))
            painter.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))

        # Draw live bbox while dragging
        if self.bbox_enabled and self._bbox_drag_start is not None and self._bbox_drag_current is not None:
            drag_rect = QRect(self._bbox_drag_start, self._bbox_drag_current).normalized().intersected(r)
            if drag_rect.width() > 1 and drag_rect.height() > 1:
                live_pen = QPen(QColor(255, 90, 0, 230))
                live_pen.setWidth(2)
                painter.setPen(live_pen)
                painter.fillRect(drag_rect, QColor(255, 90, 0, 45))
                painter.drawRect(drag_rect)

        painter.end()
    
    def _click_inside_bbox(self, pos) -> bool:
        """Check if a click position is inside the current saved bbox."""
        if self.bbox_norm is None or self._frame_rect.isEmpty():
            return False
        r = self._frame_rect
        bx1, by1, bx2, by2 = self.bbox_norm
        px = r.x() + bx1 * r.width()
        py = r.y() + by1 * r.height()
        pw = (bx2 - bx1) * r.width()
        ph = (by2 - by1) * r.height()
        return (px <= pos.x() <= px + pw) and (py <= pos.y() <= py + ph)

    def mousePressEvent(self, event):
        if self.bbox_enabled and event.button() == Qt.MouseButton.LeftButton:
            if self._frame_rect.contains(event.pos()):
                if self._click_inside_bbox(event.pos()):
                    self._bbox_moving = True
                    self._bbox_move_start = event.pos()
                else:
                    self._bbox_moving = False
                    self._bbox_drag_start = event.pos()
                    self._bbox_drag_current = event.pos()
                self.update()
            return
        return super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.bbox_enabled and (event.buttons() & Qt.MouseButton.LeftButton):
            if getattr(self, '_bbox_moving', False) and self.bbox_norm is not None:
                # Move mode: translate bbox by mouse delta
                r = self._frame_rect
                if r.width() > 0 and r.height() > 0:
                    dx_norm = (event.pos().x() - self._bbox_move_start.x()) / r.width()
                    dy_norm = (event.pos().y() - self._bbox_move_start.y()) / r.height()
                    bx1, by1, bx2, by2 = self.bbox_norm
                    w, h = bx2 - bx1, by2 - by1
                    new_x1 = bx1 + dx_norm
                    new_y1 = by1 + dy_norm
                    # Clamp to frame bounds
                    new_x1 = max(0.0, min(1.0 - w, new_x1))
                    new_y1 = max(0.0, min(1.0 - h, new_y1))
                    self.bbox_norm = (new_x1, new_y1, new_x1 + w, new_y1 + h)
                    self._bbox_move_start = event.pos()
                self.update()
                return
            if self._bbox_drag_start is not None:
                self._bbox_drag_current = event.pos()
                self.update()
            return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.bbox_enabled and event.button() == Qt.MouseButton.LeftButton:
            if getattr(self, '_bbox_moving', False):
                # End move mode — bbox_norm already updated in mouseMoveEvent
                self._bbox_moving = False
                self._bbox_move_start = None
                self.update()
                self.bbox_moved.emit()
                return
            if self._bbox_drag_start is not None and self._bbox_drag_current is not None:
                drag_rect = QRect(self._bbox_drag_start, self._bbox_drag_current).normalized().intersected(self._frame_rect)
                if drag_rect.width() >= 4 and drag_rect.height() >= 4 and self._frame_rect.width() > 0 and self._frame_rect.height() > 0:
                    x1 = (drag_rect.x() - self._frame_rect.x()) / self._frame_rect.width()
                    y1 = (drag_rect.y() - self._frame_rect.y()) / self._frame_rect.height()
                    x2 = (drag_rect.right() - self._frame_rect.x()) / self._frame_rect.width()
                    y2 = (drag_rect.bottom() - self._frame_rect.y()) / self._frame_rect.height()
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))
                    if x2 > x1 and y2 > y1:
                        self.bbox_norm = (x1, y1, x2, y2)
            self._bbox_drag_start = None
            self._bbox_drag_current = None
            self.update()
            return
        return super().mouseReleaseEvent(event)


class FullScreenLabelingDialog(QDialog):
    """Full screen dialog for video labeling."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Full Screen Labeling")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Video Area (Main content)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label)
        
        # Controls Overlay (Bottom bar)
        controls_container = QWidget()
        controls_container.setStyleSheet("""
            QWidget { background-color: rgba(40, 40, 40, 200); }
            QPushButton { color: white; border: 1px solid #666; padding: 5px 15px; border-radius: 4px; background-color: #333; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #222; }
            QLabel { color: white; font-weight: bold; }
            QComboBox { background-color: #333; color: white; border: 1px solid #666; padding: 5px; }
            QComboBox::drop-down { border: none; }
        """)
        controls_root = QVBoxLayout(controls_container)
        controls_root.setContentsMargins(10, 8, 10, 8)
        controls_root.setSpacing(8)

        # Row 1: Playback + class
        row1 = QHBoxLayout()
        self.info_label = QLabel()
        row1.addWidget(self.info_label)
        row1.addStretch()
        self.prev_btn = QPushButton("◀ Prev (A)")
        self.prev_btn.clicked.connect(self.parent_widget._prev_clip)
        row1.addWidget(self.prev_btn)
        self.play_btn = QPushButton("Play/Pause (Space)")
        self.play_btn.clicked.connect(self.parent_widget._toggle_play)
        row1.addWidget(self.play_btn)
        self.next_btn = QPushButton("Next ▶ (D)")
        self.next_btn.clicked.connect(self.parent_widget._next_clip)
        row1.addWidget(self.next_btn)
        self.random_btn = QPushButton("Random (R)")
        self.random_btn.clicked.connect(self.parent_widget._random_clip)
        row1.addWidget(self.random_btn)

        class_col = QVBoxLayout()
        class_col.setSpacing(4)
        fs_frame_nav_row = QHBoxLayout()
        fs_frame_nav_row.setSpacing(4)
        self.fs_prev_frame_btn = QPushButton("Prev frame (Q)")
        self.fs_prev_frame_btn.clicked.connect(self.parent_widget._prev_frame)
        fs_frame_nav_row.addWidget(self.fs_prev_frame_btn)
        self.fs_next_frame_btn = QPushButton("Next frame (E)")
        self.fs_next_frame_btn.clicked.connect(self.parent_widget._next_frame)
        fs_frame_nav_row.addWidget(self.fs_next_frame_btn)
        class_col.addLayout(fs_frame_nav_row)

        fs_class_row = QHBoxLayout()
        fs_class_row.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(180)
        self.update_classes()
        self.class_combo.activated.connect(self._on_class_selected)
        fs_class_row.addWidget(self.class_combo)
        class_col.addLayout(fs_class_row)
        row1.addSpacing(20)
        row1.addLayout(class_col)
        controls_root.addLayout(row1)

        # Row 2: Per-frame labeling for clip mode
        row_pf = QHBoxLayout()
        self.fs_frame_info_label = QLabel("Frame: - / -")
        self.fs_frame_info_label.setMinimumWidth(180)
        row_pf.addWidget(self.fs_frame_info_label)
        self.fs_frame_label_bar = QLabel()
        self.fs_frame_label_bar.setFixedHeight(14)
        self.fs_frame_label_bar.setMinimumWidth(180)
        self.fs_frame_label_bar.setStyleSheet("background: #4b5563; border: 1px solid #6b7280; border-radius: 3px;")
        row_pf.addWidget(self.fs_frame_label_bar, 1)
        self.fs_mark_in_btn = QPushButton("In")
        self.fs_mark_in_btn.setFixedWidth(34)
        self.fs_mark_in_btn.clicked.connect(self.parent_widget._fl_mark_in)
        row_pf.addWidget(self.fs_mark_in_btn)
        self.fs_range_label = QLabel("—")
        self.fs_range_label.setFixedWidth(60)
        self.fs_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_pf.addWidget(self.fs_range_label)
        self.fs_mark_out_btn = QPushButton("Out")
        self.fs_mark_out_btn.setFixedWidth(40)
        self.fs_mark_out_btn.clicked.connect(self.parent_widget._fl_mark_out)
        row_pf.addWidget(self.fs_mark_out_btn)
        self.fs_fl_class_combo = QComboBox()
        self.fs_fl_class_combo.setMinimumWidth(140)
        self.fs_fl_class_combo.currentTextChanged.connect(self._on_fs_fl_class_changed)
        row_pf.addWidget(self.fs_fl_class_combo)
        self.fs_apply_btn = QPushButton("Apply")
        self.fs_apply_btn.clicked.connect(self._apply_fs_frame_label)
        row_pf.addWidget(self.fs_apply_btn)
        self.fs_clear_btn = QPushButton("Clr")
        self.fs_clear_btn.clicked.connect(self.parent_widget._fl_clear_labels)
        row_pf.addWidget(self.fs_clear_btn)
        controls_root.addLayout(row_pf)

        # Row 3: Position scroll bar (drag to move through video / clip)
        row3 = QHBoxLayout()
        self.fs_position_label = QLabel("Position:")
        row3.addWidget(self.fs_position_label)
        self.fs_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.fs_position_slider.setMinimum(0)
        self.fs_position_slider.setMaximum(0)
        self.fs_position_slider.setMinimumHeight(22)
        self.fs_position_slider.valueChanged.connect(self._on_fs_position_slider_changed)
        row3.addWidget(self.fs_position_slider, 1)
        controls_root.addLayout(row3)

        # Add container to layout
        layout.addWidget(controls_container)
        self.setLayout(layout)
        
        # Floating Close Button (Top-Right)
        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFixedSize(50, 50)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 100); 
                color: white; 
                font-weight: bold; 
                font-size: 24px;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: rgba(200, 0, 0, 150);
            }
        """)
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position top-right with margin
        self.close_btn.move(self.width() - 60, 10)

    def showEvent(self, event):
        super().showEvent(event)
        # Deferred refresh so layout/geometry is ready (fixes freeze on reopen).
        QTimer.singleShot(0, self.parent_widget._refresh_fullscreen_from_current_state)
        QTimer.singleShot(80, self.parent_widget._refresh_fullscreen_from_current_state)

    def update_classes(self):
        """Sync class list with parent."""
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        # Copy items from parent's combo
        parent_combo = self.parent_widget.class_combo
        for i in range(parent_combo.count()):
            self.class_combo.addItem(parent_combo.itemText(i))
        self.class_combo.setCurrentIndex(parent_combo.currentIndex())
        self.class_combo.blockSignals(False)
        if hasattr(self.parent_widget, "fl_class_combo") and hasattr(self, "fs_fl_class_combo"):
            current = self.parent_widget.fl_class_combo.currentText()
            self.fs_fl_class_combo.blockSignals(True)
            self.fs_fl_class_combo.clear()
            for i in range(self.parent_widget.fl_class_combo.count()):
                self.fs_fl_class_combo.addItem(self.parent_widget.fl_class_combo.itemText(i))
            idx = self.fs_fl_class_combo.findText(current)
            if idx >= 0:
                self.fs_fl_class_combo.setCurrentIndex(idx)
            self.fs_fl_class_combo.blockSignals(False)

    def update_info(self, text):
        self.info_label.setText(text)

    def _on_class_selected(self):
        # Sync selection back to parent and save
        idx = self.class_combo.currentIndex()
        self.parent_widget.class_combo.setCurrentIndex(idx)
        self.parent_widget._save_label()

    def _on_fs_fl_class_changed(self, text: str):
        if hasattr(self.parent_widget, "fl_class_combo"):
            idx = self.parent_widget.fl_class_combo.findText(text)
            if idx >= 0:
                self.parent_widget.fl_class_combo.setCurrentIndex(idx)

    def _apply_fs_frame_label(self):
        self._on_fs_fl_class_changed(self.fs_fl_class_combo.currentText())
        self.parent_widget._fl_apply_label()

    def sync_per_frame_controls(self):
        clip_mode = bool(self.parent_widget.current_frames) and not self.parent_widget._is_source_video_mode()
        self.fs_frame_info_label.setVisible(clip_mode)
        self.fs_frame_label_bar.setVisible(clip_mode)
        self.fs_mark_in_btn.setVisible(clip_mode)
        self.fs_range_label.setVisible(clip_mode)
        self.fs_mark_out_btn.setVisible(clip_mode)
        self.fs_fl_class_combo.setVisible(clip_mode)
        self.fs_apply_btn.setVisible(clip_mode)
        self.fs_clear_btn.setVisible(clip_mode)
        self.fs_prev_frame_btn.setVisible(clip_mode)
        self.fs_next_frame_btn.setVisible(clip_mode)
        if not clip_mode:
            return
        self.fs_frame_info_label.setText(self.parent_widget.frame_label.text())
        self.fs_range_label.setText(self.parent_widget.fl_range_label.text())
        pixmap = self.parent_widget.frame_label_bar.pixmap()
        self.fs_frame_label_bar.setPixmap(pixmap if pixmap is not None else QPixmap())
        current = self.parent_widget.fl_class_combo.currentText()
        idx = self.fs_fl_class_combo.findText(current)
        if idx >= 0:
            self.fs_fl_class_combo.blockSignals(True)
            self.fs_fl_class_combo.setCurrentIndex(idx)
            self.fs_fl_class_combo.blockSignals(False)

    def update_scrubber(self):
        """Sync position slider with parent's current mode."""
        source_mode = self.parent_widget._is_source_video_mode()
        clip_mode = bool(self.parent_widget.current_frames) and not source_mode
        self.fs_position_slider.setEnabled(source_mode or clip_mode)
        self.fs_position_label.setText("Position:" if source_mode else "Frame:")
        self.sync_per_frame_controls()
        if clip_mode:
            self.fs_position_slider.blockSignals(True)
            self.fs_position_slider.setMinimum(0)
            self.fs_position_slider.setMaximum(max(0, len(self.parent_widget.current_frames) - 1))
            self.fs_position_slider.setValue(int(self.parent_widget.current_frame_idx))
            self.fs_position_slider.blockSignals(False)
            return
        if source_mode:
            max_frame = max(0, int(self.parent_widget.current_source_frame_count) - 1)
            curr = int(self.parent_widget.current_source_frame)
            self.fs_position_slider.blockSignals(True)
            self.fs_position_slider.setMinimum(0)
            self.fs_position_slider.setMaximum(max_frame)
            self.fs_position_slider.setValue(max(0, min(max_frame, curr)))
            self.fs_position_slider.blockSignals(False)
            return
        self.fs_position_slider.blockSignals(True)
        self.fs_position_slider.setMaximum(0)
        self.fs_position_slider.setValue(0)
        self.fs_position_slider.blockSignals(False)

    def _on_fs_position_slider_changed(self, value: int):
        if self.parent_widget._is_source_video_mode():
            self.parent_widget._display_source_frame(value)
            if hasattr(self.parent_widget, "source_scrub_slider"):
                self.parent_widget.source_scrub_slider.blockSignals(True)
                self.parent_widget.source_scrub_slider.setValue(int(value))
                self.parent_widget.source_scrub_slider.blockSignals(False)
            return
        if not self.parent_widget.current_frames:
            return
        self.parent_widget._capture_current_frame_bbox()
        self.parent_widget.current_frame_idx = int(value)
        self.parent_widget.frame_slider.blockSignals(True)
        self.parent_widget.frame_slider.setValue(int(value))
        self.parent_widget.frame_slider.blockSignals(False)
        self.parent_widget._sync_bbox_to_frame()
        self.parent_widget._display_frame()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.parent_widget._toggle_play()
        elif event.key() == Qt.Key.Key_A:
            self.parent_widget._prev_clip()
        elif event.key() == Qt.Key.Key_D:
            self.parent_widget._next_clip()
        elif event.key() == Qt.Key.Key_R:
            self.parent_widget._random_clip()
        elif event.key() == Qt.Key.Key_Q:
            self.parent_widget._prev_frame()
        elif event.key() == Qt.Key.Key_E:
            self.parent_widget._next_frame()
        elif event.key() >= Qt.Key.Key_1 and event.key() <= Qt.Key.Key_9:
             idx = event.key() - Qt.Key.Key_1
             if idx < self.class_combo.count():
                 self.class_combo.setCurrentIndex(idx)
                 self._on_class_selected()
        else:
            super().keyPressEvent(event)


class LabelingWidget(QWidget):
    """Widget for labeling video clips."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.annotation_manager = AnnotationManager(
            self.config.get("annotation_file", "data/annotations/annotations.json")
        )
        self.current_clip_path = None
        self.current_frames = []
        self.current_frame_idx = 0
        self.is_playing = False
        self.frame_bboxes = {}  # {frame_idx: [x1, y1, x2, y2]} per-frame bboxes
        self.clip_base_dir = self.config.get("clips_dir", "data/clips")
        self.fullscreen_dialog = None
        self.zoom_factor = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 4.0
        self.zoom_step = 0.2
        self._base_display_pixmap = None
        self.source_video_paths = []
        self.current_source_video_path = None
        self.current_source_cap = None
        self.current_source_frame = 0
        self.current_source_frame_count = 0
        self._setup_ui()
        self.refresh_clip_list()
    
    def _setup_ui(self):
        """Setup UI components."""
        main_layout = QHBoxLayout()
        left_column = QWidget()
        left_column.setMaximumWidth(420)
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)

        def _wrap_scroll(widget: QWidget, max_height: int, min_height: int = 0) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            if min_height > 0:
                scroll.setMinimumHeight(min_height)
            scroll.setMaximumHeight(max_height)
            scroll.setWidget(widget)
            return scroll

        source_group = QGroupBox("Source Videos")
        source_layout = QVBoxLayout()
        source_video_row = QHBoxLayout()
        self.source_video_list = QListWidget()
        self.source_video_list.setMaximumHeight(90)
        self.source_video_list.itemSelectionChanged.connect(self._on_source_video_selected)
        source_video_row.addWidget(self.source_video_list, 1)
        source_video_btn_col = QVBoxLayout()
        self.add_source_video_btn = QPushButton("Add videos...")
        self.add_source_video_btn.clicked.connect(self.open_timeline_import_dialog)
        source_video_btn_col.addWidget(self.add_source_video_btn)
        self.remove_source_video_btn = QPushButton("Remove selected")
        self.remove_source_video_btn.clicked.connect(self._remove_selected_source_video)
        source_video_btn_col.addWidget(self.remove_source_video_btn)
        self.clear_source_video_btn = QPushButton("Clear all")
        self.clear_source_video_btn.clicked.connect(self._clear_source_videos)
        source_video_btn_col.addWidget(self.clear_source_video_btn)
        source_video_btn_col.addStretch()
        source_video_row.addLayout(source_video_btn_col)
        source_layout.addLayout(source_video_row)
        source_group.setLayout(source_layout)
        left_column_layout.addWidget(source_group)

        extract_group = QGroupBox("Clip Extraction")
        extract_layout = QFormLayout()
        self.ea_target_fps_spin = QSpinBox()
        self.ea_target_fps_spin.setRange(1, 240)
        self.ea_target_fps_spin.setValue(int(self.config.get("default_target_fps", 12)))
        extract_layout.addRow("Target FPS:", self.ea_target_fps_spin)
        self.ea_clip_length_spin = QSpinBox()
        self.ea_clip_length_spin.setRange(1, 128)
        self.ea_clip_length_spin.setValue(int(self.config.get("default_clip_length", 8)))
        extract_layout.addRow("Frames/clip:", self.ea_clip_length_spin)
        self.ea_step_spin = QSpinBox()
        self.ea_step_spin.setRange(1, 1000)
        self.ea_step_spin.setValue(int(self.config.get("default_step_frames", 8)))
        extract_layout.addRow("Step (subsampled frames):", self.ea_step_spin)
        self.ea_max_clips_spin = QSpinBox()
        self.ea_max_clips_spin.setRange(0, 100000)
        self.ea_max_clips_spin.setValue(0)
        self.ea_max_clips_spin.setToolTip("Max clips per video (0 = unlimited)")
        extract_layout.addRow("Max clips/video:", self.ea_max_clips_spin)
        self.ea_output_dir_edit = QLineEdit(self.clip_base_dir)
        ea_output_browse_btn = QPushButton("...")
        ea_output_browse_btn.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.ea_output_dir_edit, 1)
        output_row.addWidget(ea_output_browse_btn)
        output_widget = QWidget()
        output_widget.setLayout(output_row)
        extract_layout.addRow("Output:", output_widget)
        ea_btn_row = QHBoxLayout()
        self.extract_all_btn = QPushButton("Extract all clips")
        self.extract_all_btn.setToolTip(
            "Slide a window over the entire video and extract every clip as unlabeled.\n"
            "Use 'Show unlabeled only' and 'Next unlabeled' to browse and label them."
        )
        self.extract_all_btn.clicked.connect(self._extract_all_clips_from_videos)
        ea_btn_row.addWidget(self.extract_all_btn)
        self.ea_progress = QProgressBar()
        self.ea_progress.setVisible(False)
        ea_btn_row.addWidget(self.ea_progress, 1)
        extract_layout.addRow(ea_btn_row)
        self.ea_status_label = QLabel("")
        self.ea_status_label.setWordWrap(True)
        extract_layout.addRow(self.ea_status_label)
        extract_group.setLayout(extract_layout)
        left_column_layout.addWidget(extract_group)

        left_group = QGroupBox("Clips")
        left_group_layout = QVBoxLayout()
        video_filter_layout = QHBoxLayout()
        video_filter_layout.addWidget(QLabel("Filter by video:"))
        self.video_filter_combo = QComboBox()
        self.video_filter_combo.addItem("All Videos")
        self.video_filter_combo.currentTextChanged.connect(self._on_video_filter_changed)
        video_filter_layout.addWidget(self.video_filter_combo)
        left_group_layout.addLayout(video_filter_layout)
        class_filter_layout = QHBoxLayout()
        class_filter_layout.addWidget(QLabel("Filter by class:"))
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Classes")
        self.class_filter_combo.currentTextChanged.connect(self._on_class_filter_changed)
        class_filter_layout.addWidget(self.class_filter_combo)
        left_group_layout.addLayout(class_filter_layout)
        self.clip_list = QListWidget()
        self.clip_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.clip_list.itemClicked.connect(self._on_clip_selected)
        left_group_layout.addWidget(self.clip_list)
        filter_layout = QHBoxLayout()
        self.show_unlabeled_btn = QPushButton("Show unlabeled only")
        self.show_unlabeled_btn.clicked.connect(self._filter_unlabeled)
        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.clicked.connect(self._show_all_clips)
        filter_layout.addWidget(self.show_unlabeled_btn)
        filter_layout.addWidget(self.show_all_btn)
        left_group_layout.addLayout(filter_layout)
        self.next_unlabeled_btn = QPushButton("Next unlabeled")
        self.next_unlabeled_btn.clicked.connect(self._next_unlabeled)
        left_group_layout.addWidget(self.next_unlabeled_btn)
        left_group.setLayout(left_group_layout)
        left_column_layout.addWidget(left_group, 1)

        main_layout.addWidget(left_column, 0)
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        video_group = QGroupBox("Video player")
        video_layout = QVBoxLayout()
        
        self.video_scroll = QScrollArea()
        self.video_scroll.setWidgetResizable(False)
        self.video_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_scroll.setMinimumSize(480, 320)
        self.video_scroll.setStyleSheet("background-color: black; border: none;")
        self.video_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.video_label = GridOverlayLabel("No clip selected")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setMouseTracking(True)
        self.video_label.bbox_moved.connect(self._on_bbox_moved)
        self.video_scroll.setWidget(self.video_label)
        video_layout.addWidget(self.video_scroll)

        self.btn_zoom_in = QPushButton("+", self.video_scroll.viewport())
        self.btn_zoom_out = QPushButton("-", self.video_scroll.viewport())
        self._style_zoom_button(self.btn_zoom_in)
        self._style_zoom_button(self.btn_zoom_out)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self._position_zoom_buttons()
        
        controls_layout = QHBoxLayout()
        
        self.prev_clip_btn = QPushButton("< Previous")
        self.prev_clip_btn.setToolTip("Go to previous clip")
        self.prev_clip_btn.clicked.connect(self._prev_clip)
        controls_layout.addWidget(self.prev_clip_btn)
        
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self._toggle_play)
        self.play_pause_btn.setEnabled(False)
        controls_layout.addWidget(self.play_pause_btn)
        
        self.next_clip_btn = QPushButton("Next >")
        self.next_clip_btn.setToolTip("Go to next clip")
        self.next_clip_btn.clicked.connect(self._next_clip)
        controls_layout.addWidget(self.next_clip_btn)
        
        self.random_clip_btn = QPushButton("Random")
        self.random_clip_btn.setToolTip("Select a random clip")
        self.random_clip_btn.clicked.connect(self._random_clip)
        controls_layout.addWidget(self.random_clip_btn)
        
        controls_layout.addStretch()
        
        self.fullscreen_btn = QPushButton("⛶ Full Screen")
        self.fullscreen_btn.setToolTip("Open clip in full screen mode for easier labeling")
        self.fullscreen_btn.clicked.connect(self._open_fullscreen)
        controls_layout.addWidget(self.fullscreen_btn)
        
        video_layout.addLayout(controls_layout)
        video_layout.addSpacing(6)

        # Frame scrubber for per-frame bbox annotation
        frame_slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setFixedWidth(90)
        frame_slider_layout.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        frame_slider_layout.addWidget(self.frame_slider)
        self.frame_prev_btn = QPushButton("◀")
        self.frame_prev_btn.setFixedWidth(32)
        self.frame_prev_btn.setToolTip("Previous frame (Q)")
        self.frame_prev_btn.clicked.connect(self._prev_frame)
        frame_slider_layout.addWidget(self.frame_prev_btn)
        self.frame_next_btn = QPushButton("▶")
        self.frame_next_btn.setFixedWidth(32)
        self.frame_next_btn.setToolTip("Next frame (E)")
        self.frame_next_btn.clicked.connect(self._next_frame)
        frame_slider_layout.addWidget(self.frame_next_btn)
        video_layout.addLayout(frame_slider_layout)
        
        video_group.setLayout(video_layout)
        right_panel.addWidget(video_group, 3)

        # Source video scrubber (visible only when a source video is loaded)
        self.source_scrub_widget = QWidget()
        scrub_layout = QHBoxLayout(self.source_scrub_widget)
        scrub_layout.setContentsMargins(4, 2, 4, 2)
        self.source_frame_label = QLabel("Frame: — / —")
        self.source_frame_label.setFixedWidth(130)
        self.source_frame_label.setStyleSheet("font-weight: 600;")
        scrub_layout.addWidget(self.source_frame_label)
        self.source_scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.source_scrub_slider.setMinimum(0)
        self.source_scrub_slider.setMaximum(0)
        self.source_scrub_slider.setMinimumHeight(24)
        self.source_scrub_slider.valueChanged.connect(self._on_source_scrub_changed)
        scrub_layout.addWidget(self.source_scrub_slider, 1)
        self.source_scrub_widget.setVisible(False)
        right_panel.addWidget(self.source_scrub_widget)

        label_group = QGroupBox("Labeling")
        label_layout = QVBoxLayout()
        
        label_controls = QHBoxLayout()
        label_controls.addWidget(QLabel("Behavior class:"))
        self.class_combo = QComboBox()
        self.class_combo.activated.connect(self._save_label)
        label_controls.addWidget(self.class_combo)
        self.multi_label_btn = QPushButton("Multi")
        self.multi_label_btn.setToolTip("Assign multiple labels to this clip (for OvR multi-label training)")
        self.multi_label_btn.setFixedWidth(50)
        self.multi_label_btn.clicked.connect(self._open_multi_label_dialog)
        label_controls.addWidget(self.multi_label_btn)
        label_layout.addLayout(label_controls)
        
        class_buttons_layout = QHBoxLayout()
        self.add_class_btn = QPushButton("Add...")
        self.add_class_btn.setToolTip("Add new behavior class")
        self.add_class_btn.clicked.connect(self._add_class)
        self.rename_class_btn = QPushButton("Rename...")
        self.rename_class_btn.setToolTip("Rename selected behavior class")
        self.rename_class_btn.clicked.connect(self._rename_class)
        self.remove_class_btn = QPushButton("Remove...")
        self.remove_class_btn.setToolTip("Remove selected behavior class")
        self.remove_class_btn.clicked.connect(self._remove_class)
        class_buttons_layout.addWidget(self.add_class_btn)
        class_buttons_layout.addWidget(self.rename_class_btn)
        class_buttons_layout.addWidget(self.remove_class_btn)
        label_layout.addLayout(class_buttons_layout)
        
        self._update_class_combo()
        
        label_layout.addWidget(QLabel("Keyboard shortcuts: 1-9 for classes, Space to play/pause"))

        # Per-frame labeling controls (integrated into label group)
        fl_separator = QLabel("Per-frame labeling:")
        fl_separator.setStyleSheet("font-weight: 600; font-size: 11px; margin-top: 4px;")
        fl_separator.setToolTip(
            "Label individual frame ranges within a clip.\n"
            "Navigate to a frame, click 'Mark In', navigate to end, click 'Mark Out',\n"
            "then click 'Apply' to assign the selected class to that range."
        )
        label_layout.addWidget(fl_separator)

        self.frame_label_bar = QLabel()
        self.frame_label_bar.setFixedHeight(14)
        self.frame_label_bar.setStyleSheet("background: #e5e7eb; border: 1px solid #c0c8d2; border-radius: 3px;")
        label_layout.addWidget(self.frame_label_bar)

        fl_controls = QHBoxLayout()
        fl_controls.setSpacing(3)
        self.fl_mark_in_btn = QPushButton("In")
        self.fl_mark_in_btn.setToolTip("Set start of frame range (current frame)")
        self.fl_mark_in_btn.setFixedWidth(30)
        self.fl_mark_in_btn.clicked.connect(self._fl_mark_in)
        fl_controls.addWidget(self.fl_mark_in_btn)

        self.fl_range_label = QLabel("—")
        self.fl_range_label.setFixedWidth(55)
        self.fl_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fl_range_label.setStyleSheet("font-size: 11px;")
        fl_controls.addWidget(self.fl_range_label)

        self.fl_mark_out_btn = QPushButton("Out")
        self.fl_mark_out_btn.setToolTip("Set end of frame range (current frame)")
        self.fl_mark_out_btn.setFixedWidth(32)
        self.fl_mark_out_btn.clicked.connect(self._fl_mark_out)
        fl_controls.addWidget(self.fl_mark_out_btn)

        self.fl_class_combo = QComboBox()
        self.fl_class_combo.setMinimumWidth(60)
        fl_controls.addWidget(self.fl_class_combo, 1)

        self.fl_apply_btn = QPushButton("Apply")
        self.fl_apply_btn.setToolTip("Assign selected class to marked frame range")
        self.fl_apply_btn.setFixedWidth(45)
        self.fl_apply_btn.setStyleSheet(
            "QPushButton { background: #4a5568; color: #e2e8f0; font-weight: 600; border-radius: 4px; }"
            "QPushButton:hover { background: #5a6578; }"
        )
        self.fl_apply_btn.clicked.connect(self._fl_apply_label)
        fl_controls.addWidget(self.fl_apply_btn)

        self.fl_clear_btn = QPushButton("Clr")
        self.fl_clear_btn.setToolTip("Clear all per-frame labels for this clip")
        self.fl_clear_btn.setFixedWidth(30)
        self.fl_clear_btn.clicked.connect(self._fl_clear_labels)
        fl_controls.addWidget(self.fl_clear_btn)

        label_layout.addLayout(fl_controls)

        label_group.setLayout(label_layout)
        right_panel.addWidget(_wrap_scroll(label_group, max_height=260, min_height=200))

        # State for per-frame labeling
        self._fl_mark_in_frame = None
        self._fl_mark_out_frame = None
        self._fl_current_labels = []  # list of class name or None per frame

        # Bbox for Localization and Hard-Negative Round side by side
        spatial_round_row = QHBoxLayout()
        spatial_round_row.setContentsMargins(0, 0, 0, 0)
        spatial_group = QGroupBox("Bbox for Localization (optional)")
        spatial_layout = QVBoxLayout()

        bbox_toggle_layout = QHBoxLayout()
        self.bbox_check = QCheckBox("Draw localization bbox")
        self.bbox_check.setToolTip(
            "Draw a bounding box ROI for this clip. The box is saved in normalized "
            "coordinates and can supervise localization during training."
        )
        self.bbox_check.toggled.connect(self._on_bbox_toggled)
        bbox_toggle_layout.addWidget(self.bbox_check)
        bbox_toggle_layout.addStretch()
        spatial_layout.addLayout(bbox_toggle_layout)

        bbox_btn_layout = QHBoxLayout()
        self.save_bbox_btn = QPushButton("Save bbox")
        self.save_bbox_btn.setToolTip("Save current drawn bbox for this clip")
        self.save_bbox_btn.clicked.connect(self._save_spatial_bbox)
        self.save_bbox_btn.setEnabled(False)
        bbox_btn_layout.addWidget(self.save_bbox_btn)

        self.clear_bbox_btn = QPushButton("Clear bbox")
        self.clear_bbox_btn.setToolTip("Clear saved bbox from this clip")
        self.clear_bbox_btn.clicked.connect(self._clear_spatial_bbox)
        self.clear_bbox_btn.setEnabled(False)
        bbox_btn_layout.addWidget(self.clear_bbox_btn)

        self.bbox_info_label = QLabel("")
        bbox_btn_layout.addWidget(self.bbox_info_label)
        bbox_btn_layout.addStretch()
        spatial_layout.addLayout(bbox_btn_layout)

        spatial_group.setLayout(spatial_layout)

        self.round_group = QGroupBox("Hard-Negative Round Dataset (optional)")
        round_layout = QFormLayout()
        self.round_name_edit = QLineEdit("rearing_round2")
        round_layout.addRow("Round name:", self.round_name_edit)
        self.round_target_list = QListWidget()
        self.round_target_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.round_target_list.setMaximumHeight(80)
        round_layout.addRow("Target classes (1+):", self.round_target_list)
        self.round_near_list = QListWidget()
        self.round_near_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.round_near_list.setMaximumHeight(80)
        self.round_near_list.setToolTip(
            "Select one or more near-negative classes to build the negative pool."
        )
        round_layout.addRow("Near-negative classes (1+):", self.round_near_list)
        self.round_neg_per_pos_spin = QSpinBox()
        self.round_neg_per_pos_spin.setRange(1, 10)
        self.round_neg_per_pos_spin.setValue(1)
        round_layout.addRow("Negatives per positive:", self.round_neg_per_pos_spin)
        self.round_negative_output_edit = QLineEdit("other")
        round_layout.addRow("Output negative label:", self.round_negative_output_edit)
        round_build_row = QHBoxLayout()
        self.build_round_btn = QPushButton("Build round dataset")
        self.build_round_btn.clicked.connect(self._build_hard_negative_round_dataset)
        round_build_row.addWidget(self.build_round_btn)
        self.round_status_label = QLabel("")
        self.round_status_label.setWordWrap(True)
        round_build_row.addWidget(self.round_status_label, 1)
        round_layout.addRow(round_build_row)
        self.round_group.setLayout(round_layout)

        self.round_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        spatial_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        spatial_round_row.addWidget(_wrap_scroll(spatial_group, max_height=220, min_height=160), 1)
        spatial_round_row.addWidget(_wrap_scroll(self.round_group, max_height=220, min_height=160), 1)
        right_panel.addLayout(spatial_round_row)
        
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.setInterval(100)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_zoom_buttons()
        if hasattr(self, '_fl_current_labels'):
            self._fl_update_bar()

    def _style_zoom_button(self, btn):
        btn.setFixedSize(34, 34)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(20, 20, 20, 190);"
            "color: white;"
            "border: 1px solid rgba(255,255,255,110);"
            "border-radius: 17px;"
            "font-size: 18px;"
            "font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "background-color: rgba(45, 45, 45, 220);"
            "}"
        )

    def _position_zoom_buttons(self):
        if not hasattr(self, "video_scroll") or not hasattr(self, "btn_zoom_in"):
            return
        viewport = self.video_scroll.viewport()
        margin = 10
        spacing = 8
        x = viewport.width() - self.btn_zoom_in.width() - margin
        y = margin
        self.btn_zoom_in.move(x, y)
        self.btn_zoom_out.move(x, y + self.btn_zoom_in.height() + spacing)
        self.btn_zoom_in.raise_()
        self.btn_zoom_out.raise_()

    def _zoom_in(self):
        self.zoom_factor = min(self.zoom_max, self.zoom_factor + self.zoom_step)
        self._apply_zoom()

    def _zoom_out(self):
        self.zoom_factor = max(self.zoom_min, self.zoom_factor - self.zoom_step)
        self._apply_zoom()

    def _apply_zoom(self):
        if self._base_display_pixmap is None or self._base_display_pixmap.isNull():
            return
        w = max(1, int(self._base_display_pixmap.width() * self.zoom_factor))
        h = max(1, int(self._base_display_pixmap.height() * self.zoom_factor))
        scaled = self._base_display_pixmap.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self.video_label.resize(scaled.size())
        self._position_zoom_buttons()
    
    def _update_class_combo(self):
        """Update class combo box with current classes."""
        self.class_combo.clear()
        classes = self.annotation_manager.get_classes()
        if classes:
            self.class_combo.addItem("(No Label)")
            self.class_combo.addItems(classes)
            self.class_combo.setEnabled(True)
        else:
            self.class_combo.addItem("(No classes - add one first)")
            self.class_combo.setEnabled(False)
            
        if self.fullscreen_dialog:
            self.fullscreen_dialog.update_classes()
        self._sync_round_builder_classes(classes)
        if hasattr(self, 'fl_class_combo'):
            self._fl_update_class_combo()
    
    def _add_class(self):
        """Add a new behavior class."""
        from PyQt6.QtWidgets import QInputDialog
        class_name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if ok and class_name.strip():
            class_name = class_name.strip()
            existing_classes = self.annotation_manager.get_classes()
            if class_name in existing_classes:
                QMessageBox.warning(self, "Error", f"Class '{class_name}' already exists.")
                return
            self.annotation_manager.add_class(class_name)
            self._update_class_combo()
            self.class_combo.setEnabled(True)
    
    def _remove_class(self):
        """Remove a behavior class."""
        classes = self.annotation_manager.get_classes()
        if not classes:
            QMessageBox.information(self, "Info", "No classes to remove.")
            return
        
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        class_name, ok = QInputDialog.getItem(
            self,
            "Remove Class",
            "Select class to remove:",
            classes,
            0,
            False
        )
        
        if ok and class_name:
            clips_with_class = [
                clip for clip in self.annotation_manager.get_all_clips()
                if clip.get("label") == class_name
            ]
            
            if clips_with_class:
                reply = QMessageBox.question(
                    self,
                    "Confirm Removal",
                    f"Class '{class_name}' is used by {len(clips_with_class)} clip(s).\n\n"
                    "Removing this class will also remove labels from those clips.\n\n"
                    "Do you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply != QMessageBox.StandardButton.Yes:
                    return
                
                for clip in clips_with_class:
                    clip_id = clip["id"]
                    self.annotation_manager.add_clip(clip_id, "", clip.get("meta"))
            
            self.annotation_manager.remove_class(class_name)
            self._update_class_combo()
            
            if not self.annotation_manager.get_classes():
                self.class_combo.setEnabled(False)
    
    def _rename_class(self):
        """Rename the currently selected class."""
        current_class = self.class_combo.currentText()
        if not current_class or current_class.startswith("(No"):
            QMessageBox.warning(self, "Error", "Please select a valid class to rename.")
            return

        from PyQt6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename Class", 
                                          f"Rename '{current_class}' to:", 
                                          text=current_class)
        
        if ok and new_name.strip():
            new_name = new_name.strip()
            if new_name == current_class:
                return
                
            if self.annotation_manager.rename_class(current_class, new_name):
                self._update_class_combo()
                self.refresh_clip_list()
                
                # Reselect the renamed class
                idx = self.class_combo.findText(new_name)
                if idx >= 0:
                    self.class_combo.setCurrentIndex(idx)
            else:
                QMessageBox.warning(self, "Error", "Failed to rename class.")

    def _on_class_filter_changed(self, class_name):
        self.refresh_clip_list()
    
    def refresh_clip_list(self):
        """Refresh the clip list from disk, applying all filters."""
        self.annotation_manager.reload()
        # Store current selection
        current_item = self.clip_list.currentItem()
        current_clip_path = current_item.data(Qt.ItemDataRole.UserRole) if current_item else None
        
        self.clip_list.clear()
        
        if not os.path.exists(self.clip_base_dir):
            return
        
        # 1. Gather all clips
        clips = []
        video_dirs = set()
        for root, dirs, files in os.walk(self.clip_base_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    rel_path = os.path.relpath(os.path.join(root, file), self.clip_base_dir)
                    clips.append(rel_path.replace('\\', '/'))
                    video_dir = os.path.dirname(rel_path)
                    if video_dir:
                        video_dirs.add(video_dir)
        
        clips.sort()
        video_dirs = sorted(video_dirs)
        
        # 2. Update Video Filter Combo (preserve selection)
        selected_video = self.video_filter_combo.currentText()
        self.video_filter_combo.blockSignals(True)
        self.video_filter_combo.clear()
        self.video_filter_combo.addItem("All Videos")
        for video_dir in video_dirs:
            self.video_filter_combo.addItem(video_dir)
        
        if selected_video in [self.video_filter_combo.itemText(i) for i in range(self.video_filter_combo.count())]:
            self.video_filter_combo.setCurrentText(selected_video)
        else:
            self.video_filter_combo.setCurrentText("All Videos")
            selected_video = "All Videos"
        self.video_filter_combo.blockSignals(False)
        
        # 3. Update Class Filter Combo (preserve selection)
        classes = self.annotation_manager.get_classes()
        selected_class = self.class_filter_combo.currentText()
        
        self.class_filter_combo.blockSignals(True)
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Classes")
        self.class_filter_combo.addItem("Unlabeled")
        self.class_filter_combo.addItems(classes)
        
        if selected_class in [self.class_filter_combo.itemText(i) for i in range(self.class_filter_combo.count())]:
            self.class_filter_combo.setCurrentText(selected_class)
        else:
            self.class_filter_combo.setCurrentText("All Classes")
            selected_class = "All Classes"
        self.class_filter_combo.blockSignals(False)
        
        # 4. Filter Clips
        filtered_clips = clips
        if selected_video != "All Videos":
            filtered_clips = [c for c in filtered_clips if c.startswith(selected_video + "/")]
        
        for clip_path in filtered_clips:
            label = self.annotation_manager.get_clip_label(clip_path)
            clip_labels = self.annotation_manager.get_clip_labels(clip_path)
            
            # Apply Class Filter
            if selected_class == "All Classes":
                pass
            elif selected_class == "Unlabeled":
                if label:
                    continue
            else:
                if selected_class not in clip_labels:
                    continue
            
            display_text = clip_path
            if clip_labels and len(clip_labels) > 1:
                display_text += f" [{', '.join(clip_labels)}]"
            elif label:
                display_text += f" [{label}]"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, clip_path)
            self.clip_list.addItem(item)
            
            # Restore selection
            if current_clip_path and clip_path == current_clip_path:
                self.clip_list.setCurrentItem(item)
    
    def _on_video_filter_changed(self, video_name: str):
            self.refresh_clip_list()
    
    def _filter_unlabeled(self):
        """Filter to show only unlabeled clips."""
        # Find "Unlabeled" in combo box and select it
        idx = self.class_filter_combo.findText("Unlabeled")
        if idx >= 0:
            self.class_filter_combo.setCurrentIndex(idx)
        else:
            # Fallback if somehow missing
            self.refresh_clip_list()

    def _show_all_clips(self):
        """Reset class filter to show all clips. Video filter is preserved."""
        self.class_filter_combo.setCurrentIndex(0)
        self.refresh_clip_list()
    
    def _on_clip_selected(self, item: QListWidgetItem):
        """Handle clip selection."""
        clip_path = item.data(Qt.ItemDataRole.UserRole)
        self._load_clip(clip_path)
    
    def _load_clip(self, clip_path: str):
        """Load a clip for viewing."""
        full_path = os.path.join(self.clip_base_dir, clip_path)
        if not os.path.exists(full_path):
            QMessageBox.warning(self, "Error", f"Clip not found: {full_path}")
            return
        
        self.current_clip_path = clip_path
        cap = cv2.VideoCapture(full_path)
        self.current_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frames.append(frame_rgb)
        
        cap.release()
        
        if not self.current_frames:
            QMessageBox.warning(self, "Error", "Could not load frames from clip.")
            return
        
        self.current_frame_idx = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, len(self.current_frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(len(self.current_frames) > 1)
        self.frame_slider.blockSignals(False)
        self._load_per_frame_bboxes()
        self._display_frame()
        self.play_pause_btn.setEnabled(True)
        
        # Update Full Screen Info
        if self.fullscreen_dialog and self.fullscreen_dialog.isVisible():
            self.fullscreen_dialog.update_info(os.path.basename(clip_path))
        
        label = self.annotation_manager.get_clip_label(clip_path)
        target_idx = -1
        
        if label:
            idx = self.class_combo.findText(label)
            if idx >= 0:
                target_idx = idx
        else:
            # Set to "No Label" if unlabeled
            idx = self.class_combo.findText("(No Label)")
            if idx >= 0:
                target_idx = idx
        
        if target_idx >= 0:
            self.class_combo.setCurrentIndex(target_idx)
            # Sync to fullscreen
            if self.fullscreen_dialog:
                self.fullscreen_dialog.class_combo.setCurrentIndex(target_idx)
        
        if self.bbox_check.isChecked():
            self._load_spatial_bbox_for_clip()
        # Load per-frame labels
        self._fl_load_labels()
        # Sync per-frame class dropdown to this clip (avoid showing previous clip's class)
        if hasattr(self, "fl_class_combo") and self.fl_class_combo.count():
            if self._fl_current_labels:
                from collections import Counter
                labels = [l for l in self._fl_current_labels if l is not None]
                target = Counter(labels).most_common(1)[0][0] if labels else label
            else:
                target = label
            if target and (idx := self.fl_class_combo.findText(target)) >= 0:
                self.fl_class_combo.setCurrentIndex(idx)

    def _open_fullscreen(self):
        """Open full screen labeling view."""
        if not self.fullscreen_dialog:
            self.fullscreen_dialog = FullScreenLabelingDialog(self)
        
        self.fullscreen_dialog.update_classes()
        self.fullscreen_dialog.update_scrubber()
        self.fullscreen_dialog.showFullScreen()
        self._refresh_fullscreen_from_current_state()
        QTimer.singleShot(0, self._refresh_fullscreen_from_current_state)
        
        # Update info text
        if self.current_clip_path:
            self.fullscreen_dialog.update_info(os.path.basename(self.current_clip_path))

    def _update_fullscreen_view(self, pixmap: QPixmap, info_text: str = None):
        if not (self.fullscreen_dialog and self.fullscreen_dialog.isVisible()):
            return
        fs_label = self.fullscreen_dialog.video_label
        target_size = fs_label.size()
        if target_size.width() < 10 or target_size.height() < 10:
            target_size = self.fullscreen_dialog.size()
        if target_size.width() < 10 or target_size.height() < 10:
            screen = self.fullscreen_dialog.screen()
            target_size = screen.availableGeometry().size() if screen else pixmap.size()
        fs_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        fs_label.setPixmap(fs_pixmap)
        if info_text:
            self.fullscreen_dialog.update_info(info_text)
        self.fullscreen_dialog.update_scrubber()
    
    def _display_frame(self):
        """Display current frame."""
        if not self.current_frames:
            return
        
        fl_txt = ""
        if (hasattr(self, '_fl_current_labels') and self._fl_current_labels
                and self.current_frame_idx < len(self._fl_current_labels)
                and self._fl_current_labels[self.current_frame_idx] is not None):
            fl_txt = f" [{self._fl_current_labels[self.current_frame_idx]}]"
        self.frame_label.setText(f"Frame: {self.current_frame_idx + 1}/{len(self.current_frames)}{fl_txt}")
        self.frame_label.setFixedWidth(max(90, self.frame_label.sizeHint().width()))
        frame = self.current_frames[self.current_frame_idx]
        h, w, c = frame.shape
        bytes_per_line = c * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Update Main Window (fit-to-viewport base, then apply zoom)
        if hasattr(self, "video_scroll") and self.video_scroll.viewport().width() > 1:
            viewport_size = self.video_scroll.viewport().size()
            self._base_display_pixmap = pixmap.scaled(
                viewport_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            self._base_display_pixmap = pixmap
        self._apply_zoom()
        
        # Update Full Screen Dialog if open
        self._update_fullscreen_view(pixmap)
        if self.fullscreen_dialog and self.fullscreen_dialog.isVisible():
            self.fullscreen_dialog.sync_per_frame_controls()
            
        if self.fullscreen_dialog and self.fullscreen_dialog.isVisible():
            txt = "Pause (Space)" if self.is_playing else "Play (Space)"
            self.fullscreen_dialog.play_btn.setText(txt)
    
    def _toggle_play(self):
        """Toggle play/pause."""
        if not self.current_frames:
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.setText("Pause")
            self.timer.start()
        else:
            self.play_pause_btn.setText("Play")
            self.timer.stop()
    
        if self.fullscreen_dialog:
            txt = "Pause (Space)" if self.is_playing else "Play (Space)"
            self.fullscreen_dialog.play_btn.setText(txt)
            
    def _prev_clip(self):
        """Go to previous clip."""
        current_row = self.clip_list.currentRow()
        if current_row > 0:
            self.clip_list.setCurrentRow(current_row - 1)
            self._on_clip_selected(self.clip_list.currentItem())
            
    def _next_clip(self):
        """Go to next clip."""
        current_row = self.clip_list.currentRow()
        if current_row < self.clip_list.count() - 1:
            self.clip_list.setCurrentRow(current_row + 1)
            self._on_clip_selected(self.clip_list.currentItem())
            
    def _random_clip(self):
        """Go to a random clip."""
        count = self.clip_list.count()
        if count > 0:
            idx = random.randint(0, count - 1)
            self.clip_list.setCurrentRow(idx)
            self._on_clip_selected(self.clip_list.currentItem())

    # -- Per-frame bbox helpers --

    def _on_frame_slider_changed(self, value):
        """User dragged the frame slider."""
        if not self.current_frames:
            return
        self._capture_current_frame_bbox()
        self.current_frame_idx = value
        self._sync_bbox_to_frame()
        self._display_frame()

    def _prev_frame(self):
        if not self.current_frames:
            return
        self._capture_current_frame_bbox()
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)
        self._sync_bbox_to_frame()
        self._display_frame()

    def _next_frame(self):
        if not self.current_frames:
            return
        self._capture_current_frame_bbox()
        self.current_frame_idx = min(len(self.current_frames) - 1, self.current_frame_idx + 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)
        self._sync_bbox_to_frame()
        self._display_frame()

    def _on_bbox_moved(self):
        """Auto-advance to next frame after user finishes repositioning the bbox."""
        self._capture_current_frame_bbox()
        if self.current_frames and self.current_frame_idx < len(self.current_frames) - 1:
            self.current_frame_idx += 1
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_idx)
            self.frame_slider.blockSignals(False)
            self._sync_bbox_to_frame()
            self._display_frame()

    def _capture_current_frame_bbox(self):
        """Store the current on-screen bbox into frame_bboxes for the current frame."""
        bbox = self.video_label.get_bbox_norm()
        if bbox:
            self.frame_bboxes[self.current_frame_idx] = bbox
        # Don't remove if bbox is None — user may not have drawn one for this frame

    def _sync_bbox_to_frame(self):
        """Load the per-frame bbox for current_frame_idx into the overlay."""
        if not self.bbox_check.isChecked():
            return
        bbox = self.frame_bboxes.get(self.current_frame_idx)
        if bbox:
            self.video_label.set_bbox_norm(tuple(bbox))
        else:
            # Carry forward from nearest previous frame that has a bbox
            prev_bbox = None
            for fi in range(self.current_frame_idx - 1, -1, -1):
                if fi in self.frame_bboxes:
                    prev_bbox = self.frame_bboxes[fi]
                    break
            if prev_bbox:
                self.video_label.set_bbox_norm(tuple(prev_bbox))
            else:
                self.video_label.clear_bbox()
        self._update_bbox_info()

    def _load_per_frame_bboxes(self):
        """Load saved per-frame bboxes from annotations into self.frame_bboxes."""
        self.frame_bboxes = {}
        if not self.current_clip_path:
            return
        frames_data = self.annotation_manager.get_spatial_bbox_frames(self.current_clip_path)
        if frames_data:
            for i, b in enumerate(frames_data):
                if b is not None and len(b) == 4:
                    self.frame_bboxes[i] = list(b)
        else:
            # Fall back to legacy single bbox (apply to frame 0)
            bbox = self.annotation_manager.get_spatial_bbox(self.current_clip_path)
            if bbox:
                self.frame_bboxes[0] = list(bbox)

    def _update_frame(self):
        """Update frame during playback."""
        if not self.current_frames:
            return
        
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.current_frames)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)
        self._sync_bbox_to_frame()
        self._display_frame()
    
    def _save_label(self):
        """Save label for current clip."""
        if not self.current_clip_path:
            return
        
        label = self.class_combo.currentText()
        if label == "(No classes - add one first)":
            QMessageBox.warning(self, "Error", "Please add a class first, then select a label.")
            return
            
        if label == "(No Label)":
            # Unlabel the clip (remove from annotations)
            self.annotation_manager.remove_clip(self.current_clip_path)
            self.refresh_clip_list()
            return
        
        if not label:
            return
        
        video_name = os.path.dirname(self.current_clip_path)
        clip_name = os.path.basename(self.current_clip_path)
        
        meta = {
            "source_video": video_name,
            "clip_name": clip_name
        }
        
        self.annotation_manager.add_clip(self.current_clip_path, label, meta)
        # Keep per-frame labels in sync: set all frames to this clip-level label
        if self.current_frames:
            T = len(self.current_frames)
            self.annotation_manager.set_frame_labels(self.current_clip_path, [label] * T)
            self._fl_current_labels = [label] * T
            self._fl_update_bar()
        self.refresh_clip_list()

    # ---- Per-frame labeling methods ----

    def _fl_mark_in(self):
        """Set the start of a frame range for per-frame labeling."""
        if not self.current_frames:
            return
        self._fl_mark_in_frame = self.current_frame_idx
        if self._fl_mark_out_frame is not None and self._fl_mark_out_frame < self._fl_mark_in_frame:
            self._fl_mark_out_frame = None
        self._fl_update_range_label()

    def _fl_mark_out(self):
        """Set the end of a frame range for per-frame labeling."""
        if not self.current_frames:
            return
        self._fl_mark_out_frame = self.current_frame_idx
        if self._fl_mark_in_frame is not None and self._fl_mark_in_frame > self._fl_mark_out_frame:
            self._fl_mark_in_frame = None
        self._fl_update_range_label()

    def _fl_update_range_label(self):
        """Update the range display label."""
        i = self._fl_mark_in_frame
        o = self._fl_mark_out_frame
        if i is not None and o is not None:
            self.fl_range_label.setText(f"{i+1}–{o+1}")
        elif i is not None:
            self.fl_range_label.setText(f"{i+1}–?")
        elif o is not None:
            self.fl_range_label.setText(f"?–{o+1}")
        else:
            self.fl_range_label.setText("—")
        if self.fullscreen_dialog and self.fullscreen_dialog.isVisible():
            self.fullscreen_dialog.sync_per_frame_controls()

    def _fl_apply_label(self):
        """Apply selected class to the marked frame range."""
        if not self.current_frames or not self.current_clip_path:
            return
        if self._fl_mark_in_frame is None or self._fl_mark_out_frame is None:
            QMessageBox.warning(self, "No range", "Use 'Mark In' and 'Mark Out' to select a frame range first.")
            return
        cls = self.fl_class_combo.currentText()
        if not cls:
            return
        T = len(self.current_frames)
        # Initialize labels if empty
        if not self._fl_current_labels or len(self._fl_current_labels) != T:
            self._fl_current_labels = [None] * T
        start = max(0, self._fl_mark_in_frame)
        end = min(T - 1, self._fl_mark_out_frame)
        for fi in range(start, end + 1):
            self._fl_current_labels[fi] = cls
        self._fl_save_labels()
        self._fl_update_bar()
        # Auto-derive clip-level label from majority class
        self._fl_sync_clip_label()

    def _fl_clear_labels(self):
        """Clear all per-frame labels for the current clip."""
        if not self.current_clip_path:
            return
        self._fl_current_labels = []
        self._fl_mark_in_frame = None
        self._fl_mark_out_frame = None
        self._fl_update_range_label()
        self.annotation_manager.clear_frame_labels(self.current_clip_path)
        self._fl_update_bar()

    def _fl_save_labels(self):
        """Save current per-frame labels to annotation store."""
        if not self.current_clip_path or not self._fl_current_labels:
            return
        self.annotation_manager.set_frame_labels(self.current_clip_path, self._fl_current_labels)

    def _fl_load_labels(self):
        """Load per-frame labels for the current clip from annotation store."""
        self._fl_current_labels = []
        self._fl_mark_in_frame = None
        self._fl_mark_out_frame = None
        self._fl_update_range_label()
        if not self.current_clip_path or not self.current_frames:
            self._fl_update_bar()
            return
        saved = self.annotation_manager.get_frame_labels(self.current_clip_path)
        if saved and isinstance(saved, list) and len(saved) == len(self.current_frames):
            self._fl_current_labels = list(saved)
        else:
            # Clip has no or mismatched frame_labels; fill from clip-level label for consistency
            clip_label = self.annotation_manager.get_clip_label(self.current_clip_path)
            if clip_label:
                T = len(self.current_frames)
                self._fl_current_labels = [clip_label] * T
                self.annotation_manager.set_frame_labels(self.current_clip_path, self._fl_current_labels)
        self._fl_update_bar()

    def _fl_sync_clip_label(self):
        """Set the clip-level label to the majority per-frame class."""
        labels = [l for l in self._fl_current_labels if l is not None]
        if not labels:
            return
        from collections import Counter
        majority = Counter(labels).most_common(1)[0][0]
        idx = self.class_combo.findText(majority)
        if idx >= 0:
            self.class_combo.setCurrentIndex(idx)
        meta = {
            "source_video": os.path.dirname(self.current_clip_path),
            "clip_name": os.path.basename(self.current_clip_path),
        }
        self.annotation_manager.add_clip(self.current_clip_path, majority, meta)
        self.refresh_clip_list()

    def _fl_update_bar(self):
        """Redraw the color bar showing per-frame labels."""
        if not self.current_frames:
            self.frame_label_bar.setPixmap(QPixmap())
            if self.fullscreen_dialog and self.fullscreen_dialog.isVisible():
                self.fullscreen_dialog.sync_per_frame_controls()
            return
        T = len(self.current_frames)
        bar_w = max(self.frame_label_bar.width(), 100)
        bar_h = self.frame_label_bar.height()
        pixmap = QPixmap(bar_w, bar_h)
        pixmap.fill(QColor("#e5e7eb"))
        if self._fl_current_labels and len(self._fl_current_labels) == T:
            from PyQt6.QtGui import QPainter
            painter = QPainter(pixmap)
            classes = self.annotation_manager.get_classes()
            # Consistent colors per class
            palette = [
                "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7",
                "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
            ]
            cls_color = {}
            for i, c in enumerate(classes):
                cls_color[c] = QColor(palette[i % len(palette)])
            for fi in range(T):
                lbl = self._fl_current_labels[fi]
                if lbl is not None and lbl in cls_color:
                    x0 = int(fi * bar_w / T)
                    x1 = int((fi + 1) * bar_w / T)
                    painter.fillRect(x0, 0, x1 - x0, bar_h, cls_color[lbl])
            painter.end()
        self.frame_label_bar.setPixmap(pixmap)
        if self.fullscreen_dialog and self.fullscreen_dialog.isVisible():
            self.fullscreen_dialog.sync_per_frame_controls()

    def _fl_update_class_combo(self):
        """Sync the per-frame class combo with available classes."""
        current = self.fl_class_combo.currentText()
        self.fl_class_combo.clear()
        classes = self.annotation_manager.get_classes()
        self.fl_class_combo.addItems(classes)
        idx = self.fl_class_combo.findText(current)
        if idx >= 0:
            self.fl_class_combo.setCurrentIndex(idx)
        if self.fullscreen_dialog:
            self.fullscreen_dialog.update_classes()

    def _open_multi_label_dialog(self):
        """Open dialog to assign multiple labels to selected clips."""
        selected_items = self.clip_list.selectedItems()
        selected_paths = [it.data(Qt.ItemDataRole.UserRole) for it in selected_items if it.data(Qt.ItemDataRole.UserRole)]
        if not selected_paths and self.current_clip_path:
            selected_paths = [self.current_clip_path]
        if not selected_paths:
            QMessageBox.warning(self, "No clip", "Select one or more clips first.")
            return
        classes = self.annotation_manager.get_classes()
        if not classes:
            QMessageBox.warning(self, "No classes", "Add behavior classes first.")
            return

        current_labels = self.annotation_manager.get_clip_labels(selected_paths[0])

        dlg = QDialog(self)
        dlg.setWindowTitle("Multi-label assignment")
        dlg.setMinimumWidth(260)
        layout = QVBoxLayout(dlg)
        if len(selected_paths) == 1:
            prompt = f"Select labels for:\n{os.path.basename(selected_paths[0])}"
        else:
            prompt = f"Select labels for {len(selected_paths)} selected clips"
        layout.addWidget(QLabel(prompt))

        checks = {}
        for cls in classes:
            cb = QCheckBox(cls)
            cb.setChecked(cls in current_labels)
            layout.addWidget(cb)
            checks[cls] = cb

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec():
            selected = [cls for cls, cb in checks.items() if cb.isChecked()]
            for clip_path in selected_paths:
                if not selected:
                    self.annotation_manager.remove_clip(clip_path)
                    continue
                video_name = os.path.dirname(clip_path)
                clip_name = os.path.basename(clip_path)
                meta = {"source_video": video_name, "clip_name": clip_name}
                self.annotation_manager.add_clip(clip_path, selected, meta)
            # Sync combo to first selected label
            if selected:
                idx = self.class_combo.findText(selected[0])
                if idx >= 0:
                    self.class_combo.setCurrentIndex(idx)
            self.refresh_clip_list()

    def _next_unlabeled(self):
        """Select next unlabeled clip."""
        unlabeled_items = []
        for i in range(self.clip_list.count()):
            item = self.clip_list.item(i)
            clip_path = item.data(Qt.ItemDataRole.UserRole)
            if not self.annotation_manager.get_clip_label(clip_path):
                unlabeled_items.append((i, item))
        
        if not unlabeled_items:
            QMessageBox.information(self, "Info", "No unlabeled clips found.")
            return
        
        current_row = self.clip_list.currentRow()
        for idx, item in unlabeled_items:
            if idx > current_row:
                self.clip_list.setCurrentItem(item)
                self._on_clip_selected(item)
                return
        
        idx, item = unlabeled_items[0]
        self.clip_list.setCurrentItem(item)
        self._on_clip_selected(item)
    
    def _on_bbox_toggled(self, enabled: bool):
        """Toggle bbox drawing overlay."""
        self.video_label.set_bbox_enabled(enabled)
        self.save_bbox_btn.setEnabled(enabled)
        self.clear_bbox_btn.setEnabled(enabled)
        if enabled:
            self._load_spatial_bbox_for_clip()

    def _save_spatial_bbox(self):
        """Save per-frame bboxes for this clip. Captures the current frame's bbox first."""
        if not self.current_clip_path:
            return
        self._capture_current_frame_bbox()
        if not self.frame_bboxes:
            QMessageBox.information(self, "Info", "No bbox drawn. Enable bbox mode and drag on frames first.")
            return
        # Build full per-frame list (None for unannotated frames)
        num_frames = len(self.current_frames) if self.current_frames else 1
        frame_list = [self.frame_bboxes.get(i) for i in range(num_frames)]
        self.annotation_manager.set_spatial_bbox_frames(self.current_clip_path, frame_list)
        self._update_bbox_info()

    def _clear_spatial_bbox(self):
        """Clear all saved bboxes (per-frame and legacy) from the current clip."""
        if not self.current_clip_path:
            return
        self.annotation_manager.clear_spatial_bbox(self.current_clip_path)
        self.annotation_manager.clear_spatial_bbox_frames(self.current_clip_path)
        self.frame_bboxes = {}
        self.video_label.clear_bbox()
        self._update_bbox_info()

    def _update_bbox_info(self):
        """Update bbox info label with per-frame status."""
        if not self.current_clip_path:
            self.bbox_info_label.setText("")
            return
        saved_frames = self.annotation_manager.get_spatial_bbox_frames(self.current_clip_path)
        n_total = len(self.current_frames) if self.current_frames else 0
        if saved_frames:
            n_saved = sum(1 for b in saved_frames if b is not None)
            self.bbox_info_label.setText(f"Saved: {n_saved}/{len(saved_frames)} frames")
        elif self.frame_bboxes:
            n_drawn = len(self.frame_bboxes)
            self.bbox_info_label.setText(f"Drawn: {n_drawn}/{n_total} frames (unsaved)")
        elif self.annotation_manager.get_spatial_bbox(self.current_clip_path):
            self.bbox_info_label.setText("Saved bbox (legacy single-frame)")
        else:
            self.bbox_info_label.setText("")

    def _load_spatial_bbox_for_clip(self):
        """Load saved bbox(es) for current clip into overlay."""
        if not self.current_clip_path:
            return
        self._load_per_frame_bboxes()
        self._sync_bbox_to_frame()
        self._update_bbox_info()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() >= Qt.Key.Key_1 and event.key() <= Qt.Key.Key_9:
            idx = event.key() - Qt.Key.Key_1
            if idx < self.class_combo.count():
                self.class_combo.setCurrentIndex(idx)
        elif event.key() == Qt.Key.Key_Space:
            self._toggle_play()
        elif event.key() == Qt.Key.Key_Q:
            self._prev_frame()
        elif event.key() == Qt.Key.Key_E:
            self._next_frame()
        elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._save_label()
        else:
            super().keyPressEvent(event)

    def _normalize_video_path(self, path: str) -> str:
        if not path:
            return ""
        return os.path.abspath(path).replace("\\", "/")

    def _sync_round_builder_classes(self, classes):
        if not hasattr(self, "round_target_list"):
            return
        classes = list(classes or [])
        selected = {item.text() for item in self.round_target_list.selectedItems()}
        selected_near = set()
        if hasattr(self, "round_near_list"):
            selected_near = {item.text() for item in self.round_near_list.selectedItems()}
        self.round_target_list.clear()
        self.round_target_list.addItems(classes)
        for i in range(self.round_target_list.count()):
            if self.round_target_list.item(i).text() in selected:
                self.round_target_list.item(i).setSelected(True)
        if hasattr(self, "round_near_list"):
            self.round_near_list.clear()
            self.round_near_list.addItems(classes)
            default_near = {c for c in classes if c.startswith("near_negative")}
            to_select = selected_near or default_near
            for i in range(self.round_near_list.count()):
                if self.round_near_list.item(i).text() in to_select:
                    self.round_near_list.item(i).setSelected(True)

    def open_timeline_import_dialog(self):
        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Source Videos for Timeline Labeling",
            self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos")),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not video_paths:
            return
        self.add_source_videos(video_paths, select_last=True)

    def add_source_videos(self, video_paths, select_last=False):
        if not video_paths:
            return
        from .video_utils import ensure_videos_in_experiment
        ensured = ensure_videos_in_experiment(video_paths, self.config, self)
        for vp in ensured:
            npath = self._normalize_video_path(vp)
            if not os.path.exists(npath):
                continue
            if npath not in self.source_video_paths:
                self.source_video_paths.append(npath)
        self._refresh_source_video_list(select_path=self._normalize_video_path(ensured[-1]) if (ensured and select_last) else None)

    def _refresh_source_video_list(self, select_path=None):
        if not hasattr(self, "source_video_list"):
            return
        self.source_video_list.clear()
        deduped = []
        for p in self.source_video_paths:
            if p not in deduped:
                deduped.append(p)
        self.source_video_paths = deduped
        for path in sorted(self.source_video_paths):
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.source_video_list.addItem(item)
            if select_path and self._normalize_video_path(path) == self._normalize_video_path(select_path):
                self.source_video_list.setCurrentItem(item)

    def _remove_selected_source_video(self):
        item = self.source_video_list.currentItem()
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        self.source_video_paths = [p for p in self.source_video_paths if self._normalize_video_path(p) != self._normalize_video_path(path)]
        if self.current_source_video_path and self._normalize_video_path(self.current_source_video_path) == self._normalize_video_path(path):
            self._close_current_source_video()
        self._refresh_source_video_list()

    def _clear_source_videos(self):
        self.source_video_paths = []
        self._close_current_source_video()
        self._refresh_source_video_list()

    def _close_current_source_video(self):
        if self.current_source_cap is not None:
            self.current_source_cap.release()
        self.current_source_cap = None
        self.current_source_video_path = None
        self.current_source_frame = 0
        self.current_source_frame_count = 0
        if hasattr(self, "source_scrub_slider"):
            self.source_scrub_slider.setMaximum(0)
            self.source_scrub_slider.setValue(0)
        if hasattr(self, "source_scrub_widget"):
            self.source_scrub_widget.setVisible(False)
        self.source_frame_label.setText("Frame: — / —")
        if not self.current_clip_path:
            self.video_label.setText("No clip selected")
            self.video_label.setPixmap(QPixmap())
        if self.fullscreen_dialog:
            self.fullscreen_dialog.update_scrubber()

    def _on_source_video_selected(self):
        item = self.source_video_list.currentItem()
        if not item:
            return
        self._load_source_video(item.data(Qt.ItemDataRole.UserRole))

    def _load_source_video(self, path: str):
        npath = self._normalize_video_path(path)
        if not os.path.exists(npath):
            QMessageBox.warning(self, "Missing video", f"Video not found:\n{npath}")
            return
        if self.current_source_cap is not None:
            self.current_source_cap.release()
        cap = cv2.VideoCapture(npath)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", f"Could not open video:\n{npath}")
            return
        self.current_source_cap = cap
        self.current_source_video_path = npath
        self.current_source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_source_frame = 0
        self.current_clip_path = None
        self.current_frames = []
        self.is_playing = False
        self.timer.stop()
        self.play_pause_btn.setText("Play")
        self.play_pause_btn.setEnabled(False)
        max_frame = max(0, self.current_source_frame_count - 1)
        self.source_scrub_slider.blockSignals(True)
        self.source_scrub_slider.setMinimum(0)
        self.source_scrub_slider.setMaximum(max_frame)
        self.source_scrub_slider.setValue(0)
        self.source_scrub_slider.blockSignals(False)
        self.source_scrub_widget.setVisible(True)
        self._display_source_frame(0)

    def _read_frame_at(self, frame_idx: int):
        if self.current_source_cap is None:
            return None
        idx = max(0, min(frame_idx, max(0, self.current_source_frame_count - 1)))
        self.current_source_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.current_source_cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _display_source_frame(self, frame_idx: int):
        frame_rgb = self._read_frame_at(frame_idx)
        if frame_rgb is None:
            return
        self.current_source_frame = int(frame_idx)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if hasattr(self, "video_scroll") and self.video_scroll.viewport().width() > 1:
            viewport_size = self.video_scroll.viewport().size()
            self._base_display_pixmap = pixmap.scaled(
                viewport_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            self._base_display_pixmap = pixmap
        self._apply_zoom()
        self.source_frame_label.setText(f"Frame: {self.current_source_frame + 1}/{max(1, self.current_source_frame_count)}")
        self._update_fullscreen_view(
            pixmap,
            info_text=(
                f"{os.path.basename(self.current_source_video_path or '')}  "
                f"[{self.current_source_frame + 1}/{max(1, self.current_source_frame_count)}]"
            ),
        )

    def _on_source_scrub_changed(self, value: int):
        if not self._is_source_video_mode():
            return
        self._display_source_frame(value)
        if self.fullscreen_dialog and hasattr(self.fullscreen_dialog, "fs_position_slider"):
            self.fullscreen_dialog.fs_position_slider.blockSignals(True)
            self.fullscreen_dialog.fs_position_slider.setValue(int(value))
            self.fullscreen_dialog.fs_position_slider.blockSignals(False)

    def _is_source_video_mode(self) -> bool:
        """True when timeline source-video labeling is active (not clip playback)."""
        return self.current_source_cap is not None and not self.current_frames

    def _refresh_fullscreen_from_current_state(self):
        """Refresh fullscreen content using current active mode/frame."""
        if not (self.fullscreen_dialog and self.fullscreen_dialog.isVisible()):
            return
        if self._is_source_video_mode():
            self._display_source_frame(self.current_source_frame)
        elif self.current_frames:
            self._display_frame()

    def _browse_output_dir(self):
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select output clips directory",
            self.ea_output_dir_edit.text().strip() or self.clip_base_dir,
        )
        if out_dir:
            self.ea_output_dir_edit.setText(out_dir)

    def _save_clip_from_frames(self, frames, out_path, fps):
        if not frames:
            return False
        h, w, _ = frames[0].shape
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return True

    def _extract_window_to_clip(self, video_path, sub_start, clip_length, frame_interval, target_fps, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        orig_start = int(sub_start * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_start)
        frames = []
        needed = clip_length
        idx = orig_start
        while needed > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if ((idx - orig_start) % frame_interval) == 0:
                frames.append(frame)
                needed -= 1
            idx += 1
        cap.release()
        if len(frames) != clip_length:
            return False
        return self._save_clip_from_frames(frames, output_path, target_fps)

    def _extract_all_clips_from_videos(self):
        """Extract all clips from source videos using a sliding window. Clips are saved as unlabeled."""
        if not self.source_video_paths:
            QMessageBox.warning(self, "No source videos", "Add source videos in the timeline section first.")
            return

        target_fps = int(self.ea_target_fps_spin.value())
        clip_length = int(self.ea_clip_length_spin.value())
        step_frames = int(self.ea_step_spin.value())
        max_clips = int(self.ea_max_clips_spin.value())

        if step_frames > clip_length:
            QMessageBox.warning(self, "Invalid params", "Step frames cannot exceed clip length.")
            return

        output_root = self.ea_output_dir_edit.text().strip() or self.clip_base_dir
        os.makedirs(output_root, exist_ok=True)
        if os.path.abspath(output_root) != os.path.abspath(self.clip_base_dir):
            self.clip_base_dir = output_root
            self.config["clips_dir"] = output_root

        self.ea_progress.setVisible(True)
        self.ea_progress.setValue(0)
        self.ea_status_label.setText("Extracting...")
        QApplication.processEvents()

        total_generated = 0

        for vid_idx, video_path in enumerate(self.source_video_paths):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                continue

            frame_interval = max(1, int(round(orig_fps / max(1, target_fps))))
            total_sub = max(1, frame_count // frame_interval)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_out_dir = os.path.join(output_root, video_name)
            os.makedirs(video_out_dir, exist_ok=True)

            # Sliding window over the whole video
            windows = []
            pos = 0
            while pos + clip_length <= total_sub:
                windows.append(pos)
                pos += max(1, step_frames)

            if max_clips > 0 and len(windows) > max_clips:
                stride = len(windows) / float(max_clips)
                windows = [windows[int(i * stride)] for i in range(max_clips)]

            per_video = 0
            for win_idx, sub_start in enumerate(windows):
                out_name = f"clip_{sub_start:06d}_{win_idx:05d}.mp4"
                out_path = os.path.join(video_out_dir, out_name)
                if os.path.exists(out_path):
                    rel_id = os.path.relpath(out_path, self.clip_base_dir).replace("\\", "/")
                    if not self.annotation_manager.get_clip_label(rel_id):
                        self.annotation_manager.add_clip(rel_id, "", meta={
                            "source_video": video_name,
                            "bulk_extracted": True,
                            "sub_start_frame": sub_start,
                        }, _defer_save=True)
                    per_video += 1
                    total_generated += 1
                    continue

                ok = self._extract_window_to_clip(
                    video_path, sub_start, clip_length, frame_interval, target_fps, out_path
                )
                if not ok:
                    continue
                rel_id = os.path.relpath(out_path, self.clip_base_dir).replace("\\", "/")
                self.annotation_manager.add_clip(rel_id, "", meta={
                    "source_video": video_name,
                    "bulk_extracted": True,
                    "sub_start_frame": sub_start,
                }, _defer_save=True)
                per_video += 1
                total_generated += 1

                if win_idx % 20 == 0:
                    pct = int(100 * (vid_idx + (win_idx / max(1, len(windows)))) / len(self.source_video_paths))
                    self.ea_progress.setValue(min(pct, 99))
                    self.ea_status_label.setText(f"{video_name}: {per_video}/{len(windows)} clips")
                    QApplication.processEvents()

            self.ea_progress.setValue(int(100 * (vid_idx + 1) / len(self.source_video_paths)))
            QApplication.processEvents()

        self.annotation_manager.save()
        self.ea_progress.setVisible(False)
        self.ea_status_label.setText(f"Extracted {total_generated} unlabeled clips")
        self.refresh_clip_list()
        QMessageBox.information(
            self, "Done",
            f"Extracted {total_generated} clips.\n"
            f"Use 'Show unlabeled only' to browse and label them."
        )

    def _build_hard_negative_round_dataset(self):
        """Create a separate round dataset using target class(es) + sampled near negatives."""
        target_labels = [item.text().strip() for item in self.round_target_list.selectedItems() if item.text().strip()]
        near_labels = [item.text().strip() for item in self.round_near_list.selectedItems() if item.text().strip()]
        negative_output_label = self.round_negative_output_edit.text().strip() or "other"
        round_name = self.round_name_edit.text().strip()
        neg_per_pos = int(self.round_neg_per_pos_spin.value())

        if not target_labels:
            QMessageBox.warning(self, "Missing target", "Select at least one target class.")
            return
        if not round_name:
            QMessageBox.warning(self, "Missing round name", "Enter a round name.")
            return
        if not near_labels:
            QMessageBox.warning(self, "Missing near negatives", "Select at least one near-negative class.")
            return
        exp_path = self.config.get("experiment_path")
        if not exp_path:
            QMessageBox.warning(self, "No experiment", "Load/create an experiment first.")
            return

        all_clips = self.annotation_manager.get_all_clips()
        target_set = set(target_labels)
        near_set = set(near_labels)
        positives = [c for c in all_clips if c.get("label") in target_set]
        near_pool = [c for c in all_clips if c.get("label") in near_set]
        if not positives:
            QMessageBox.warning(self, "No positives", f"No clips found for target(s) {target_labels}.")
            return
        if not near_pool:
            QMessageBox.warning(
                self,
                "No negatives",
                "No near-negative clips found for the selected near-negative classes.",
            )
            return

        desired_neg = max(1, len(positives) * max(1, neg_per_pos))

        rnd = random.Random(42)
        near_shuf = near_pool.copy()
        rnd.shuffle(near_shuf)

        selected_near = near_shuf[: min(desired_neg, len(near_shuf))]
        selected_neg = selected_near
        if not selected_neg:
            QMessageBox.warning(self, "No sampled negatives", "Could not sample negatives with the selected settings.")
            return

        round_root = os.path.join(exp_path, "data", "rounds", round_name)
        round_clips_dir = os.path.join(round_root, "clips")
        round_ann_dir = os.path.join(round_root, "annotations")
        round_ann_file = os.path.join(round_ann_dir, "annotations.json")
        os.makedirs(round_clips_dir, exist_ok=True)
        os.makedirs(round_ann_dir, exist_ok=True)

        selected = positives + selected_neg
        new_clips = []
        copied = 0
        missing = 0

        for clip in selected:
            clip_id = (clip.get("id") or "").replace("\\", "/")
            if not clip_id:
                continue
            src_path = os.path.join(self.clip_base_dir, clip_id)
            if not os.path.exists(src_path):
                missing += 1
                continue
            dst_path = os.path.join(round_clips_dir, clip_id)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied += 1

            out_label = clip.get("label") if clip.get("label") in target_set else negative_output_label
            meta = dict(clip.get("meta") or {})
            meta["round_dataset"] = round_name
            if out_label == negative_output_label:
                meta["hard_negative_source"] = "near"
            new_clips.append({
                "id": clip_id,
                "label": out_label,
                "meta": meta,
            })

        with open(round_ann_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "classes": list(target_labels) + [negative_output_label],
                    "clips": new_clips,
                },
                f,
                indent=2,
            )

        # Keep round dataset separate, but set explicit training overrides.
        self.config["training_clips_dir"] = round_clips_dir
        self.config["training_annotation_file"] = round_ann_file
        shortage = max(0, desired_neg - len(selected_neg))
        shortage_note = f", shortage {shortage}" if shortage else ""
        self.round_status_label.setText(
            f"Round ready: {len(positives)} pos, {len(selected_neg)} neg (near {len(selected_near)}{shortage_note})"
        )
        QMessageBox.information(
            self,
            "Round dataset built",
            "Created new round dataset and pointed Training tab to it.\n\n"
            f"Round: {round_name}\n"
            f"Clips: {round_clips_dir}\n"
            f"Annotations: {round_ann_file}\n"
            f"Copied: {copied}, Missing: {missing}",
        )

    def update_config(self, config: dict):
        """Apply new configuration (experiment change)."""
        self.config = config
        self.clip_base_dir = self.config.get("clips_dir", "data/clips")
        if hasattr(self, "ea_output_dir_edit"):
            self.ea_output_dir_edit.setText(self.clip_base_dir)
        if hasattr(self, "ea_target_fps_spin"):
            self.ea_target_fps_spin.setValue(int(self.config.get("default_target_fps", 12)))
            self.ea_clip_length_spin.setValue(int(self.config.get("default_clip_length", 8)))
            self.ea_step_spin.setValue(int(self.config.get("default_step_frames", 8)))
        self.annotation_manager = AnnotationManager(
            self.config.get("annotation_file", "data/annotations/annotations.json")
        )
        self.refresh_clip_list()
        self._update_class_combo()

