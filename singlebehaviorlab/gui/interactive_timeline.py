"""QGraphicsView-based interactive behavior timeline.

Renders a SegmentsModel as colored rectangles in a scrollable, zoomable
scene. Phase B: read-only rendering and click-to-signal. Phase C adds
drag-to-edit, right-click menus, and keyboard shortcuts.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QFont, QPen, QBrush, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsLineItem,
)

from singlebehaviorlab.backend.segments import SegmentsModel

__all__ = ["InteractiveTimeline", "SegmentItem"]

_EDGE_ZONE = 5
_ROW_HEIGHT = 60
_LABEL_MIN_WIDTH = 50


class SegmentItem(QGraphicsRectItem):
    """Visual rectangle for one segment. Stores segment index and class."""

    def __init__(self, index: int, x: float, y: float, w: float, h: float,
                 class_idx: int, color: QColor, label: str, confidence: float):
        super().__init__(x, y, w, h)
        self.seg_index = index
        self.class_idx = class_idx
        self.label = label
        self.confidence = confidence

        alpha = int(min(255, 128 + 127 * min(1.0, confidence)))
        fill = QColor(color)
        fill.setAlpha(alpha)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(QColor(0, 0, 0, 80), 0.5))
        self.setAcceptHoverEvents(True)

        if w >= _LABEL_MIN_WIDTH:
            text = QGraphicsSimpleTextItem(label, self)
            text.setFont(QFont("Arial", 8))
            text.setBrush(QBrush(QColor(0, 0, 0)))
            text.setPos(x + 3, y + 3)

    def hoverEnterEvent(self, event):
        pen = self.pen()
        pen.setWidthF(1.5)
        pen.setColor(QColor(0, 0, 0, 200))
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        pen = self.pen()
        pen.setWidthF(0.5)
        pen.setColor(QColor(0, 0, 0, 80))
        self.setPen(pen)
        super().hoverLeaveEvent(event)


class InteractiveTimeline(QGraphicsView):
    """Scrollable, zoomable behavior timeline backed by SegmentsModel."""

    segment_clicked = pyqtSignal(int)
    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHints(self.renderHints())
        self.setMinimumHeight(_ROW_HEIGHT + 30)
        self.setStyleSheet("border: 1px solid #ccc; background: white;")

        self._model: SegmentsModel | None = None
        self._colors: list[QColor] = []
        self._pixels_per_frame: float = 1.0
        self._items: list[SegmentItem] = []
        self._filter_class: int | None = None

    def set_model(self, model: SegmentsModel, colors: list[QColor]) -> None:
        self._model = model
        self._colors = list(colors)
        self._rebuild()

    def set_zoom(self, pixels_per_frame: float) -> None:
        self._pixels_per_frame = max(0.1, pixels_per_frame)
        self._rebuild()

    def set_filter(self, class_idx: int | None) -> None:
        self._filter_class = class_idx
        self._rebuild()

    def _rebuild(self) -> None:
        self._scene.clear()
        self._items.clear()
        if self._model is None:
            return

        ppf = self._pixels_per_frame
        total_width = max(100, self._model.total_frames * ppf)
        h = _ROW_HEIGHT

        self._scene.setSceneRect(0, 0, total_width, h + 20)

        for i, seg in enumerate(self._model.segments):
            x = seg.start * ppf
            w = max(1.0, seg.length * ppf)

            if seg.class_idx < 0:
                label = "Filtered"
                color = QColor(180, 180, 180)
            elif seg.class_idx < len(self._model.classes):
                label = self._model.classes[seg.class_idx]
                color = self._colors[seg.class_idx % len(self._colors)] if self._colors else QColor(100, 100, 200)
            else:
                continue

            dimmed = (
                self._filter_class is not None
                and self._filter_class >= 0
                and seg.class_idx != self._filter_class
            )
            if dimmed:
                color = QColor(240, 240, 240)

            item = SegmentItem(i, x, 0, w, h, seg.class_idx, color, label, seg.confidence)
            self._scene.addItem(item)
            self._items.append(item)

        fps = self._model.orig_fps or 30.0
        dur = self._model.total_frames / fps
        info = QGraphicsSimpleTextItem(
            f"{len(self._model)} segments | {self._model.total_frames} frames | {dur:.1f}s"
        )
        info.setFont(QFont("Arial", 7))
        info.setBrush(QBrush(QColor(80, 80, 80)))
        info.setPos(4, h + 2)
        self._scene.addItem(info)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        item = self._scene.itemAt(scene_pos, self.transform())

        if isinstance(item, SegmentItem):
            self.segment_clicked.emit(item.seg_index)
        elif self._model and self._pixels_per_frame > 0:
            frame = int(scene_pos.x() / self._pixels_per_frame)
            frame = max(0, min(self._model.total_frames - 1, frame))
            self.frame_clicked.emit(frame)
        else:
            super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            self._pixels_per_frame = max(0.1, min(50.0, self._pixels_per_frame * factor))
            self._rebuild()
            event.accept()
        else:
            super().wheelEvent(event)
