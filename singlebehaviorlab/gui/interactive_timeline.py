"""QGraphicsView-based interactive behavior timeline.

Renders a SegmentsModel as colored rectangles. Supports drag-to-resize
edges, drag-to-move, right-click context menus (delete, split, merge,
change class), keyboard shortcuts, and Ctrl+Z / Ctrl+Y undo/redo.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMenu,
)

from singlebehaviorlab.backend.segments import SegmentsModel

__all__ = ["InteractiveTimeline", "SegmentItem"]

_EDGE_PX = 6
_LANE_H = 36
_LANE_PAD = 2
_LABEL_W = 90
_TICK_H = 18
_MIN_LABEL_W = 48


def _hit_edge(item_rect, scene_x):
    if abs(scene_x - item_rect.x()) <= _EDGE_PX:
        return "left"
    if abs(scene_x - (item_rect.x() + item_rect.width())) <= _EDGE_PX:
        return "right"
    return None


class SegmentItem(QGraphicsRectItem):

    def __init__(self, index, x, y, w, h, class_idx, color, label, confidence):
        super().__init__(x, y, w, h)
        self.seg_index = index
        self.class_idx = class_idx
        self.label = label
        self.confidence = confidence
        self._base_color = QColor(color)

        alpha = int(min(255, 140 + 115 * min(1.0, confidence)))
        fill = QColor(color)
        fill.setAlpha(alpha)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(QColor(40, 40, 40, 100), 0.5))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)

        if w >= _MIN_LABEL_W:
            txt = QGraphicsSimpleTextItem(label, self)
            txt.setFont(QFont("Sans", 9, QFont.Weight.Bold))
            txt.setBrush(QBrush(QColor(255, 255, 255)))
            txt.setPos(x + 4, y + h / 2 - 7)

    def _set_highlight(self, on):
        c = QColor(self._base_color)
        if on:
            c = c.lighter(130)
        alpha = int(min(255, 140 + 115 * min(1.0, self.confidence)))
        c.setAlpha(alpha)
        self.setBrush(QBrush(c))
        pen = self.pen()
        pen.setWidthF(2.0 if on else 0.5)
        pen.setColor(QColor(0, 0, 0, 220) if on else QColor(40, 40, 40, 100))
        self.setPen(pen)

    def hoverEnterEvent(self, event):
        self._set_highlight(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._set_highlight(False)
        super().hoverLeaveEvent(event)


class InteractiveTimeline(QGraphicsView):

    segment_clicked = pyqtSignal(int)
    frame_clicked = pyqtSignal(int)
    model_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumHeight(180)
        self.setStyleSheet("background: #fafafa; border: 1px solid #bbb;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._model: SegmentsModel | None = None
        self._colors: list[QColor] = []
        self._ppf: float = 1.0
        self._items: list[SegmentItem] = []
        self._filter_class: int | None = None

        self._drag_mode: str | None = None
        self._drag_idx: int = -1
        self._drag_origin: float = 0.0
        self._drag_start_frame: int = 0
        self._drag_moved: bool = False

    def set_model(self, model: SegmentsModel, colors: list[QColor]) -> None:
        self._model = model
        self._colors = list(colors)
        self._rebuild()

    def set_zoom(self, ppf: float) -> None:
        self._ppf = max(0.1, ppf)
        if self._model:
            self._rebuild()

    def set_filter(self, class_idx: int | None) -> None:
        self._filter_class = class_idx
        self._rebuild()

    def _lane_y(self, class_idx: int) -> float:
        if not self._model or class_idx < 0 or class_idx >= len(self._model.classes):
            return 0.0
        return class_idx * (_LANE_H + _LANE_PAD)

    def _class_from_y(self, y: float) -> int:
        if not self._model:
            return -1
        idx = int(y / (_LANE_H + _LANE_PAD))
        return max(0, min(len(self._model.classes) - 1, idx))

    def _rebuild(self) -> None:
        self._scene.clear()
        self._items.clear()
        if not self._model:
            return

        n_classes = len(self._model.classes)
        ppf = self._ppf
        tw = max(200, self._model.total_frames * ppf) + _LABEL_W
        lanes_h = n_classes * (_LANE_H + _LANE_PAD)
        scene_h = lanes_h + _TICK_H + 10
        self._scene.setSceneRect(0, 0, tw, scene_h)

        self._draw_lane_backgrounds(n_classes, tw, lanes_h)

        for i, seg in enumerate(self._model.segments):
            x = _LABEL_W + seg.start * ppf
            w = max(1.0, seg.length * ppf)
            y = self._lane_y(seg.class_idx)

            if seg.class_idx < 0:
                label, color = "Filtered", QColor(180, 180, 180)
            elif seg.class_idx < n_classes:
                label = self._model.classes[seg.class_idx]
                color = self._colors[seg.class_idx % len(self._colors)] if self._colors else QColor(100, 100, 200)
            else:
                continue

            if (self._filter_class is not None and self._filter_class >= 0
                    and seg.class_idx != self._filter_class):
                color = QColor(235, 235, 235)

            item = SegmentItem(i, x, y, w, _LANE_H, seg.class_idx, color, label, seg.confidence)
            self._scene.addItem(item)
            self._items.append(item)

        self._draw_time_axis(tw, lanes_h)

    def _draw_lane_backgrounds(self, n_classes, total_width, lanes_h):
        label_font = QFont("Sans", 9)
        for ci in range(n_classes):
            y = self._lane_y(ci)
            bg = QColor(245, 245, 245) if ci % 2 == 0 else QColor(252, 252, 252)
            rect = self._scene.addRect(
                _LABEL_W, y, total_width - _LABEL_W, _LANE_H,
                QPen(Qt.PenStyle.NoPen), QBrush(bg),
            )
            rect.setZValue(-1)
            color = self._colors[ci % len(self._colors)] if self._colors else QColor(100, 100, 200)
            indicator = self._scene.addRect(
                0, y + 2, 6, _LANE_H - 4,
                QPen(Qt.PenStyle.NoPen), QBrush(color),
            )
            name = self._model.classes[ci] if ci < len(self._model.classes) else f"class {ci}"
            txt = QGraphicsSimpleTextItem(name)
            txt.setFont(label_font)
            txt.setBrush(QBrush(QColor(40, 40, 40)))
            txt.setPos(10, y + _LANE_H / 2 - 7)
            self._scene.addItem(txt)

    def _draw_time_axis(self, total_width, lanes_h):
        fps = (self._model.orig_fps or 30.0) if self._model else 30.0
        ppf = self._ppf
        pps = ppf * fps
        y = lanes_h + 2

        if pps <= 0:
            return
        interval = 1.0
        for candidate in (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600):
            if candidate * pps >= 60:
                interval = candidate
                break

        t = 0.0
        total_sec = (self._model.total_frames / fps) if self._model else 0
        pen = QPen(QColor(120, 120, 120), 0.5)
        font = QFont("Sans", 7)
        while t <= total_sec:
            px = _LABEL_W + t * pps
            line = self._scene.addLine(px, lanes_h, px, lanes_h + 5, pen)
            if interval >= 60:
                m, s = divmod(int(t), 60)
                lbl = f"{m}:{s:02d}"
            else:
                lbl = f"{t:.1f}s" if interval < 1 else f"{t:.0f}s"
            txt = QGraphicsSimpleTextItem(lbl)
            txt.setFont(font)
            txt.setBrush(QBrush(QColor(100, 100, 100)))
            txt.setPos(px + 2, y + 2)
            self._scene.addItem(txt)
            t += interval

    def _frame_at(self, scene_x: float) -> int:
        if self._ppf <= 0 or not self._model:
            return 0
        return max(0, min(self._model.total_frames - 1, int((scene_x - _LABEL_W) / self._ppf)))

    def _item_at(self, pos) -> SegmentItem | None:
        scene_pos = self.mapToScene(pos)
        item = self._scene.itemAt(scene_pos, self.transform())
        return item if isinstance(item, SegmentItem) else None

    # ---------------------------------------------------------------- mouse

    def mousePressEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.MouseButton.RightButton:
            item = self._item_at(event.pos())
            if item:
                self._show_context_menu(event.globalPosition().toPoint(), item, scene_pos)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        item = self._item_at(event.pos())
        if item and self._model:
            self._drag_moved = False
            edge = _hit_edge(item.rect(), scene_pos.x())
            if edge:
                self._drag_mode = edge
                self._drag_idx = item.seg_index
                self._drag_origin = scene_pos.x()
                self._drag_start_frame = self._model[item.seg_index].start if edge == "left" else self._model[item.seg_index].end
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
            self._drag_mode = "move"
            self._drag_idx = item.seg_index
            self._drag_origin = scene_pos.x()
            self._drag_start_frame = self._model[item.seg_index].start
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self._model and self._ppf > 0:
            frame = self._frame_at(scene_pos.x())
            self.frame_clicked.emit(frame)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode and self._model and self._drag_idx >= 0:
            scene_pos = self.mapToScene(event.pos())
            delta_px = scene_pos.x() - self._drag_origin
            delta_frames = int(delta_px / self._ppf) if self._ppf > 0 else 0

            if self._drag_mode in ("left", "right"):
                if delta_frames != 0:
                    self._drag_moved = True
                    self._model.resize(self._drag_idx, self._drag_mode, delta_frames)
                    self._drag_origin = scene_pos.x()
                    self._rebuild()
                    self.model_changed.emit()
            elif self._drag_mode == "move":
                new_class = self._class_from_y(scene_pos.y())
                seg = self._model[self._drag_idx]
                changed = False
                if new_class != seg.class_idx and 0 <= new_class < len(self._model.classes):
                    self._model.reclass(self._drag_idx, new_class)
                    changed = True
                if delta_frames != 0:
                    self._model.move(self._drag_idx, delta_frames)
                    self._drag_origin = scene_pos.x()
                    changed = True
                if changed:
                    self._drag_moved = True
                    self._rebuild()
                    self.model_changed.emit()
            return

        item = self._item_at(event.pos())
        if item:
            scene_pos = self.mapToScene(event.pos())
            edge = _hit_edge(item.rect(), scene_pos.x())
            if edge:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode and self._model and self._drag_idx >= 0:
            idx = self._drag_idx
            moved = self._drag_moved
            self._drag_mode = None
            self._drag_idx = -1
            self._drag_moved = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if not moved and event.button() == Qt.MouseButton.LeftButton:
                self.segment_clicked.emit(idx)
            return
        super().mouseReleaseEvent(event)

    # --------------------------------------------------------- context menu

    def _show_context_menu(self, global_pos, item: SegmentItem, scene_pos: QPointF):
        if not self._model:
            return
        menu = QMenu(self)

        act_delete = menu.addAction("Delete segment")
        act_split = menu.addAction("Split here")
        menu.addSeparator()
        act_merge_prev = menu.addAction("Merge with previous")
        act_merge_next = menu.addAction("Merge with next")
        menu.addSeparator()

        class_menu = menu.addMenu("Change class")
        class_actions: list[tuple[QAction, int]] = []
        for ci, cname in enumerate(self._model.classes):
            a = class_menu.addAction(cname)
            if ci == item.class_idx:
                a.setEnabled(False)
            class_actions.append((a, ci))

        act_merge_prev.setEnabled(item.seg_index > 0)
        act_merge_next.setEnabled(item.seg_index < len(self._model) - 1)

        chosen = menu.exec(global_pos)
        if not chosen:
            return

        if chosen is act_delete:
            self._model.delete(item.seg_index)
        elif chosen is act_split:
            frame = self._frame_at(scene_pos.x())
            self._model.split(item.seg_index, frame)
        elif chosen is act_merge_prev:
            self._model.merge_with_next(item.seg_index - 1)
        elif chosen is act_merge_next:
            self._model.merge_with_next(item.seg_index)
        else:
            for a, ci in class_actions:
                if chosen is a:
                    self._model.reclass(item.seg_index, ci)
                    break
            else:
                return

        self._rebuild()
        self.model_changed.emit()

    # ------------------------------------------------------------ keyboard

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._model:
            super().keyPressEvent(event)
            return

        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        key = event.key()

        if ctrl and key == Qt.Key.Key_Z:
            if self._model.undo():
                self._rebuild()
                self.model_changed.emit()
            return
        if ctrl and key == Qt.Key.Key_Y:
            if self._model.redo():
                self._rebuild()
                self.model_changed.emit()
            return

        selected = [it for it in self._items if it.isSelected()]
        if not selected:
            super().keyPressEvent(event)
            return
        item = selected[0]

        if key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            self._model.delete(item.seg_index)
            self._rebuild()
            self.model_changed.emit()
            return

        if key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4,
                   Qt.Key.Key_5, Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9):
            ci = key - Qt.Key.Key_1
            if ci < len(self._model.classes):
                self._model.reclass(item.seg_index, ci)
                self._rebuild()
                self.model_changed.emit()
            return

        super().keyPressEvent(event)

    # --------------------------------------------------------------- zoom

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            self._ppf = max(0.1, min(50.0, self._ppf * factor))
            self._rebuild()
            event.accept()
        else:
            super().wheelEvent(event)
