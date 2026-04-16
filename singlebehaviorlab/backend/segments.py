"""Mutable segment list that drives the interactive timeline editor.

Segments are the single source of truth: per-frame label arrays, CSV rows,
and SVG rectangles are all derived from them. Editing operations enforce
non-overlap and boundary constraints, and every mutation is tracked by an
undo/redo stack.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

__all__ = ["Segment", "SegmentsModel"]

UNDO_LIMIT = 50


@dataclass
class Segment:
    class_idx: int
    start: int
    end: int
    confidence: float = 1.0

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "class": self.class_idx,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(
            class_idx=int(d["class"]),
            start=int(d["start"]),
            end=int(d["end"]),
            confidence=float(d.get("confidence", 1.0)),
        )


class SegmentsModel:
    """Ordered, non-overlapping segment list with undo/redo."""

    def __init__(
        self,
        segments: list[dict] | list[Segment],
        classes: list[str],
        total_frames: int,
        orig_fps: float = 30.0,
    ):
        self.classes = list(classes)
        self.total_frames = max(0, total_frames)
        self.orig_fps = float(orig_fps)

        raw = []
        for s in segments:
            raw.append(s if isinstance(s, Segment) else Segment.from_dict(s))
        raw.sort(key=lambda s: s.start)
        self._segments: list[Segment] = raw
        self._undo: list[list[Segment]] = []
        self._redo: list[list[Segment]] = []

    @property
    def segments(self) -> list[Segment]:
        return self._segments

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> Segment:
        return self._segments[idx]

    def _snapshot(self) -> None:
        self._undo.append(copy.deepcopy(self._segments))
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()

    def _clamp(self, val: int) -> int:
        return max(0, min(self.total_frames, val))

    def _prev_end(self, idx: int) -> int:
        return self._segments[idx - 1].end if idx > 0 else 0

    def _next_start(self, idx: int) -> int:
        if idx < len(self._segments) - 1:
            return self._segments[idx + 1].start
        return self.total_frames

    # ------------------------------------------------------------------ undo

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self._segments))
        self._segments = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self._segments))
        self._segments = self._redo.pop()
        return True

    # --------------------------------------------------------------- editing

    def resize(self, idx: int, edge: Literal["left", "right"], delta: int) -> bool:
        if idx < 0 or idx >= len(self._segments):
            return False
        seg = self._segments[idx]
        self._snapshot()

        if edge == "left":
            new_start = self._clamp(seg.start + delta)
            new_start = max(new_start, self._prev_end(idx))
            if new_start >= seg.end:
                new_start = seg.end - 1
            seg.start = new_start
        else:
            new_end = self._clamp(seg.end + delta)
            new_end = min(new_end, self._next_start(idx))
            if new_end <= seg.start:
                new_end = seg.start + 1
            seg.end = new_end
        return True

    def move(self, idx: int, delta: int) -> bool:
        if idx < 0 or idx >= len(self._segments):
            return False
        seg = self._segments[idx]
        length = seg.length
        lo = self._prev_end(idx)
        hi = self._next_start(idx)
        if hi - lo < length:
            return False
        self._snapshot()
        new_start = self._clamp(seg.start + delta)
        new_start = max(new_start, lo)
        if new_start + length > hi:
            new_start = hi - length
        seg.start = new_start
        seg.end = new_start + length
        return True

    def reclass(self, idx: int, new_class_idx: int) -> bool:
        if idx < 0 or idx >= len(self._segments):
            return False
        if new_class_idx < 0 or new_class_idx >= len(self.classes):
            return False
        self._snapshot()
        self._segments[idx].class_idx = new_class_idx
        return True

    def delete(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._segments):
            return False
        self._snapshot()
        self._segments.pop(idx)
        return True

    def split(self, idx: int, at_frame: int) -> bool:
        if idx < 0 or idx >= len(self._segments):
            return False
        seg = self._segments[idx]
        if at_frame <= seg.start or at_frame >= seg.end:
            return False
        self._snapshot()
        left = Segment(seg.class_idx, seg.start, at_frame, seg.confidence)
        right = Segment(seg.class_idx, at_frame, seg.end, seg.confidence)
        self._segments[idx:idx + 1] = [left, right]
        return True

    def merge_with_next(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._segments) - 1:
            return False
        self._snapshot()
        left = self._segments[idx]
        right = self._segments[idx + 1]
        left.end = right.end
        self._segments.pop(idx + 1)
        return True

    # -------------------------------------------------------- derived outputs

    def to_frame_labels(self) -> np.ndarray:
        labels = np.full(self.total_frames, -1, dtype=np.int32)
        for seg in self._segments:
            labels[seg.start:seg.end] = seg.class_idx
        return labels

    def to_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self._segments]

    def to_csv_rows(self) -> list[dict]:
        rows = []
        for seg in self._segments:
            if seg.class_idx < 0 or seg.class_idx >= len(self.classes):
                name = f"class_{seg.class_idx}"
            else:
                name = self.classes[seg.class_idx]
            fps = max(1e-6, self.orig_fps)
            rows.append({
                "Behavior": name,
                "Start Time (s)": round(seg.start / fps, 4),
                "End Time (s)": round(seg.end / fps, 4),
                "Start Frame": seg.start,
                "End Frame": seg.end,
                "Duration (s)": round(seg.length / fps, 4),
                "Confidence": round(seg.confidence, 4),
            })
        return rows

    @classmethod
    def from_frame_labels(
        cls,
        labels: np.ndarray,
        classes: list[str],
        total_frames: int,
        orig_fps: float = 30.0,
        confidences: Optional[np.ndarray] = None,
    ) -> "SegmentsModel":
        """Build from a per-frame label array (e.g. argmax output)."""
        segments: list[Segment] = []
        if len(labels) == 0:
            return cls([], classes, total_frames, orig_fps)
        current_class = int(labels[0])
        start = 0
        for i in range(1, len(labels)):
            if int(labels[i]) != current_class:
                if current_class >= 0:
                    conf = float(confidences[start:i].mean()) if confidences is not None else 1.0
                    segments.append(Segment(current_class, start, i, conf))
                current_class = int(labels[i])
                start = i
        if current_class >= 0:
            conf = float(confidences[start:].mean()) if confidences is not None else 1.0
            segments.append(Segment(current_class, start, len(labels), conf))
        return cls(segments, classes, total_frames, orig_fps)
