"""
Active-learning review widget.

After inference, ranks clips by model uncertainty per class and lets the user
accept (correct label) or reassign clips.  Accepted clips are extracted from
the original video and added to the experiment's annotations.json so they can
be included in the next training run.
"""

import os
import json
import re
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QComboBox, QFileDialog,
    QMessageBox, QProgressBar, QScrollArea, QSizePolicy, QFrame,
    QSpinBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont

from singlebehaviorlab.backend.uncertainty import (
    save_uncertainty_report,
    rank_clips_for_review,
    rank_clips_per_video_for_review,
    rank_confident_clips_for_review,
    rank_confident_clips_per_video_for_review,
    rank_transition_clips_for_review,
    rank_transition_clips_per_video_for_review,
)
from singlebehaviorlab.backend.video_utils import save_clip
from singlebehaviorlab.backend.data_store import AnnotationManager


# Helpers.

def _frames_from_video(video_path: str, start_frame: int, n_frames: int,
                       frame_interval: int = 1) -> list:
    """Extract n_frames starting at start_frame (original-fps index)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames * frame_interval):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # Subsample at frame_interval
    return frames[::frame_interval] if frame_interval > 1 else frames


def _score_bar_html(scores: dict, predicted: str, width: int = 260) -> str:
    """Build a minimal HTML table with per-class score bars."""
    colours = {
        predicted: "#2ecc71",
    }
    default_colour = "#5dade2"
    rows = ""
    for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        col = colours.get(cls, default_colour)
        bar_w = max(2, int(score * width))
        rows += (
            f"<tr>"
            f"<td style='padding-right:6px;width:110px;'>{cls}</td>"
            f"<td><div style='background:{col};width:{bar_w}px;height:10px;"
            f"border-radius:3px;'></div></td>"
            f"<td style='padding-left:6px;'>{score:.1%}</td>"
            f"</tr>"
        )
    return f"<table cellspacing='2'>{rows}</table>"


# Mini video player.

class _ClipPlayer(QWidget):
    """Loops through a list of BGR frames displayed as a QLabel."""

    DISPLAY_SIZE = (560, 400)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list = []
        self._idx = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self._label = QLabel(self)
        self._label.setFixedSize(*self.DISPLAY_SIZE)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background:#111;border-radius:4px;")

        self._fps_label = QLabel("–", self)
        self._fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self._fps_label)

    def load(self, frames: list, playback_fps: float = 8.0):
        self._timer.stop()
        self._frames = frames
        self._idx = 0
        self._fps_label.setText(f"{len(frames)} frames @ {playback_fps:.0f} fps")
        if frames:
            self._show_frame(0)
            interval_ms = max(33, int(1000 / playback_fps))
            self._timer.start(interval_ms)
        else:
            self._label.setText("No preview")

    def stop(self):
        self._timer.stop()

    def _next_frame(self):
        if not self._frames:
            return
        self._idx = (self._idx + 1) % len(self._frames)
        self._show_frame(self._idx)

    def _show_frame(self, idx: int):
        frame = self._frames[idx]
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            *self.DISPLAY_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pix)


# Main widget.

class ReviewWidget(QWidget):
    """Tab for reviewing uncertain inference clips and adding them to the dataset."""

    # Emitted when the user saves accepted clips so other tabs can refresh.
    annotations_updated = pyqtSignal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

        # State
        self._report: dict = {}           # loaded uncertainty report
        self._pending: list = []          # [(entry, assigned_label), ...]
        self._current_entry: dict = {}
        self._current_frames: list = []
        self._review_mode = "uncertain"
        self._review_scope = "overall"
        self._accepted_keys = set()
        self._hard_negative_keys = set()
        self._pending_hard_negatives: list = []  # [(entry, target_class), ...]
        self._pending_transitions: list = []  # [(entry, primary_label, frame_labels), ...]
        self._transition_frame_combos = []

        self._setup_ui()

    # Public API.

    def update_config(self, config: dict):
        self.config = config

    def _arrays_sidecar_path(self, json_path: str) -> str:
        base, ext = os.path.splitext(json_path)
        if ext.lower() == ".json":
            return base + ".arrays.npz"
        return json_path + ".arrays.npz"

    def _restore_external_arrays(self, file_path: str, data: dict):
        results = data.get("results")
        if not isinstance(results, dict):
            return
        store_info = data.get("external_array_store", {})
        sidecar_file = store_info.get("file") if isinstance(store_info, dict) else None
        sidecar_path = (
            os.path.join(os.path.dirname(file_path), sidecar_file)
            if sidecar_file else self._arrays_sidecar_path(file_path)
        )
        if not os.path.exists(sidecar_path):
            return
        try:
            with np.load(sidecar_path, allow_pickle=False) as npz_file:
                for entry in results.values():
                    if not isinstance(entry, dict):
                        continue
                    refs = entry.pop("_external_arrays", None)
                    if not isinstance(refs, dict):
                        continue
                    for field_name, store_key in refs.items():
                        if store_key not in npz_file:
                            continue
                        arr = np.asarray(npz_file[store_key])
                        if field_name == "aggregated_frame_probs":
                            entry[field_name] = arr.astype(np.float32, copy=False)
                        else:
                            entry[field_name] = arr.tolist()
        except Exception:
            pass

    def load_from_inference(
        self,
        results: dict,
        classes: list,
        is_ovr: bool,
        clip_length: int,
        target_fps: int,
    ):
        """Called directly from InferenceWidget when inference finishes."""
        ranked = rank_clips_for_review(
            results, classes, n_per_class=25, is_ovr=is_ovr
        )
        ranked_per_video = rank_clips_per_video_for_review(
            results, classes, n_per_class=25, is_ovr=is_ovr
        )
        confident_per_class = rank_confident_clips_for_review(
            results, classes, n_per_class=200, is_ovr=is_ovr, clip_length=clip_length
        )
        confident_per_class_per_video = rank_confident_clips_per_video_for_review(
            results, classes, n_per_class=200, is_ovr=is_ovr, clip_length=clip_length
        )
        transition_per_class = rank_transition_clips_for_review(
            results, classes, clip_length=clip_length, is_ovr=is_ovr, n_per_class=50
        )
        transition_per_class_per_video = rank_transition_clips_per_video_for_review(
            results, classes, clip_length=clip_length, is_ovr=is_ovr, n_per_class=50
        )
        self._report = {
            "classes": classes,
            "is_ovr": is_ovr,
            "clip_length": clip_length,
            "target_fps": target_fps,
            "per_class": ranked,
            "per_class_per_video": ranked_per_video,
            "confident_per_class": confident_per_class,
            "confident_per_class_per_video": confident_per_class_per_video,
            "transition_per_class": transition_per_class,
            "transition_per_class_per_video": transition_per_class_per_video,
        }
        self._pending.clear()
        self._pending_hard_negatives.clear()
        self._pending_transitions.clear()
        self._accepted_keys.clear()
        self._hard_negative_keys.clear()
        self._review_mode = "uncertain"
        self._mode_combo.setCurrentIndex(0)
        self._populate_class_combo()
        self._update_status_label("Loaded from inference")
        self._update_save_btn()

    def load_from_file(self, path: str):
        """Load a previously saved _uncertainty.json file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._report = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        if "confident_per_class" not in self._report:
            legacy_confident = self._report.get("confident_candidates", []) or []
            confident_per_class = {cls: [] for cls in self._report.get("classes", [])}
            for entry in legacy_confident:
                pred = entry.get("predicted_class")
                if pred in confident_per_class:
                    confident_per_class[pred].append(entry)
            self._report["confident_per_class"] = confident_per_class
        if "per_class_per_video" not in self._report:
            per_video = {cls: {} for cls in self._report.get("classes", [])}
            for cls, entries in (self._report.get("per_class", {}) or {}).items():
                video_map = {}
                for entry in entries:
                    video_map.setdefault(entry.get("video", ""), []).append(entry)
                per_video[cls] = video_map
            self._report["per_class_per_video"] = per_video
        if "confident_per_class_per_video" not in self._report:
            per_video = {cls: {} for cls in self._report.get("classes", [])}
            for cls, entries in (self._report.get("confident_per_class", {}) or {}).items():
                video_map = {}
                for entry in entries:
                    video_map.setdefault(entry.get("video", ""), []).append(entry)
                per_video[cls] = video_map
            self._report["confident_per_class_per_video"] = per_video
        if "transition_per_class" not in self._report:
            self._report["transition_per_class"] = {
                cls: [] for cls in self._report.get("classes", [])
            }
        if "transition_per_class_per_video" not in self._report:
            per_video = {cls: {} for cls in self._report.get("classes", [])}
            for cls, entries in (self._report.get("transition_per_class", {}) or {}).items():
                video_map = {}
                for entry in entries:
                    video_map.setdefault(entry.get("video", ""), []).append(entry)
                per_video[cls] = video_map
            self._report["transition_per_class_per_video"] = per_video
        self._pending.clear()
        self._pending_hard_negatives.clear()
        self._pending_transitions.clear()
        self._accepted_keys.clear()
        self._hard_negative_keys.clear()
        self._review_mode = "uncertain"
        self._mode_combo.setCurrentIndex(0)
        self._populate_class_combo()
        self._update_status_label(f"Loaded {path}")
        self._update_save_btn()

    # UI construction.

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Top bar.
        top = QHBoxLayout()
        self._status_label = QLabel("No inference results loaded.")
        self._status_label.setStyleSheet("color:#aaa;font-style:italic;")
        top.addWidget(self._status_label, 1)

        load_btn = QPushButton("Load results file…")
        load_btn.setToolTip("Load a saved _uncertainty.json or inference_results.json")
        load_btn.clicked.connect(self._on_load_file)
        top.addWidget(load_btn)

        self._save_btn = QPushButton("Save 0 accepted clips to annotations")
        self._save_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;padding:4px 10px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_to_annotations)
        top.addWidget(self._save_btn)

        root.addLayout(top)

        # Horizontal splitter.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # LEFT: class selector + clip list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Uncertain review", "uncertain")
        self._mode_combo.addItem("Confident enrichment", "confident")
        self._mode_combo.addItem("Transition mining", "transition")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        class_row.addWidget(self._mode_combo)

        class_row.addSpacing(10)
        class_row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        class_row.addWidget(self._class_combo, 1)

        self._confident_count_label = QLabel("Top:")
        class_row.addWidget(self._confident_count_label)
        self._confident_count_spin = QSpinBox()
        self._confident_count_spin.setRange(1, 500)
        self._confident_count_spin.setValue(25)
        self._confident_count_spin.setToolTip("Number of high-confidence suggestions to show.")
        self._confident_count_spin.valueChanged.connect(self._refresh_clip_list)
        class_row.addWidget(self._confident_count_spin)

        self._scope_label = QLabel("Scope:")
        class_row.addWidget(self._scope_label)
        self._confident_scope_combo = QComboBox()
        self._confident_scope_combo.addItem("Across videos", "overall")
        self._confident_scope_combo.addItem("Per video", "per_video")
        self._confident_scope_combo.currentIndexChanged.connect(self._on_confident_scope_changed)
        class_row.addWidget(self._confident_scope_combo)

        self._confident_video_label = QLabel("Video:")
        class_row.addWidget(self._confident_video_label)
        self._confident_video_combo = QComboBox()
        self._confident_video_combo.currentIndexChanged.connect(self._refresh_clip_list)
        class_row.addWidget(self._confident_video_combo, 1)
        left_layout.addLayout(class_row)

        self._clip_list = QListWidget()
        self._clip_list.currentRowChanged.connect(self._on_clip_selected)
        left_layout.addWidget(self._clip_list, 1)

        # pending queue summary
        self._pending_label = QLabel("Pending: 0 accepted")
        self._pending_label.setStyleSheet("color:#f39c12;font-size:11px;")
        left_layout.addWidget(self._pending_label)

        splitter.addWidget(left_panel)

        # RIGHT: preview + info + actions
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)

        # video player
        player_row = QHBoxLayout()
        player_row.addStretch()
        self._player = _ClipPlayer()
        player_row.addWidget(self._player)
        player_row.addStretch()
        right_layout.addLayout(player_row)

        # clip info
        self._info_label = QLabel("Select a clip from the list.")
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._info_label.setMinimumHeight(60)
        right_layout.addWidget(self._info_label)

        # score bars
        self._score_label = QLabel("")
        self._score_label.setWordWrap(True)
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self._score_label)

        # action box — single unified flow
        action_box = QGroupBox("Label & add to dataset")
        action_layout = QVBoxLayout(action_box)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label this clip as:"))
        self._label_combo = QComboBox()
        self._label_combo.setMinimumWidth(160)
        self._label_combo.setToolTip(
            "Pre-filled with the model's prediction. Change it to reassign."
        )
        label_row.addWidget(self._label_combo, 1)
        action_layout.addLayout(label_row)

        self._transition_box = QGroupBox("Transition frame labels")
        transition_layout = QVBoxLayout(self._transition_box)
        self._transition_help_label = QLabel(
            "Review the proposed per-frame labels for this transition window before saving."
        )
        self._transition_help_label.setWordWrap(True)
        transition_layout.addWidget(self._transition_help_label)
        self._transition_frames_widget = QWidget()
        self._transition_frames_layout = QHBoxLayout(self._transition_frames_widget)
        self._transition_frames_layout.setContentsMargins(0, 0, 0, 0)
        self._transition_frames_layout.setSpacing(4)
        transition_layout.addWidget(self._transition_frames_widget)
        action_layout.addWidget(self._transition_box)

        self._add_btn = QPushButton("Add to dataset with this label")
        self._add_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;"
            "padding:8px;font-size:13px;}"
            "QPushButton:disabled{background:#555;color:#999;}"
        )
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._on_add)
        action_layout.addWidget(self._add_btn)

        self._skip_btn = QPushButton("→  Reject and save as hard negative")
        self._skip_btn.setStyleSheet("padding:6px;color:#aaa;")
        self._skip_btn.clicked.connect(self._on_skip)
        action_layout.addWidget(self._skip_btn)

        # feedback line shown briefly after adding
        self._feedback_label = QLabel("")
        self._feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feedback_label.setStyleSheet("color:#2ecc71;font-weight:bold;font-size:12px;")
        action_layout.addWidget(self._feedback_label)
        self._feedback_timer = QTimer(self)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.timeout.connect(lambda: self._feedback_label.setText(""))

        right_layout.addWidget(action_box)
        splitter.addWidget(right_panel)

        splitter.setSizes([360, 680])

    # Population helpers.

    def _populate_class_combo(self):
        classes = self._report.get("classes", [])
        current_index = self._class_combo.currentIndex()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        if self._review_mode == "confident":
            confident_per_class = self._report.get("confident_per_class", {}) or {}
            for cls in classes:
                n = len(confident_per_class.get(cls, []))
                self._class_combo.addItem(f"{cls}  ({n})")
        elif self._review_mode == "transition":
            transition_per_class = self._report.get("transition_per_class", {}) or {}
            for cls in classes:
                n = len(transition_per_class.get(cls, []))
                self._class_combo.addItem(f"{cls}  ({n})")
        else:
            per_class = self._report.get("per_class", {})
            for cls in classes:
                n = len(per_class.get(cls, []))
                self._class_combo.addItem(f"{cls}  ({n})")
        self._class_combo.blockSignals(False)

        self._label_combo.clear()
        self._label_combo.addItems(classes)
        self._populate_video_combo()

        if classes:
            self._class_combo.setCurrentIndex(max(0, min(current_index, len(classes) - 1)))
        self._update_mode_controls()
        self._refresh_clip_list()

    def _populate_video_combo(self):
        current_video = self._confident_video_combo.currentData()
        all_videos = set()
        if self._review_mode == "confident":
            report_key = "confident_per_class_per_video"
        elif self._review_mode == "transition":
            report_key = "transition_per_class_per_video"
        else:
            report_key = "per_class_per_video"
        per_class_per_video = self._report.get(report_key, {}) or {}
        for per_video in per_class_per_video.values():
            if isinstance(per_video, dict):
                all_videos.update(per_video.keys())
        ordered_videos = sorted(all_videos, key=lambda path: os.path.basename(path))
        self._confident_video_combo.blockSignals(True)
        self._confident_video_combo.clear()
        for video_path in ordered_videos:
            self._confident_video_combo.addItem(os.path.basename(video_path), video_path)
        self._confident_video_combo.blockSignals(False)
        if ordered_videos:
            idx = self._confident_video_combo.findData(current_video)
            if idx < 0:
                idx = 0
            self._confident_video_combo.setCurrentIndex(idx)

    def _current_video(self):
        return self._confident_video_combo.currentData()

    def _entry_key(self, entry: dict):
        return (entry.get("video", ""), int(entry.get("clip_idx", -1)))

    def _entry_class_key(self, entry: dict, class_name: str | None = None):
        target_class = class_name or entry.get("review_target_class") or self._current_target_class()
        return (entry.get("video", ""), int(entry.get("clip_idx", -1)), target_class or "")

    def _current_target_class(self):
        classes = self._report.get("classes", [])
        index = self._class_combo.currentIndex()
        if 0 <= index < len(classes):
            return classes[index]
        return ""

    def _apply_item_review_state(self, item: QListWidgetItem, entry: dict):
        key = self._entry_key(entry)
        if key in self._accepted_keys:
            item.setText("+ " + item.text())
            item.setForeground(QColor("#2ecc71"))
        elif self._entry_class_key(entry) in self._hard_negative_keys:
            item.setText("! " + item.text())
            item.setForeground(QColor("#e67e22"))

    def _update_mode_controls(self):
        is_confident = (self._review_mode == "confident")
        is_transition = (self._review_mode == "transition")
        self._class_combo.setEnabled(True)
        self._confident_count_label.setVisible(is_confident)
        self._confident_count_spin.setVisible(is_confident)
        self._scope_label.setVisible(True)
        self._confident_scope_combo.setVisible(True)
        show_video = self._review_scope == "per_video"
        self._confident_video_label.setVisible(show_video)
        self._confident_video_combo.setVisible(show_video)
        self._transition_box.setVisible(is_transition)
        self._label_combo.setVisible(not is_transition)
        self._transition_help_label.setVisible(is_transition)

    def _update_status_label(self, prefix: str = "Loaded"):
        classes = self._report.get("classes", [])
        per_class = self._report.get("per_class", {})
        uncertain_total = sum(len(v) for v in per_class.values())
        confident_per_class = self._report.get("confident_per_class", {}) or {}
        confident_total = sum(len(v) for v in confident_per_class.values())
        transition_per_class = self._report.get("transition_per_class", {}) or {}
        transition_total = sum(len(v) for v in transition_per_class.values())
        if self._review_mode == "uncertain":
            mode_txt = "uncertain review"
        elif self._review_mode == "confident":
            mode_txt = "confident enrichment"
        else:
            mode_txt = "transition mining"
        if self._review_scope == "per_video":
            sel_video = self._current_video()
            if sel_video:
                mode_txt += f" / {os.path.basename(sel_video)}"
        self._status_label.setText(
            f"{prefix} – {len(classes)} classes, {uncertain_total} uncertain candidates, "
            f"{confident_total} confident candidates, {transition_total} transitions ({mode_txt})"
        )

    def _clear_transition_editor(self):
        while self._transition_frames_layout.count():
            item = self._transition_frames_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._transition_frame_combos = []

    def _populate_transition_editor(self, frame_labels: list):
        self._clear_transition_editor()
        classes = self._report.get("classes", [])
        for idx, label in enumerate(frame_labels):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            frame_label = QLabel(f"F{idx+1}")
            frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            combo = QComboBox()
            combo.addItem("Ignore", None)
            for cls in classes:
                combo.addItem(cls, cls)
            sel_idx = combo.findData(label)
            if sel_idx >= 0:
                combo.setCurrentIndex(sel_idx)
            cell_layout.addWidget(frame_label)
            cell_layout.addWidget(combo)
            self._transition_frames_layout.addWidget(cell)
            self._transition_frame_combos.append(combo)
        self._transition_frames_layout.addStretch(1)

    def _current_transition_frame_labels(self):
        labels = []
        for combo in self._transition_frame_combos:
            labels.append(combo.currentData())
        return labels

    def _refresh_clip_list(self):
        self._clip_list.clear()
        self._current_entry = {}
        self._current_frames = []
        self._player.stop()
        self._add_btn.setEnabled(False)
        self._skip_btn.setEnabled(False)
        self._info_label.setText("Select a clip from the list.")
        self._score_label.setText("")

        classes = self._report.get("classes", [])
        index = self._class_combo.currentIndex()
        if index < 0 or index >= len(classes):
            self._update_status_label()
            return
        class_name = classes[index]

        if self._review_mode == "confident":
            if self._review_scope == "per_video":
                confident_per_class_per_video = self._report.get("confident_per_class_per_video", {}) or {}
                per_video = confident_per_class_per_video.get(class_name, {}) or {}
                entries = list(per_video.get(self._current_video(), []) or [])
            else:
                confident_per_class = self._report.get("confident_per_class", {}) or {}
                entries = list(confident_per_class.get(class_name, []) or [])
            limit = int(self._confident_count_spin.value())
            entries = entries[:limit]
            for rank, e in enumerate(entries, start=1):
                video_name = os.path.basename(e.get("video", ""))
                pred = e.get("predicted_class", class_name)
                conf = e.get("confidence_score", e.get("class_score", e.get("top_score", 0.0)))
                text = (
                    f"#{rank}  {video_name}  frame {e.get('start_frame', 0)}\n"
                    f"  class={pred}  score={conf:.0%}"
                )
                item = QListWidgetItem(text)
                if conf > 0.9:
                    item.setForeground(QColor("#2ecc71"))
                elif conf > 0.75:
                    item.setForeground(QColor("#5dade2"))
                item.setData(Qt.ItemDataRole.UserRole, e)
                self._apply_item_review_state(item, e)
                self._clip_list.addItem(item)
        elif self._review_mode == "transition":
            if self._review_scope == "per_video":
                per_class_per_video = self._report.get("transition_per_class_per_video", {}) or {}
                per_video = per_class_per_video.get(class_name, {}) or {}
                entries = list(per_video.get(self._current_video(), []) or [])
            else:
                per_class = self._report.get("transition_per_class", {}) or {}
                entries = list(per_class.get(class_name, []) or [])
            for rank, e in enumerate(entries, start=1):
                video_name = os.path.basename(e.get("video", ""))
                text = (
                    f"#{rank}  {video_name}  frame {e.get('transition_frame', e.get('start_frame', 0))}\n"
                    f"  {e.get('left_class', '?')} -> {e.get('right_class', '?')}  score={e.get('transition_score', 0.0):.0%}"
                )
                item = QListWidgetItem(text)
                item.setForeground(QColor("#9b59b6"))
                item.setData(Qt.ItemDataRole.UserRole, e)
                self._apply_item_review_state(item, e)
                self._clip_list.addItem(item)
        else:
            if self._review_scope == "per_video":
                per_class_per_video = self._report.get("per_class_per_video", {}) or {}
                per_video = per_class_per_video.get(class_name, {}) or {}
                entries = list(per_video.get(self._current_video(), []) or [])
            else:
                per_class = self._report.get("per_class", {})
                entries = list(per_class.get(class_name, []) or [])
            for e in entries:
                video_name = os.path.basename(e.get("video", ""))
                pred = e.get("predicted_class", "?")
                top_s = e.get("top_score", 0.0)
                margin = e.get("margin", 0.0)
                u_score = e.get("uncertainty_score", 0.0)
                text = (
                    f"{video_name}  frame {e.get('start_frame', 0)}\n"
                    f"  pred={pred} ({top_s:.0%})  margin={margin:.0%}  "
                    f"uncertainty={u_score:.0%}"
                )
                item = QListWidgetItem(text)
                if u_score > 0.7:
                    item.setForeground(QColor("#e74c3c"))
                elif u_score > 0.4:
                    item.setForeground(QColor("#f39c12"))
                item.setData(Qt.ItemDataRole.UserRole, e)
                self._apply_item_review_state(item, e)
                self._clip_list.addItem(item)

        if self._clip_list.count():
            self._clip_list.setCurrentRow(0)
        self._update_status_label()

    def _on_mode_changed(self, index: int):
        mode = self._mode_combo.itemData(index)
        self._review_mode = mode or "uncertain"
        self._populate_class_combo()
        self._update_mode_controls()

    def _on_confident_scope_changed(self, index: int):
        scope = self._confident_scope_combo.itemData(index)
        self._review_scope = scope or "overall"
        self._populate_class_combo()
        self._update_mode_controls()

    def _on_class_changed(self, index: int):
        classes = self._report.get("classes", [])
        if index < 0 or index >= len(classes):
            self._clip_list.clear()
            return
        self._refresh_clip_list()

    # Clip selection / preview.

    def _on_clip_selected(self, row: int):
        self._player.stop()
        if row < 0:
            self._current_entry = {}
            self._add_btn.setEnabled(False)
            self._clear_transition_editor()
            self._info_label.setText("Select a clip from the list.")
            self._score_label.setText("")
            return

        item = self._clip_list.item(row)
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not entry:
            return
        self._current_entry = entry

        # Update info label
        pred = entry.get("predicted_class", "?")
        top_s = entry.get("top_score", 0.0)
        margin = entry.get("margin", 0.0)
        u_score = entry.get("uncertainty_score", 0.0)
        conf_score = entry.get("confidence_score", top_s)
        review_kind = entry.get("review_kind", "uncertain")
        video = entry.get("video", "")
        start = entry.get("start_frame", 0)
        if review_kind == "transition":
            self._info_label.setText(
                f"<b>Video:</b> {os.path.basename(video)}<br>"
                f"<b>Window start:</b> {start}<br>"
                f"<b>Transition:</b> {entry.get('left_class', '?')} -> {entry.get('right_class', '?')}<br>"
                f"<b>Boundary score:</b> {entry.get('transition_score', 0.0):.1%}"
            )
        elif review_kind == "confident":
            self._info_label.setText(
                f"<b>Video:</b> {os.path.basename(video)}<br>"
                f"<b>Start frame:</b> {start}<br>"
                f"<b>Predicted:</b> {pred} ({top_s:.1%})<br>"
                f"<b>Confidence:</b> {conf_score:.1%}"
            )
        else:
            self._info_label.setText(
                f"<b>Video:</b> {os.path.basename(video)}<br>"
                f"<b>Start frame:</b> {start}<br>"
                f"<b>Predicted:</b> {pred} ({top_s:.1%})<br>"
                f"<b>Margin:</b> {margin:.1%}  |  <b>Uncertainty:</b> {u_score:.1%}"
            )

        scores = entry.get("scores", {})
        self._score_label.setText(_score_bar_html(scores, pred))

        # Pre-fill the label combo with the model's prediction
        idx = self._label_combo.findText(pred)
        if idx >= 0:
            self._label_combo.setCurrentIndex(idx)
        if review_kind == "transition":
            self._populate_transition_editor(
                list(entry.get("proposed_frame_labels", []) or [None] * int(self._report.get("clip_length", 8)))
            )
        else:
            self._clear_transition_editor()

        key = self._entry_key(entry)
        class_key = self._entry_class_key(entry)
        self._add_btn.setEnabled(key not in self._accepted_keys)
        if review_kind == "transition":
            self._skip_btn.setEnabled(True)
            self._skip_btn.setText("→  Skip transition candidate")
        else:
            self._skip_btn.setEnabled((key not in self._accepted_keys) and (class_key not in self._hard_negative_keys))
            target_class = self._current_target_class()
            if target_class:
                self._skip_btn.setText(f"→  Reject and save as hard negative for '{target_class}'")
            else:
                self._skip_btn.setText("→  Reject and save as hard negative")

        # Load frames for preview
        self._load_preview(entry)

    def _load_preview(self, entry: dict):
        video_path = entry.get("video", "")
        start_frame = entry.get("start_frame", 0)
        clip_length = self._report.get("clip_length", 8)
        frame_interval = entry.get("frame_interval", 1)
        target_fps = self._report.get("target_fps", 8)

        if not os.path.exists(video_path):
            self._player.load([])
            self._info_label.setText(
                self._info_label.text()
                + "<br><span style='color:red;'>Video file not found.</span>"
            )
            return

        frames = _frames_from_video(
            video_path, start_frame, clip_length, frame_interval
        )
        playback_fps = max(4, min(target_fps, 16))
        self._player.load(frames, playback_fps)
        self._current_frames = frames

    # Actions.

    def _on_add(self):
        """Register the current clip with the selected label and advance."""
        if not self._current_entry:
            return
        if self._current_entry.get("review_kind") == "transition":
            frame_labels = self._current_transition_frame_labels()
            valid_labels = [lbl for lbl in frame_labels if lbl]
            if not valid_labels:
                return
            label_counts = {}
            for lbl in valid_labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            label = max(sorted(label_counts), key=lambda lbl: label_counts[lbl])
            key = self._entry_key(self._current_entry)
            self._accepted_keys.add(key)
            self._pending = [
                (entry, entry_label)
                for entry, entry_label in self._pending
                if self._entry_key(entry) != key
            ]
            self._pending_transitions = [
                (entry, entry_label, entry_frame_labels)
                for entry, entry_label, entry_frame_labels in self._pending_transitions
                if self._entry_key(entry) != key
            ]
            transition_entry = dict(self._current_entry)
            transition_entry["proposed_frame_labels"] = list(frame_labels)
            self._pending_transitions.append((transition_entry, label, list(frame_labels)))
            self._update_save_btn()
            row = self._clip_list.currentRow()
            item = self._clip_list.item(row)
            if item is not None and not item.text().startswith("+ "):
                item.setText("+ " + item.text().lstrip("! ").strip())
                item.setForeground(QColor("#2ecc71"))
            self._feedback_label.setText("Added transition clip with frame labels")
            self._feedback_timer.start(1800)
            self._advance_list()
            return
        label = self._label_combo.currentText().strip()
        if not label:
            return
        key = self._entry_key(self._current_entry)
        self._accepted_keys.add(key)
        self._pending = [
            (entry, entry_label)
            for entry, entry_label in self._pending
            if self._entry_key(entry) != key
        ]
        self._hard_negative_keys = {hn_key for hn_key in self._hard_negative_keys if hn_key[:2] != key}
        self._pending_hard_negatives = [
            (entry, target_class)
            for entry, target_class in self._pending_hard_negatives
            if self._entry_key(entry) != key
        ]
        self._pending.append((dict(self._current_entry), label))
        self._update_save_btn()

        # Mark the list item so it's clear it was registered
        row = self._clip_list.currentRow()
        item = self._clip_list.item(row)
        if item is not None:
            if not item.text().startswith("+ "):
                item.setText("+ " + item.text().lstrip("! ").strip())
            item.setForeground(QColor("#2ecc71"))

        self._feedback_label.setText(f"Added as '{label}'")
        self._feedback_timer.start(1800)

        self._advance_list()

    def _on_skip(self):
        """Queue the current clip as a hard negative for the currently viewed class."""
        if self._current_entry and self._current_entry.get("review_kind") == "transition":
            self._feedback_label.setText("Skipped transition candidate")
            self._feedback_timer.start(1200)
            self._advance_list()
            return
        if self._current_entry:
            clip_key = self._entry_key(self._current_entry)
            if clip_key in self._accepted_keys:
                return
            target_class = self._current_target_class()
            if not target_class:
                return
            class_key = self._entry_class_key(self._current_entry, target_class)
            self._hard_negative_keys.add(class_key)
            self._pending_hard_negatives = [
                (entry, cls_name)
                for entry, cls_name in self._pending_hard_negatives
                if self._entry_class_key(entry, cls_name) != class_key
            ]
            hard_negative_entry = dict(self._current_entry)
            hard_negative_entry["review_target_class"] = target_class
            self._pending_hard_negatives.append((hard_negative_entry, target_class))
            self._update_save_btn()
        row = self._clip_list.currentRow()
        item = self._clip_list.item(row)
        if item is not None:
            if not item.text().startswith("! "):
                base_text = item.text()
                if base_text.startswith("+ "):
                    base_text = base_text[2:]
                item.setText("! " + base_text)
            item.setForeground(QColor("#e67e22"))
        self._feedback_label.setText("Saved as hard-negative candidate")
        self._feedback_timer.start(1800)
        self._advance_list()

    def _advance_list(self):
        self._add_btn.setEnabled(False)
        row = self._clip_list.currentRow()
        total = self._clip_list.count()
        if total == 0:
            return
        if row < total - 1:
            self._clip_list.setCurrentRow(row + 1)
        else:
            self._clip_list.setCurrentRow(-1)
            self._player.stop()
            self._feedback_label.setText("All clips in this view reviewed.")

    def _update_save_btn(self):
        n_accept = len(self._pending)
        n_hn = len(self._pending_hard_negatives)
        n_transitions = len(self._pending_transitions)
        total = n_accept + n_hn + n_transitions
        parts = []
        if n_accept:
            parts.append(f"{n_accept} accepted")
        if n_hn:
            parts.append(f"{n_hn} hard negative{'s' if n_hn != 1 else ''}")
        if n_transitions:
            parts.append(f"{n_transitions} transition clip{'s' if n_transitions != 1 else ''}")
        detail = " + ".join(parts) if parts else "0 items"
        self._save_btn.setText(f"Save {detail} to annotations")
        self._save_btn.setEnabled(total > 0)
        self._pending_label.setText(
            f"Pending: {n_accept} accepted, {n_hn} hard negatives, {n_transitions} transitions | "
            f"{len(self._accepted_keys)} accepted, {len(self._hard_negative_keys)} hard negatives"
        )

    # Load from file.

    def _on_load_file(self):
        start_dir = self.config.get("experiment_path", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load uncertainty / inference results",
            start_dir,
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return

        # If it's a full inference_results.json, compute uncertainty on the fly
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._restore_external_arrays(path, data)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        if "per_class" in data:
            # Already an uncertainty report
            self._report = data
            if "per_class_per_video" not in self._report:
                per_video = {cls: {} for cls in self._report.get("classes", [])}
                for cls, entries in (self._report.get("per_class", {}) or {}).items():
                    video_map = {}
                    for entry in entries:
                        video_map.setdefault(entry.get("video", ""), []).append(entry)
                    per_video[cls] = video_map
                self._report["per_class_per_video"] = per_video
            if "confident_per_class" not in self._report:
                legacy_confident = self._report.get("confident_candidates", []) or []
                confident_per_class = {cls: [] for cls in self._report.get("classes", [])}
                for entry in legacy_confident:
                    pred = entry.get("predicted_class")
                    if pred in confident_per_class:
                        confident_per_class[pred].append(entry)
                self._report["confident_per_class"] = confident_per_class
            if "confident_per_class_per_video" not in self._report:
                per_video = {cls: {} for cls in self._report.get("classes", [])}
                for cls, entries in (self._report.get("confident_per_class", {}) or {}).items():
                    video_map = {}
                    for entry in entries:
                        video_map.setdefault(entry.get("video", ""), []).append(entry)
                    per_video[cls] = video_map
                self._report["confident_per_class_per_video"] = per_video
            if "transition_per_class" not in self._report:
                self._report["transition_per_class"] = {
                    cls: [] for cls in self._report.get("classes", [])
                }
            if "transition_per_class_per_video" not in self._report:
                per_video = {cls: {} for cls in self._report.get("classes", [])}
                for cls, entries in (self._report.get("transition_per_class", {}) or {}).items():
                    video_map = {}
                    for entry in entries:
                        video_map.setdefault(entry.get("video", ""), []).append(entry)
                    per_video[cls] = video_map
                self._report["transition_per_class_per_video"] = per_video
        elif "classes" in data:
            # Full inference_results.json — compute uncertainty
            classes = data.get("classes", [])
            params = data.get("parameters", {})
            is_ovr = params.get("use_ovr", False)
            clip_length = params.get("clip_length", 8)
            target_fps = params.get("target_fps", 16)
            # results keyed by video path
            nested_results = data.get("results")
            if isinstance(nested_results, dict) and nested_results:
                results = nested_results
            else:
                results = {
                    k: v for k, v in data.items()
                    if k not in ("classes", "parameters", "inference_time")
                }
            ranked = rank_clips_for_review(results, classes, n_per_class=25, is_ovr=is_ovr)
            ranked_per_video = rank_clips_per_video_for_review(
                results, classes, n_per_class=25, is_ovr=is_ovr
            )
            confident_per_class = rank_confident_clips_for_review(
                results, classes, n_per_class=200, is_ovr=is_ovr, clip_length=clip_length
            )
            confident_per_class_per_video = rank_confident_clips_per_video_for_review(
                results, classes, n_per_class=200, is_ovr=is_ovr, clip_length=clip_length
            )
            transition_per_class = rank_transition_clips_for_review(
                results, classes, clip_length=clip_length, is_ovr=is_ovr, n_per_class=50
            )
            transition_per_class_per_video = rank_transition_clips_per_video_for_review(
                results, classes, clip_length=clip_length, is_ovr=is_ovr, n_per_class=50
            )
            self._report = {
                "classes": classes,
                "is_ovr": is_ovr,
                "clip_length": clip_length,
                "target_fps": target_fps,
                "per_class": ranked,
                "per_class_per_video": ranked_per_video,
                "confident_per_class": confident_per_class,
                "confident_per_class_per_video": confident_per_class_per_video,
                "transition_per_class": transition_per_class,
                "transition_per_class_per_video": transition_per_class_per_video,
            }
        else:
            QMessageBox.warning(
                self, "Unknown format",
                "File doesn't look like an inference_results.json or uncertainty report."
            )
            return

        self._pending.clear()
        self._pending_hard_negatives.clear()
        self._pending_transitions.clear()
        self._accepted_keys.clear()
        self._hard_negative_keys.clear()
        self._populate_class_combo()
        self._update_status_label(f"Loaded: {path}")
        self._update_save_btn()

    # Save accepted clips to annotations.

    def _on_save_to_annotations(self):
        if not self._pending and not self._pending_hard_negatives and not self._pending_transitions:
            return

        annotation_file = self.config.get("annotation_file", "")
        clips_dir = self.config.get("clips_dir", "")

        if not annotation_file or not clips_dir:
            QMessageBox.warning(
                self, "No experiment",
                "Load an experiment first (annotation file and clips dir must be set)."
            )
            return

        reviewed_dir = os.path.join(clips_dir, "reviewed_clips")
        os.makedirs(reviewed_dir, exist_ok=True)

        am = AnnotationManager(annotation_file)
        added = 0
        hard_negatives_added = 0
        transitions_added = 0
        skipped = 0
        errors = []

        clip_length = self._report.get("clip_length", 8)
        target_fps = self._report.get("target_fps", 8)
        existing = {c.get("id") for c in am.get_all_clips()}

        def ensure_saved_clip(out_path: str, video_path: str, start_frame: int, frame_interval: int):
            frames = []
            if not os.path.exists(out_path):
                frames = _frames_from_video(video_path, start_frame, clip_length, frame_interval)
                if not frames:
                    raise RuntimeError(
                        f"No frames extracted: {os.path.basename(video_path)} @ {start_frame}"
                    )
                save_clip(frames, out_path, fps=float(target_fps))
            else:
                cap = cv2.VideoCapture(out_path)
                n_existing = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if n_existing <= 0:
                    n_existing = clip_length
                frames = [None] * n_existing
            return frames

        for entry, label in self._pending:
            video_path = entry.get("video", "")
            start_frame = entry.get("start_frame", 0)
            frame_interval = entry.get("frame_interval", 1)

            if not os.path.exists(video_path):
                errors.append(f"Not found: {video_path}")
                skipped += 1
                continue

            # Build a safe output filename
            video_stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(video_path))[0])
            out_name = f"{video_stem}_f{start_frame:07d}.mp4"
            out_path = os.path.join(reviewed_dir, out_name)

            try:
                frames = ensure_saved_clip(out_path, video_path, start_frame, frame_interval)
            except Exception as exc:
                errors.append(str(exc))
                skipped += 1
                continue

            # Relative ID from clips_dir
            try:
                rel_id = os.path.relpath(out_path, clips_dir).replace("\\", "/")
            except ValueError:
                rel_id = os.path.basename(out_path)

            # Skip if already annotated
            if rel_id in existing:
                skipped += 1
                continue

            actual_frames = len(frames)
            am.add_clip(rel_id, label, meta={
                "source_video": video_path,
                "start_frame": start_frame,
                "origin": "active_learning_review",
                "review_mode": entry.get("review_kind", "uncertain"),
            }, _defer_save=True)
            am.set_frame_labels(rel_id, [label] * actual_frames, _defer_save=True)
            am.add_class(label)
            added += 1
            existing.add(rel_id)

        transition_dir = os.path.join(reviewed_dir, "transitions")
        os.makedirs(transition_dir, exist_ok=True)
        for entry, label, frame_labels in self._pending_transitions:
            video_path = entry.get("video", "")
            start_frame = int(entry.get("start_frame", 0))
            frame_interval = int(entry.get("frame_interval", 1) or 1)
            if not os.path.exists(video_path):
                errors.append(f"Not found: {video_path}")
                skipped += 1
                continue

            video_stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(video_path))[0])
            left_slug = re.sub(r"[^\w\-]", "_", str(entry.get("left_class", "left"))).strip("_") or "left"
            right_slug = re.sub(r"[^\w\-]", "_", str(entry.get("right_class", "right"))).strip("_") or "right"
            out_name = f"{video_stem}_f{start_frame:07d}_transition_{left_slug}_to_{right_slug}.mp4"
            out_path = os.path.join(transition_dir, out_name)
            try:
                frames = ensure_saved_clip(out_path, video_path, start_frame, frame_interval)
            except Exception as exc:
                errors.append(str(exc))
                skipped += 1
                continue

            try:
                rel_id = os.path.relpath(out_path, clips_dir).replace("\\", "/")
            except ValueError:
                rel_id = os.path.basename(out_path)
            if rel_id in existing:
                skipped += 1
                continue

            actual_frames = len(frames)
            normalized_frame_labels = list(frame_labels[:actual_frames])
            if len(normalized_frame_labels) < actual_frames:
                normalized_frame_labels.extend([None] * (actual_frames - len(normalized_frame_labels)))
            am.add_clip(rel_id, label, meta={
                "source_video": video_path,
                "start_frame": start_frame,
                "origin": "active_learning_transition_review",
                "review_mode": "transition",
                "transition_frame": int(entry.get("transition_frame", start_frame)),
                "transition_left_class": entry.get("left_class"),
                "transition_right_class": entry.get("right_class"),
            }, _defer_save=True)
            am.set_frame_labels(rel_id, normalized_frame_labels, _defer_save=True)
            am.add_class(label)
            transitions_added += 1
            existing.add(rel_id)

        hard_negative_dir = os.path.join(reviewed_dir, "hard_negatives")
        os.makedirs(hard_negative_dir, exist_ok=True)
        for entry, target_class in self._pending_hard_negatives:
            video_path = entry.get("video", "")
            start_frame = entry.get("start_frame", 0)
            frame_interval = entry.get("frame_interval", 1)

            if not os.path.exists(video_path):
                errors.append(f"Not found: {video_path}")
                skipped += 1
                continue

            video_stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(video_path))[0])
            class_slug = re.sub(r"[^\w\-]", "_", target_class.strip()).strip("_") or "unknown"
            class_dir = os.path.join(hard_negative_dir, class_slug)
            os.makedirs(class_dir, exist_ok=True)
            out_name = f"{video_stem}_f{start_frame:07d}_hn_{class_slug}.mp4"
            out_path = os.path.join(class_dir, out_name)

            try:
                ensure_saved_clip(out_path, video_path, start_frame, frame_interval)
            except Exception as exc:
                errors.append(str(exc))
                skipped += 1
                continue

            try:
                rel_id = os.path.relpath(out_path, clips_dir).replace("\\", "/")
            except ValueError:
                rel_id = os.path.basename(out_path)

            if rel_id in existing:
                skipped += 1
                continue

            hn_label = f"near_negative_{class_slug}"
            am.add_clip(rel_id, hn_label, meta={
                "source_video": video_path,
                "start_frame": start_frame,
                "origin": "active_learning_review",
                "review_mode": entry.get("review_kind", "uncertain"),
                "review_target_class": target_class,
                "hard_negative_for_class": target_class,
            }, _defer_save=True)
            hard_negatives_added += 1
            existing.add(rel_id)

        am.save()  # flush any deferred writes
        self._pending.clear()
        self._pending_hard_negatives.clear()
        self._pending_transitions.clear()
        self._update_save_btn()

        msg = (
            f"Added {added} accepted clip{'s' if added != 1 else ''} and "
            f"{hard_negatives_added} hard negative{'s' if hard_negatives_added != 1 else ''} and "
            f"{transitions_added} transition clip{'s' if transitions_added != 1 else ''}."
        )
        if skipped:
            msg += f"  {skipped} skipped (already present or file missing)."
        if errors:
            msg += f"\n\nErrors:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Saved", msg)

        if added or hard_negatives_added or transitions_added:
            self.annotations_updated.emit()
