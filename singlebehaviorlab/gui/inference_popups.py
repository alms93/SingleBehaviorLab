"""Popup dialogs for inference clip and frame-segment inspection."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QMessageBox, QSizePolicy, QSpinBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import os
import numpy as np
from singlebehaviorlab.backend.video_utils import save_clip
from singlebehaviorlab.backend.data_store import AnnotationManager


class ClipPopupDialog(QDialog):
    """Dialog showing a single inference clip with label, correction, and training controls."""

    def __init__(self, parent, widget, clip_idx):
        super().__init__(parent)
        self._widget = widget
        self._clip_idx = clip_idx

        pred_idx = self._widget._effective_prediction_for_clip(self._clip_idx)
        conf = self._widget.confidences[self._clip_idx]
        label = self._widget.classes[pred_idx] if (0 <= pred_idx < len(self._widget.classes)) else self._widget.ignore_label_name

        attr_info = ""
        attr_idx = self._widget._get_attr_idx(self._clip_idx)
        if self._widget.attributes and isinstance(attr_idx, int) and attr_idx < len(self._widget.attributes):
            attr_label = self._widget.attributes[attr_idx]
            attr_conf = 0.0
            if self._widget.attr_confidences and self._clip_idx < len(self._widget.attr_confidences):
                attr_conf = self._widget.attr_confidences[self._clip_idx]
            attr_info = f"<br><b>Attribute:</b> {attr_label} ({attr_conf:.2%})"

        bbox_info = "<br><b>Localization BBox:</b> unavailable"
        if self._widget._get_localization_bbox_for_clip_frame(self._clip_idx, 0) is not None:
            bbox_info = "<br><b>Localization BBox:</b> shown in green"

        self.setWindowTitle(f"Clip {self._clip_idx + 1}: {label} ({conf:.1%} confidence)")
        self.setMinimumSize(700, 650)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)

        if self._widget.clip_popup_maximized:
            self.showMaximized()
        else:
            self.show()

        layout = QVBoxLayout()

        info_layout = QHBoxLayout()
        ovr_scores = ""
        if self._widget._use_ovr and self._clip_idx < len(self._widget.clip_probabilities):
            probs = self._widget.clip_probabilities[self._clip_idx]
            if isinstance(probs, (list, tuple)):
                scored = []
                for ci, sc in enumerate(probs):
                    if ci < len(self._widget.classes):
                        scored.append((self._widget.classes[ci], float(sc)))
                scored.sort(key=lambda x: x[1], reverse=True)
                parts = [f"{name}: {sc:.1%}" for name, sc in scored]
                ovr_scores = "<br><b>All scores:</b> " + " | ".join(parts)
        info_label = QLabel(f"<b>Predicted Label:</b> {label}<br><b>Confidence:</b> {conf:.2%}{attr_info}{bbox_info}{ovr_scores}")
        info_label.setStyleSheet("font-size: 14px; padding: 10px;")
        info_layout.addWidget(info_label)

        if self._clip_idx in self._widget.corrected_labels or self._clip_idx in self._widget.corrected_attr_labels:
            corrected_label = QLabel("<b style='color: green;'>Corrected</b>")
            corrected_label.setStyleSheet("font-size: 14px; padding: 10px;")
            info_layout.addWidget(corrected_label)

        layout.addLayout(info_layout)

        correction_group = QGroupBox("Correct label")
        correction_layout = QVBoxLayout()

        label_combo = QComboBox()
        label_combo.addItems(self._widget.classes)
        if 0 <= pred_idx < len(self._widget.classes):
            label_combo.setCurrentIndex(pred_idx)

        attr_combo = QComboBox()
        attr_combo.addItem("No attribute")
        if self._widget.attributes:
            attr_combo.addItems(self._widget.attributes)
        if isinstance(attr_idx, int) and attr_idx < len(self._widget.attributes):
            attr_combo.setCurrentIndex(attr_idx + 1)

        colors = self._widget._get_timeline_palette()
        color_indicator = QLabel()
        color_indicator.setFixedSize(20, 20)

        def update_indicator(index):
            if index < len(colors):
                c = colors[index % len(colors)]
                color_hex = f"rgb({c[0]},{c[1]},{c[2]})"
                color_indicator.setStyleSheet(f"background-color: {color_hex}; border-radius: 10px; border: 1px solid #666;")

        label_combo.currentIndexChanged.connect(update_indicator)
        update_indicator(pred_idx)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Select correct label:"))
        combo_layout.addWidget(color_indicator)
        combo_layout.addWidget(label_combo)
        correction_layout.addLayout(combo_layout)

        attr_layout = QHBoxLayout()
        attr_layout.addWidget(QLabel("Select attribute:"))
        attr_layout.addWidget(attr_combo)
        correction_layout.addLayout(attr_layout)

        save_correction_btn = QPushButton("Save correction")
        save_correction_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")

        def save_correction():
            new_label_idx = label_combo.currentIndex()
            original_pred_idx = self._widget.predictions[self._clip_idx]

            if new_label_idx != original_pred_idx:
                self._widget.corrected_labels[self._clip_idx] = new_label_idx
            else:
                if self._clip_idx in self._widget.corrected_labels:
                    del self._widget.corrected_labels[self._clip_idx]

            new_attr_idx = attr_combo.currentIndex() - 1
            original_attr_idx = self._widget.attr_predictions[self._clip_idx] if (self._widget.attr_predictions and self._clip_idx < len(self._widget.attr_predictions)) else None
            if new_attr_idx >= 0 and new_attr_idx != original_attr_idx:
                self._widget.corrected_attr_labels[self._clip_idx] = new_attr_idx
            else:
                if self._clip_idx in self._widget.corrected_attr_labels:
                    del self._widget.corrected_attr_labels[self._clip_idx]

            self._widget._draw_timeline()
            QMessageBox.information(self, "Correction Saved", "Corrections saved.\n\nTimeline updated.")

        save_correction_btn.clicked.connect(save_correction)
        correction_layout.addWidget(save_correction_btn)

        correction_group.setLayout(correction_layout)
        layout.addWidget(correction_group)

        training_group = QGroupBox("Add to training dataset")
        training_layout = QVBoxLayout()

        add_to_training_btn = QPushButton("Add to training dataset")
        add_to_training_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")

        def _extract_and_store_clip(selected_label: str, extra_meta: dict = None):
            clips_dir = self._widget._get_clips_dir()

            cap = cv2.VideoCapture(self._widget.video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "Could not open video file.")
                return None, None

            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            if orig_fps <= 0:
                orig_fps = 30.0

            target_fps = self._widget.target_fps_spin.value()
            frame_interval = self._widget._get_saved_frame_interval(self._widget.video_path, orig_fps)
            clip_length = self._widget.clip_length_spin.value()
            clip_start_frame = self._widget.clip_starts[self._clip_idx]

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
            frames = []
            frame_count = 0
            while len(frames) < clip_length:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frames.append(frame.copy())
                frame_count += 1
            cap.release()

            if not frames:
                QMessageBox.warning(self, "Error", "Could not extract frames from clip.")
                return None, None

            video_basename = self._widget._video_basename()
            clip_filename = f"{video_basename}_clip_{self._clip_idx:06d}_frame_{clip_start_frame}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)
            counter = 1
            while os.path.exists(clip_path):
                clip_filename = f"{video_basename}_clip_{self._clip_idx:06d}_frame_{clip_start_frame}_{counter}.mp4"
                clip_path = os.path.join(clips_dir, clip_filename)
                counter += 1

            save_clip(frames, clip_path, target_fps)
            if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                QMessageBox.warning(
                    self, "Error",
                    f"Failed to save clip to disk.\nPath: {clip_path}\n"
                    "Check write permissions and disk space."
                )
                return None, None

            annotation_manager = AnnotationManager(self._widget._get_annotation_file())
            clip_id = self._widget._clip_path_to_id(clip_path, clips_dir)

            annotation_manager.add_class(selected_label)
            meta = {
                "source_video": os.path.basename(self._widget.video_path),
                "source_frame": clip_start_frame,
                "target_fps": target_fps,
                "clip_length": clip_length,
                "added_from_inference": True,
            }
            if extra_meta:
                meta.update(extra_meta)
            used_clip_id = annotation_manager.add_clip(clip_id, selected_label, meta=meta)
            frame_labels = [selected_label] * len(frames)
            annotation_manager.set_frame_labels(used_clip_id, frame_labels)
            return clip_path, selected_label

        def add_to_training():
            try:
                selected_label_idx = label_combo.currentIndex()
                selected_label = self._widget.classes[selected_label_idx]
                if self._widget.attributes:
                    selected_attr_idx = attr_combo.currentIndex() - 1
                    if 0 <= selected_attr_idx < len(self._widget.attributes):
                        selected_label = self._widget.attributes[selected_attr_idx]
                clip_path, label = _extract_and_store_clip(selected_label)
                if clip_path:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Clip added to training dataset!\n\n"
                        f"Label: {label}\n"
                        f"Saved to: {clip_path}\n\n"
                        f"You can now retrain the model with this new data.",
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add clip to training dataset:\n{str(e)}")

        def add_as_near_negative():
            try:
                prefix = str(self._widget.config.get("near_negative_label", "near_negative")).strip() or "near_negative"
                target_class = None

                sel_idx = label_combo.currentIndex()
                if 0 <= sel_idx < len(self._widget.classes):
                    target_class = self._widget.classes[sel_idx]
                else:
                    try:
                        raw_pred_idx = int(self._widget.predictions[self._clip_idx])
                    except Exception:
                        raw_pred_idx = -1
                    if 0 <= raw_pred_idx < len(self._widget.classes):
                        target_class = self._widget.classes[raw_pred_idx]

                if target_class:
                    class_token = str(target_class).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
                    while "__" in class_token:
                        class_token = class_token.replace("__", "_")
                    class_token = class_token.strip("_")
                    if class_token:
                        near_label = f"{prefix}_{class_token}" if not prefix.endswith(f"_{class_token}") else prefix
                    else:
                        near_label = prefix
                else:
                    near_label = prefix

                clip_path, label = _extract_and_store_clip(
                    near_label,
                    extra_meta={
                        "near_negative": True,
                        "hard_negative_candidate": True,
                        "hard_negative_for_class": target_class,
                    },
                )
                if clip_path:
                    self._widget.log_text.append(f"Near negative saved: label='{label}', clip='{clip_path}'")
                    QMessageBox.information(
                        self,
                        "Near negative added",
                        f"Clip saved as hard negative for this prediction.\n\n"
                        f"Label: {label}\n"
                        f"Saved to: {clip_path}\n\n"
                        "Train with multiple near_negative_* classes; at inference ignore them as background.",
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add near negative clip:\n{str(e)}")

        add_to_training_btn.clicked.connect(add_to_training)
        training_layout.addWidget(add_to_training_btn)
        add_near_negative_btn = QPushButton("Mark as near negative")
        add_near_negative_btn.setToolTip(
            "Saves clip as near_negative_<predicted_class> (e.g. near_negative_jump). "
            "Train with all near_negative_* classes; at inference treat them as background."
        )
        add_near_negative_btn.setStyleSheet("background-color: #5a6578; color: white; font-weight: bold; padding: 5px;")
        add_near_negative_btn.clicked.connect(add_as_near_negative)
        training_layout.addWidget(add_near_negative_btn)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        video_label = QLabel("Loading clip...")
        video_label.setMinimumSize(640, 360)
        video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(video_label, 1)

        controls_layout = QHBoxLayout()

        prev_btn = QPushButton("< Previous")
        prev_btn.setToolTip("Previous Clip")

        has_prev = False
        selected_behavior = self._widget.filter_behavior_combo.currentText()
        selected_attr = None
        if selected_behavior.startswith("Attr: "):
            selected_attr = selected_behavior.replace("Attr: ", "", 1)
        prev_search_idx = self._clip_idx - 1

        if selected_behavior == "All Behaviors":
            if prev_search_idx >= 0:
                has_prev = True
        else:
            while prev_search_idx >= 0:
                p_idx = self._widget._effective_prediction_for_clip(prev_search_idx)

                if selected_attr is not None:
                    if selected_attr in self._widget.attributes:
                        attr_target_idx = self._widget.attributes.index(selected_attr)
                        if self._widget._get_attr_idx(prev_search_idx) == attr_target_idx:
                            has_prev = True
                            break
                else:
                    if selected_behavior == self._widget.ignore_label_name and p_idx < 0:
                        has_prev = True
                        break
                    if p_idx < len(self._widget.classes) and p_idx >= 0 and self._widget.classes[p_idx] == selected_behavior:
                        has_prev = True
                        break
                prev_search_idx -= 1

        if not has_prev:
            prev_btn.setEnabled(False)

        def go_prev():
            self._widget.clip_popup_maximized = self.isMaximized()
            self.close()

            target_idx = self._clip_idx - 1
            if selected_behavior == "All Behaviors":
                if target_idx >= 0:
                    self._widget._show_clip_popup(target_idx)
            else:
                while target_idx >= 0:
                    p_idx = self._widget._effective_prediction_for_clip(target_idx)

                    if selected_attr is not None:
                        if selected_attr in self._widget.attributes:
                            attr_target_idx = self._widget.attributes.index(selected_attr)
                            if self._widget._get_attr_idx(target_idx) == attr_target_idx:
                                self._widget._show_clip_popup(target_idx)
                                return
                    else:
                        if selected_behavior == self._widget.ignore_label_name and p_idx < 0:
                            self._widget._show_clip_popup(target_idx)
                            return
                        if p_idx < len(self._widget.classes) and p_idx >= 0 and self._widget.classes[p_idx] == selected_behavior:
                            self._widget._show_clip_popup(target_idx)
                            return
                    target_idx -= 1

        prev_btn.clicked.connect(go_prev)
        controls_layout.addWidget(prev_btn)

        play_pause_btn = QPushButton("Play")
        is_playing = [False]
        current_frame_idx = [0]
        video_frames = []
        self._play_timer = QTimer()

        def load_clip_frames():
            try:
                cap = cv2.VideoCapture(self._widget.video_path)
                if not cap.isOpened():
                    video_label.setText("Error: Could not open video")
                    return []

                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                if orig_fps <= 0:
                    orig_fps = 30.0

                target_fps = self._widget.target_fps_spin.value()
                frame_interval = self._widget._get_saved_frame_interval(self._widget.video_path, orig_fps)
                clip_length = self._widget.clip_length_spin.value()

                clip_start_frame = self._widget.clip_starts[self._clip_idx]

                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

                frames = []
                frame_count = 0

                while len(frames) < clip_length:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        bbox = self._widget._get_localization_bbox_for_clip_frame(self._clip_idx, len(frames))
                        crop_bbox = self._widget._get_classification_roi_bbox_for_clip_frame(self._clip_idx)
                        if bbox is not None:
                            h, w = frame_rgb.shape[:2]
                            x1, y1, x2, y2 = bbox
                            fx1 = max(0, min(int(round(x1 * w)), w - 1))
                            fy1 = max(0, min(int(round(y1 * h)), h - 1))
                            fx2 = max(fx1 + 1, min(int(round(x2 * w)), w))
                            fy2 = max(fy1 + 1, min(int(round(y2 * h)), h))
                            cv2.rectangle(frame_rgb, (fx1, fy1), (fx2 - 1, fy2 - 1), (0, 255, 0), 2)
                            cv2.putText(
                                frame_rgb,
                                "Raw localization",
                                (fx1, max(18, fy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )
                        if crop_bbox is not None:
                            h, w = frame_rgb.shape[:2]
                            x1, y1, x2, y2 = crop_bbox
                            fx1 = max(0, min(int(round(x1 * w)), w - 1))
                            fy1 = max(0, min(int(round(y1 * h)), h - 1))
                            fx2 = max(fx1 + 1, min(int(round(x2 * w)), w))
                            fy2 = max(fy1 + 1, min(int(round(y2 * h)), h))
                            cv2.rectangle(frame_rgb, (fx1, fy1), (fx2 - 1, fy2 - 1), (0, 255, 255), 2)
                            cv2.putText(
                                frame_rgb,
                                "Cls crop ROI",
                                (fx1, min(h - 8, fy2 + 18)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        frames.append(frame_rgb)

                    frame_count += 1

                cap.release()

                if frames:
                    fps = target_fps
                    self._play_timer.setInterval(int(1000 / fps))

                return frames
            except Exception as e:
                video_label.setText(f"Error loading clip: {str(e)}")
                return []

        def update_frame():
            if current_frame_idx[0] < len(video_frames):
                frame = video_frames[current_frame_idx[0]]
                h, w, c = frame.shape
                bytes_per_line = int(c * w)
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)

                q_image = QImage(frame.data, int(w), int(h), bytes_per_line, QImage.Format.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(
                    video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                video_label.setPixmap(scaled_pixmap)
                current_frame_idx[0] = (current_frame_idx[0] + 1) % len(video_frames)
            else:
                self._play_timer.stop()
                is_playing[0] = False
                play_pause_btn.setText("Play")

        def toggle_play():
            if not video_frames:
                loaded_frames = load_clip_frames()
                video_frames.extend(loaded_frames)
                if video_frames:
                    update_frame()

            if is_playing[0]:
                play_pause_btn.setText("Play")
                self._play_timer.stop()
                is_playing[0] = False
            else:
                play_pause_btn.setText("Pause")
                self._play_timer.start()
                is_playing[0] = True

        self._play_timer.timeout.connect(update_frame)
        self._play_timer.setInterval(33)

        play_pause_btn.clicked.connect(toggle_play)
        controls_layout.addWidget(play_pause_btn)

        restart_btn = QPushButton("Restart")

        def restart():
            current_frame_idx[0] = 0
            if is_playing[0]:
                toggle_play()
            update_frame()

        restart_btn.clicked.connect(restart)
        controls_layout.addWidget(restart_btn)

        next_btn = QPushButton("Next >")
        next_btn.setToolTip("Next Clip")

        has_next = False
        next_search_idx = self._clip_idx + 1

        if selected_behavior == "All Behaviors":
            if next_search_idx < len(self._widget.predictions):
                has_next = True
        else:
            while next_search_idx < len(self._widget.predictions):
                p_idx = self._widget._effective_prediction_for_clip(next_search_idx)

                if selected_attr is not None:
                    if selected_attr in self._widget.attributes:
                        attr_target_idx = self._widget.attributes.index(selected_attr)
                        if self._widget._get_attr_idx(next_search_idx) == attr_target_idx:
                            has_next = True
                            break
                else:
                    if selected_behavior == self._widget.ignore_label_name and p_idx < 0:
                        has_next = True
                        break
                    if p_idx < len(self._widget.classes) and p_idx >= 0 and self._widget.classes[p_idx] == selected_behavior:
                        has_next = True
                        break
                next_search_idx += 1

        if not has_next:
            next_btn.setEnabled(False)

        def go_next():
            self._widget.clip_popup_maximized = self.isMaximized()
            self.close()

            target_idx = self._clip_idx + 1
            if selected_behavior == "All Behaviors":
                if target_idx < len(self._widget.predictions):
                    self._widget._show_clip_popup(target_idx)
            else:
                while target_idx < len(self._widget.predictions):
                    p_idx = self._widget._effective_prediction_for_clip(target_idx)

                    if selected_attr is not None:
                        if selected_attr in self._widget.attributes:
                            attr_target_idx = self._widget.attributes.index(selected_attr)
                            if self._widget._get_attr_idx(target_idx) == attr_target_idx:
                                self._widget._show_clip_popup(target_idx)
                                return
                    else:
                        if selected_behavior == self._widget.ignore_label_name and p_idx < 0:
                            self._widget._show_clip_popup(target_idx)
                            return
                        if p_idx < len(self._widget.classes) and p_idx >= 0 and self._widget.classes[p_idx] == selected_behavior:
                            self._widget._show_clip_popup(target_idx)
                            return
                    target_idx += 1

        next_btn.clicked.connect(go_next)
        controls_layout.addWidget(next_btn)

        controls_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        controls_layout.addWidget(close_btn)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

        def load_and_show_first_frame():
            loaded_frames = load_clip_frames()
            video_frames.extend(loaded_frames)
            if video_frames:
                update_frame()

        QTimer.singleShot(100, load_and_show_first_frame)

        self.exec()

        if self._play_timer.isActive():
            self._play_timer.stop()

    def closeEvent(self, event):
        if hasattr(self, '_play_timer') and self._play_timer.isActive():
            self._play_timer.stop()
        super().closeEvent(event)


class FrameSegmentPopupDialog(QDialog):
    """Dialog showing a frame-aggregated segment with training and transition controls."""

    def __init__(self, parent, widget, frame_idx, segment, segment_idx):
        super().__init__(parent)
        self._widget = widget
        self._frame_idx = frame_idx
        self._segment = segment
        self._segment_idx = segment_idx

        if not self._widget.video_path:
            self.close()
            return

        if self._segment_idx is None:
            for i, seg in enumerate(self._widget.aggregated_segments):
                if seg['start'] == self._segment['start'] and seg['end'] == self._segment['end']:
                    self._segment_idx = i
                    break

        pred_idx = self._segment['class']
        conf = self._segment.get('confidence', 1.0)
        start_frame = self._segment['start']
        end_frame = self._segment['end']

        if pred_idx >= len(self._widget.classes):
            self.close()
            return

        label = self._widget.classes[pred_idx]

        cap = cv2.VideoCapture(self._widget.video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30.0
        cap.release()

        start_time = start_frame / orig_fps
        end_time = (end_frame + 1) / orig_fps
        duration = end_time - start_time
        clicked_time = self._frame_idx / orig_fps

        self.setWindowTitle(f"Segment {(self._segment_idx + 1) if self._segment_idx is not None else '?'}/{len(self._widget.aggregated_segments)}: {label} (frames {start_frame}-{end_frame})")
        self.setMinimumSize(700, 650)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout()

        info_text = (
            f"<b>Behavior:</b> {label}<br>"
            f"<b>Aggregated Confidence:</b> {conf:.2f}<br>"
            f"<b>Frame Range:</b> {start_frame} - {end_frame} ({end_frame - start_frame + 1} frames)<br>"
            f"<b>Time Range:</b> {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s duration)<br>"
            f"<b>Clicked Frame:</b> {self._frame_idx} ({clicked_time:.2f}s)"
        )
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(info_label)

        training_group = QGroupBox("Add segment to training dataset")
        training_layout = QVBoxLayout()
        seg_label_row = QHBoxLayout()
        seg_label_row.addWidget(QLabel("Training label:"))
        seg_label_combo = QComboBox()
        seg_label_combo.addItems(self._widget.classes)
        if 0 <= pred_idx < len(self._widget.classes):
            seg_label_combo.setCurrentIndex(pred_idx)
        seg_label_row.addWidget(seg_label_combo)
        training_layout.addLayout(seg_label_row)

        add_segment_btn = QPushButton("Add segment chunks to training dataset")
        add_segment_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        add_segment_btn.setToolTip(
            "Creates consecutive clips over this segment.\n"
            "Frames inside the segment are labeled; frames outside are set to None (ignored)."
        )

        def add_segment_chunks_to_training():
            try:
                selected_label = seg_label_combo.currentText().strip()
                if not selected_label:
                    QMessageBox.warning(self, "Missing label", "Select a training label first.")
                    return

                clips_dir = self._widget._get_clips_dir()
                annotation_manager = AnnotationManager(self._widget._get_annotation_file())
                annotation_manager.add_class(selected_label)

                clip_length = int(self._widget.clip_length_spin.value())
                if clip_length <= 0:
                    QMessageBox.warning(self, "Invalid clip length", "Clip length must be > 0.")
                    return

                frame_interval = int(max(1, self._widget._get_saved_frame_interval(self._widget.video_path, orig_fps)))
                segment_sampled_frames = ((end_frame - start_frame) // frame_interval) + 1
                if segment_sampled_frames <= 0:
                    QMessageBox.warning(self, "Empty segment", "Segment has no usable frames.")
                    return

                num_chunks = (segment_sampled_frames + clip_length - 1) // clip_length
                if num_chunks <= 0:
                    QMessageBox.warning(self, "No chunks", "Could not create segment chunks.")
                    return

                cap = cv2.VideoCapture(self._widget.video_path)
                if not cap.isOpened():
                    QMessageBox.warning(self, "Error", "Could not open video file.")
                    return

                target_fps = int(self._widget.target_fps_spin.value())
                video_basename = self._widget._video_basename()
                added_paths = []

                for chunk_idx in range(num_chunks):
                    chunk_start_vid_frame = int(start_frame + chunk_idx * clip_length * frame_interval)
                    frames_in_segment = min(clip_length, segment_sampled_frames - chunk_idx * clip_length)

                    if frames_in_segment <= 0:
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start_vid_frame)
                    frames = []
                    frame_count = 0

                    while len(frames) < clip_length:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % frame_interval == 0:
                            frames.append(frame.copy())
                        frame_count += 1

                    if frames and len(frames) < clip_length:
                        last_frame = frames[-1]
                        while len(frames) < clip_length:
                            frames.append(last_frame.copy())

                    if not frames:
                        continue

                    clip_filename = (
                        f"{video_basename}_seg_{start_frame}_{end_frame}_"
                        f"chunk_{chunk_idx:03d}_frame_{chunk_start_vid_frame}.mp4"
                    )
                    clip_path = os.path.join(clips_dir, clip_filename)
                    clip_path = self._widget._unique_clip_path(clip_path)

                    save_clip(frames, clip_path, target_fps)
                    if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                        continue

                    clip_id = self._widget._clip_path_to_id(clip_path, clips_dir)

                    frame_labels = []
                    for i in range(clip_length):
                        if i < frames_in_segment:
                            frame_labels.append(selected_label)
                        else:
                            frame_labels.append(None)

                    meta = {
                        "source_video": os.path.basename(self._widget.video_path),
                        "source_segment_start_frame": int(start_frame),
                        "source_segment_end_frame": int(end_frame),
                        "source_chunk_index": int(chunk_idx),
                        "source_frame": int(chunk_start_vid_frame),
                        "target_fps": int(target_fps),
                        "clip_length": int(clip_length),
                        "added_from_inference_segment": True,
                        "segment_label_frames": int(frames_in_segment),
                    }

                    annotation_manager.add_clip(clip_id, selected_label, meta=meta, _defer_save=True)
                    annotation_manager.set_frame_labels(clip_id, frame_labels, _defer_save=True)
                    added_paths.append(clip_path)

                cap.release()

                if not added_paths:
                    QMessageBox.warning(self, "Nothing added", "No segment clips were added.")
                    return

                annotation_manager.save()
                self._widget.log_text.append(
                    f"Added {len(added_paths)} segment chunk(s) to training dataset "
                    f"for '{selected_label}' (segment {start_frame}-{end_frame})."
                )
                QMessageBox.information(
                    self,
                    "Segment added",
                    f"Added {len(added_paths)} clip(s) to training dataset.\n\n"
                    f"Label: {selected_label}\n"
                    f"Segment: frames {start_frame}-{end_frame}\n\n"
                    f"Each clip has frame labels only where behavior is inside this segment; "
                    "other frames are set to None (ignored).",
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add segment chunks:\n{str(e)}")

        add_segment_btn.clicked.connect(add_segment_chunks_to_training)
        training_layout.addWidget(add_segment_btn)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        video_label = QLabel("Loading segment...")
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setMinimumSize(640, 360)
        video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(video_label)

        controls_layout = QHBoxLayout()

        prev_seg_btn = QPushButton("< Prev Segment")
        prev_seg_btn.setToolTip("Go to previous behavior segment")
        has_prev = self._segment_idx is not None and self._segment_idx > 0
        prev_seg_btn.setEnabled(has_prev)

        def go_prev_segment():
            if self._segment_idx is not None and self._segment_idx > 0:
                self.close()
                prev_seg = self._widget.aggregated_segments[self._segment_idx - 1]
                mid_frame = (prev_seg['start'] + prev_seg['end']) // 2
                self._widget._show_frame_segment_popup(mid_frame, prev_seg, self._segment_idx - 1)

        prev_seg_btn.clicked.connect(go_prev_segment)
        controls_layout.addWidget(prev_seg_btn)

        frame_slider = QSpinBox()
        frame_slider.setRange(start_frame, end_frame)
        frame_slider.setValue(self._frame_idx)
        controls_layout.addWidget(QLabel("Frame:"))
        controls_layout.addWidget(frame_slider)

        play_pause_btn = QPushButton("Play")
        is_playing = [False]
        current_frame_idx = [self._frame_idx]
        video_frames = {}
        self._play_timer = QTimer()

        def load_frame(f_idx):
            if f_idx in video_frames:
                return video_frames[f_idx]

            cap = cv2.VideoCapture(self._widget.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames[f_idx] = frame_rgb
                return frame_rgb
            return None

        def update_frame():
            frame = load_frame(current_frame_idx[0])
            if frame is not None:
                h, w, c = frame.shape
                q_img = QImage(frame.data, w, h, w * c, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(
                    video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                video_label.setPixmap(scaled_pixmap)

        def on_frame_changed(val):
            current_frame_idx[0] = val
            update_frame()

        frame_slider.valueChanged.connect(on_frame_changed)

        def toggle_play():
            if is_playing[0]:
                self._play_timer.stop()
                play_pause_btn.setText("Play")
                is_playing[0] = False
            else:
                self._play_timer.start(int(1000 / orig_fps))
                play_pause_btn.setText("Pause")
                is_playing[0] = True

        def advance_frame():
            current_frame_idx[0] += 1
            if current_frame_idx[0] > end_frame:
                current_frame_idx[0] = start_frame
            frame_slider.setValue(current_frame_idx[0])
            update_frame()

        self._play_timer.timeout.connect(advance_frame)
        play_pause_btn.clicked.connect(toggle_play)
        controls_layout.addWidget(play_pause_btn)

        next_seg_btn = QPushButton("Next Segment >")
        next_seg_btn.setToolTip("Go to next behavior segment")
        has_next = self._segment_idx is not None and self._segment_idx < len(self._widget.aggregated_segments) - 1
        next_seg_btn.setEnabled(has_next)

        def go_next_segment():
            if self._segment_idx is not None and self._segment_idx < len(self._widget.aggregated_segments) - 1:
                self.close()
                next_seg = self._widget.aggregated_segments[self._segment_idx + 1]
                mid_frame = (next_seg['start'] + next_seg['end']) // 2
                self._widget._show_frame_segment_popup(mid_frame, next_seg, self._segment_idx + 1)

        next_seg_btn.clicked.connect(go_next_segment)
        controls_layout.addWidget(next_seg_btn)

        controls_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        controls_layout.addWidget(close_btn)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

        QTimer.singleShot(100, update_frame)

        self.exec()

        if self._play_timer.isActive():
            self._play_timer.stop()

    def closeEvent(self, event):
        if hasattr(self, '_play_timer') and self._play_timer.isActive():
            self._play_timer.stop()
        super().closeEvent(event)
