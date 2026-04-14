"""
Overlay export and video preview for inference results.
Options dialog, export loop, and preview player live here to keep inference_widget lean.
"""

import os
import shutil
import subprocess
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QGroupBox, QFormLayout,
    QScrollArea, QWidget, QPushButton, QDialogButtonBox, QMessageBox, QProgressDialog,
    QFileDialog, QApplication, QRadioButton, QSpinBox, QSlider, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


_NVENC_AVAILABLE = None


def _ffmpeg_nvenc_available():
    """Return True when ffmpeg with NVIDIA NVENC encoding is usable."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        _NVENC_AVAILABLE = False
        return _NVENC_AVAILABLE

    # NVENC rejects very small frame sizes, so use a normal tiny test clip.
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.2",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        _NVENC_AVAILABLE = (res.returncode == 0)
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def _open_overlay_writer(output_path: str, fps: float, width: int, height: int):
    """Open a video writer, preferring ffmpeg NVENC and falling back to OpenCV."""
    if _ffmpeg_nvenc_available():
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            cmd = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                f"{float(fps):.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                output_path,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                return "ffmpeg_nvenc", proc
            except Exception:
                pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return "opencv_mp4v", out


def _write_overlay_frame(writer_kind: str, writer_obj, frame: np.ndarray):
    if writer_kind == "ffmpeg_nvenc":
        try:
            writer_obj.stdin.write(frame.tobytes())
        except Exception as e:
            stderr_text = ""
            try:
                if writer_obj.stderr is not None:
                    stderr_text = writer_obj.stderr.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
            raise RuntimeError(stderr_text or f"ffmpeg NVENC write failed: {e}") from e
        return
    writer_obj.write(frame)


def _close_overlay_writer(writer_kind: str, writer_obj, abort: bool = False):
    """Close writer and return a short encoder label."""
    if writer_kind == "ffmpeg_nvenc":
        stderr_text = ""
        try:
            if abort:
                if writer_obj.stdin is not None:
                    writer_obj.stdin.close()
                writer_obj.kill()
                writer_obj.wait(timeout=5)
                return "ffmpeg NVENC"

            if writer_obj.stdin is not None:
                writer_obj.stdin.close()
            return_code = writer_obj.wait(timeout=30)
            if writer_obj.stderr is not None:
                stderr_text = writer_obj.stderr.read().decode("utf-8", errors="ignore").strip()
            if return_code != 0:
                raise RuntimeError(stderr_text or f"ffmpeg NVENC exited with code {return_code}")
            return "ffmpeg NVENC"
        finally:
            try:
                if writer_obj.stderr is not None:
                    writer_obj.stderr.close()
            except Exception:
                pass

    writer_obj.release()
    return "OpenCV mp4v"


def ask_overlay_export_options(widget):
    """Ask user which behaviors to include and whether to use precise boundaries."""
    dlg = QDialog(widget)
    dlg.setWindowTitle("Overlay export options")
    dlg.resize(760, 560)
    layout = QVBoxLayout(dlg)

    available_videos = [
        vp for vp in getattr(widget, "results_cache", {}).keys()
        if isinstance(getattr(widget, "results_cache", {}).get(vp), dict)
    ]
    if not available_videos and getattr(widget, "video_path", None):
        available_videos = [widget.video_path]
    current_video = getattr(widget, "video_path", None)

    layout.addWidget(QLabel("Videos to export:"))
    video_button_row = QHBoxLayout()
    video_all_btn = QPushButton("Select all")
    video_current_btn = QPushButton("Current video")
    video_clear_btn = QPushButton("Clear")
    video_button_row.addWidget(video_all_btn)
    video_button_row.addWidget(video_current_btn)
    video_button_row.addWidget(video_clear_btn)
    video_button_row.addStretch()
    layout.addLayout(video_button_row)

    video_list = QListWidget()
    for vp in available_videos:
        item = QListWidgetItem(os.path.basename(vp))
        item.setData(Qt.ItemDataRole.UserRole, vp)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        default_checked = len(available_videos) == 1 or vp == current_video
        item.setCheckState(Qt.CheckState.Checked if default_checked else Qt.CheckState.Unchecked)
        video_list.addItem(item)
    video_list.setMaximumHeight(120)
    layout.addWidget(video_list)

    def _set_video_checks(state):
        for i in range(video_list.count()):
            video_list.item(i).setCheckState(state)

    def _select_current_video():
        for i in range(video_list.count()):
            item = video_list.item(i)
            is_current = item.data(Qt.ItemDataRole.UserRole) == current_video
            item.setCheckState(Qt.CheckState.Checked if is_current else Qt.CheckState.Unchecked)

    video_all_btn.clicked.connect(lambda: _set_video_checks(Qt.CheckState.Checked))
    video_clear_btn.clicked.connect(lambda: _set_video_checks(Qt.CheckState.Unchecked))
    video_current_btn.clicked.connect(_select_current_video)

    use_precise_cb = QCheckBox("Use precise boundary timeline when available")
    use_precise_cb.setChecked(
        bool(getattr(widget, "aggregated_segments", None))
        or (getattr(widget, "frame_aggregation_check", None) and widget.frame_aggregation_check.isChecked())
    )
    layout.addWidget(use_precise_cb)

    ignore_label = getattr(widget, "ignore_label_name", "Filtered")
    include_ignore_cb = QCheckBox(f"Include '{ignore_label}' overlays")
    include_ignore_cb.setChecked(False)
    layout.addWidget(include_ignore_cb)

    video_duration_s = 0.0
    video_fps = 30.0
    video_path = getattr(widget, "video_path", None)
    if video_path:
        try:
            _cap = cv2.VideoCapture(video_path)
            _fps = _cap.get(cv2.CAP_PROP_FPS)
            if _fps > 0:
                video_fps = _fps
            _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration_s = _total / video_fps
            _cap.release()
        except Exception:
            pass

    range_group = QGroupBox("Export range")
    range_layout = QVBoxLayout(range_group)

    def _fmt_time(s):
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    rb_full = QRadioButton(f"Full video ({_fmt_time(video_duration_s)})")
    rb_samples = QRadioButton("Quick-check samples")
    rb_full.setChecked(True)
    range_layout.addWidget(rb_full)
    range_layout.addWidget(rb_samples)

    samples_widget = QWidget()
    samples_form = QFormLayout(samples_widget)
    samples_form.setContentsMargins(20, 4, 0, 4)
    sample_dur_spin = QSpinBox()
    sample_dur_spin.setRange(10, 300)
    sample_dur_spin.setValue(60)
    sample_dur_spin.setSuffix(" s")
    sample_dur_spin.setToolTip("Duration of each sample clip")
    num_samples_spin = QSpinBox()
    num_samples_spin.setRange(1, 50)
    num_samples_spin.setValue(min(5, max(1, int(video_duration_s / 120))))
    num_samples_spin.setToolTip("Number of sample clips spread evenly across the video")
    samples_form.addRow("Clip duration:", sample_dur_spin)
    samples_form.addRow("Number of samples:", num_samples_spin)
    samples_info = QLabel()
    samples_form.addRow(samples_info)

    def _update_samples_info():
        n = num_samples_spin.value()
        d = sample_dur_spin.value()
        total = n * d
        samples_info.setText(
            f"{n} × {d}s clips = {_fmt_time(total)} of {_fmt_time(video_duration_s)} total\n"
            f"Spread evenly across the video, saved to a folder"
        )

    num_samples_spin.valueChanged.connect(lambda: _update_samples_info())
    sample_dur_spin.valueChanged.connect(lambda: _update_samples_info())
    _update_samples_info()

    samples_widget.setVisible(False)
    range_layout.addWidget(samples_widget)
    rb_samples.toggled.connect(samples_widget.setVisible)
    layout.addWidget(range_group)

    layout.addWidget(QLabel("Behaviors to render in exported video:"))
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    body = QWidget()
    body_layout = QVBoxLayout(body)
    behavior_checks = []
    for cls in getattr(widget, "classes", []):
        cb = QCheckBox(cls)
        cb.setChecked(True)
        behavior_checks.append(cb)
        body_layout.addWidget(cb)
    body_layout.addStretch()
    scroll.setWidget(body)
    layout.addWidget(scroll)

    btn_row = QHBoxLayout()
    sel_all = QPushButton("Select all")
    sel_none = QPushButton("Select none")
    btn_row.addWidget(sel_all)
    btn_row.addWidget(sel_none)
    btn_row.addStretch()
    layout.addLayout(btn_row)

    def _set_all(v: bool):
        for cb in behavior_checks:
            cb.setChecked(v)

    sel_all.clicked.connect(lambda: _set_all(True))
    sel_none.clicked.connect(lambda: _set_all(False))

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None

    selected_videos = []
    for i in range(video_list.count()):
        item = video_list.item(i)
        if item.checkState() == Qt.CheckState.Checked:
            selected_videos.append(item.data(Qt.ItemDataRole.UserRole))
    if not selected_videos:
        QMessageBox.warning(widget, "No videos selected", "Select at least one video to export.")
        return None

    selected = {cb.text() for cb in behavior_checks if cb.isChecked()}
    if not selected and not include_ignore_cb.isChecked():
        QMessageBox.warning(
            widget, "Nothing selected",
            "Select at least one behavior or include ignore overlays.",
        )
        return None

    result = {
        "selected_videos": selected_videos,
        "selected_behaviors": selected,
        "include_ignore": bool(include_ignore_cb.isChecked()),
        "use_precise": bool(use_precise_cb.isChecked()),
        "mode": "full",
    }
    if rb_samples.isChecked():
        result["mode"] = "samples"
        result["sample_duration_seconds"] = int(sample_dur_spin.value())
        result["sample_num_clips"] = int(num_samples_spin.value())
    return result


def run_export_video_with_overlay(widget):
    """Export video with configurable overlays (entry point from inference widget)."""
    if not getattr(widget, "video_path", None) or not getattr(widget, "predictions", None):
        QMessageBox.warning(widget, "Error", "No predictions available to export.")
        return

    if hasattr(widget, "_persist_current_video_state"):
        widget._persist_current_video_state()

    opts = ask_overlay_export_options(widget)
    if not opts:
        return

    selected_videos = list(opts.get("selected_videos", []))
    if not selected_videos:
        return

    original_video_path = getattr(widget, "video_path", None)
    original_threshold_settings = (
        widget._current_threshold_settings()
        if hasattr(widget, "_current_threshold_settings")
        else None
    )
    shared_threshold_settings = dict(original_threshold_settings or {})
    multi_video = len(selected_videos) > 1

    if opts.get("mode") == "samples":
        default_dir = os.path.dirname(original_video_path) if original_video_path else os.getcwd()
        folder = QFileDialog.getExistingDirectory(
            widget, "Select folder for sample clips", default_dir,
        )
        if not folder:
            return
    elif multi_video:
        default_dir = os.path.dirname(original_video_path) if original_video_path else os.getcwd()
        folder = QFileDialog.getExistingDirectory(
            widget, "Select folder for overlay videos", default_dir,
        )
        if not folder:
            return
    else:
        video_path = selected_videos[0]
        output_path, _ = QFileDialog.getSaveFileName(
            widget,
            "Save Video with Overlays",
            os.path.splitext(video_path)[0] + "_annotated.mp4",
            "Video Files (*.mp4);;All Files (*)",
        )
        if not output_path:
            return
        folder = None

    exported = []
    encoders_used = []
    try:
        for video_path in selected_videos:
            entry = getattr(widget, "results_cache", {}).get(video_path, {})
            threshold_override = None
            if not isinstance(entry.get("threshold_settings"), dict) and shared_threshold_settings:
                threshold_override = shared_threshold_settings
            if hasattr(widget, "_load_video_from_cache"):
                ok = widget._load_video_from_cache(
                    video_path,
                    refresh_display=False,
                    persist_current=False,
                    threshold_settings_override=threshold_override,
                    persist_loaded_thresholds=False,
                )
                if not ok:
                    continue

            if opts.get("mode") == "samples":
                out_dir = os.path.join(
                    folder,
                    f"{os.path.splitext(os.path.basename(video_path))[0]}_overlay_samples",
                )
                os.makedirs(out_dir, exist_ok=True)
                sample_ranges, fps = _compute_sample_ranges_for_video(
                    video_path,
                    int(opts.get("sample_duration_seconds", 60)),
                    int(opts.get("sample_num_clips", 5)),
                )
                video_exported = 0
                for si, (sf, ef) in enumerate(sample_ranges):
                    start_s = sf / fps
                    end_s = ef / fps
                    sample_path = os.path.join(
                        out_dir, f"sample_{si+1:02d}_{start_s:.0f}s-{end_s:.0f}s.mp4"
                    )
                    sample_opts = dict(opts)
                    sample_opts["mode"] = "range"
                    sample_opts["start_frame"] = sf
                    sample_opts["end_frame"] = ef
                    encoder_label = run_export_single_overlay(
                        widget,
                        sample_path,
                        sample_opts,
                        sample_label=f"{os.path.basename(video_path)} sample {si+1}/{len(sample_ranges)}",
                    )
                    if not encoder_label:
                        break
                    exported.append(sample_path)
                    encoders_used.append(str(encoder_label))
                    video_exported += 1
                widget.log_text.append(
                    f"Exported {video_exported} sample overlay clip(s) for {os.path.basename(video_path)} to {out_dir}"
                )
            else:
                if multi_video:
                    output_path = os.path.join(
                        folder,
                        os.path.splitext(os.path.basename(video_path))[0] + "_annotated.mp4",
                    )
                export_opts = dict(opts)
                if multi_video:
                    export_opts["show_success_popup"] = False
                encoder_label = run_export_single_overlay(widget, output_path, export_opts)
                if not encoder_label:
                    break
                exported.append(output_path)
                encoders_used.append(str(encoder_label))
    finally:
        if (
            original_video_path
            and hasattr(widget, "_load_video_from_cache")
            and original_video_path in getattr(widget, "results_cache", {})
        ):
            widget._load_video_from_cache(
                original_video_path,
                refresh_display=True,
                persist_current=False,
                threshold_settings_override=original_threshold_settings,
                persist_loaded_thresholds=False,
            )
            if getattr(widget, "filter_video_combo", None) is not None:
                idx = widget.filter_video_combo.findData(original_video_path)
                if idx >= 0:
                    widget.filter_video_combo.blockSignals(True)
                    widget.filter_video_combo.setCurrentIndex(idx)
                    widget.filter_video_combo.blockSignals(False)

    if not exported:
        return

    if opts.get("mode") == "samples":
        encoder_summary = ", ".join(sorted(set(encoders_used))) if encoders_used else "unknown"
        QMessageBox.information(
            widget,
            "Success",
            f"Exported {len(exported)} sample overlay clip(s) to:\n{folder}\n\n"
            f"Encoder: {encoder_summary}",
        )
    elif multi_video:
        encoder_summary = ", ".join(sorted(set(encoders_used))) if encoders_used else "unknown"
        QMessageBox.information(
            widget,
            "Success",
            f"Exported overlay videos for {len(exported)} video(s) to:\n{folder}\n\n"
            f"Encoder: {encoder_summary}",
        )


def _compute_sample_ranges_for_video(video_path: str, duration_seconds: int, num_samples: int):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    dur_frames = max(1, int(round(max(1, duration_seconds) * fps)))
    dur_frames = max(1, min(dur_frames, max(1, total_frames)))
    usable = max(0, total_frames - dur_frames)
    n = max(1, int(num_samples))
    if n == 1:
        starts = [usable // 2]
    else:
        starts = [int(round(i * usable / (n - 1))) for i in range(n)]
    return [(s, min(s + dur_frames, total_frames)) for s in starts], fps


def run_export_single_overlay(widget, output_path, opts, sample_label=None):
    """Export a single overlay video file for the given options/range."""
    cap = None
    progress = None
    writer_kind = None
    out = None
    try:
        if opts["use_precise"] and not getattr(widget, "aggregated_segments", None):
            widget._compute_aggregated_timeline()
        if (
            opts["use_precise"]
            and getattr(widget, "_use_ovr", False)
            and not isinstance(getattr(widget, "_aggregated_frame_scores_norm", None), np.ndarray)
        ):
            widget._compute_aggregated_timeline()

        use_precise = bool(opts["use_precise"] and getattr(widget, "aggregated_segments", None))
        selected_behaviors = set(opts["selected_behaviors"])
        include_ignore = bool(opts["include_ignore"])
        palette = widget._get_timeline_palette()

        video_path = widget.video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = widget._get_saved_frame_interval(video_path, fps)

        MIN_EXPORT_WIDTH = 480
        if raw_width < MIN_EXPORT_WIDTH:
            scale = MIN_EXPORT_WIDTH / raw_width
            width = MIN_EXPORT_WIDTH
            height = int(round(raw_height * scale))
        else:
            width = raw_width
            height = raw_height
        clip_length = widget.clip_length_spin.value()

        export_start_frame = int(opts.get("start_frame", 0))
        export_end_frame = int(opts.get("end_frame", total_frames))
        export_end_frame = min(export_end_frame, total_frames)
        export_frame_count = max(1, export_end_frame - export_start_frame)
        is_partial = export_start_frame > 0 or export_end_frame < total_frames

        top_panel_h = 110
        out_h = height + top_panel_h
        writer_kind, out = _open_overlay_writer(output_path, fps, width, out_h)

        if export_start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, export_start_frame)

        range_label = ""
        if is_partial:
            start_s = export_start_frame / fps
            end_s = export_end_frame / fps
            range_label = f" ({start_s:.1f}s – {end_s:.1f}s)"

        progress_title = f"{sample_label} — " if sample_label else ""
        progress = QProgressDialog(
            f"{progress_title}Exporting{range_label}...",
            "Cancel",
            0,
            max(1, export_frame_count),
            widget,
        )
        progress.setWindowTitle(f"{progress_title}Export progress")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        segments = getattr(widget, "aggregated_segments", None) or []
        frame_idx = export_start_frame
        clip_idx = 0
        seg_idx = 0
        update_interval = max(1, export_frame_count // 100)

        if export_start_frame > 0 and getattr(widget, "clip_starts", None):
            for ci_tmp in range(len(widget.clip_starts)):
                if int(widget.clip_starts[ci_tmp]) <= export_start_frame:
                    clip_idx = ci_tmp
                else:
                    break
        if export_start_frame > 0 and segments:
            for si_tmp in range(len(segments)):
                if int(segments[si_tmp]["end"]) < export_start_frame:
                    seg_idx = si_tmp + 1
                else:
                    break

        has_loc = bool(
            getattr(widget, "localization_bboxes", None)
            and len(widget.localization_bboxes) > 0
        )
        trail = []
        trail_max = 50
        label_anchor = None
        segment_start_frame = None
        last_cx_px, last_cy_px = None, None
        smooth_alpha = 0.35
        prev_primary_idx = None
        max_rows = 6
        timeline_window_frames = max(60, int(round(fps * 8.0)))
        selected_cls_idx = [
            i for i, name in enumerate(widget.classes) if name in selected_behaviors
        ][:max_rows]
        row_history = {ci: [] for ci in selected_cls_idx}

        while True:
            if frame_idx >= export_end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                is_upscale = (frame.shape[1] < width) or (frame.shape[0] < height)
                interp = cv2.INTER_LANCZOS4 if is_upscale else cv2.INTER_AREA
                frame = cv2.resize(frame, (width, height), interpolation=interp)
                if is_upscale:
                    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
                    frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

            frames_done = frame_idx - export_start_frame
            if frames_done % update_interval == 0 or frame_idx == export_end_frame - 1:
                progress.setValue(frames_done)
                progress.setLabelText(
                    f"{progress_title}Exporting{range_label}... frame {frames_done + 1} / {export_frame_count}"
                )
                QApplication.processEvents()
                if progress.wasCanceled():
                    break

            pred_idx = None
            conf = None
            mode_tag = "clip"
            seg_start_this_frame = False
            active_infos = []

            if (
                use_precise
                and getattr(widget, "_use_ovr", False)
                and isinstance(getattr(widget, "_aggregated_frame_scores_norm", None), np.ndarray)
            ):
                mode_tag = "precise-ovr"
                active_infos = widget._get_precise_active_for_frame(frame_idx)
            elif use_precise:
                mode_tag = "precise"
                while seg_idx < len(segments) and frame_idx > int(segments[seg_idx]["end"]):
                    seg_idx += 1
                if seg_idx < len(segments):
                    seg = segments[seg_idx]
                    s0, s1 = int(seg["start"]), int(seg["end"])
                    if s0 <= frame_idx <= s1:
                        pred_idx = int(seg["class"])
                        conf = float(seg.get("confidence", 0.0))
                        if frame_idx == s0:
                            seg_start_this_frame = True
                        active_infos = [(pred_idx, conf)]
            else:
                clip_starts = widget.clip_starts
                while clip_idx + 1 < len(clip_starts) and frame_idx >= int(clip_starts[clip_idx + 1]):
                    clip_idx += 1
                if 0 <= clip_idx < len(clip_starts):
                    start_f = int(clip_starts[clip_idx])
                    if clip_idx + 1 < len(clip_starts):
                        end_exclusive = int(clip_starts[clip_idx + 1])
                    else:
                        end_exclusive = start_f + (clip_length * frame_interval)
                    if start_f <= frame_idx < end_exclusive:
                        pred_idx = int(widget._effective_prediction_for_clip(clip_idx))
                        conf = (
                            float(widget.confidences[clip_idx])
                            if clip_idx < len(widget.confidences)
                            else 0.0
                        )
                        if frame_idx == start_f:
                            seg_start_this_frame = True
                        active_infos = [(pred_idx, conf)]

            ignore_label = getattr(widget, "ignore_label_name", "Filtered")
            filtered_infos = []
            for ci, sc in active_infos:
                if ci < 0:
                    if include_ignore:
                        filtered_infos.append((ci, sc, ignore_label))
                elif 0 <= ci < len(widget.classes):
                    lbl = widget.classes[ci]
                    if lbl in selected_behaviors:
                        filtered_infos.append((ci, sc, lbl))

            draw_label = None
            draw_color = (160, 160, 160)
            extra_overlay_labels = []
            if filtered_infos:
                primary_idx, conf, primary_label = filtered_infos[0]
                draw_label = primary_label
                pred_idx = primary_idx
                if primary_idx >= 0:
                    pr, pg, pb = palette[primary_idx % len(palette)]
                    draw_color = (int(pb), int(pg), int(pr))
                extra_overlay_labels = [
                    f"{lbl}:{float(sc):.0%}" for _, sc, lbl in filtered_infos[1:4]
                ]
                if len(filtered_infos) > 4:
                    extra_overlay_labels.append(f"+{len(filtered_infos) - 4} more")
                if primary_idx != prev_primary_idx:
                    seg_start_this_frame = True
                prev_primary_idx = primary_idx
            else:
                prev_primary_idx = None

            if draw_label is not None and has_loc:
                clip_for_bbox = clip_idx
                if use_precise and seg_idx < len(segments):
                    start_f = int(segments[seg_idx]["start"])
                    clip_for_bbox = 0
                    for ci in range(len(widget.clip_starts)):
                        cs = int(widget.clip_starts[ci])
                        end_c = (
                            cs + (clip_length * frame_interval)
                            if ci + 1 >= len(widget.clip_starts)
                            else int(widget.clip_starts[ci + 1])
                        )
                        if cs <= frame_idx < end_c:
                            clip_for_bbox = ci
                            break
                frame_within_clip = 0
                if clip_for_bbox < len(widget.clip_starts):
                    cs = int(widget.clip_starts[clip_for_bbox])
                    frame_within_clip = (frame_idx - cs) // max(1, frame_interval)
                    frame_within_clip = max(0, min(clip_length - 1, frame_within_clip))
                bbox = widget._get_localization_bbox_for_clip_frame(clip_for_bbox, frame_within_clip)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cx_px = int(round(cx * width))
                    cy_px = int(round(cy * height))
                    if last_cx_px is not None and last_cy_px is not None:
                        cx_px = int(smooth_alpha * cx_px + (1 - smooth_alpha) * last_cx_px)
                        cy_px = int(smooth_alpha * cy_px + (1 - smooth_alpha) * last_cy_px)
                    last_cx_px, last_cy_px = cx_px, cy_px
                    cx_px = max(0, min(width - 1, cx_px))
                    cy_px = max(0, min(height - 1, cy_px))
                    trail.append((cx_px, cy_px))
                    if len(trail) > trail_max:
                        trail.pop(0)
                    if seg_start_this_frame or label_anchor is None:
                        label_anchor = (cx_px, cy_px)

                    if len(trail) >= 2:
                        overlay_trail = frame.copy()
                        pts = np.array(trail, dtype=np.int32)
                        cv2.polylines(overlay_trail, [pts], False, draw_color, 4, cv2.LINE_AA)
                        frame = cv2.addWeighted(overlay_trail, 0.4, frame, 0.6, 0.0)
                    dot_overlay = frame.copy()
                    cv2.circle(dot_overlay, (cx_px, cy_px), 10, draw_color, -1, cv2.LINE_AA)
                    cv2.circle(dot_overlay, (cx_px, cy_px), 10, (255, 255, 255), 1, cv2.LINE_AA)
                    frame = cv2.addWeighted(dot_overlay, 0.75, frame, 0.25, 0.0)
                    if label_anchor is not None:
                        lx, ly = label_anchor
                        ly = max(24, ly - 14)
                        (tw, th), _ = cv2.getTextSize(
                            draw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        has_extra = bool(extra_overlay_labels)
                        box_h = th + 8 + (16 if has_extra else 0)
                        cv2.rectangle(
                            frame,
                            (lx - 4, ly - box_h + 4),
                            (lx + tw + 4, ly + 4),
                            (30, 30, 30),
                            -1,
                        )
                        cv2.rectangle(
                            frame,
                            (lx - 4, ly - box_h + 4),
                            (lx + tw + 4, ly + 4),
                            draw_color,
                            1,
                        )
                        cv2.putText(
                            frame, draw_label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2, cv2.LINE_AA,
                        )
                        if has_extra:
                            cv2.putText(
                                frame,
                                ", ".join(extra_overlay_labels),
                                (lx, ly + 14),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (220, 220, 220),
                                1,
                                cv2.LINE_AA,
                            )
            elif draw_label is not None:
                segment_start_frame = None
                label_anchor = None
                trail.clear()
                last_cx_px, last_cy_px = None, None
                panel_h = 88 + (22 if extra_overlay_labels else 0)
                x0, y0 = 12, 12
                x1 = min(width - 12, 540)
                y1 = min(height - 12, y0 + panel_h)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
                frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), draw_color, 2)
                conf_txt = f"{conf:.1%}" if conf is not None else "n/a"
                cv2.putText(
                    frame, draw_label, (x0 + 12, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, draw_color, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"conf: {conf_txt} | {mode_tag} | t={frame_idx / fps:.2f}s",
                    (x0 + 12, y0 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                if extra_overlay_labels:
                    cv2.putText(
                        frame,
                        "co-occur: " + ", ".join(extra_overlay_labels),
                        (x0 + 12, y0 + 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (210, 210, 210),
                        1,
                        cv2.LINE_AA,
                    )
            else:
                trail.clear()
                label_anchor = None
                segment_start_frame = None
                last_cx_px, last_cy_px = None, None

            composed = np.zeros((out_h, width, 3), dtype=np.uint8)
            panel = composed[:top_panel_h]
            panel[:] = (22, 22, 22)
            mode_txt = "precise" if use_precise else "clip"
            cv2.putText(
                panel,
                f"Behavior sequencing ({mode_txt})  t={frame_idx / max(1e-6, fps):.2f}s",
                (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            tool_title = "SingleBehavior Lab"
            (title_w, _), _ = cv2.getTextSize(tool_title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            title_x = max(10, width - title_w - 12)
            cv2.putText(
                panel, tool_title, (title_x, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
            )
            if selected_cls_idx:
                row_top = 28
                row_h = max(12, (top_panel_h - row_top - 8) // max_rows)
                x0, x1 = 130, width - 12
                active_set = {ci for ci, _sc, _lbl in filtered_infos if ci >= 0}
                for ci in selected_cls_idx:
                    is_active = 1 if ci in active_set else 0
                    hist = row_history[ci]
                    hist.append(is_active)
                    if len(hist) > timeline_window_frames:
                        del hist[0 : len(hist) - timeline_window_frames]
                for ri, ci in enumerate(selected_cls_idx):
                    y = row_top + ri * row_h
                    name = widget.classes[ci]
                    pr, pg, pb = palette[ci % len(palette)]
                    bgr = (int(pb), int(pg), int(pr))
                    cv2.putText(
                        panel, name, (10, y + row_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA,
                    )
                    cv2.rectangle(
                        panel, (x0, y + 1), (x1, y + row_h - 2), (55, 55, 55), 1
                    )
                    hist = row_history[ci]
                    if len(hist) >= 1:
                        xs = np.linspace(
                            x0 + 1, x1 - 1, num=len(hist) + 1, dtype=np.int32
                        )
                        run_start = None
                        for hi, hv in enumerate(hist):
                            on = bool(hv)
                            if on and run_start is None:
                                run_start = hi
                            if run_start is not None and (
                                (not on) or hi == len(hist) - 1
                            ):
                                run_end = (
                                    hi if (on and hi == len(hist) - 1) else (hi - 1)
                                )
                                if run_end >= run_start:
                                    xa = int(xs[run_start])
                                    xb = int(max(xa + 1, xs[run_end + 1]))
                                    cv2.rectangle(
                                        panel,
                                        (xa, y + 2),
                                        (xb, y + row_h - 3),
                                        bgr,
                                        -1,
                                    )
                                run_start = None
                        cv2.line(
                            panel,
                            (x1 - 1, y + 1),
                            (x1 - 1, y + row_h - 2),
                            (235, 235, 235),
                            1,
                            cv2.LINE_AA,
                        )

            composed[top_panel_h : top_panel_h + height, :, :] = frame
            _write_overlay_frame(writer_kind, out, composed)
            frame_idx += 1

        cap.release()
        encoder_label = _close_overlay_writer(writer_kind, out, abort=progress.wasCanceled())
        user_canceled = progress.wasCanceled()
        progress.close()

        if user_canceled:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return False

        widget.exported_video_path = output_path
        widget.preview_btn.setEnabled(True)

        mode_txt = "precise boundary" if use_precise else "clip-based"
        range_msg = f" | range: {range_label.strip()}" if is_partial else ""
        widget.log_text.append(
            f"Overlay export complete ({mode_txt}{range_msg}); "
            f"behaviors={sorted(selected_behaviors)}; include_ignore={include_ignore}; "
            f"encoder={encoder_label}"
        )
        show_success_popup = bool(opts.get("show_success_popup", sample_label is None))
        if show_success_popup:
            QMessageBox.information(
                widget,
                "Success",
                f"Video exported to:\n{output_path}\n\nMode: {mode_txt}{range_msg}\n"
                f"Encoder: {encoder_label}\n"
                "Click 'Preview Video with Overlays' to watch it.",
            )
        return encoder_label

    except Exception as e:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if writer_kind is not None and out is not None:
                _close_overlay_writer(writer_kind, out, abort=True)
        except Exception:
            pass
        try:
            if progress is not None:
                progress.close()
        except Exception:
            pass
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        QMessageBox.critical(widget, "Error", f"Failed to export video: {str(e)}")
        return False


class VideoPreviewDialog(QDialog):
    """Video player for exported overlay video. Streams frames on demand to avoid memory crash."""

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.setWindowTitle("Video preview with overlays")
        self.setMinimumSize(900, 700)

        self._cap = None
        self._total_frames = 0
        self._fps = 30.0

        layout = QVBoxLayout(self)

        self.video_label = QLabel("Loading video...")
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label)

        slider_layout = QHBoxLayout()
        self._frame_label = QLabel("0 / 0")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self._slider)
        slider_layout.addWidget(self._frame_label)
        layout.addLayout(slider_layout)

        controls = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play")
        self._playing = False
        self._frame_idx = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._slider_dragging = False

        self.play_pause_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_pause_btn)

        restart_btn = QPushButton("Restart")
        restart_btn.clicked.connect(self._restart)
        controls.addWidget(restart_btn)

        controls.addStretch()
        controls.addWidget(QLabel(f"Video: {os.path.basename(video_path)}"))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._close_and_stop)
        controls.addWidget(close_btn)

        layout.addLayout(controls)

        QTimer.singleShot(100, self._load_and_show_first)

    def _open_video(self):
        if self._cap is not None:
            return True
        try:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                self.video_label.setText("Error: Could not open video")
                return False
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            if self._fps <= 0:
                self._fps = 30.0
            self._timer.setInterval(int(1000 / self._fps))
            self._slider.setMaximum(max(0, self._total_frames - 1))
            return True
        except Exception as e:
            self.video_label.setText(f"Error: {e}")
            return False

    def _read_frame(self, idx):
        if self._cap is None or idx < 0 or idx >= self._total_frames:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        return frame

    def _show_frame(self, frame):
        if frame is None:
            return
        h, w, c = frame.shape
        q_image = QImage(frame.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self._frame_label.setText(f"{self._frame_idx + 1} / {self._total_frames}")

    def _update_frame(self):
        if self._cap is None or self._total_frames == 0:
            return
        frame = self._read_frame(self._frame_idx)
        if frame is not None:
            self._show_frame(frame)
            self._slider.blockSignals(True)
            self._slider.setValue(self._frame_idx)
            self._slider.blockSignals(False)
        self._frame_idx = (self._frame_idx + 1) % self._total_frames

    def _on_slider_changed(self, value):
        if self._cap is None:
            return
        self._frame_idx = value
        frame = self._read_frame(self._frame_idx)
        if frame is not None:
            self._show_frame(frame)

    def _toggle_play(self):
        if self._cap is None:
            if not self._open_video():
                return
            self._update_frame()

        if self._playing:
            self.play_pause_btn.setText("Play")
            self._timer.stop()
            self._playing = False
        else:
            self.play_pause_btn.setText("Pause")
            self._timer.start()
            self._playing = True

    def _restart(self):
        self._frame_idx = 0
        if self._cap is not None:
            self._slider.setValue(0)
            self._update_frame()
        if self._playing:
            self._toggle_play()

    def _load_and_show_first(self):
        if self._open_video():
            self._update_frame()

    def _close_and_stop(self):
        if self._timer.isActive():
            self._timer.stop()
        self.close()

    def closeEvent(self, event):
        if self._timer.isActive():
            self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)


def run_preview_video_with_overlay(widget):
    """Open video player to preview the exported video with overlays."""
    video_path = getattr(widget, "exported_video_path", None)

    if not video_path or not os.path.exists(video_path):
        QMessageBox.warning(
            widget,
            "Error",
            "No exported video found. Please export a video with overlays first.",
        )
        return

    try:
        dialog = VideoPreviewDialog(video_path, widget)
        dialog.exec()
    except Exception as e:
        QMessageBox.warning(
            widget,
            "Video Player Error",
            f"Could not open built-in video player:\n{str(e)}\n\n"
            "Opening with system default player instead.",
        )
        import subprocess
        import platform

        try:
            if platform.system() == "Darwin":
                subprocess.call(("open", video_path))
            elif platform.system() == "Windows":
                os.startfile(video_path)
            else:
                subprocess.call(("xdg-open", video_path))
        except Exception as e2:
            QMessageBox.critical(
                widget,
                "Error",
                f"Could not open video player:\n{str(e2)}\n\nVideo saved at:\n{video_path}",
            )
