from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QGroupBox, QFormLayout, QMessageBox, QListWidget, QListWidgetItem,
    QSpinBox, QComboBox, QTextEdit, QScrollArea, QDialog, QCheckBox, QSizePolicy,
    QProgressBar, QProgressDialog, QDoubleSpinBox, QDialogButtonBox, QApplication,
    QSplitter, QGridLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QMouseEvent
import copy
import cv2
import os
import torch
import numpy as np
import json
from singlebehaviorlab.backend.model import VideoPrismBackbone, BehaviorClassifier
from singlebehaviorlab.backend.video_utils import get_video_info, save_clip
from singlebehaviorlab.backend.data_store import AnnotationManager
from singlebehaviorlab.backend.uncertainty import save_uncertainty_report
from .timeline_themes import TIMELINE_COLOR_THEMES, DEFAULT_THEME, get_palette as get_timeline_palette
from .inference_worker import InferenceWorker, _sanitize_bbox_coords
from .inference_popups import ClipPopupDialog, FrameSegmentPopupDialog




class InferenceWidget(QWidget):
    """Widget for running inference on videos."""

    # Emitted when inference finishes so the Review tab can load results.
    # Payload: (results_dict, classes, is_ovr, clip_length, target_fps)
    review_ready = pyqtSignal(dict, list, bool, int, int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = None
        self.classes = []
        self.attributes = []
        self.label_mapping = None
        self.video_path = None
        self.video_paths = []  # List of selected videos
        self.predictions = []
        self.confidences = []
        self.clip_probabilities = []
        self.clip_frame_probabilities = []  # per-frame probs from FrameClassificationHead
        self.attr_predictions = [] # Store attribute indices
        self.attr_confidences = []
        self.clip_starts = []
        self.localization_bboxes = []
        self.results_cache = {}  # Cache for multi-video results
        self.worker = None
        self.exported_video_path = None
        self.corrected_labels = {}  # Map clip_idx -> corrected_label_index
        self.corrected_attr_labels = {}  # Map clip_idx -> corrected_attr_index
        self.attributes_registry = None # NEW: Store registry
        self.hierarchy_registry = None
        self.aggregated_segments = []  # Frame-level top-1 segments: [{'class': int, 'start': int, 'end': int}, ...]
        self.aggregated_multiclass_segments = []  # OvR frame-level per-class segments
        self._aggregated_frame_scores_norm = None  # np.ndarray [frames, classes]
        self._aggregated_active_mask = None  # np.ndarray [frames, classes], bool
        self._aggregated_last_covered_frame = 0
        self.total_frames = 0  # Total frames in current video
        self.clip_popup_maximized = False  # Track if popup should be maximized
        self.infer_resolution = 288
        self._bbox_ema_alpha = 0.85  # Minimal smoothing: 85% current frame, 15% previous
        self._min_segment_frames = int(self.config.get("inference_min_segment_frames", 1))
        self._merge_gap_frames = int(self.config.get("inference_merge_gap_frames", 0))
        raw_smooth_win = int(self.config.get("inference_temporal_smoothing_window_frames", 1))
        self._temporal_smoothing_window_frames = max(1, raw_smooth_win)
        if self._temporal_smoothing_window_frames % 2 == 0:
            self._temporal_smoothing_window_frames += 1
        self.use_ignore_threshold = bool(self.config.get("inference_use_ignore_threshold", False))
        self.global_ignore_threshold = float(self.config.get("inference_ignore_threshold", 0.60))
        self.class_ignore_thresholds = dict(self.config.get("inference_class_ignore_thresholds", {}))
        self.class_min_segment_frames = {
            str(k): int(v) for k, v in self.config.get("inference_class_min_segment_frames", {}).items()
        }
        self.class_merge_gap_frames = {
            str(k): int(v) for k, v in self.config.get("inference_class_merge_gap_frames", {}).items()
        }
        self.class_smoothing_window_frames = {
            str(k): int(v) for k, v in self.config.get("inference_class_smoothing_window_frames", {}).items()
        }
        self.use_viterbi_decode = bool(self.config.get("inference_use_viterbi_decode", False))
        self.viterbi_switch_penalty = float(self.config.get("inference_viterbi_switch_penalty", 0.35))
        self.ignore_label_name = "Ignored / Unknown"
        self.model_training_config = {}
        self._use_ovr = False
        self._allowed_cooccurrence = set()
        self._ignore_threshold_user_modified = False
        self._applying_auto_threshold = False
        self._setup_ui()
        self._update_viterbi_ui_state()
    
    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()
        
        model_group = QGroupBox("Model")
        model_layout = QFormLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_browse_btn = QPushButton("Load head weights...")
        self.model_browse_btn.clicked.connect(self._load_model)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.model_browse_btn)
        model_layout.addRow("Model:", model_path_layout)
        
        self.classes_label = QLabel("No model loaded")
        model_layout.addRow("Classes:", self.classes_label)
        
        model_group.setLayout(model_layout)
        
        video_group = QGroupBox("Video selection")
        video_layout = QFormLayout()
        
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_browse_btn = QPushButton("Select video(s)...")
        self.video_browse_btn.clicked.connect(self._select_video)
        video_path_layout = QHBoxLayout()
        video_path_layout.addWidget(self.video_path_edit)
        video_path_layout.addWidget(self.video_browse_btn)
        video_layout.addRow("Video:", video_path_layout)
        
        self.video_info_label = QLabel("No video selected")
        video_layout.addRow("Info:", self.video_info_label)
        
        video_group.setLayout(video_layout)
        
        model_video_layout = QVBoxLayout()
        model_video_layout.addWidget(model_group)
        model_video_layout.addWidget(video_group)
        model_video_widget = QWidget()
        model_video_widget.setLayout(model_video_layout)
        
        params_group = QGroupBox("Inference parameters")
        params_layout = QFormLayout()
        
        self.target_fps_spin = QSpinBox()
        self.target_fps_spin.setRange(1, 60)
        self.target_fps_spin.setValue(16)
        params_layout.addRow("Target FPS:", self.target_fps_spin)
        
        self.clip_length_spin = QSpinBox()
        self.clip_length_spin.setRange(1, 64)
        self.clip_length_spin.setValue(16)
        params_layout.addRow("Frames per clip:", self.clip_length_spin)
        
        self.step_frames_spin = QSpinBox()
        self.step_frames_spin.setRange(1, 64)
        self.step_frames_spin.setValue(max(1, self.clip_length_spin.value() // 2))
        self.step_frames_spin.setToolTip(
            "Number of subsampled frames to advance between clips.\n"
            "Defaults to half of 'Frames per clip' (50% overlap) but can be changed.\n"
            "Smaller values mean more overlap and finer temporal resolution."
        )
        self.step_frames_spin.valueChanged.connect(self._on_step_or_clip_changed)
        params_layout.addRow("Step frames:", self.step_frames_spin)

        self.clip_length_spin.valueChanged.connect(self._on_clip_length_changed)

        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(64, 1024)
        self.resolution_spin.setValue(288)
        params_layout.addRow("Resolution:", self.resolution_spin)

        self.override_resolution_check = QCheckBox("Override model resolution")
        self.override_resolution_check.setToolTip("Use the resolution above instead of model metadata")
        params_layout.addRow("", self.override_resolution_check)

        self.collect_attention_check = QCheckBox("Collect attention maps")
        self.collect_attention_check.setToolTip(
            "Record spatial attention weights during inference.\n"
            "Enables exporting a heatmap video showing what the model focuses on.\n"
            "Minimal performance impact."
        )
        params_layout.addRow("", self.collect_attention_check)

        self.sample_inference_check = QCheckBox("Quick-check sampled inference")
        self.sample_inference_check.setToolTip(
            "Run inference only on evenly spread chunks of each video.\n"
            "Useful for checking model behavior on long videos without running full inference."
        )
        self.sample_inference_check.toggled.connect(self._update_sample_inference_controls)
        params_layout.addRow("", self.sample_inference_check)

        self.sample_duration_spin = QSpinBox()
        self.sample_duration_spin.setRange(10, 300)
        self.sample_duration_spin.setValue(60)
        self.sample_duration_spin.setSuffix(" s")
        self.sample_duration_spin.setToolTip("Duration of each sampled chunk per video.")
        params_layout.addRow("Chunk duration:", self.sample_duration_spin)

        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(1, 50)
        self.sample_count_spin.setValue(5)
        self.sample_count_spin.setToolTip("Number of sampled chunks spread evenly across each video.")
        params_layout.addRow("Number of chunks:", self.sample_count_spin)
        self._update_sample_inference_controls(False)

        params_group.setLayout(params_layout)
        
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(model_video_widget)
        top_splitter.addWidget(params_group)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        layout.addWidget(top_splitter)
        
        button_layout = QHBoxLayout()
        self.run_inference_btn = QPushButton("Run inference")
        self.run_inference_btn.clicked.connect(self._run_inference)
        self.run_inference_btn.setEnabled(False)
        button_layout.addWidget(self.run_inference_btn)

        self.stop_inference_btn = QPushButton("Stop inference")
        self.stop_inference_btn.clicked.connect(self._stop_inference)
        self.stop_inference_btn.setEnabled(False)
        self.stop_inference_btn.setToolTip("Stop batch inference; already-completed videos are kept.")
        button_layout.addWidget(self.stop_inference_btn)
        
        self.load_timeline_btn = QPushButton("Load timeline results")
        self.load_timeline_btn.clicked.connect(self._load_timeline_results)
        button_layout.addWidget(self.load_timeline_btn)
        
        self.export_btn = QPushButton("Export video with overlays")
        self.export_btn.clicked.connect(self._export_video_with_overlay)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        self.preview_btn = QPushButton("Preview video with overlays")
        self.preview_btn.clicked.connect(self._preview_video_with_overlay)
        self.preview_btn.setEnabled(False)
        button_layout.addWidget(self.preview_btn)

        self.export_attention_btn = QPushButton("Export attention heatmap")
        self.export_attention_btn.clicked.connect(self._export_attention_heatmap)
        self.export_attention_btn.setEnabled(False)
        self.export_attention_btn.setToolTip("Export video with spatial attention overlay showing what the model focuses on")
        button_layout.addWidget(self.export_attention_btn)
        
        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)
        
        timeline_group = QGroupBox("Timeline")
        timeline_group_layout = QVBoxLayout()
        timeline_controls_layout = QHBoxLayout()
        timeline_controls_layout.setSpacing(8)
        timeline_controls_layout.setContentsMargins(4, 2, 4, 2)
        timeline_label_text = QLabel("Timeline:")
        timeline_controls_layout.addWidget(timeline_label_text)
        
        timeline_controls_layout.addWidget(QLabel("Video:"))
        self.filter_video_combo = QComboBox()
        self.filter_video_combo.currentIndexChanged.connect(self._on_video_selection_changed)
        timeline_controls_layout.addWidget(self.filter_video_combo)
        
        timeline_controls_layout.addStretch()
        
        timeline_controls_layout.addWidget(QLabel("Show behavior:"))
        self.filter_behavior_combo = QComboBox()
        self.filter_behavior_combo.addItem("All Behaviors")
        self.filter_behavior_combo.currentIndexChanged.connect(self._on_filter_changed)
        timeline_controls_layout.addWidget(self.filter_behavior_combo)
        timeline_controls_layout.addWidget(QLabel("Cleanup:"))
        self.cleanup_preset_combo = QComboBox()
        self.cleanup_preset_combo.addItems(["None", "Light", "Standard", "Aggressive", "Custom"])
        self.cleanup_preset_combo.setCurrentText("Custom")
        self.cleanup_preset_combo.setToolTip(
            "Quick post-processing preset. Sets smoothing, gap fill, min segment, and Viterbi.\n"
            "Pick a preset or tweak individual settings (switches to 'Custom' automatically)."
        )
        self.cleanup_preset_combo.currentTextChanged.connect(self._on_cleanup_preset_changed)
        timeline_controls_layout.addWidget(self.cleanup_preset_combo)

        self.use_ignore_threshold_check = QCheckBox("Ignore low-confidence")
        self.use_ignore_threshold_check.setChecked(self.use_ignore_threshold)
        self.use_ignore_threshold_check.stateChanged.connect(self._on_ignore_threshold_changed)
        timeline_controls_layout.addWidget(self.use_ignore_threshold_check)
        self.ignore_threshold_spin = QDoubleSpinBox()
        self.ignore_threshold_spin.setDecimals(2)
        self.ignore_threshold_spin.setRange(0.0, 1.0)
        self.ignore_threshold_spin.setSingleStep(0.05)
        self.ignore_threshold_spin.setValue(self.global_ignore_threshold)
        self.ignore_threshold_spin.setToolTip(
            "Fallback threshold for classes without a per-class threshold.\n"
            "Clips below this confidence are grayed out."
        )
        self.ignore_threshold_spin.valueChanged.connect(self._on_ignore_threshold_changed)
        timeline_controls_layout.addWidget(QLabel("Default τ:"))
        timeline_controls_layout.addWidget(self.ignore_threshold_spin)
        self.per_class_thresh_btn = QPushButton("Per-class τ")
        self.per_class_thresh_btn.clicked.connect(self._open_per_class_thresholds_dialog)
        timeline_controls_layout.addWidget(self.per_class_thresh_btn)

        timeline_controls_layout.addWidget(QLabel("Theme:"))
        self.timeline_theme_combo = QComboBox()
        self.timeline_theme_combo.addItems(list(TIMELINE_COLOR_THEMES.keys()))
        self.timeline_theme_combo.setCurrentText(DEFAULT_THEME)
        self.timeline_theme_combo.setToolTip("Change timeline color theme")
        self.timeline_theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        timeline_controls_layout.addWidget(self.timeline_theme_combo)
        
        self.merge_timeline_check = QCheckBox("Merge consecutive identical behaviors")
        self.merge_timeline_check.setToolTip("Merge consecutive clips with the same predicted behavior")
        self.merge_timeline_check.stateChanged.connect(self._on_merge_changed)
        timeline_controls_layout.addWidget(self.merge_timeline_check)
        
        self.frame_aggregation_check = QCheckBox("Precise frame boundaries")
        self.frame_aggregation_check.setToolTip(
            "Use overlapping clip votes to determine precise behavior boundaries.\n"
            "Best used with step_frames < clip_length (e.g., step=4 for 16-frame clips).\n"
            "Each frame gets votes from all clips that cover it, weighted by confidence."
        )
        self.frame_aggregation_check.stateChanged.connect(self._on_frame_aggregation_changed)
        timeline_controls_layout.addWidget(self.frame_aggregation_check)

        self.use_viterbi_check = QCheckBox("Viterbi decode")
        self.use_viterbi_check.setToolTip(
            "Inference-only sequence decoding on the merged frame probabilities.\n"
            "Single-label models use classic Viterbi over classes.\n"
            "OvR models use binary per-class Viterbi with co-occurrence-aware cleanup."
        )
        self.use_viterbi_check.setChecked(self.use_viterbi_decode)
        self.use_viterbi_check.stateChanged.connect(self._on_viterbi_changed)
        timeline_controls_layout.addWidget(self.use_viterbi_check)

        timeline_controls_layout.addWidget(QLabel("Viterbi switch:"))
        self.viterbi_switch_penalty_spin = QDoubleSpinBox()
        self.viterbi_switch_penalty_spin.setDecimals(2)
        self.viterbi_switch_penalty_spin.setRange(0.0, 5.0)
        self.viterbi_switch_penalty_spin.setSingleStep(0.05)
        self.viterbi_switch_penalty_spin.setValue(self.viterbi_switch_penalty)
        self.viterbi_switch_penalty_spin.setToolTip(
            "Penalty for changing behavior between adjacent frames.\n"
            "Higher values make the decoded sequence more stable."
        )
        self.viterbi_switch_penalty_spin.valueChanged.connect(self._on_viterbi_changed)
        timeline_controls_layout.addWidget(self.viterbi_switch_penalty_spin)

        self.per_class_seg_btn = QPushButton("Per-class seg rules")
        self.per_class_seg_btn.setToolTip(
            "Set smooth window, gap fill, and minimum segment length per behavior class."
        )
        self.per_class_seg_btn.clicked.connect(self._open_per_class_segment_rules_dialog)
        timeline_controls_layout.addWidget(self.per_class_seg_btn)

        self.ovr_rows_check = QCheckBox("Per-class rows")
        self.ovr_rows_check.setChecked(True)
        self.ovr_rows_check.setToolTip(
            "Per-class rows: one row per class. Uncheck for single-row timeline (OvR models only)."
        )
        self.ovr_rows_check.stateChanged.connect(lambda: self._draw_timeline())
        timeline_controls_layout.addWidget(self.ovr_rows_check)

        self.ovr_show_all_check = QCheckBox("Show all classes")
        self.ovr_show_all_check.setChecked(False)
        self.ovr_show_all_check.setToolTip(
            "When enabled, every class above its threshold is shown independently\n"
            "(no mutual exclusivity). When disabled, only the top-1 class and\n"
            "allowed co-occurring classes are shown."
        )
        self.ovr_show_all_check.stateChanged.connect(self._on_ovr_show_all_changed)
        timeline_controls_layout.addWidget(self.ovr_show_all_check)

        timeline_controls_layout.addWidget(QLabel("Zoom (px/s):"))
        self.timeline_zoom_spin = QSpinBox()
        self.timeline_zoom_spin.setRange(10, 2000)
        self.timeline_zoom_spin.setValue(100)
        self.timeline_zoom_spin.setSingleStep(20)
        self.timeline_zoom_spin.setToolTip(
            "Timeline horizontal zoom: pixels per second of video.\n"
            "Increase to spread the timeline out and make short segments clickable.\n"
            "The timeline area scrolls horizontally when zoomed in."
        )
        self.timeline_zoom_spin.valueChanged.connect(self._on_timeline_zoom_changed)
        timeline_controls_layout.addWidget(self.timeline_zoom_spin)

        self.export_timeline_btn = QPushButton("Export CSV/SVG")
        self.export_timeline_btn.setToolTip("Export timeline as CSV (behavior segments) and SVG (visualization) for external analysis")
        self.export_timeline_btn.clicked.connect(self._export_timeline)
        self.export_timeline_btn.setEnabled(False)
        timeline_controls_layout.addWidget(self.export_timeline_btn)
        
        self.save_results_btn = QPushButton("Save results")
        self.save_results_btn.setToolTip("Save inference results (including corrections) for downstream analysis")
        self.save_results_btn.clicked.connect(self._save_results)
        self.save_results_btn.setEnabled(False)
        timeline_controls_layout.addWidget(self.save_results_btn)

        # Put controls in a horizontal scroll area so all settings stay accessible
        # on smaller screens instead of being cramped into one visible row.
        timeline_controls_widget = QWidget()
        timeline_controls_widget.setLayout(timeline_controls_layout)
        timeline_controls_widget.setMinimumHeight(40)

        timeline_controls_scroll = QScrollArea()
        timeline_controls_scroll.setWidget(timeline_controls_widget)
        timeline_controls_scroll.setWidgetResizable(True)
        timeline_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        timeline_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        timeline_controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        timeline_controls_scroll.setMinimumHeight(48)
        timeline_controls_scroll.setMaximumHeight(62)
        
        from singlebehaviorlab.gui.interactive_timeline import InteractiveTimeline
        self._interactive_timeline = InteractiveTimeline()
        self._interactive_timeline.setMinimumHeight(180)
        self._interactive_timeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._interactive_timeline.segment_clicked.connect(self._on_interactive_segment_clicked)
        self._interactive_timeline.frame_clicked.connect(self._on_interactive_frame_clicked)
        self._interactive_timeline.model_changed.connect(self._on_segments_edited)
        self._segments_model = None
        
        timeline_group_layout.addWidget(timeline_controls_scroll)
        timeline_group_layout.addWidget(self._interactive_timeline, 1)
        timeline_preset_row = QHBoxLayout()
        timeline_preset_row.addStretch()
        self.save_timeline_settings_btn = QPushButton("Save timeline settings")
        self.save_timeline_settings_btn.setToolTip(
            "Save ignore filtering and timeline postprocessing/display settings as a reusable preset."
        )
        self.save_timeline_settings_btn.clicked.connect(self._save_timeline_settings_preset)
        timeline_preset_row.addWidget(self.save_timeline_settings_btn)
        self.load_timeline_settings_btn = QPushButton("Load timeline settings")
        self.load_timeline_settings_btn.setToolTip(
            "Load a previously saved timeline settings preset."
        )
        self.load_timeline_settings_btn.clicked.connect(self._load_timeline_settings_preset)
        timeline_preset_row.addWidget(self.load_timeline_settings_btn)
        timeline_group_layout.addLayout(timeline_preset_row)
        timeline_group.setLayout(timeline_group_layout)
        timeline_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(timeline_group, 1)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_list = QListWidget()
        results_layout.addWidget(self.results_list)
        results_group.setLayout(results_layout)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        results_logs_splitter = QSplitter(Qt.Orientation.Horizontal)
        results_logs_splitter.addWidget(results_group)
        results_logs_splitter.addWidget(log_group)
        results_logs_splitter.setStretchFactor(0, 1)
        results_logs_splitter.setStretchFactor(1, 1)
        results_logs_splitter.setMaximumHeight(320)
        layout.addWidget(results_logs_splitter, 0)
        
        self.setLayout(layout)

    def _collect_timeline_settings_payload(self) -> dict:
        return {
            "frame_aggregation_enabled": bool(self.frame_aggregation_check.isChecked()),
            "merge_consecutive_enabled": bool(self.merge_timeline_check.isChecked()),
            "use_viterbi_decode": bool(self.use_viterbi_decode),
            "viterbi_switch_penalty": float(self.viterbi_switch_penalty),
            "use_ignore_threshold": bool(self.use_ignore_threshold),
            "ignore_threshold": float(self.global_ignore_threshold),
            "class_ignore_thresholds": {
                cls: float(t) for cls, t in self.class_ignore_thresholds.items()
            },
            "timeline_zoom": int(self.timeline_zoom_spin.value()),
            "ovr_rows": bool(getattr(self, "ovr_rows_check", None) is not None and self.ovr_rows_check.isChecked()),
            "ovr_show_all": bool(getattr(self, "ovr_show_all_check", None) is not None and self.ovr_show_all_check.isChecked()),
            "min_segment_frames": int(self._min_segment_frames),
            "merge_gap_frames": int(self._merge_gap_frames),
            "temporal_smoothing_window_frames": int(self._temporal_smoothing_window_frames),
            "class_min_segment_frames": {
                cls: int(v) for cls, v in self.class_min_segment_frames.items()
            },
            "class_merge_gap_frames": {
                cls: int(v) for cls, v in self.class_merge_gap_frames.items()
            },
            "class_smoothing_window_frames": {
                cls: int(v) for cls, v in self.class_smoothing_window_frames.items()
            },
        }

    def _apply_timeline_settings_payload(self, payload: dict):
        if not isinstance(payload, dict):
            return
        self.use_ignore_threshold = bool(payload.get("use_ignore_threshold", self.use_ignore_threshold))
        self.use_ignore_threshold_check.setChecked(self.use_ignore_threshold)
        if "ignore_threshold" in payload:
            self.global_ignore_threshold = float(payload["ignore_threshold"])
            self.ignore_threshold_spin.setValue(self.global_ignore_threshold)
        if "class_ignore_thresholds" in payload and isinstance(payload["class_ignore_thresholds"], dict):
            self.class_ignore_thresholds = {
                str(cls): float(t) for cls, t in payload["class_ignore_thresholds"].items()
            }
        if "timeline_zoom" in payload:
            self.timeline_zoom_spin.setValue(int(payload["timeline_zoom"]))
        if "frame_aggregation_enabled" in payload:
            self.frame_aggregation_check.setChecked(bool(payload["frame_aggregation_enabled"]))
        if "merge_consecutive_enabled" in payload:
            self.merge_timeline_check.setChecked(bool(payload["merge_consecutive_enabled"]))
        if "use_viterbi_decode" in payload:
            self.use_viterbi_decode = bool(payload["use_viterbi_decode"])
            self.use_viterbi_check.setChecked(self.use_viterbi_decode)
        if "viterbi_switch_penalty" in payload:
            self.viterbi_switch_penalty = float(payload["viterbi_switch_penalty"])
            self.viterbi_switch_penalty_spin.setValue(self.viterbi_switch_penalty)
        if getattr(self, "ovr_rows_check", None) is not None and "ovr_rows" in payload:
            self.ovr_rows_check.setChecked(bool(payload["ovr_rows"]))
        if getattr(self, "ovr_show_all_check", None) is not None and "ovr_show_all" in payload:
            self.ovr_show_all_check.setChecked(bool(payload["ovr_show_all"]))
        if "min_segment_frames" in payload:
            self._min_segment_frames = int(payload["min_segment_frames"])
        if "merge_gap_frames" in payload:
            self._merge_gap_frames = int(payload["merge_gap_frames"])
        if "temporal_smoothing_window_frames" in payload:
            smooth = int(payload["temporal_smoothing_window_frames"])
            if smooth % 2 == 0:
                smooth += 1
            self._temporal_smoothing_window_frames = max(1, smooth)
        if "class_min_segment_frames" in payload and isinstance(payload["class_min_segment_frames"], dict):
            self.class_min_segment_frames = {
                str(cls): int(v) for cls, v in payload["class_min_segment_frames"].items()
            }
        if "class_merge_gap_frames" in payload and isinstance(payload["class_merge_gap_frames"], dict):
            self.class_merge_gap_frames = {
                str(cls): int(v) for cls, v in payload["class_merge_gap_frames"].items()
            }
        if "class_smoothing_window_frames" in payload and isinstance(payload["class_smoothing_window_frames"], dict):
            self.class_smoothing_window_frames = {
                str(cls): int(v) for cls, v in payload["class_smoothing_window_frames"].items()
            }
        self.config["inference_ignore_threshold"] = self.global_ignore_threshold
        self.config["inference_class_ignore_thresholds"] = dict(self.class_ignore_thresholds)
        self.config["inference_use_viterbi_decode"] = self.use_viterbi_decode
        self.config["inference_viterbi_switch_penalty"] = self.viterbi_switch_penalty
        self._sync_per_class_segment_rule_config()
        if self.predictions:
            if self.frame_aggregation_check.isChecked():
                self._compute_aggregated_timeline()
            self._display_results()

    def _save_timeline_settings_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Timeline Settings",
            self.config.get("experiment_path", os.getcwd()),
            "JSON Files (*.json)"
        )
        if not path:
            return
        payload = {
            "classes": list(self.classes),
            "timeline_settings": self._collect_timeline_settings_payload(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.log_text.append(f"Timeline settings saved: {path}")

    def _load_timeline_settings_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Timeline Settings",
            self.config.get("experiment_path", os.getcwd()),
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            payload = data.get("timeline_settings", data)
            self._apply_timeline_settings_payload(payload)
            self.log_text.append(f"Timeline settings loaded: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Failed to load timeline settings:\n{e}")
    
    def _load_model(self):
        """Load trained model head."""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model Head",
            self.config.get("models_dir", "models/behavior_heads"),
            "PyTorch Files (*.pt);;All Files (*)"
        )
        
        if not model_path:
            return
        
        try:
            self.log_text.append("Loading model...")
            
            pre_state_dict = None
            inferred_embed_dim = None
            inferred_backbone = None
            if os.path.exists(model_path):
                try:
                    pre_state_dict = torch.load(model_path, map_location='cpu')
                    for key in (
                        "head_root.query",
                        "head_class.query",
                        "head_root.ln1.weight",
                        "head_class.ln1.weight",
                        "head_root.fc.weight",
                        "head_class.fc.weight",
                        "query",
                        "ln1.weight",
                        "fc.weight",
                    ):
                        if key in pre_state_dict:
                            tensor = pre_state_dict[key]
                            if tensor is None:
                                continue
                            if tensor.ndim == 3:
                                inferred_embed_dim = tensor.shape[-1]
                            elif tensor.ndim == 2:
                                inferred_embed_dim = tensor.shape[1]
                            elif tensor.ndim == 1:
                                inferred_embed_dim = tensor.shape[0]
                            if inferred_embed_dim:
                                break
                    embed_to_backbone = {
                        768: "videoprism_public_v1_base",
                        1024: "videoprism_public_v1_large",
                    }
                    inferred_backbone = embed_to_backbone.get(inferred_embed_dim)
                    if inferred_backbone:
                        self.log_text.append(
                            f"Inferred backbone from weights: {inferred_backbone} (embed_dim={inferred_embed_dim})"
                        )
                except Exception:
                    pre_state_dict = None
            # Attempt to load metadata (preferred)
            meta_classes = None
            meta_path = model_path + ".meta.json"
            metadata_backbone = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as meta_file:
                        meta_data = json.load(meta_file)
                    meta_classes = meta_data.get("classes")
                    self.attributes = meta_data.get("attributes", []) # Load attributes
                    self.attributes_registry = meta_data.get("attributes_registry", None)
                    self.hierarchy_registry = meta_data.get("hierarchy_registry", None)
                    self.label_mapping = meta_data.get("label_mapping", None)
                    if meta_classes:
                        self.log_text.append(f"Loaded {len(meta_classes)} classes from metadata: {meta_classes}")
                    if self.attributes:
                        self.log_text.append(f"Loaded {len(self.attributes)} attributes: {self.attributes}")
                    if self.hierarchy_registry:
                        self.log_text.append("Loaded deep hierarchy registry")
                        if not self.attributes:
                            leaf_nodes = [
                                n for n, children in self.hierarchy_registry.items()
                                if not children and n != "__root__"
                            ]
                            self.attributes = sorted(set(leaf_nodes))
                            if self.attributes:
                                self.log_text.append(f"Derived {len(self.attributes)} leaf attributes from hierarchy")
                    if self.attributes_registry:
                        self.log_text.append(f"Loaded hierarchical registry for {len(self.attributes_registry)} classes")
                    elif self.attributes and meta_classes:
                        # Fallback: Reconstruct registry from attribute names if missing (for models trained before registry was saved)
                        self.log_text.append("Registry missing from metadata. Attempting to reconstruct from attribute names...")
                        reconstructed = {}
                        for cls in meta_classes:
                            reconstructed[cls] = []
                        
                        for attr in self.attributes:
                            # Match attribute to class based on prefix (e.g. "Walk_Start" -> "Walk")
                            # We look for the longest matching class name prefix
                            best_match = None
                            for cls in meta_classes:
                                if attr.startswith(cls + "_") or attr == cls:
                                    if best_match is None or len(cls) > len(best_match):
                                        best_match = cls
                            
                            if best_match:
                                reconstructed[best_match].append(attr)
                        
                        # Only use if we found matches
                        if any(reconstructed.values()):
                            for cls in reconstructed:
                                reconstructed[cls].sort()
                            self.attributes_registry = reconstructed
                            self.log_text.append(f"Reconstructed registry: { {k: len(v) for k, v in reconstructed.items()} }")
                    elif self.label_mapping and meta_classes:
                        if any("path" in v for v in self.label_mapping.values()):
                            # Fallback: Build hierarchy from label_mapping paths
                            self.log_text.append("Hierarchy missing from metadata. Reconstructing from label_mapping...")
                            reconstructed = {"__root__": []}
                            for _, info in self.label_mapping.items():
                                path = info.get("path", [])
                                if not path:
                                    continue
                                if path[0] not in reconstructed["__root__"]:
                                    reconstructed["__root__"].append(path[0])
                                for i in range(1, len(path)):
                                    parent = path[i - 1]
                                    child = path[i]
                                    reconstructed.setdefault(parent, [])
                                    if child not in reconstructed[parent]:
                                        reconstructed[parent].append(child)
                                reconstructed.setdefault(path[-1], [])
                            for node in reconstructed:
                                reconstructed[node] = sorted(set(reconstructed[node]))
                            self.hierarchy_registry = reconstructed
                            leaf_nodes = [
                                n for n, children in reconstructed.items()
                                if not children and n != "__root__"
                            ]
                            self.attributes = sorted(set(leaf_nodes))
                            self.log_text.append("Reconstructed deep hierarchy registry from label_mapping")
                        else:
                            # Fallback: Build registry directly from label_mapping
                            self.log_text.append("Registry missing from metadata. Reconstructing from label_mapping...")
                            reconstructed = {cls: [] for cls in meta_classes}
                            for raw_label, info in self.label_mapping.items():
                                cls_name = info.get("class")
                                attr_name = info.get("attribute")
                                if cls_name in reconstructed and attr_name:
                                    reconstructed[cls_name].append(attr_name)
                            if any(reconstructed.values()):
                                for cls in reconstructed:
                                    reconstructed[cls] = sorted(set(reconstructed[cls]))
                                self.attributes_registry = reconstructed
                                self.attributes = sorted({a for attrs in reconstructed.values() for a in attrs})
                                self.log_text.append(f"Reconstructed registry from label_mapping: { {k: len(v) for k, v in reconstructed.items()} }")
                    
                    # Load clip length from metadata if available
                    clip_len = meta_data.get("clip_length")
                    if clip_len:
                        self.clip_length_spin.setValue(int(clip_len))
                        self.log_text.append(f"Automatically set 'Frames per clip' to {clip_len} from model metadata.")

                    meta_fps = meta_data.get("target_fps") or meta_data.get("training_config", {}).get("target_fps")
                    if meta_fps:
                        self.target_fps_spin.setValue(int(meta_fps))
                        self.log_text.append(f"Automatically set 'Target FPS' to {meta_fps} from model metadata.")
                    
                    resolution = meta_data.get("resolution") or meta_data.get("training_config", {}).get("resolution")
                    if resolution:
                        self.infer_resolution = int(resolution)
                        self.resolution_spin.setValue(self.infer_resolution)
                        self.log_text.append(f"Using inference resolution {self.infer_resolution}x{self.infer_resolution} from model metadata.")
                    
                    if not resolution:
                        config_path = os.path.splitext(model_path)[0] + "_training_config.json"
                        if os.path.exists(config_path):
                            try:
                                with open(config_path, "r", encoding="utf-8") as cfg_file:
                                    cfg_data = json.load(cfg_file)
                                if isinstance(cfg_data, dict):
                                    cfg_train = cfg_data.get("training_config", cfg_data)
                                    if isinstance(cfg_train, dict):
                                        self.model_training_config = dict(cfg_train)
                                cfg_resolution = cfg_data.get("resolution") or cfg_data.get("training_config", {}).get("resolution")
                                if cfg_resolution:
                                    self.infer_resolution = int(cfg_resolution)
                                    self.resolution_spin.setValue(self.infer_resolution)
                                    self.log_text.append(f"Using inference resolution {self.infer_resolution}x{self.infer_resolution} from training config.")
                            except Exception as cfg_err:
                                self.log_text.append(f"Warning: Failed to read training config {config_path}: {cfg_err}")
                    
                    # Load localization crop parameters from training metadata
                    train_cfg = meta_data.get("training_config", {})
                    self.model_training_config = dict(train_cfg) if isinstance(train_cfg, dict) else {}
                    self._crop_padding = float(train_cfg.get("classification_crop_padding", 0.35) or 0.35)
                    self._crop_min_size = float(train_cfg.get("classification_crop_min_size_norm", 0.04) or 0.04)

                    # OvR: use sigmoid scoring instead of softmax at inference
                    self._use_ovr = bool(train_cfg.get("use_ovr", False))
                    self._update_viterbi_ui_state()
                    self._allowed_cooccurrence = set()
                    self._ovr_temperatures = {}
                    if self._use_ovr:
                        if getattr(self, "ovr_rows_check", None) is not None:
                            self.ovr_rows_check.setChecked(True)
                        self.log_text.append("OvR model detected: using sigmoid scoring at inference.")
                        for pair in train_cfg.get("allowed_cooccurrence", []):
                            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                self._allowed_cooccurrence.add((pair[0], pair[1]))
                                self._allowed_cooccurrence.add((pair[1], pair[0]))
                        if self._allowed_cooccurrence:
                            unique = train_cfg.get("allowed_cooccurrence", [])
                            pairs_str = ", ".join(f"{a}+{b}" for a, b in unique)
                            self.log_text.append(f"Allowed co-occurrence pairs: {pairs_str}")

                        # Per-head calibrated temperatures from training
                        ovr_temps = meta_data.get("ovr_temperatures", {})
                        if ovr_temps:
                            self._ovr_temperatures = {k: float(v) for k, v in ovr_temps.items()}
                            t_str = ", ".join(f"{k}={v}" for k, v in self._ovr_temperatures.items())
                            self.log_text.append(f"OvR calibrated temperatures: {t_str}")

                    # Validate backbone model
                    head_backbone = meta_data.get("backbone_model")
                    current_backbone = self.config.get("backbone_model", "videoprism_public_v1_base")
                    
                    # Prefer inferred backbone if metadata disagrees with weights
                    if inferred_backbone and head_backbone and head_backbone != inferred_backbone:
                        self.log_text.append(
                            f"WARNING: Metadata backbone '{head_backbone}' does not match weights. "
                            f"Using inferred backbone '{inferred_backbone}'."
                        )
                        head_backbone = inferred_backbone
                    
                    if head_backbone and head_backbone != current_backbone:
                        msg = (
                            f"Backbone Model Mismatch (auto-corrected for inference).\n"
                            f"Head backbone: '{head_backbone}', current config: '{current_backbone}'.\n"
                            f"Using '{head_backbone}' for inference."
                        )
                        self.log_text.append(f"WARNING: {msg}")
                    
                    metadata_backbone = head_backbone
                        
                except Exception as meta_err:
                    self.log_text.append(f"Warning: Failed to read metadata file {meta_path}: {meta_err}")
            
            if meta_classes:
                self.classes = meta_classes
            else:
                annotation_file = self.config.get("annotation_file", "data/annotations/annotations.json")
                if os.path.exists(annotation_file):
                    annotation_manager = AnnotationManager(annotation_file)
                    self.classes = annotation_manager.get_classes()
                else:
                    QMessageBox.warning(self, "Warning", "Annotation file not found. Cannot determine classes.")
                    return
            
            if not self.classes:
                QMessageBox.warning(self, "Error", "No classes found for this model.")
                return
            
            self.log_text.append(f"Using {len(self.classes)} classes: {self.classes}")
            
            resolved_backbone = inferred_backbone or metadata_backbone
            if not resolved_backbone:
                resolved_backbone = self.config.get("backbone_model", "videoprism_public_v1_base")
            model_name = resolved_backbone
            self.log_text.append(f"Loading VideoPrism backbone ({model_name}) at resolution {self.infer_resolution}×{self.infer_resolution}...")
            
            backbone = VideoPrismBackbone(model_name=model_name, resolution=self.infer_resolution)
            
            head_kwargs = {"num_heads": 4}
            dropout = 0.1
            num_attributes = 0
            use_localization = False
            use_frame_head = True
            frame_head_temporal_layers = 1
            temporal_pool_frames = 1
            num_stages = 3
            proj_dim = 256
            localization_hidden_dim = 256
            multi_scale = False
            use_temporal_decoder = True
            
            try:
                if os.path.exists(meta_path):
                    num_attributes = meta_data.get("num_attributes", 0)
                    head_cfg = meta_data.get("head", {}) if 'meta_data' in locals() and isinstance(meta_data, dict) else {}
                    if isinstance(head_cfg, dict):
                        if "num_heads" in head_cfg:
                            head_kwargs["num_heads"] = head_cfg.get("num_heads", head_kwargs["num_heads"])
                        if "dropout" in head_cfg:
                            dropout = head_cfg.get("dropout", dropout)
                        use_localization = bool(
                            head_cfg.get("use_localization", False)
                            or meta_data.get("training_config", {}).get("use_localization", False)
                        )
                        localization_hidden_dim = int(head_cfg.get("localization_hidden_dim", 256))
                        use_temporal_decoder = bool(
                            head_cfg.get(
                                "use_temporal_decoder",
                                meta_data.get("training_config", {}).get("use_temporal_decoder", True),
                            )
                        )
                        frame_head_temporal_layers = int(head_cfg.get("frame_head_temporal_layers", 1))
                        temporal_pool_frames = int(head_cfg.get("temporal_pool_frames", 1))
                        num_stages = int(head_cfg.get("num_stages", 3))
                        proj_dim = int(head_cfg.get("proj_dim", 256))
                        multi_scale = bool(head_cfg.get("multi_scale", False))
                        if frame_head_temporal_layers <= 0:
                            frame_head_temporal_layers = int(
                                meta_data.get("training_config", {}).get("frame_head_temporal_layers", 1)
                            )
                        if temporal_pool_frames <= 0:
                            temporal_pool_frames = int(
                                meta_data.get("training_config", {}).get("temporal_pool_frames", 1)
                            )
                        if num_stages <= 0:
                            num_stages = int(
                                meta_data.get("training_config", {}).get("num_stages", 3)
                            )
            except Exception:
                pass
            
            if frame_head_temporal_layers <= 0:
                frame_head_temporal_layers = 1
            if use_temporal_decoder:
                self.log_text.append(f"Frame head: temporal decoder on, layers={frame_head_temporal_layers}, proj_dim={proj_dim}")
            else:
                self.log_text.append(f"Frame head: direct per-frame classifier, proj_dim={proj_dim}")
            
            # Peek at checkpoint to detect multi_scale (most reliable source)
            try:
                _ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(_ckpt, dict) and "multi_scale" in _ckpt:
                    multi_scale = bool(_ckpt["multi_scale"])
                if isinstance(_ckpt, dict) and "use_temporal_decoder" in _ckpt:
                    use_temporal_decoder = bool(_ckpt["use_temporal_decoder"])
                del _ckpt
            except Exception:
                pass

            self.log_text.append(f"Creating classifier (kwargs={head_kwargs}, multi_scale={multi_scale})...")
            self.model = BehaviorClassifier(
                backbone,
                num_classes=len(self.classes),
                class_names=self.classes,
                dropout=dropout,
                freeze_backbone=True,
                head_kwargs=head_kwargs,
                use_localization=use_localization,
                localization_hidden_dim=localization_hidden_dim,
                use_frame_head=use_frame_head,
                use_temporal_decoder=use_temporal_decoder,
                frame_head_temporal_layers=frame_head_temporal_layers,
                temporal_pool_frames=temporal_pool_frames,
                proj_dim=proj_dim,
                num_stages=num_stages,
                multi_scale=multi_scale,
            )
            
            self.log_text.append(f"Loading head weights from {model_path}...")
            if getattr(self, "_use_ovr", False) and getattr(self.model, "frame_head", None) is not None:
                self.model.frame_head.use_ovr = True
            try:
                self.model.load_head(model_path)
            except RuntimeError as e:
                err_str = str(e)
                if "size mismatch" in err_str:
                    msg = (
                        f"Model architecture mismatch.\n"
                        f"Error Details: {err_str}\n\n"
                        "Please ensure the weights file matches the current architecture."
                    )
                    self.log_text.append(f"ERROR: {msg}")
                    QMessageBox.critical(self, "Model Load Error", msg)
                    self.model = None
                    return
                raise
            
            if multi_scale:
                self.log_text.append(
                    "Multi-scale mode active: inference runs backbone twice per clip "
                    "(full fps + half fps). Expect ~2x backbone time per clip."
                )
            self.model_path_edit.setText(model_path)
            self.classes_label.setText(", ".join(self.classes))
            
            # Populate filter combo
            self.filter_behavior_combo.blockSignals(True)
            self.filter_behavior_combo.clear()
            self.filter_behavior_combo.addItem("All Behaviors")
            self.filter_behavior_combo.addItem(self.ignore_label_name)
            self.filter_behavior_combo.addItems(self.classes)
            self.filter_behavior_combo.blockSignals(False)
            
            self.log_text.append("Model loaded successfully!")
            QMessageBox.information(self, "Success", "Model loaded successfully!")

            # Offer autosave recovery if a crash interrupted a previous batch run
            self._check_autosave_recovery(model_path)

            if self.video_path:
                self.run_inference_btn.setEnabled(True)
        
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.log_text.append(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
    
    def _select_video(self):
        """Select video file(s) for inference."""
        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video File(s)",
            self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos")),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if video_paths:
            # Ensure videos are in experiment folder (batch operation)
            from .video_utils import ensure_videos_in_experiment
            self.video_paths = ensure_videos_in_experiment(video_paths, self.config, self)
            if len(video_paths) == 1:
                self.video_path = video_paths[0]
                self.video_path_edit.setText(self.video_path)
                
                info = get_video_info(self.video_path)
                if info:
                    info_text = f"FPS: {info['fps']:.2f}, Frames: {info['frame_count']}, Size: {info['width']}x{info['height']}"
                    self.video_info_label.setText(info_text)
            else:
                self.video_path = video_paths[0] # Set first as default for preview
                self.video_path_edit.setText(f"{len(video_paths)} videos selected")
                self.video_info_label.setText(f"Selected {len(video_paths)} videos")
            
            if self.model:
                self.run_inference_btn.setEnabled(True)

    def _update_sample_inference_controls(self, enabled: bool):
        self.sample_duration_spin.setEnabled(bool(enabled))
        self.sample_count_spin.setEnabled(bool(enabled))

    def _compute_sample_ranges_for_video(self, video_path: str):
        """Build evenly spread sample ranges [(start_frame, end_frame), ...] for one video."""
        if not video_path:
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            return []

        dur_frames = int(round(self.sample_duration_spin.value() * fps))
        dur_frames = max(1, min(dur_frames, total_frames))
        n = max(1, int(self.sample_count_spin.value()))
        usable = max(0, total_frames - dur_frames)
        if n == 1 or usable == 0:
            starts = [usable // 2]
        else:
            starts = [int(round(i * usable / (n - 1))) for i in range(n)]
        return [(start, min(start + dur_frames, total_frames)) for start in starts]
    
    def _run_inference(self):
        """Run inference on selected video."""
        if not self.model:
            QMessageBox.warning(self, "Error", "Please load a model first.")
            return
        
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video first.")
            return
        
        self.run_inference_btn.setEnabled(False)
        self.load_timeline_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Running inference...")
        self.results_list.clear()
        self.log_text.clear()
        self.log_text.append("Starting inference...")
        if self.override_resolution_check.isChecked():
            self.infer_resolution = int(self.resolution_spin.value())
            self.log_text.append(f"Inference resolution (override): {self.infer_resolution}x{self.infer_resolution}")
        else:
            self.log_text.append(f"Inference resolution: {self.infer_resolution}x{self.infer_resolution}")
        
        video_paths = self.video_paths if self.video_paths else ([self.video_path] if self.video_path else [])
        sample_ranges_by_video = {}
        if self.sample_inference_check.isChecked():
            for v_path in video_paths:
                sample_ranges = self._compute_sample_ranges_for_video(v_path)
                if sample_ranges:
                    sample_ranges_by_video[v_path] = sample_ranges
                    cap = cv2.VideoCapture(v_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if fps <= 0:
                        fps = 30.0
                    summary = ", ".join(
                        f"{start / fps:.1f}s-{end / fps:.1f}s" for start, end in sample_ranges[:6]
                    )
                    if len(sample_ranges) > 6:
                        summary += ", ..."
                    self.log_text.append(
                        f"Quick-check sampled inference for {os.path.basename(v_path)}: {summary}"
                    )
                else:
                    self.log_text.append(
                        f"Warning: could not build sample ranges for {os.path.basename(v_path)}. "
                        "Falling back to full video."
                    )
        
        model_path = self.model_path_edit.text()
        
        self.worker = InferenceWorker(
            self.model,
            video_paths,
            self.target_fps_spin.value(),
            self.clip_length_spin.value(),
            self.step_frames_spin.value(),
            resolution=self.infer_resolution,
            classes=self.classes,
            use_localization_pipeline=getattr(self.model, "use_localization", False),
            crop_padding=getattr(self, "_crop_padding", 0.35),
            crop_min_size=getattr(self, "_crop_min_size", 0.04),
            use_ovr=getattr(self, "_use_ovr", False),
            ovr_temperatures=getattr(self, "_ovr_temperatures", {}),
            collect_attention=self.collect_attention_check.isChecked(),
            sample_ranges_by_video=sample_ranges_by_video,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log_message.connect(self._on_log)
        self.worker.video_done.connect(self._on_video_done)
        self.run_inference_btn.setEnabled(False)
        self.stop_inference_btn.setEnabled(True)
        self.worker.start()
    
    def _on_progress(self, current: int, total: int):
        """Update progress (capped at 100%)."""
        if total > 0:
            progress = min(100, int(100 * current / total))
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"Processing: {current}/{total} items ({progress}%)")
    
    def _on_finished(self, results: dict):
        """Handle inference completion."""
        self.results_cache = results
        self._ignore_threshold_user_modified = False
        
        # Initialize corrections storage for each video if not present
        for v_path in self.results_cache:
            if "corrected_labels" not in self.results_cache[v_path]:
                self.results_cache[v_path]["corrected_labels"] = {}
            if "corrected_attr_labels" not in self.results_cache[v_path]:
                self.results_cache[v_path]["corrected_attr_labels"] = {}
        
        # Populate video dropdown
        self.filter_video_combo.blockSignals(True)
        self.filter_video_combo.clear()
        for path in sorted(results.keys()):
            self.filter_video_combo.addItem(os.path.basename(path), path)
        self.filter_video_combo.blockSignals(False)
        
        # Select first video
        if results:
            has_any_frame_probs = any(
                isinstance(res.get("clip_frame_probabilities"), list) and len(res.get("clip_frame_probabilities", [])) > 0
                for res in results.values()
            )
            if has_any_frame_probs and not self.frame_aggregation_check.isChecked():
                # Automatically switch to precise frame-boundary mode when
                # frame-head outputs are available.
                self.frame_aggregation_check.setChecked(True)
                self.log_text.append("Detected frame-head predictions. Enabled 'Precise frame boundaries' mode.")
            first_video = sorted(results.keys())[0]
            idx = self.filter_video_combo.findData(first_video)
            if idx >= 0:
                self.filter_video_combo.setCurrentIndex(idx)
                self._on_video_selection_changed(idx)
        
        total_clips = sum(len(res["predictions"]) for res in results.values())
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        if results:
            self.progress_label.setText(f"Inference complete! Processed {len(results)} videos, {total_clips} clips")
        else:
            self.progress_label.setText("Stopped. No videos were completed.")
        self.run_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(False)
        self.load_timeline_btn.setEnabled(True)
        
        self.export_btn.setEnabled(bool(results))
        self.export_timeline_btn.setEnabled(bool(results))
        self.save_results_btn.setEnabled(bool(results))
        has_attn = any(
            "clip_attention_maps" in res for res in results.values()
        ) if results else False
        self.export_attention_btn.setEnabled(has_attn)
        
        if results:
            self._save_results(silent=True)
            autosave_path = self._autosave_path()
            if autosave_path and os.path.exists(autosave_path):
                try:
                    os.remove(autosave_path)
                except Exception:
                    pass

            # Save uncertainty report and notify the Review tab.
            try:
                model_path = self.model_path_edit.text()
                if model_path:
                    uncertainty_path = os.path.join(
                        os.path.dirname(model_path),
                        os.path.splitext(os.path.basename(model_path))[0] + "_uncertainty.json",
                    )
                else:
                    exp_path = self.config.get("experiment_path", "")
                    uncertainty_path = os.path.join(exp_path, "results", "_uncertainty.json")
                save_uncertainty_report(
                    results=results,
                    classes=self.classes,
                    output_path=uncertainty_path,
                    is_ovr=bool(self._use_ovr),
                    n_per_class=25,
                    clip_length=self.clip_length_spin.value(),
                    target_fps=self.target_fps_spin.value(),
                )
                self.log_text.append(f"Uncertainty report saved to: {uncertainty_path}")
            except Exception as _ue:
                self.log_text.append(f"Warning: could not save uncertainty report: {_ue}")

            self.review_ready.emit(
                results,
                list(self.classes),
                bool(self._use_ovr),
                self.clip_length_spin.value(),
                self.target_fps_spin.value(),
            )

            QMessageBox.information(self, "Success", f"Inference complete! Processed {len(results)} videos.")

    def _autosave_path(self) -> str:
        """Return the path for the incremental autosave file next to the loaded model."""
        model_path = self.model_path_edit.text()
        if not model_path:
            return ""
        return os.path.join(os.path.dirname(model_path), "_inference_autosave.json")

    def _on_video_done(self, video_path: str, res_entry: dict):
        """Called after each video finishes — accumulate and autosave incrementally."""
        # Initialise corrections fields
        res_entry.setdefault("corrected_labels", {})
        res_entry.setdefault("corrected_attr_labels", {})
        self.results_cache[video_path] = res_entry
        self.export_btn.setEnabled(True)
        self.export_timeline_btn.setEnabled(True)
        self.save_results_btn.setEnabled(True)
        if "clip_attention_maps" in res_entry:
            self.export_attention_btn.setEnabled(True)

        self.filter_video_combo.blockSignals(True)
        existing_idx = self.filter_video_combo.findData(video_path)
        if existing_idx < 0:
            self.filter_video_combo.addItem(os.path.basename(video_path), video_path)
            model = self.filter_video_combo.model()
            if model is not None:
                self.filter_video_combo.model().sort(0)
        self.filter_video_combo.blockSignals(False)

        current_video = self.filter_video_combo.currentData()
        if not current_video:
            idx = self.filter_video_combo.findData(video_path)
            if idx >= 0:
                self.filter_video_combo.setCurrentIndex(idx)
                self._on_video_selection_changed(idx)
        elif current_video == video_path:
            idx = self.filter_video_combo.currentIndex()
            self._on_video_selection_changed(idx)

        finished_count = len(self.results_cache)
        clip_count = len(res_entry.get("predictions", []))
        self.log_text.append(
            f"Finished {os.path.basename(video_path)}: {clip_count} clips ready to review "
            f"({finished_count} video{'s' if finished_count != 1 else ''} available so far)."
        )

        autosave_path = self._autosave_path()
        if not autosave_path:
            return
        try:
            save_data = {
                "classes": self.classes,
                "parameters": {
                    "target_fps": self.target_fps_spin.value(),
                    "clip_length": self.clip_length_spin.value(),
                    "step_frames": self.step_frames_spin.value(),
                    "sample_inference_enabled": bool(self.sample_inference_check.isChecked()),
                    "sample_duration_seconds": int(self.sample_duration_spin.value()),
                    "sample_num_chunks": int(self.sample_count_spin.value()),
                    "frame_aggregation_enabled": self.frame_aggregation_check.isChecked(),
                    "merge_consecutive_enabled": bool(self.merge_timeline_check.isChecked()),
                    "use_viterbi_decode": bool(self.use_viterbi_decode),
                    "viterbi_switch_penalty": float(self.viterbi_switch_penalty),
                    "min_segment_frames": int(self._min_segment_frames),
                    "merge_gap_frames": int(self._merge_gap_frames),
                    "temporal_smoothing_window_frames": int(self._temporal_smoothing_window_frames),
                    "class_min_segment_frames": {
                        cls: int(v) for cls, v in self.class_min_segment_frames.items()
                    },
                    "class_merge_gap_frames": {
                        cls: int(v) for cls, v in self.class_merge_gap_frames.items()
                    },
                    "class_smoothing_window_frames": {
                        cls: int(v) for cls, v in self.class_smoothing_window_frames.items()
                    },
                    "use_ovr": bool(self._use_ovr),
                    "ovr_rows": bool(getattr(self, "ovr_rows_check", None) is not None and self.ovr_rows_check.isChecked()),
                    "ovr_show_all": bool(getattr(self, "ovr_show_all_check", None) is not None and self.ovr_show_all_check.isChecked()),
                    "use_ignore_threshold": bool(self.use_ignore_threshold),
                    "ignore_threshold": float(self.global_ignore_threshold),
                    "class_ignore_thresholds": {
                        cls: float(t) for cls, t in self.class_ignore_thresholds.items()
                    },
                    "timeline_zoom": int(self.timeline_zoom_spin.value()),
                },
                "results": self.results_cache,
                "_autosave": True,
            }
            self._write_results_bundle(autosave_path, save_data, pretty=False)
            self.log_text.append(
                f"  [Autosave] {len(self.results_cache)} video(s) saved → {os.path.basename(autosave_path)}"
            )
        except Exception as e:
            self.log_text.append(f"  [Autosave] Warning: could not write autosave: {e}")

    def _check_autosave_recovery(self, model_path: str):
        """After loading a model, check if an autosave from a crashed batch run exists."""
        autosave_path = os.path.join(os.path.dirname(model_path), "_inference_autosave.json")
        if not os.path.exists(autosave_path):
            return
        try:
            import json as _json
            with open(autosave_path) as f:
                data = _json.load(f)
            n_videos = len(data.get("results", {}))
            if n_videos == 0:
                return
            reply = QMessageBox.question(
                self,
                "Recover interrupted batch run?",
                f"Found an autosave from a previous session with {n_videos} video(s) already processed.\n\n"
                f"Would you like to restore those results now?\n\n"
                f"(Autosave file: {os.path.basename(autosave_path)})",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._load_timeline_results(path=autosave_path)
                self.log_text.append(
                    f"Recovered {n_videos} video(s) from autosave. "
                    "You can continue running inference on remaining videos."
                )
        except Exception as e:
            self.log_text.append(f"[Autosave] Could not read autosave file: {e}")

    def _save_results(self, silent=False):
        """Save current inference results to JSON."""
        if not self.results_cache:
            return
            
        try:
            # Determine default save path based on experiment config
            exp_path = self.config.get("experiment_path")
            if exp_path:
                results_dir = os.path.join(exp_path, "results")
            else:
                results_dir = os.path.join(self.config.get("data_dir", "data"), "results")
            
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, "inference_results.json")
            
            # If manual save and loaded from file, ask user
            if not silent and hasattr(self, 'loaded_results_path') and self.loaded_results_path:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Question)
                msg.setText("Save timeline results")
                msg.setInformativeText(f"Results loaded from:\n{self.loaded_results_path}\n\nDo you want to overwrite or save as new?")
                msg.setWindowTitle("Save results")
                overwrite_btn = msg.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
                save_as_btn = msg.addButton("Save As...", QMessageBox.ButtonRole.ActionRole)
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                
                if msg.clickedButton() == overwrite_btn:
                    results_path = self.loaded_results_path
                elif msg.clickedButton() == save_as_btn:
                    path, _ = QFileDialog.getSaveFileName(
                        self, "Save Results As", self.loaded_results_path, "JSON Files (*.json)"
                    )
                    if not path: return
                    results_path = path
                else:
                    return
            
            # Persist current video state into cache so it is saved
            self._persist_current_video_state()

            # If frame aggregation is enabled, persist aggregated segments for all videos.
            frame_aggregation_enabled = self.frame_aggregation_check.isChecked()
            if frame_aggregation_enabled:
                state = {
                    "video_path": self.video_path,
                    "predictions": self.predictions,
                    "confidences": self.confidences,
                    "clip_probabilities": self.clip_probabilities,
                    "clip_starts": self.clip_starts,
                    "clip_frame_probabilities": getattr(self, "clip_frame_probabilities", []),
                    "total_frames": self.total_frames,
                    "corrected_labels": self.corrected_labels,
                    "aggregated_segments": self.aggregated_segments,
                    "aggregated_multiclass_segments": self.aggregated_multiclass_segments,
                }
                try:
                    for v_path, entry in self.results_cache.items():
                        if not isinstance(entry, dict):
                            continue
                        preds = entry.get("predictions", [])
                        starts = entry.get("clip_starts", [])
                        if not preds or not starts:
                            continue
                        self.video_path = v_path
                        self.predictions = preds
                        self.confidences = entry.get("confidences", [])
                        self.clip_probabilities = entry.get("clip_probabilities", [])
                        self.clip_starts = starts
                        self.total_frames = int(entry.get("total_frames", 0) or 0)
                        self.corrected_labels = entry.get("corrected_labels", {})
                        precomputed = entry.get("aggregated_frame_probs")
                        if isinstance(precomputed, list):
                            precomputed = np.asarray(precomputed, dtype=np.float32)
                        if isinstance(precomputed, np.ndarray) and precomputed.ndim == 2 and precomputed.shape[0] > 0:
                            self._aggregated_frame_scores_norm = precomputed
                            self._aggregated_last_covered_frame = precomputed.shape[0]
                            self._build_timeline_segments()
                        else:
                            self.clip_frame_probabilities = entry.get("clip_frame_probabilities", [])
                            self.aggregated_segments = []
                            self.aggregated_multiclass_segments = []
                            self._compute_aggregated_timeline()
                        entry["aggregated_segments"] = copy.deepcopy(self.aggregated_segments)
                        entry["aggregated_multiclass_segments"] = copy.deepcopy(self.aggregated_multiclass_segments)
                finally:
                    self.video_path = state["video_path"]
                    self.predictions = state["predictions"]
                    self.confidences = state["confidences"]
                    self.clip_probabilities = state["clip_probabilities"]
                    self.clip_starts = state["clip_starts"]
                    self.clip_frame_probabilities = state["clip_frame_probabilities"]
                    self.total_frames = state["total_frames"]
                    self.corrected_labels = state["corrected_labels"]
                    self.aggregated_segments = state["aggregated_segments"]
                    self.aggregated_multiclass_segments = state["aggregated_multiclass_segments"]
            
            save_data = {
                "classes": self.classes,
                "parameters": {
                    "target_fps": self.target_fps_spin.value(),
                    "clip_length": self.clip_length_spin.value(),
                    "step_frames": self.step_frames_spin.value(),
                    "sample_inference_enabled": bool(self.sample_inference_check.isChecked()),
                    "sample_duration_seconds": int(self.sample_duration_spin.value()),
                    "sample_num_chunks": int(self.sample_count_spin.value()),
                    "frame_aggregation_enabled": frame_aggregation_enabled,
                    "merge_consecutive_enabled": bool(self.merge_timeline_check.isChecked()),
                    "use_viterbi_decode": bool(self.use_viterbi_decode),
                    "viterbi_switch_penalty": float(self.viterbi_switch_penalty),
                    "min_segment_frames": int(self._min_segment_frames),
                    "merge_gap_frames": int(self._merge_gap_frames),
                    "temporal_smoothing_window_frames": int(self._temporal_smoothing_window_frames),
                    "class_min_segment_frames": {
                        cls: int(v) for cls, v in self.class_min_segment_frames.items()
                    },
                    "class_merge_gap_frames": {
                        cls: int(v) for cls, v in self.class_merge_gap_frames.items()
                    },
                    "class_smoothing_window_frames": {
                        cls: int(v) for cls, v in self.class_smoothing_window_frames.items()
                    },
                    "use_ovr": bool(self._use_ovr),
                    "ovr_rows": bool(getattr(self, "ovr_rows_check", None) is not None and self.ovr_rows_check.isChecked()),
                    "ovr_show_all": bool(getattr(self, "ovr_show_all_check", None) is not None and self.ovr_show_all_check.isChecked()),
                    "use_ignore_threshold": bool(self.use_ignore_threshold),
                    "ignore_threshold": float(self.global_ignore_threshold),
                    "class_ignore_thresholds": {
                        cls: float(t) for cls, t in self.class_ignore_thresholds.items()
                    },
                    "allowed_cooccurrence": [
                        [a, b] for (a, b) in sorted(self._allowed_cooccurrence) if a <= b
                    ],
                    "timeline_zoom": int(self.timeline_zoom_spin.value()),
                },
                "results": self.results_cache
            }
            self._write_results_bundle(results_path, save_data, pretty=True)
            
            if frame_aggregation_enabled and self.aggregated_segments:
                self.log_text.append(f"Inference results saved to: {results_path} (with {len(self.aggregated_segments)} aggregated segments)")
            else:
                self.log_text.append(f"Inference results saved to: {results_path}")
            
            if not silent:
                QMessageBox.information(self, "Saved", f"Results saved to:\n{results_path}")
                # Update loaded path if we saved to a new location
                self.loaded_results_path = results_path
                
        except Exception as e:
            self.log_text.append(f"Warning: Failed to save inference results: {e}")
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to save results:\n{str(e)}")
    
    def _load_timeline_results(self, path: str = None):
        """Load previously saved timeline results.

        If *path* is provided (e.g. autosave recovery), skip the file dialog.
        """
        if path:
            file_path = path
        else:
            # Prompt user to select results file
            exp_path = self.config.get("experiment_path")
            if exp_path:
                default_dir = os.path.join(exp_path, "results")
            else:
                default_dir = os.path.join(self.config.get("data_dir", "data"), "results")

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Timeline Results",
                default_dir,
                "JSON Files (*.json)"
            )

            if not file_path:
                return

        self.loaded_results_path = file_path

        MAX_LOAD_SIZE_MB = 300
        if os.path.getsize(file_path) > MAX_LOAD_SIZE_MB * 1024 * 1024:
            QMessageBox.critical(
                self,
                "File too large",
                f"This results file is over {MAX_LOAD_SIZE_MB} MB and may cause a crash when loading.\n\n"
                "Re-run inference and save again (attention maps are now excluded from saves).\n"
                "Or load a smaller/simpler results file.",
            )
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._restore_external_arrays(file_path, data)
            
            # Extract data
            self.classes = data.get("classes", [])
            parameters = data.get("parameters", {})
            self.results_cache = data.get("results", {})
            self._ignore_threshold_user_modified = False
            
            if not self.classes or not self.results_cache:
                QMessageBox.warning(self, "Invalid File", "The selected file does not contain valid inference results.")
                return
            
            for v_path in self.results_cache:
                if "corrected_labels" not in self.results_cache[v_path]:
                    self.results_cache[v_path]["corrected_labels"] = {}
                if "corrected_attr_labels" not in self.results_cache[v_path]:
                    self.results_cache[v_path]["corrected_attr_labels"] = {}
            
            # Update UI with loaded parameters
            self.target_fps_spin.setValue(parameters.get("target_fps", 16))
            self.clip_length_spin.setValue(parameters.get("clip_length", 16))
            self.step_frames_spin.setValue(max(1, self.clip_length_spin.value() // 2))
            self.sample_inference_check.setChecked(bool(parameters.get("sample_inference_enabled", False)))
            self.sample_duration_spin.setValue(int(parameters.get("sample_duration_seconds", 60)))
            self.sample_count_spin.setValue(int(parameters.get("sample_num_chunks", 5)))
            loaded_use_ovr = parameters.get("use_ovr", None)
            if loaded_use_ovr is None:
                loaded_use_ovr = any(
                    bool((v or {}).get("aggregated_multiclass_segments", []))
                    for v in self.results_cache.values()
                    if isinstance(v, dict)
                )
            self._use_ovr = bool(loaded_use_ovr)
            self._allowed_cooccurrence = set()
            for pair in parameters.get("allowed_cooccurrence", []):
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a, b = pair[0], pair[1]
                    self._allowed_cooccurrence.add((a, b))
                    self._allowed_cooccurrence.add((b, a))
            if getattr(self, "ovr_rows_check", None) is not None and self._use_ovr:
                self.ovr_rows_check.setChecked(True)
            if getattr(self, "ovr_rows_check", None) is not None and "ovr_rows" in parameters:
                self.ovr_rows_check.setChecked(bool(parameters.get("ovr_rows", True)))
            if getattr(self, "ovr_show_all_check", None) is not None:
                self.ovr_show_all_check.setChecked(bool(parameters.get("ovr_show_all", False)))
            if "merge_consecutive_enabled" in parameters:
                self.merge_timeline_check.setChecked(bool(parameters.get("merge_consecutive_enabled", False)))
            if "use_viterbi_decode" in parameters:
                self.use_viterbi_decode = bool(parameters.get("use_viterbi_decode", False))
                self.use_viterbi_check.setChecked(self.use_viterbi_decode)
            if "viterbi_switch_penalty" in parameters:
                self.viterbi_switch_penalty = float(parameters.get("viterbi_switch_penalty", 0.35))
                self.viterbi_switch_penalty_spin.setValue(self.viterbi_switch_penalty)
            
            # Restore frame aggregation setting
            frame_aggregation_enabled = parameters.get("frame_aggregation_enabled", False)
            self.frame_aggregation_check.setChecked(frame_aggregation_enabled)
            if "min_segment_frames" in parameters:
                self._min_segment_frames = int(parameters.get("min_segment_frames", 1))
            if "merge_gap_frames" in parameters:
                self._merge_gap_frames = int(parameters.get("merge_gap_frames", 0))
            if "temporal_smoothing_window_frames" in parameters:
                loaded_win = int(parameters.get("temporal_smoothing_window_frames", 1))
                if loaded_win % 2 == 0:
                    loaded_win += 1
                self._temporal_smoothing_window_frames = max(1, loaded_win)
            if "class_min_segment_frames" in parameters:
                self.class_min_segment_frames = {
                    str(cls): int(v) for cls, v in parameters.get("class_min_segment_frames", {}).items()
                }
            if "class_merge_gap_frames" in parameters:
                self.class_merge_gap_frames = {
                    str(cls): int(v) for cls, v in parameters.get("class_merge_gap_frames", {}).items()
                }
            if "class_smoothing_window_frames" in parameters:
                self.class_smoothing_window_frames = {
                    str(cls): int(v) for cls, v in parameters.get("class_smoothing_window_frames", {}).items()
                }
            self._sync_per_class_segment_rule_config()

            # Restore timeline zoom
            if "timeline_zoom" in parameters:
                self.timeline_zoom_spin.blockSignals(True)
                self.timeline_zoom_spin.setValue(int(parameters["timeline_zoom"]))
                self.timeline_zoom_spin.blockSignals(False)

            # Restore ignore threshold settings so the loaded timeline matches what was saved
            if "use_ignore_threshold" in parameters:
                self.use_ignore_threshold = bool(parameters["use_ignore_threshold"])
                self.use_ignore_threshold_check.setChecked(self.use_ignore_threshold)
            if "ignore_threshold" in parameters:
                self.global_ignore_threshold = float(parameters["ignore_threshold"])
                self.ignore_threshold_spin.setValue(self.global_ignore_threshold)
            if "class_ignore_thresholds" in parameters:
                self.class_ignore_thresholds = {
                    cls: float(t) for cls, t in parameters["class_ignore_thresholds"].items()
                }
                self.config["inference_class_ignore_thresholds"] = dict(self.class_ignore_thresholds)
            self.config["inference_use_viterbi_decode"] = self.use_viterbi_decode
            self.config["inference_viterbi_switch_penalty"] = self.viterbi_switch_penalty
            self._update_viterbi_ui_state()
            
            self.classes_label.setText(", ".join(self.classes))
            
            # Populate video filter combo
            self.filter_video_combo.blockSignals(True)
            self.filter_video_combo.clear()
            
            video_paths = list(self.results_cache.keys())
            for vp in video_paths:
                video_name = os.path.basename(vp)
                self.filter_video_combo.addItem(video_name, vp)
            
            self.filter_video_combo.blockSignals(False)
            
            # Load first video's results
            if video_paths:
                first_video = video_paths[0]
                self.video_path = first_video
                self.video_paths = video_paths
                
                # Update video path display
                if len(video_paths) == 1:
                    self.video_path_edit.setText(first_video)
                else:
                    self.video_path_edit.setText(f"{len(video_paths)} videos loaded")
                
                # Load first video data
                data = self.results_cache[first_video]
                self.predictions = data["predictions"]
                self.confidences = data["confidences"]
                self.clip_probabilities = data.get("clip_probabilities", [])
                self.clip_frame_probabilities = data.get("clip_frame_probabilities", [])
                self.attr_predictions = data.get("attr_predictions", [])
                self.attr_confidences = data.get("attr_confidences", [])
                self.clip_starts = data["clip_starts"]
                self.localization_bboxes = data.get("localization_bboxes", [])
                self.total_frames = data.get("total_frames", 0)
                self.corrected_labels = data.get("corrected_labels", {})
                self.corrected_attr_labels = data.get("corrected_attr_labels", {})
                
                # Backwards compatibility: if total_frames not saved, try to get from video
                if self.total_frames <= 0 and os.path.exists(first_video):
                    try:
                        cap = cv2.VideoCapture(first_video)
                        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        # Store for future use
                        data["total_frames"] = self.total_frames
                    except Exception:
                        pass
                
                # Load aggregated segments if saved, otherwise compute if frame aggregation enabled
                if "aggregated_segments" in data:
                    self.aggregated_segments = data["aggregated_segments"]
                    self.aggregated_multiclass_segments = data.get("aggregated_multiclass_segments", [])
                    self.log_text.append(f"Loaded {len(self.aggregated_segments)} saved aggregated segments")
                    
                    # Load precomputed frame probs if available (may be list when loaded from JSON)
                    precomputed_probs = data.get("aggregated_frame_probs")
                    if isinstance(precomputed_probs, list):
                        precomputed_probs = np.asarray(precomputed_probs, dtype=np.float32)
                    if isinstance(precomputed_probs, np.ndarray):
                         self._aggregated_frame_scores_norm = precomputed_probs
                         self._aggregated_last_covered_frame = len(precomputed_probs)
                    elif frame_aggregation_enabled and self._use_ovr and self.clip_probabilities:
                        self._compute_aggregated_timeline()
                elif frame_aggregation_enabled:
                    # Check for pre-computed frame probs (from worker or JSON load)
                    precomputed_probs = data.get("aggregated_frame_probs")
                    if isinstance(precomputed_probs, list):
                        precomputed_probs = np.asarray(precomputed_probs, dtype=np.float32)
                    if isinstance(precomputed_probs, np.ndarray):
                         self._aggregated_frame_scores_norm = precomputed_probs
                         self._aggregated_last_covered_frame = len(precomputed_probs)
                         self._build_timeline_segments()
                    else:
                        self._compute_aggregated_timeline()
                
                # Populate behavior filter
                self.filter_behavior_combo.blockSignals(True)
                self.filter_behavior_combo.clear()
                self.filter_behavior_combo.addItem("All Behaviors")
                self.filter_behavior_combo.addItem(self.ignore_label_name)
                self.filter_behavior_combo.addItems(self.classes)
                self.filter_behavior_combo.blockSignals(False)
                
                # Display results
                self._display_results()
                
                # Enable export/preview buttons
                self.export_btn.setEnabled(True)
                self.preview_btn.setEnabled(True)
                self.export_timeline_btn.setEnabled(True)
                self.save_results_btn.setEnabled(True) # Enable saving loaded/corrected results
                
                self.log_text.append(f"Loaded timeline results from: {file_path}")
                self.log_text.append(f"Loaded {len(video_paths)} video(s) with {len(self.predictions)} predictions")
                
                QMessageBox.information(
                    self,
                    "Loaded",
                    f"Successfully loaded timeline results.\n\n"
                    f"Videos: {len(video_paths)}\n"
                    f"Classes: {len(self.classes)}\n\n"
                    f"You can now review clips, make corrections, and re-save."
                )
            
        except Exception as e:
            self.log_text.append(f"Error loading timeline results: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load timeline results:\n{str(e)}")

    def _on_video_selection_changed(self, index: int):
        """Handle video selection change."""
        video_path = self.filter_video_combo.currentData()
        self._load_video_from_cache(video_path)

    def _current_threshold_settings(self) -> dict:
        return {
            "use_ignore_threshold": bool(self.use_ignore_threshold),
            "ignore_threshold": float(self.global_ignore_threshold),
            "class_ignore_thresholds": {
                str(cls): float(t) for cls, t in self.class_ignore_thresholds.items()
            },
            "user_modified": bool(self._ignore_threshold_user_modified),
        }

    def _apply_threshold_settings(self, settings: dict):
        use_ignore = bool(settings.get("use_ignore_threshold", self.use_ignore_threshold))
        tau = settings.get("ignore_threshold", self.global_ignore_threshold)
        try:
            tau = float(tau)
        except Exception:
            tau = self.global_ignore_threshold
        raw_per_class = settings.get("class_ignore_thresholds", {})
        if not isinstance(raw_per_class, dict):
            raw_per_class = {}
        per_class = {str(cls): float(t) for cls, t in raw_per_class.items()}

        self._applying_auto_threshold = True
        try:
            self.use_ignore_threshold_check.blockSignals(True)
            self.ignore_threshold_spin.blockSignals(True)
            try:
                self.use_ignore_threshold_check.setChecked(use_ignore)
                self.ignore_threshold_spin.setValue(tau)
            finally:
                self.use_ignore_threshold_check.blockSignals(False)
                self.ignore_threshold_spin.blockSignals(False)
        finally:
            self._applying_auto_threshold = False

        self.use_ignore_threshold = use_ignore
        self.global_ignore_threshold = tau
        self.class_ignore_thresholds = per_class
        self._ignore_threshold_user_modified = bool(settings.get("user_modified", False))
        self.config["inference_use_ignore_threshold"] = self.use_ignore_threshold
        self.config["inference_ignore_threshold"] = self.global_ignore_threshold
        self.config["inference_class_ignore_thresholds"] = dict(self.class_ignore_thresholds)

    def _persist_current_video_state(self):
        if not self.video_path or self.video_path not in self.results_cache:
            return
        entry = self.results_cache[self.video_path]
        entry["corrected_labels"] = dict(self.corrected_labels)
        entry["corrected_attr_labels"] = dict(self.corrected_attr_labels)
        entry["threshold_settings"] = self._current_threshold_settings()

    def _load_video_from_cache(
        self,
        video_path: str,
        refresh_display: bool = True,
        persist_current: bool = True,
        threshold_settings_override: dict | None = None,
        persist_loaded_thresholds: bool = True,
    ) -> bool:
        if not video_path or video_path not in self.results_cache:
            return False

        if persist_current:
            self._persist_current_video_state()

        data = self.results_cache[video_path]
        self.predictions = data["predictions"]
        self.confidences = data["confidences"]
        self.clip_probabilities = data.get("clip_probabilities", [])
        self.clip_frame_probabilities = data.get("clip_frame_probabilities", [])
        self.attr_predictions = data.get("attr_predictions", [])
        self.attr_confidences = data.get("attr_confidences", [])
        self.clip_starts = data["clip_starts"]
        self.localization_bboxes = data.get("localization_bboxes", [])
        self.total_frames = data.get("total_frames", 0)
        self.corrected_labels = data.get("corrected_labels", {})
        self.corrected_attr_labels = data.get("corrected_attr_labels", {})
        self.video_path = video_path

        threshold_settings = threshold_settings_override
        if threshold_settings is None:
            threshold_settings = data.get("threshold_settings")
        if isinstance(threshold_settings, dict):
            self._apply_threshold_settings(threshold_settings)
        else:
            self._auto_update_ignore_threshold()
            if persist_loaded_thresholds:
                data["threshold_settings"] = self._current_threshold_settings()

        if self.clip_frame_probabilities and not self.frame_aggregation_check.isChecked():
            self.frame_aggregation_check.setChecked(True)
            self.log_text.append("Frame-head outputs found for this video. Switched to precise frame-boundary mode.")

        if self.total_frames <= 0 and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                data["total_frames"] = self.total_frames
            except Exception:
                pass

        if self.frame_aggregation_check.isChecked():
            precomputed_probs = data.get("aggregated_frame_probs")
            if isinstance(precomputed_probs, list):
                precomputed_probs = np.asarray(precomputed_probs, dtype=np.float32)
            if isinstance(precomputed_probs, np.ndarray):
                 self._aggregated_frame_scores_norm = precomputed_probs
                 self._aggregated_last_covered_frame = len(precomputed_probs)
                 self._build_timeline_segments()
            elif self.use_ignore_threshold:
                self._compute_aggregated_timeline()
            elif "aggregated_segments" in data:
                self.aggregated_segments = data["aggregated_segments"]
                self.aggregated_multiclass_segments = data.get("aggregated_multiclass_segments", [])
                self._aggregated_frame_scores_norm = None
                self._aggregated_active_mask = None
                self._aggregated_last_covered_frame = 0
                if self._use_ovr:
                    self._compute_aggregated_timeline()
            else:
                self._compute_aggregated_timeline()
        else:
            self.aggregated_segments = []
            self.aggregated_multiclass_segments = []
            self._aggregated_frame_scores_norm = None
            self._aggregated_active_mask = None
            self._aggregated_last_covered_frame = 0

        if refresh_display:
            self._display_results()
        return True
    
    def _on_error(self, error_msg: str):
        """Handle inference error."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Error: {error_msg}")
        self.run_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(False)
        self.load_timeline_btn.setEnabled(True)
        QMessageBox.critical(self, "Inference Error", f"Inference failed:\n{error_msg}")

    def _stop_inference(self):
        """Request inference worker to stop; completed videos are kept."""
        if getattr(self, "worker", None) and self.worker.isRunning():
            self.worker.stop()
            self.progress_label.setText("Stopping... (keeping completed videos)")
    
    def _on_log(self, message: str):
        """Handle log message."""
        self.log_text.append(message)
    
    def _display_results(self):
        """Display inference results."""
        self._update_viterbi_ui_state()
        self.results_list.clear()

        # In precise frame mode, show segment-level results instead of per-clip
        # entries to avoid confusion when a clip contains multiple behaviors.
        if self.frame_aggregation_check.isChecked() and self.aggregated_segments:
            for i, seg in enumerate(self.aggregated_segments):
                cls_idx = int(seg.get("class", -1))
                start_f = int(seg.get("start", 0))
                end_f = int(seg.get("end", start_f))
                conf = float(seg.get("confidence", 0.0))
                if cls_idx < 0 or cls_idx >= len(self.classes):
                    label = self.ignore_label_name
                else:
                    label = self.classes[cls_idx]
                item_text = f"Segment {i+1}: {label} (frames {start_f}-{end_f}, confidence: {conf:.2%})"
                item = QListWidgetItem(item_text)
                if conf > 0.7:
                    item.setForeground(QColor(0, 150, 0))
                elif conf > 0.5:
                    item.setForeground(QColor(200, 150, 0))
                else:
                    item.setForeground(QColor(200, 0, 0))
                self.results_list.addItem(item)
            self._draw_timeline()
            return
        
        has_attrs = bool(self.attr_predictions and self.attributes)
        
        effective_preds = self._effective_predictions()
        for i, (pred_idx, conf) in enumerate(zip(effective_preds, self.confidences)):
            if pred_idx < len(self.classes) and pred_idx >= 0:
                label = self.classes[pred_idx]
                
                # Append Attribute if available
                attr_conf = None
                if has_attrs and i < len(self.attr_predictions):
                    attr_idx = self.attr_predictions[i]
                    if isinstance(attr_idx, int) and attr_idx < len(self.attributes):
                        attr_label = self.attributes[attr_idx]
                        if self.attr_confidences and i < len(self.attr_confidences):
                            attr_conf = self.attr_confidences[i]
                        label = f"{label} ({attr_label})"
                
                extra_labels = ""
                if self._use_ovr and i < len(self.clip_probabilities):
                    probs_i = self.clip_probabilities[i]
                    if isinstance(probs_i, (list, tuple)) and len(probs_i) == len(self.classes):
                        active = self._filter_cooccurrence(probs_i, 0.3)
                        above = []
                        for ci in sorted(active):
                            if ci == pred_idx:
                                continue
                            above.append(f"{self.classes[ci]}:{float(probs_i[ci]):.0%}")
                        if above:
                            extra_labels = " + " + ", ".join(above)

                if attr_conf is not None:
                    item_text = f"Clip {i+1}: {label} (class: {conf:.2%}, {attr_label}: {attr_conf:.2%}){extra_labels}"
                else:
                    item_text = f"Clip {i+1}: {label} (confidence: {conf:.2%}){extra_labels}"
                item = QListWidgetItem(item_text)
                
                if conf > 0.7:
                    item.setForeground(QColor(0, 150, 0))
                elif conf > 0.5:
                    item.setForeground(QColor(200, 150, 0))
                else:
                    item.setForeground(QColor(200, 0, 0))
                
                self.results_list.addItem(item)
            elif pred_idx < 0:
                item = QListWidgetItem(f"Clip {i+1}: {self.ignore_label_name} (confidence: {conf:.2%})")
                item.setForeground(QColor(120, 120, 120))
                self.results_list.addItem(item)
        
        self._draw_timeline()

    def _threshold_for_pred(self, pred_idx: int) -> float:
        if pred_idx < 0 or pred_idx >= len(self.classes):
            return self.global_ignore_threshold
        cls = self.classes[pred_idx]
        if cls in self.class_ignore_thresholds:
            return float(self.class_ignore_thresholds[cls])
        return self.global_ignore_threshold

    def _effective_prediction_for_clip(self, clip_idx: int) -> int:
        pred_idx = self.corrected_labels[clip_idx] if clip_idx in self.corrected_labels else self.predictions[clip_idx]
        if not self.use_ignore_threshold:
            return pred_idx
        if clip_idx >= len(self.confidences):
            return pred_idx
        # Round to 3 decimals so 0.9997 (displayed as ~100%) isn't below a 1.000 threshold
        conf = round(float(self.confidences[clip_idx]), 3)
        if conf < self._threshold_for_pred(pred_idx):
            return -1
        return pred_idx

    def _effective_predictions(self):
        if not self.predictions:
            return []
        return [self._effective_prediction_for_clip(i) for i in range(len(self.predictions))]

    def _on_ignore_threshold_changed(self, *_args):
        self.cleanup_preset_combo.blockSignals(True)
        self.cleanup_preset_combo.setCurrentText("Custom")
        self.cleanup_preset_combo.blockSignals(False)
        self.use_ignore_threshold = bool(self.use_ignore_threshold_check.isChecked())
        self.global_ignore_threshold = float(self.ignore_threshold_spin.value())
        self.config["inference_use_ignore_threshold"] = self.use_ignore_threshold
        self.config["inference_ignore_threshold"] = self.global_ignore_threshold
        if not self._applying_auto_threshold:
            self._ignore_threshold_user_modified = True
        self._persist_current_video_state()
        if self.predictions:
            if self.frame_aggregation_check.isChecked():
                self._compute_aggregated_timeline()
            self._display_results()

    def _update_viterbi_ui_state(self):
        """Keep Viterbi controls enabled and explain current mode."""
        self.use_viterbi_check.setEnabled(True)
        self.viterbi_switch_penalty_spin.setEnabled(bool(self.use_viterbi_decode))
        if self._use_ovr:
            self.use_viterbi_check.setToolTip(
                "Inference-only sequence decoding on the merged frame probabilities.\n"
                "OvR uses binary per-class Viterbi, then keeps top classes consistent\n"
                "with allowed co-occurrence rules."
            )
        else:
            self.use_viterbi_check.setToolTip(
                "Inference-only sequence decoding for the single-label timeline.\n"
                "Reduces rapid frame-to-frame class switching without retraining."
            )

    _CLEANUP_PRESETS = {
        "None": {
            "use_viterbi": False, "switch_penalty": 0.35,
            "use_ignore": False, "threshold": 0.60,
            "smooth_window": 1, "gap_fill": 0, "min_segment": 1,
        },
        "Light": {
            "use_viterbi": False, "switch_penalty": 0.35,
            "use_ignore": True, "threshold": 0.50,
            "smooth_window": 3, "gap_fill": 2, "min_segment": 3,
        },
        "Standard": {
            "use_viterbi": True, "switch_penalty": 0.35,
            "use_ignore": True, "threshold": 0.55,
            "smooth_window": 5, "gap_fill": 5, "min_segment": 8,
        },
        "Aggressive": {
            "use_viterbi": True, "switch_penalty": 0.60,
            "use_ignore": True, "threshold": 0.60,
            "smooth_window": 11, "gap_fill": 10, "min_segment": 20,
        },
    }

    def _on_cleanup_preset_changed(self, preset_name: str):
        if preset_name == "Custom":
            return
        preset = self._CLEANUP_PRESETS.get(preset_name)
        if not preset:
            return

        self.use_viterbi_check.blockSignals(True)
        self.viterbi_switch_penalty_spin.blockSignals(True)
        self.use_ignore_threshold_check.blockSignals(True)
        self.ignore_threshold_spin.blockSignals(True)
        try:
            self.use_viterbi_check.setChecked(preset["use_viterbi"])
            self.viterbi_switch_penalty_spin.setValue(preset["switch_penalty"])
            self.use_ignore_threshold_check.setChecked(preset["use_ignore"])
            self.ignore_threshold_spin.setValue(preset["threshold"])
        finally:
            self.use_viterbi_check.blockSignals(False)
            self.viterbi_switch_penalty_spin.blockSignals(False)
            self.use_ignore_threshold_check.blockSignals(False)
            self.ignore_threshold_spin.blockSignals(False)

        self.use_viterbi_decode = preset["use_viterbi"]
        self.viterbi_switch_penalty = preset["switch_penalty"]
        self.use_ignore_threshold = preset["use_ignore"]
        self.global_ignore_threshold = preset["threshold"]
        self._min_segment_frames = preset["min_segment"]
        self._merge_gap_frames = preset["gap_fill"]
        self._temporal_smoothing_window_frames = preset["smooth_window"]

        for cls in self.classes:
            self.class_min_segment_frames[cls] = preset["min_segment"]
            self.class_merge_gap_frames[cls] = preset["gap_fill"]
            self.class_smoothing_window_frames[cls] = preset["smooth_window"]

        self.config["inference_use_viterbi_decode"] = self.use_viterbi_decode
        self.config["inference_viterbi_switch_penalty"] = self.viterbi_switch_penalty
        self.config["inference_use_ignore_threshold"] = self.use_ignore_threshold
        self.config["inference_ignore_threshold"] = self.global_ignore_threshold
        self._sync_per_class_segment_rule_config()
        self._update_viterbi_ui_state()

        if self.predictions and self.frame_aggregation_check.isChecked():
            self._compute_aggregated_timeline()
            self._display_results()

    def _on_viterbi_changed(self, *_args):
        self.cleanup_preset_combo.blockSignals(True)
        self.cleanup_preset_combo.setCurrentText("Custom")
        self.cleanup_preset_combo.blockSignals(False)
        self.use_viterbi_decode = bool(self.use_viterbi_check.isChecked())
        self.viterbi_switch_penalty = float(self.viterbi_switch_penalty_spin.value())
        self.config["inference_use_viterbi_decode"] = self.use_viterbi_decode
        self.config["inference_viterbi_switch_penalty"] = self.viterbi_switch_penalty
        if self.use_viterbi_decode and not self.frame_aggregation_check.isChecked():
            self.frame_aggregation_check.setChecked(True)
        self._update_viterbi_ui_state()
        self._persist_current_video_state()
        if self.predictions and self.frame_aggregation_check.isChecked():
            self._compute_aggregated_timeline()
            self._display_results()

    def _auto_update_ignore_threshold(self):
        """Auto-update ignore thresholds, preferring validation calibration."""
        if self._ignore_threshold_user_modified:
            return
        cfg = self.model_training_config if isinstance(self.model_training_config, dict) else {}
        calibrated = cfg.get("validation_calibrated_ignore_thresholds")
        if isinstance(calibrated, dict):
            tau = calibrated.get("global_threshold", self.global_ignore_threshold)
            try:
                tau = float(tau)
            except Exception:
                tau = self.global_ignore_threshold
            tau = max(0.35, min(0.90, tau))
            raw_per_class = calibrated.get("per_class_thresholds", {})
            per_class = {}
            if isinstance(raw_per_class, dict):
                for cls_name in self.classes:
                    if cls_name in raw_per_class:
                        try:
                            per_class[cls_name] = max(0.35, min(0.90, float(raw_per_class[cls_name])))
                        except Exception:
                            per_class[cls_name] = tau
                    else:
                        per_class[cls_name] = tau
            if per_class:
                self._applying_auto_threshold = True
                try:
                    self.ignore_threshold_spin.blockSignals(True)
                    try:
                        self.ignore_threshold_spin.setValue(tau)
                    finally:
                        self.ignore_threshold_spin.blockSignals(False)
                finally:
                    self._applying_auto_threshold = False
                self.global_ignore_threshold = tau
                self.class_ignore_thresholds = per_class
                self.config["inference_ignore_threshold"] = tau
                self.config["inference_class_ignore_thresholds"] = dict(per_class)
                source = str(calibrated.get("source", "validation"))
                self.log_text.append(
                    f"Auto-set ignore thresholds from model validation: global τ={tau:.2f}, "
                    f"per-class τ for {len(per_class)} classes ({source})"
                )
                return

        if not self.confidences:
            return
        confs = np.array(self.confidences, dtype=float)
        confs = confs[np.isfinite(confs)]
        if confs.size == 0:
            return

        has_validation = bool(cfg) and (cfg.get("use_all_for_training") is False)
        if has_validation:
            best_val_f1 = cfg.get("best_val_f1")
            try:
                best_val_f1 = float(best_val_f1) if best_val_f1 is not None else None
            except Exception:
                best_val_f1 = None
            if best_val_f1 is not None:
                quality = max(0.0, min(1.0, (best_val_f1 - 40.0) / 50.0))
                quantile = 0.80 - 0.20 * quality
            else:
                quantile = 0.68
            source = "current-video fallback (validation-aware quantile)"
        else:
            quantile = 0.70
            source = "current-video fallback"

        tau = float(np.quantile(confs, quantile))
        tau = max(0.35, min(0.90, tau))
        self._applying_auto_threshold = True
        try:
            self.ignore_threshold_spin.blockSignals(True)
            try:
                self.ignore_threshold_spin.setValue(tau)
            finally:
                self.ignore_threshold_spin.blockSignals(False)
        finally:
            self._applying_auto_threshold = False
        self.global_ignore_threshold = tau
        self.config["inference_ignore_threshold"] = tau

        min_class_support = 10
        per_class = {}
        for cls_idx, cls_name in enumerate(self.classes):
            cls_confs = [
                float(self.confidences[i])
                for i, pred_idx in enumerate(self.predictions)
                if pred_idx == cls_idx and i < len(self.confidences) and np.isfinite(self.confidences[i])
            ]
            if len(cls_confs) >= min_class_support:
                cls_tau = float(np.quantile(np.array(cls_confs, dtype=float), quantile))
                cls_tau = max(0.35, min(0.90, cls_tau))
                per_class[cls_name] = cls_tau
            else:
                per_class[cls_name] = tau

        self.class_ignore_thresholds = per_class
        self.config["inference_class_ignore_thresholds"] = dict(per_class)
        self.log_text.append(
            f"Auto-set ignore thresholds: global τ={tau:.2f}, per-class τ for {len(per_class)} classes ({source})"
        )

    def _open_per_class_thresholds_dialog(self):
        if not self.classes:
            QMessageBox.information(self, "No classes", "Load a model first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Per-class Ignore Thresholds")
        layout = QFormLayout(dlg)
        spins = {}
        for cls in self.classes:
            sp = QDoubleSpinBox()
            sp.setDecimals(3)
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.01)
            sp.setValue(float(self.class_ignore_thresholds.get(cls, self.global_ignore_threshold)))
            layout.addRow(cls, sp)
            spins[cls] = sp
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec():
            self.class_ignore_thresholds = {cls: float(sp.value()) for cls, sp in spins.items()}
            self.config["inference_class_ignore_thresholds"] = dict(self.class_ignore_thresholds)
            self._on_ignore_threshold_changed()

    def _open_per_class_segment_rules_dialog(self):
        if not self.classes:
            QMessageBox.information(self, "No classes", "Load a model first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Per-class Segment Rules")
        dlg.resize(620, 520)
        root = QVBoxLayout(dlg)
        root.addWidget(QLabel(
            "Override precise-boundary postprocessing per class. "
            "Values matching the global controls act like defaults."
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.addWidget(QLabel("Class"), 0, 0)
        grid.addWidget(QLabel("Smooth"), 0, 1)
        grid.addWidget(QLabel("Gap"), 0, 2)
        grid.addWidget(QLabel("Min seg"), 0, 3)

        controls = {}
        global_smooth = int(max(1, self._temporal_smoothing_window_frames))
        if global_smooth % 2 == 0:
            global_smooth += 1
        global_gap = int(max(0, self._merge_gap_frames))
        global_min_seg = int(max(1, self._min_segment_frames))

        for row, cls in enumerate(self.classes, start=1):
            grid.addWidget(QLabel(cls), row, 0)

            sp_smooth = QSpinBox()
            sp_smooth.setRange(1, 99)
            sp_smooth.setSingleStep(2)
            smooth_val = int(self.class_smoothing_window_frames.get(cls, global_smooth))
            if smooth_val % 2 == 0:
                smooth_val += 1
            sp_smooth.setValue(max(1, smooth_val))
            grid.addWidget(sp_smooth, row, 1)

            sp_gap = QSpinBox()
            sp_gap.setRange(0, 200)
            sp_gap.setValue(int(self.class_merge_gap_frames.get(cls, global_gap)))
            grid.addWidget(sp_gap, row, 2)

            sp_min = QSpinBox()
            sp_min.setRange(1, 200)
            sp_min.setValue(int(self.class_min_segment_frames.get(cls, global_min_seg)))
            grid.addWidget(sp_min, row, 3)

            controls[cls] = (sp_smooth, sp_gap, sp_min)

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec():
            new_smooth = {}
            new_gap = {}
            new_min = {}
            for cls, (sp_smooth, sp_gap, sp_min) in controls.items():
                smooth_val = int(sp_smooth.value())
                if smooth_val % 2 == 0:
                    smooth_val += 1
                gap_val = int(sp_gap.value())
                min_val = int(sp_min.value())
                if smooth_val != global_smooth:
                    new_smooth[cls] = smooth_val
                if gap_val != global_gap:
                    new_gap[cls] = gap_val
                if min_val != global_min_seg:
                    new_min[cls] = min_val
            self.class_smoothing_window_frames = new_smooth
            self.class_merge_gap_frames = new_gap
            self.class_min_segment_frames = new_min
            self._sync_per_class_segment_rule_config()
            if self.predictions and self.frame_aggregation_check.isChecked():
                self._compute_aggregated_timeline()
                self._display_results()

    def _get_localization_bbox_for_clip_frame(self, clip_idx: int, frame_idx: int):
        """Return normalized xyxy localization bbox for a clip frame, or None."""
        if clip_idx < 0 or clip_idx >= len(self.localization_bboxes):
            return None

        raw = self.localization_bboxes[clip_idx]
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            return None

        def _ema_smooth_boxes(boxes, alpha: float):
            if not boxes:
                return boxes
            prev = [float(v) for v in boxes[0]]
            smoothed = [prev]
            for i in range(1, len(boxes)):
                curr = [float(v) for v in boxes[i]]
                prev = [
                    alpha * curr[0] + (1.0 - alpha) * prev[0],
                    alpha * curr[1] + (1.0 - alpha) * prev[1],
                    alpha * curr[2] + (1.0 - alpha) * prev[2],
                    alpha * curr[3] + (1.0 - alpha) * prev[3],
                ]
                smoothed.append(prev)
            return smoothed

        # Per-clip bbox: [4]
        if len(raw) == 4 and all(not isinstance(v, (list, tuple)) for v in raw):
            vals = raw
        # Per-frame bboxes: [T,4]
        elif isinstance(raw[0], (list, tuple)) and len(raw[0]) == 4:
            idx = max(0, min(int(frame_idx), len(raw) - 1))
            vals = _ema_smooth_boxes(raw, self._bbox_ema_alpha)[idx]
        else:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in vals]
        except Exception:
            return None

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if (x2 - x1) < 1e-4 or (y2 - y1) < 1e-4:
            return None
        return x1, y1, x2, y2

    def _get_classification_roi_bbox_for_clip_frame(self, clip_idx: int):
        """Return the normalized xyxy ROI used by classification cropping."""
        if clip_idx < 0 or clip_idx >= len(self.localization_bboxes):
            return None

        raw = self.localization_bboxes[clip_idx]
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            return None

        # Match InferenceWorker._build_refined_clips:
        # if temporal boxes exist, use first-frame box as fixed clip ROI.
        if isinstance(raw[0], (list, tuple)) and len(raw[0]) == 4:
            vals = raw[0]
        elif len(raw) == 4 and all(not isinstance(v, (list, tuple)) for v in raw):
            vals = raw
        else:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in vals]
        except Exception:
            return None

        crop_padding = getattr(self, "_crop_padding", 0.35)
        crop_min_size = getattr(self, "_crop_min_size", 0.04)
        x1, y1, x2, y2 = _sanitize_bbox_coords(x1, y1, x2, y2, crop_padding, crop_min_size)
        if (x2 - x1) < 1e-4 or (y2 - y1) < 1e-4:
            return None
        return x1, y1, x2, y2

    def _get_saved_frame_interval(self, video_path: str, orig_fps: float) -> int:
        """Return inference-time frame interval for a video when available."""
        if video_path and isinstance(self.results_cache, dict):
            entry = self.results_cache.get(video_path, {})
            if isinstance(entry, dict):
                stored = entry.get("frame_interval", None)
                try:
                    stored_val = int(stored)
                except Exception:
                    stored_val = 0
                if stored_val > 0:
                    return stored_val

        target_fps = max(1, int(self.target_fps_spin.value()))
        return max(1, int(round(float(orig_fps) / float(target_fps))))

    def _build_center_merge_weights(self, length: int) -> np.ndarray:
        """Match inference-worker overlap merge weighting."""
        if length <= 1:
            return np.ones((max(1, length),), dtype=np.float32)
        w = np.hanning(length).astype(np.float32)
        if not np.any(w > 0):
            return np.ones((length,), dtype=np.float32)
        return np.clip(0.1 + 0.9 * w, 1e-3, None).astype(np.float32)

    def _get_precomputed_aggregated_probs(self, video_path: str = None):
        """Return cached worker-built frame probabilities for the active video when available."""
        v_path = video_path or self.video_path
        if not v_path or not isinstance(self.results_cache, dict):
            return None
        entry = self.results_cache.get(v_path, {})
        if not isinstance(entry, dict):
            return None
        precomputed = entry.get("aggregated_frame_probs")
        if isinstance(precomputed, list):
            try:
                precomputed = np.asarray(precomputed, dtype=np.float32)
            except Exception:
                return None
        if isinstance(precomputed, np.ndarray) and precomputed.ndim == 2 and precomputed.shape[0] > 0:
            return precomputed
        return None

    def _smooth_win_for_class(self, cls_idx: int) -> int:
        base = int(max(1, getattr(self, "_temporal_smoothing_window_frames", 1)))
        if 0 <= cls_idx < len(self.classes):
            base = int(self.class_smoothing_window_frames.get(self.classes[cls_idx], base))
        if base % 2 == 0:
            base += 1
        return max(1, base)

    def _gap_fill_for_class(self, cls_idx: int) -> int:
        base = int(max(0, getattr(self, "_merge_gap_frames", 0)))
        if 0 <= cls_idx < len(self.classes):
            base = int(self.class_merge_gap_frames.get(self.classes[cls_idx], base))
        return max(0, base)

    def _min_seg_for_class(self, cls_idx: int) -> int:
        base = int(max(1, getattr(self, "_min_segment_frames", 1)))
        if 0 <= cls_idx < len(self.classes):
            base = int(self.class_min_segment_frames.get(self.classes[cls_idx], base))
        return max(1, base)

    def _sync_per_class_segment_rule_config(self):
        self.config["inference_class_min_segment_frames"] = {
            cls: int(v) for cls, v in self.class_min_segment_frames.items()
        }
        self.config["inference_class_merge_gap_frames"] = {
            cls: int(v) for cls, v in self.class_merge_gap_frames.items()
        }
        self.config["inference_class_smoothing_window_frames"] = {
            cls: int(v) for cls, v in self.class_smoothing_window_frames.items()
        }

    def _json_safe_result_value(self, value):
        """Recursively convert numpy values into JSON-safe Python types."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            skip = {"clip_attention_maps"}
            return {str(k): self._json_safe_result_value(v) for k, v in value.items() if str(k) not in skip}
        if isinstance(value, (list, tuple)):
            return [self._json_safe_result_value(v) for v in value]
        return value

    def _arrays_sidecar_path(self, json_path: str) -> str:
        base, ext = os.path.splitext(json_path)
        if ext.lower() == ".json":
            return base + ".arrays.npz"
        return json_path + ".arrays.npz"

    def _coerce_external_result_array(self, key: str, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        else:
            return None
        if arr.size <= 0 or arr.dtype == object:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32, copy=False)
        return np.ascontiguousarray(arr)

    def _prepare_results_for_storage(self, results_cache: dict):
        heavy_keys = {"clip_probabilities", "clip_frame_probabilities", "aggregated_frame_probs"}
        json_results = {}
        external_arrays = {}
        for video_idx, (video_path, entry) in enumerate(results_cache.items()):
            if not isinstance(entry, dict):
                json_results[str(video_path)] = self._json_safe_result_value(entry)
                continue
            entry_json = {}
            entry_refs = {}
            for key, value in entry.items():
                if str(key) == "clip_attention_maps":
                    continue
                if key in heavy_keys:
                    arr = self._coerce_external_result_array(key, value)
                    if arr is not None:
                        store_key = f"video_{video_idx:05d}__{key}"
                        external_arrays[store_key] = arr
                        entry_refs[key] = store_key
                        continue
                entry_json[str(key)] = self._json_safe_result_value(value)
            if entry_refs:
                entry_json["_external_arrays"] = entry_refs
            json_results[str(video_path)] = entry_json
        return json_results, external_arrays

    def _write_results_bundle(self, results_path: str, payload: dict, *, pretty: bool):
        results_copy = dict(payload)
        json_results, external_arrays = self._prepare_results_for_storage(payload.get("results", {}) or {})
        results_copy["results"] = json_results
        sidecar_path = self._arrays_sidecar_path(results_path)
        if external_arrays:
            np.savez_compressed(sidecar_path, **external_arrays)
            results_copy["external_array_store"] = {
                "format": "npz",
                "file": os.path.basename(sidecar_path),
            }
        else:
            results_copy.pop("external_array_store", None)
            if os.path.exists(sidecar_path):
                try:
                    os.remove(sidecar_path)
                except Exception:
                    pass
        with open(results_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(results_copy, f, indent=2)
            else:
                json.dump(results_copy, f)

    def _restore_external_arrays(self, file_path: str, data: dict):
        if not isinstance(data, dict):
            return
        results = data.get("results", {})
        if not isinstance(results, dict):
            return
        store_info = data.get("external_array_store", {})
        sidecar_file = None
        if isinstance(store_info, dict):
            sidecar_file = store_info.get("file")
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
        except Exception as exc:
            self.log_text.append(f"Warning: Failed to load companion results arrays: {exc}")

    def _get_clips_dir(self) -> str:
        """Resolved clips directory from config; creates dir if needed."""
        clips_dir = self.config.get("clips_dir", "data/clips")
        if not os.path.isabs(clips_dir):
            exp_path = self.config.get("experiment_path")
            if exp_path:
                clips_dir = os.path.join(exp_path, clips_dir)
            else:
                clips_dir = os.path.abspath(clips_dir)
        os.makedirs(clips_dir, exist_ok=True)
        return clips_dir

    def _get_annotation_file(self) -> str:
        """Path to annotations JSON from config."""
        return self.config.get("annotation_file", "data/annotations/annotations.json")

    def _clip_path_to_id(self, clip_path: str, clips_dir: str) -> str:
        """Convert absolute clip path to annotation clip ID (relative, forward slashes)."""
        clip_id = os.path.relpath(clip_path, clips_dir).replace("\\", "/")
        if clip_id.startswith("../"):
            return os.path.basename(clip_path)
        for prefix in ("../clips/", "clips/", "data/clips/"):
            if clip_id.startswith(prefix):
                return clip_id[len(prefix):]
        return clip_id

    def _get_video_fps(self, video_path: str) -> float:
        """Return video FPS from path; 30.0 if unavailable or invalid."""
        if not video_path:
            return 30.0
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return float(fps) if fps and fps > 0 else 30.0
        except Exception:
            return 30.0

    def _video_basename(self) -> str:
        """Basename of current video path without extension."""
        if not self.video_path:
            return ""
        return os.path.splitext(os.path.basename(self.video_path))[0]

    def _get_timeline_qcolors(self) -> list:
        """Timeline palette as list of QColor for drawing."""
        return [QColor(r, g, b) for r, g, b in self._get_timeline_palette()]

    def _unique_clip_path(self, clip_path: str) -> str:
        """Return a path that does not exist by appending _1, _2, ... to stem."""
        base_clip_path = clip_path
        suffix = 1
        while os.path.exists(clip_path):
            base, ext = os.path.splitext(base_clip_path)
            clip_path = f"{base}_{suffix}{ext}"
            suffix += 1
        return clip_path

    def _merge_predictions(self, predictions, confidences, clip_starts):
        """Merge consecutive identical predictions."""
        if not predictions:
            return predictions, confidences, clip_starts
        
        merged_preds = []
        merged_confs = []
        merged_starts = []
        
        current_pred = predictions[0]
        current_conf = confidences[0]
        current_start = clip_starts[0]
        
        for i in range(1, len(predictions)):
            if predictions[i] == current_pred:
                # Same behavior, continue merging (keep max confidence)
                current_conf = max(current_conf, confidences[i])
            else:
                # Different behavior, save current segment and start new one
                merged_preds.append(current_pred)
                merged_confs.append(current_conf)
                merged_starts.append(current_start)
                
                current_pred = predictions[i]
                current_conf = confidences[i]
                current_start = clip_starts[i]
        
        # Add last segment
        merged_preds.append(current_pred)
        merged_confs.append(current_conf)
        merged_starts.append(current_start)
        
        return merged_preds, merged_confs, merged_starts
    
    def _draw_timeline(self):
        if not self.predictions or not self.classes:
            return

        frame_aggregation_enabled = self.frame_aggregation_check.isChecked()

        if frame_aggregation_enabled:
            if (not self.aggregated_segments
                    or not isinstance(self._aggregated_frame_scores_norm, np.ndarray)
                    or int(self._aggregated_last_covered_frame) <= 0):
                self._compute_aggregated_timeline()

        if frame_aggregation_enabled and self.aggregated_segments:
            self._draw_frame_aggregated_timeline()
            return

        self._draw_clip_based_timeline()
    
    def _filter_cooccurrence(self, probs, threshold):
        """Return set of class indices to display, respecting co-occurrence rules.

        When "Show all classes" is on, every class above threshold is returned.
        Otherwise: top-1 class is always shown, additional classes only if they
        form an allowed co-occurrence pair with the top-1 class.
        """
        show_all = getattr(self, "ovr_show_all_check", None) is not None and self.ovr_show_all_check.isChecked()
        n = min(len(probs), len(self.classes))
        scored = [(ci, float(probs[ci])) for ci in range(n) if float(probs[ci]) >= threshold]
        if not scored:
            return set()
        if show_all:
            return {ci for ci, _ in scored}
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ci = scored[0][0]
        top_name = self.classes[top_ci]
        active = {top_ci}
        for ci, sc in scored[1:]:
            name = self.classes[ci]
            if (top_name, name) in self._allowed_cooccurrence:
                active.add(ci)
        return active

    def _active_ovr_indices_from_scores(self, probs_row, threshold_override: float | None = None):
        """Active OvR class indices at one frame using thresholds.

        When "Show all classes" is enabled, every class above its threshold is
        returned (fully independent).  Otherwise the top-1 class is returned
        plus any class that forms an allowed co-occurrence pair with it.
        """
        show_all = getattr(self, "ovr_show_all_check", None) is not None and self.ovr_show_all_check.isChecked()
        n = min(len(probs_row), len(self.classes))
        scored = []
        for ci in range(n):
            s = float(probs_row[ci])
            thr = float(threshold_override) if threshold_override is not None else self._threshold_for_pred(ci)
            if s >= thr:
                scored.append((ci, s))
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)

        if show_all:
            return [ci for ci, _ in scored]

        top_ci = scored[0][0]
        if not self._allowed_cooccurrence:
            return [top_ci]
        top_name = self.classes[top_ci]
        active = [top_ci]
        for ci, _ in scored[1:]:
            name = self.classes[ci]
            if (top_name, name) in self._allowed_cooccurrence:
                active.append(ci)
        return active

    def _get_precise_active_for_frame(self, frame_idx: int):
        """Return [(class_idx, score), ...] active at frame_idx, sorted desc.
        
        For OvR: uses _aggregated_active_mask when available so that
        min-segment and gap-fill filtering are respected.
        """
        if not isinstance(self._aggregated_frame_scores_norm, np.ndarray):
            return []
        if frame_idx < 0 or frame_idx >= int(self._aggregated_last_covered_frame):
            return []
        scores = self._aggregated_frame_scores_norm[frame_idx]
        if self._use_ovr:
            # Prefer the active mask (has min-segment + gap-fill applied)
            if isinstance(self._aggregated_active_mask, np.ndarray) and frame_idx < self._aggregated_active_mask.shape[0]:
                mask_row = self._aggregated_active_mask[frame_idx]
                out = [(ci, float(scores[ci])) for ci in range(len(mask_row)) if mask_row[ci]]
            else:
                thr = None if self.use_ignore_threshold else 0.35
                active = self._active_ovr_indices_from_scores(scores, threshold_override=thr)
                out = [(ci, float(scores[ci])) for ci in active]
            out.sort(key=lambda x: x[1], reverse=True)
            return out
        if len(scores) == 0:
            return []
        ci = int(np.argmax(scores))
        if 0 <= ci < len(self.classes):
            if self.use_ignore_threshold and float(scores[ci]) < self._threshold_for_pred(ci):
                return []
            return [(ci, float(scores[ci]))]
        return []


    def _draw_frame_aggregated_timeline(self):
        """Render the frame-level aggregated timeline via the interactive QGraphicsView widget."""
        if not self.aggregated_segments or not self.classes:
            return

        from singlebehaviorlab.backend.segments import SegmentsModel

        total_frames = self.total_frames
        if total_frames <= 0 and self.video_path and os.path.exists(self.video_path):
            try:
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                pass
        if total_frames <= 0 and self.aggregated_segments:
            total_frames = self.aggregated_segments[-1]["end"] + 1

        orig_fps = self._get_video_fps(self.video_path) if self.video_path else 30.0

        model = SegmentsModel(self.aggregated_segments, self.classes, total_frames, orig_fps)
        self._segments_model = model

        colors = self._get_timeline_qcolors()
        px_per_sec = float(self.timeline_zoom_spin.value() if hasattr(self, "timeline_zoom_spin") else 100)
        ppf = max(0.5, px_per_sec / orig_fps)

        filter_idx = None
        selected_behavior = self.filter_behavior_combo.currentText()
        if selected_behavior != "All Behaviors":
            if selected_behavior == self.ignore_label_name:
                filter_idx = -1
            elif selected_behavior in self.classes:
                filter_idx = self.classes.index(selected_behavior)

        self._interactive_timeline.set_zoom(ppf)
        self._interactive_timeline.set_filter(filter_idx)
        self._interactive_timeline.set_model(model, colors)
    
    def _on_interactive_segment_clicked(self, seg_index: int):
        if not self._segments_model or seg_index < 0 or seg_index >= len(self._segments_model):
            return
        self._show_clip_popup(self._segments_model[seg_index].start, frame_mode=True)

    def _on_interactive_frame_clicked(self, frame_idx: int):
        self._show_clip_popup(frame_idx, frame_mode=True)

    def _on_segments_edited(self):
        if not self._segments_model:
            return
        self.aggregated_segments = self._segments_model.to_dicts()

    def _draw_clip_based_timeline(self):
        from singlebehaviorlab.backend.segments import SegmentsModel
        corrected_preds = self._effective_predictions()
        if not corrected_preds:
            return
        total_frames = self.total_frames or len(corrected_preds) * max(1, self.clip_length_spin.value())
        orig_fps = self._get_video_fps(self.video_path) if self.video_path else 30.0
        step = max(1, self.step_frames_spin.value())
        segments = []
        for i, (pred, conf) in enumerate(zip(corrected_preds, self.confidences)):
            start = self.clip_starts[i] if i < len(self.clip_starts) else i * step
            end = self.clip_starts[i + 1] if i + 1 < len(self.clip_starts) else start + self.clip_length_spin.value()
            end = min(end, total_frames)
            if pred >= 0:
                segments.append({"class": int(pred), "start": int(start), "end": int(end), "confidence": float(conf)})
        model = SegmentsModel(segments, self.classes, total_frames, orig_fps)
        self._segments_model = model
        colors = self._get_timeline_qcolors()
        px_per_sec = float(self.timeline_zoom_spin.value() if hasattr(self, "timeline_zoom_spin") else 100)
        ppf = max(0.5, px_per_sec / orig_fps)
        self._interactive_timeline.set_zoom(ppf)
        self._interactive_timeline.set_filter(None)
        self._interactive_timeline.set_model(model, colors)

    def _on_filter_changed(self, index: int):
        if self.predictions:
            self._draw_timeline()

    def _on_theme_changed(self, index: int):
        if self.predictions:
            self._draw_timeline()

    def _get_timeline_palette(self) -> list[tuple[int, int, int]]:
        theme = self.timeline_theme_combo.currentText() if hasattr(self, "timeline_theme_combo") else DEFAULT_THEME
        return get_timeline_palette(theme)

    def _get_attr_idx(self, clip_idx: int):
        if clip_idx in self.corrected_attr_labels:
            return self.corrected_attr_labels[clip_idx]
        if self.attr_predictions and clip_idx < len(self.attr_predictions):
            return self.attr_predictions[clip_idx]
        return None

    def _on_clip_length_changed(self, value: int):
        clip_length = int(value)
        self.step_frames_spin.blockSignals(True)
        self.step_frames_spin.setValue(max(1, clip_length // 2))
        self.step_frames_spin.blockSignals(False)
        self._on_step_or_clip_changed(self.step_frames_spin.value())

    def _on_step_or_clip_changed(self, value: int):
        step_frames = self.step_frames_spin.value()
        clip_length = self.clip_length_spin.value()
        if step_frames != clip_length:
            if not self.frame_aggregation_check.isChecked():
                self.frame_aggregation_check.setChecked(True)
                self.log_text.append(
                    f"Auto-enabled 'Precise frame boundaries' (step={step_frames} ≠ clip={clip_length})"
                )
    
    def _on_merge_changed(self, state: int):
        """Handle merge checkbox change."""
        if self.predictions:
            self._draw_timeline()
    
    def _on_frame_aggregation_changed(self, state: int):
        """Handle frame aggregation checkbox change."""
        if self.predictions:
            if state:
                self._compute_aggregated_timeline()
            self._draw_timeline()

    def _on_timeline_zoom_changed(self, *_args):
        """Redraw timeline when zoom level changes."""
        if self.predictions:
            self._draw_timeline()

    def _on_ovr_show_all_changed(self, *_args):
        """Toggle between co-occurrence-restricted and fully independent OvR display."""
        if self.predictions and self._use_ovr:
            if self.frame_aggregation_check.isChecked():
                self._build_timeline_segments()
            self._display_results()

    def _on_smoothing_changed(self, *_args):
        """Legacy hook: keep config synced when smoothing settings change."""
        self.config["inference_min_segment_frames"] = self._min_segment_frames
        self.config["inference_merge_gap_frames"] = self._merge_gap_frames
        self.config["inference_temporal_smoothing_window_frames"] = self._temporal_smoothing_window_frames
        self._sync_per_class_segment_rule_config()
        if self.predictions and self.frame_aggregation_check.isChecked():
            self._compute_aggregated_timeline()
            self._display_results()

    def _smooth_frame_labels(self, frame_labels: np.ndarray) -> np.ndarray:
        """Apply simple temporal smoothing to frame-wise top-1 labels."""
        if frame_labels.size == 0:
            return frame_labels

        labels = frame_labels.copy()
        T = int(labels.shape[0])

        if T > 1:
            majority_labels = labels.copy()
            for i in range(T):
                center = int(labels[i])
                if center < 0:
                    continue
                win = self._smooth_win_for_class(center)
                if win <= 1:
                    continue
                half = win // 2
                left = max(0, i - half)
                right = min(T, i + half + 1)
                window_vals = labels[left:right]
                valid_vals = window_vals[window_vals >= 0]
                if valid_vals.size == 0:
                    continue
                counts = np.bincount(valid_vals.astype(np.int64))
                max_count = int(np.max(counts))
                winners = np.where(counts == max_count)[0]
                if winners.size == 1:
                    majority_labels[i] = int(winners[0])
                else:
                    majority_labels[i] = center if center in winners else int(winners[0])
            labels = majority_labels
        return labels

    def _apply_gap_merge_and_min_seg(self, frame_labels: np.ndarray, T: int) -> np.ndarray:
        """Merge short gaps between identical classes and remove short runs.

        Operates on a copy so the caller's array is unchanged if not reassigned.
        Gap-merge: fills runs of -1 that are <= merge_gap_frames long when the
        class on both sides is the same.
        Min-segment: removes runs shorter than min_segment_frames by replacing
        them with their neighbour.
        """
        max_gap = int(max(0, getattr(self, "_merge_gap_frames", 0)))
        min_len = int(max(1, getattr(self, "_min_segment_frames", 1)))
        has_class_overrides = bool(
            self.class_merge_gap_frames
            or self.class_min_segment_frames
        )
        if max_gap == 0 and min_len <= 1 and not has_class_overrides:
            return frame_labels

        labels = frame_labels.copy()

        if max_gap > 0 or self.class_merge_gap_frames:
            i = 0
            while i < T:
                if labels[i] != -1:
                    i += 1
                    continue
                j = i
                while j + 1 < T and labels[j + 1] == -1:
                    j += 1
                gap_len = j - i + 1
                left = int(labels[i - 1]) if i > 0 else -1
                right = int(labels[j + 1]) if j + 1 < T else -1
                gap_thr = self._gap_fill_for_class(left) if left >= 0 and left == right else max_gap
                if gap_len <= gap_thr and left >= 0 and left == right:
                    labels[i:j + 1] = left
                i = j + 1

        if min_len > 1 or self.class_min_segment_frames:
            changed = True
            while changed:
                changed = False
                i = 0
                while i < T:
                    cls = int(labels[i])
                    j = i
                    while j + 1 < T and int(labels[j + 1]) == cls:
                        j += 1
                    run_len = j - i + 1
                    min_len_cls = self._min_seg_for_class(cls) if cls >= 0 else min_len
                    if cls >= 0 and run_len < min_len_cls:
                        left = int(labels[i - 1]) if i > 0 else -1
                        right = int(labels[j + 1]) if j + 1 < T else -1
                        if left >= 0 and right >= 0:
                            repl = left if left == right else left
                        elif left >= 0:
                            repl = left
                        elif right >= 0:
                            repl = right
                        else:
                            repl = -1
                        if repl != cls:
                            labels[i:j + 1] = repl
                            changed = True
                    i = j + 1

        return labels

    def _viterbi_decode_dense(self, probs: np.ndarray) -> np.ndarray:
        """Single-label Viterbi decode for a contiguous covered frame range."""
        if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0:
            return np.zeros((0,), dtype=np.int32)
        T, C = probs.shape
        if C == 1:
            return np.zeros((T,), dtype=np.int32)

        emissions = np.log(np.clip(probs.astype(np.float32, copy=False), 1e-8, 1.0))
        switch_penalty = float(max(0.0, self.viterbi_switch_penalty))
        trans = np.full((C, C), -switch_penalty, dtype=np.float32)
        np.fill_diagonal(trans, 0.0)

        dp = np.empty((T, C), dtype=np.float32)
        backptr = np.zeros((T, C), dtype=np.int32)
        dp[0] = emissions[0]

        for t in range(1, T):
            scores = dp[t - 1][:, np.newaxis] + trans
            backptr[t] = np.argmax(scores, axis=0).astype(np.int32)
            dp[t] = emissions[t] + scores[backptr[t], np.arange(C)]

        labels = np.zeros((T,), dtype=np.int32)
        labels[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            labels[t] = int(backptr[t + 1, labels[t + 1]])
        return labels

    def _decode_viterbi_labels(self, fs: np.ndarray, covered_mask: np.ndarray) -> np.ndarray:
        """Run Viterbi only on contiguous covered ranges, leaving gaps as -1."""
        T = int(fs.shape[0])
        decoded = np.full((T,), -1, dtype=np.int32)
        i = 0
        while i < T:
            if not bool(covered_mask[i]):
                i += 1
                continue
            j = i
            while j + 1 < T and bool(covered_mask[j + 1]):
                j += 1
            decoded[i:j + 1] = self._viterbi_decode_dense(fs[i:j + 1])
            i = j + 1
        return decoded

    def _binary_viterbi_decode(self, probs: np.ndarray, threshold: float) -> np.ndarray:
        """Binary Viterbi decode for one OvR class over a contiguous covered range."""
        p = np.clip(np.asarray(probs, dtype=np.float32).reshape(-1), 1e-6, 1.0 - 1e-6)
        T = int(p.shape[0])
        if T == 0:
            return np.zeros((0,), dtype=bool)

        tau = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
        emit_off = np.log(1.0 - p) - np.log(1.0 - tau)
        emit_on = np.log(p) - np.log(tau)
        switch_penalty = float(max(0.0, self.viterbi_switch_penalty))

        dp = np.empty((T, 2), dtype=np.float32)
        backptr = np.zeros((T, 2), dtype=np.int8)
        dp[0, 0] = emit_off[0]
        dp[0, 1] = emit_on[0]

        for t in range(1, T):
            stay_off = dp[t - 1, 0]
            on_to_off = dp[t - 1, 1] - switch_penalty
            if stay_off >= on_to_off:
                dp[t, 0] = emit_off[t] + stay_off
                backptr[t, 0] = 0
            else:
                dp[t, 0] = emit_off[t] + on_to_off
                backptr[t, 0] = 1

            off_to_on = dp[t - 1, 0] - switch_penalty
            stay_on = dp[t - 1, 1]
            if off_to_on >= stay_on:
                dp[t, 1] = emit_on[t] + off_to_on
                backptr[t, 1] = 0
            else:
                dp[t, 1] = emit_on[t] + stay_on
                backptr[t, 1] = 1

        states = np.zeros((T,), dtype=np.int8)
        states[-1] = 1 if dp[-1, 1] >= dp[-1, 0] else 0
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]
        return states.astype(bool)

    def _build_timeline_segments(self):
        """Build timeline segments from precomputed frame probabilities."""
        if not isinstance(self._aggregated_frame_scores_norm, np.ndarray):
            self.aggregated_segments = []
            self.aggregated_multiclass_segments = []
            self._aggregated_active_mask = None
            self._aggregated_last_covered_frame = 0
            return

        fs = self._aggregated_frame_scores_norm
        if fs.ndim != 2 or fs.shape[0] == 0 or fs.shape[1] == 0:
            self.aggregated_segments = []
            self.aggregated_multiclass_segments = []
            self._aggregated_active_mask = None
            self._aggregated_last_covered_frame = 0
            return

        T, C = fs.shape
        self._aggregated_last_covered_frame = T
        covered_mask = np.sum(fs, axis=1) > 1e-8
        if self.use_viterbi_decode and not self._use_ovr:
            frame_labels = self._decode_viterbi_labels(fs, covered_mask)
        else:
            frame_labels = np.argmax(fs, axis=1)
            frame_labels[~covered_mask] = -1
            # Smooth raw argmax labels (majority-vote temporal smoothing).
            frame_labels = self._smooth_frame_labels(frame_labels)

        # Apply ignore threshold after smoothing.
        if self.use_ignore_threshold and not self._use_ovr:
            for fi in range(T):
                ci = int(frame_labels[fi])
                if ci < 0:
                    continue
                thr = self._threshold_for_pred(ci)
                if float(fs[fi, ci]) < thr:
                    frame_labels[fi] = -1

        # Merge-gap and min-segment run ONCE as the final cleanup, after all
        # other preprocessing (smoothing + threshold) is done.
        frame_labels = self._apply_gap_merge_and_min_seg(frame_labels, T)

        segments = []
        cur_cls = int(frame_labels[0])
        cur_start = 0
        for i in range(1, T):
            if int(frame_labels[i]) != cur_cls:
                conf = float(np.mean(fs[cur_start:i, cur_cls])) if cur_cls >= 0 else 0.0
                segments.append({
                    "class": int(cur_cls),
                    "start": int(cur_start),
                    "end": int(i - 1),
                    "confidence": conf,
                })
                cur_cls = int(frame_labels[i])
                cur_start = i
        conf = float(np.mean(fs[cur_start:T, cur_cls])) if cur_cls >= 0 else 0.0
        segments.append({
            "class": int(cur_cls),
            "start": int(cur_start),
            "end": int(T - 1),
            "confidence": conf,
        })
        self.aggregated_segments = segments

        self.aggregated_multiclass_segments = []
        self._aggregated_active_mask = None
        if self._use_ovr:
            active_mask = np.zeros((T, C), dtype=bool)
            ovr_threshold = None if self.use_ignore_threshold else 0.35
            show_all = getattr(self, "ovr_show_all_check", None) is not None and self.ovr_show_all_check.isChecked()
            if self.use_viterbi_decode:
                for ci in range(C):
                    thr = self._threshold_for_pred(ci) if self.use_ignore_threshold else float(ovr_threshold)
                    active_mask[:, ci] = self._binary_viterbi_decode(fs[:, ci], thr)
                if not show_all:
                    pruned_mask = np.zeros_like(active_mask)
                    for fi in range(T):
                        active_idx = np.flatnonzero(active_mask[fi])
                        if active_idx.size == 0:
                            continue
                        scores = fs[fi, active_idx]
                        top_ci = int(active_idx[int(np.argmax(scores))])
                        pruned_mask[fi, top_ci] = True
                        if self._allowed_cooccurrence:
                            top_name = self.classes[top_ci]
                            for ci in active_idx:
                                ci = int(ci)
                                if ci == top_ci:
                                    continue
                                name = self.classes[ci]
                                if (top_name, name) in self._allowed_cooccurrence:
                                    pruned_mask[fi, ci] = True
                    active_mask = pruned_mask
            else:
                for fi in range(T):
                    for ci in self._active_ovr_indices_from_scores(fs[fi], threshold_override=ovr_threshold):
                        if 0 <= ci < C:
                            active_mask[fi, ci] = True
            self._aggregated_active_mask = active_mask

            max_gap = int(max(0, getattr(self, "_merge_gap_frames", 0)))
            min_len = int(max(1, getattr(self, "_min_segment_frames", 1)))

            # Per-class gap-fill and min-segment cleanup for OvR.
            for ci in range(C):
                col = active_mask[:, ci]
                smooth_win = self._smooth_win_for_class(ci)
                if (not self.use_viterbi_decode) and smooth_win > 1 and T > 1:
                    half = smooth_win // 2
                    smooth_col = col.copy()
                    for i in range(T):
                        left = max(0, i - half)
                        right = min(T, i + half + 1)
                        window = col[left:right]
                        on_count = int(np.count_nonzero(window))
                        off_count = int(window.size - on_count)
                        if on_count > off_count:
                            smooth_col[i] = True
                        elif off_count > on_count:
                            smooth_col[i] = False
                    col = smooth_col

                gap_thr = self._gap_fill_for_class(ci)
                min_len_cls = self._min_seg_for_class(ci)
                # Gap-fill: a short "off" gap between two "on" runs of the same class.
                if gap_thr > 0:
                    i = 0
                    while i < T:
                        if col[i]:
                            i += 1
                            continue
                        j = i
                        while j + 1 < T and not col[j + 1]:
                            j += 1
                        gap_len = j - i + 1
                        left_on = col[i - 1] if i > 0 else False
                        right_on = col[j + 1] if j + 1 < T else False
                        if gap_len <= gap_thr and left_on and right_on:
                            col[i:j + 1] = True
                        i = j + 1
                # Min-segment: remove short "on" runs.
                if min_len_cls > 1:
                    i = 0
                    while i < T:
                        if not col[i]:
                            i += 1
                            continue
                        j = i
                        while j + 1 < T and col[j + 1]:
                            j += 1
                        run_len = j - i + 1
                        if run_len < min_len_cls:
                            col[i:j + 1] = False
                        i = j + 1
                active_mask[:, ci] = col

            if not show_all:
                pruned_mask = np.zeros_like(active_mask)
                for fi in range(T):
                    active_idx = np.flatnonzero(active_mask[fi])
                    if active_idx.size == 0:
                        continue
                    scores = fs[fi, active_idx]
                    top_ci = int(active_idx[int(np.argmax(scores))])
                    pruned_mask[fi, top_ci] = True
                    if self._allowed_cooccurrence:
                        top_name = self.classes[top_ci]
                        for ci in active_idx:
                            ci = int(ci)
                            if ci == top_ci:
                                continue
                            name = self.classes[ci]
                            if (top_name, name) in self._allowed_cooccurrence:
                                pruned_mask[fi, ci] = True
                active_mask = pruned_mask
                self._aggregated_active_mask = active_mask

            multi_segments = []
            for ci in range(C):
                run_start = None
                for fi in range(T):
                    is_on = bool(active_mask[fi, ci])
                    if is_on and run_start is None:
                        run_start = fi
                    if run_start is not None and (not is_on or fi == T - 1):
                        run_end = fi if (is_on and fi == T - 1) else fi - 1
                        if run_end >= run_start:
                            conf = float(np.mean(fs[run_start:run_end + 1, ci]))
                            multi_segments.append({
                                "class": int(ci),
                                "start": int(run_start),
                                "end": int(run_end),
                                "confidence": conf,
                            })
                        run_start = None
            self.aggregated_multiclass_segments = multi_segments
    
    def _compute_aggregated_timeline(self):
        """
        Aggregate overlapping clip predictions into precise frame-level segments.
        Uses confidence-weighted voting: each clip votes for its predicted class
        with weight equal to its confidence score.
        """
        if not self.predictions or not self.classes:
            self.aggregated_segments = []
            self.aggregated_multiclass_segments = []
            self._aggregated_frame_scores_norm = None
            self._aggregated_active_mask = None
            self._aggregated_last_covered_frame = 0
            return

        # Prefer the exact worker-built merged frame timeline when available.
        # This preserves center-weighted overlap merge.
        if not self.corrected_labels:
            precomputed = self._get_precomputed_aggregated_probs(self.video_path)
            if precomputed is not None:
                self._aggregated_frame_scores_norm = precomputed.copy()
                self._aggregated_last_covered_frame = int(precomputed.shape[0])
                self._build_timeline_segments()
                return

        # Get parameters
        clip_length = self.clip_length_spin.value()
        step_frames = self.step_frames_spin.value()
        target_fps = self.target_fps_spin.value()
        num_classes = len(self.classes)
        
        # Get total frames from video or estimate from clip_starts
        if self.total_frames > 0:
            total_frames = self.total_frames
        elif self.clip_starts:
            # Estimate: last clip start + clip_length
            total_frames = self.clip_starts[-1] + clip_length + 1
        else:
            self.aggregated_segments = []
            self.aggregated_multiclass_segments = []
            self._aggregated_frame_scores_norm = None
            self._aggregated_active_mask = None
            self._aggregated_last_covered_frame = 0
            return
        
        orig_fps = self._get_video_fps(self.video_path) if self.video_path else 30.0
        frame_interval = self._get_saved_frame_interval(self.video_path, orig_fps)
        
        # Apply corrections + ignore-threshold gating
        corrected_preds = self._effective_predictions()
        
        # Initialize score matrix: [total_frames, num_classes] and coverage count
        frame_scores = np.zeros((total_frames, num_classes), dtype=np.float32)
        frame_coverage = np.zeros(total_frames, dtype=np.float32)
        used_full_probability_voting = False
        probs_available = (
            isinstance(self.clip_probabilities, list)
            and len(self.clip_probabilities) > 0
        )
        # Per-frame probabilities from FrameClassificationHead give much finer
        # temporal resolution: each frame within a clip gets its own prediction
        # instead of the entire clip being smeared with one probability vector.
        frame_probs_available = (
            isinstance(self.clip_frame_probabilities, list)
            and len(self.clip_frame_probabilities) > 0
        )
        
        # Accumulate votes from each clip
        for clip_i, (pred_class, conf, start_frame) in enumerate(zip(corrected_preds, self.confidences, self.clip_starts)):
            end_frame = min(start_frame + clip_length * frame_interval, total_frames)

            if start_frame >= total_frames:
                continue

            # Best path: per-frame probabilities from frame head. Do NOT gate this
            # by clip-level class/confidence, since a mixed clip can contain multiple
            # behaviors with low clip confidence but useful frame-wise predictions.
            if frame_probs_available and clip_i < len(self.clip_frame_probabilities):
                fp = self.clip_frame_probabilities[clip_i]
                if isinstance(fp, (list, np.ndarray)):
                    fp_arr = np.asarray(fp, dtype=np.float32)  # [T, C]
                    if fp_arr.ndim == 2 and fp_arr.shape[1] == num_classes:
                        T_clip = fp_arr.shape[0]
                        merge_w = self._build_center_merge_weights(T_clip)
                        for t in range(T_clip):
                            f_start = start_frame + t * frame_interval
                            f_end = min(f_start + frame_interval, total_frames)
                            if f_start >= total_frames:
                                break
                            if f_end <= f_start:
                                continue
                            probs_t = np.clip(fp_arr[t], 0.0, None)
                            w = float(merge_w[t])
                            frame_scores[f_start:f_end, :] += probs_t[np.newaxis, :] * w
                            frame_coverage[f_start:f_end] += w
                        used_full_probability_voting = True
                        continue

            # Remaining paths are clip-level and require a valid clip prediction.
            if not (0 <= pred_class < num_classes):
                continue

            # Fallback: clip-level probabilities smeared across the clip
            if probs_available and clip_i < len(self.clip_probabilities):
                raw_probs = self.clip_probabilities[clip_i]
                if isinstance(raw_probs, (list, tuple, np.ndarray)) and len(raw_probs) == num_classes:
                    probs_vec = np.asarray(raw_probs, dtype=np.float32)
                    if np.all(np.isfinite(probs_vec)):
                        probs_vec = np.clip(probs_vec, 0.0, None)
                        s = float(np.sum(probs_vec))
                        if s > 1e-8:
                            if not self._use_ovr:
                                probs_vec = probs_vec / s
                            if (not self._use_ovr) and int(np.argmax(probs_vec)) != int(pred_class):
                                probs_vec = np.zeros(num_classes, dtype=np.float32)
                                probs_vec[int(pred_class)] = 1.0
                            frame_scores[start_frame:end_frame, :] += probs_vec[np.newaxis, :]
                            frame_coverage[start_frame:end_frame] += 1.0
                            used_full_probability_voting = True
                            continue

            # Last fallback for older result files without per-class probabilities.
            frame_scores[start_frame:end_frame, pred_class] += float(conf)
            frame_coverage[start_frame:end_frame] += 1.0
        
        # Normalize scores by clip coverage so confidence stays in [0, 1]
        frame_scores_norm = np.divide(
            frame_scores,
            np.maximum(frame_coverage[:, np.newaxis], 1.0),
        )
        
        # For Softmax, ensure final probabilities sum to 1 (averaging usually does, but small errors can creep in)
        if not self._use_ovr:
            sums = np.sum(frame_scores_norm, axis=1, keepdims=True)
            valid_sums = sums > 1e-8
            frame_scores_norm = np.where(
                valid_sums,
                frame_scores_norm / sums,
                frame_scores_norm
            )
        
        # Find the last frame with actual clip coverage to avoid phantom
        # class-0 labels from uncovered tail frames (argmax of all-zeros = 0).
        # Interior uncovered frames (filtered clips) are marked -1 ("Filtered").
        # Use coverage count (number of clips that touched each frame) rather than
        # score sum, which can be > 0 for uncovered frames in OvR (sigmoid > 0 always).
        covered_mask = frame_coverage > 0
        if not np.any(covered_mask):
            self.aggregated_segments = []
            self.aggregated_multiclass_segments = []
            self._aggregated_frame_scores_norm = None
            self._aggregated_active_mask = None
            self._aggregated_last_covered_frame = 0
            return
        last_covered_frame = int(np.max(np.where(covered_mask))) + 1  # exclusive end
        self._aggregated_last_covered_frame = last_covered_frame
        self._aggregated_frame_scores_norm = frame_scores_norm[:last_covered_frame].copy()
        
        self._build_timeline_segments()
        
        # Log summary
        if self.aggregated_segments:
            self.log_text.append(f"Frame aggregation: {len(self.aggregated_segments)} segments from {len(self.predictions)} clips")
            if self._use_ovr and self.aggregated_multiclass_segments:
                self.log_text.append(
                    f"  OvR precise multi-class segments: {len(self.aggregated_multiclass_segments)}"
                )
            if used_full_probability_voting:
                mode_name = "sigmoid" if getattr(self, "_use_ovr", False) else "softmax"
                self.log_text.append(f"  Evidence mode: full {mode_name} probability voting")
            else:
                self.log_text.append("  Evidence mode: top-1 confidence voting")
            step_info = f"step={step_frames}, clip_length={clip_length}"
            if step_frames >= clip_length:
                self.log_text.append(f"  Note: No overlap ({step_info}). For better boundaries, use step_frames < clip_length.")
            else:
                overlap_pct = (1 - step_frames / clip_length) * 100
                self.log_text.append(f"  Overlap: {overlap_pct:.0f}% ({step_info})")
    
    def _export_timeline(self):
        """Export timeline as SVG/PDF and CSV with behavior segments."""
        if not self.predictions or not self.video_path:
            QMessageBox.warning(self, "Error", "No predictions available to export.")
            return
        
        self._persist_current_video_state()
        available_videos = [vp for vp in self.results_cache.keys() if isinstance(self.results_cache.get(vp), dict)]
        if not available_videos:
            available_videos = [self.video_path]

        export_selection = self._prompt_timeline_export_options(available_videos)
        if not export_selection:
            return
        selected_videos, selected_classes = export_selection

        if len(selected_videos) == 1:
            default_base = os.path.splitext(selected_videos[0])[0] + "_timeline"
            base_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Timeline",
                default_base,
                "All Files (*)"
            )
            if not base_path:
                return
            export_root = os.path.splitext(base_path)[0]
        else:
            default_dir = os.path.dirname(self.video_path) if self.video_path else os.getcwd()
            export_root = QFileDialog.getExistingDirectory(self, "Select Export Folder", default_dir)
            if not export_root:
                return

        original_video_path = self.video_path
        original_threshold_settings = self._current_threshold_settings()
        shared_threshold_settings = dict(original_threshold_settings)
        exported = []
        n_videos = len(selected_videos)
        show_progress = n_videos > 1
        progress = None
        export_canceled = False
        if show_progress:
            progress = QProgressDialog(
                "Exporting timeline 1 / {}...".format(n_videos),
                "Cancel",
                0,
                n_videos,
                self,
            )
            progress.setWindowTitle("Timeline export")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

        try:
            for vi, video_path in enumerate(selected_videos):
                if progress and progress.wasCanceled():
                    export_canceled = True
                    break
                if show_progress:
                    progress.setValue(vi)
                    progress.setLabelText(
                        "Exporting timeline {} / {}: {}".format(
                            vi + 1, n_videos, os.path.basename(video_path)
                        )
                    )
                    QApplication.processEvents()

                entry = self.results_cache.get(video_path, {})
                threshold_override = None
                if not isinstance(entry.get("threshold_settings"), dict):
                    threshold_override = shared_threshold_settings
                ok = self._load_video_from_cache(
                    video_path,
                    refresh_display=False,
                    persist_current=False,
                    threshold_settings_override=threshold_override,
                    persist_loaded_thresholds=False,
                )
                if not ok:
                    continue

                if len(selected_videos) == 1:
                    base_path = export_root
                else:
                    base_path = os.path.join(
                        export_root,
                        os.path.splitext(os.path.basename(video_path))[0] + "_timeline",
                    )
                csv_path, svg_path, mode_text = self._export_current_timeline_to_base_path(
                    base_path,
                    selected_classes=selected_classes,
                )
                exported.append((video_path, csv_path, svg_path, mode_text))
                if show_progress:
                    progress.setValue(vi + 1)
                    QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export timeline:\n{str(e)}")
            if progress:
                progress.close()
            return
        finally:
            if progress:
                progress.close()
            if original_video_path and original_video_path in self.results_cache:
                self._load_video_from_cache(
                    original_video_path,
                    refresh_display=True,
                    persist_current=False,
                    threshold_settings_override=original_threshold_settings,
                    persist_loaded_thresholds=False,
                )
                idx = self.filter_video_combo.findData(original_video_path)
                if idx >= 0:
                    self.filter_video_combo.blockSignals(True)
                    self.filter_video_combo.setCurrentIndex(idx)
                    self.filter_video_combo.blockSignals(False)

        if not exported:
            msg = "Export canceled." if export_canceled else "No timelines were exported."
            QMessageBox.warning(self, "Export", msg)
            return

        if len(exported) == 1:
            _, csv_path, svg_path, mode_text = exported[0]
            QMessageBox.information(
                self,
                "Export Complete",
                f"Timeline exported successfully!\n\n"
                f"Mode: {mode_text}\n"
                f"CSV: {csv_path}\n"
                f"SVG: {svg_path}"
            )
        else:
            QMessageBox.information(
                self,
                "Batch Export Complete",
                f"Exported timelines for {len(exported)} videos to:\n{export_root}"
            )

    def _prompt_timeline_export_options(self, available_videos):
        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Timeline Export")
        dlg.resize(760, 460)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose which videos and classes to export. All classes are included by default."))

        content_row = QHBoxLayout()

        video_col = QVBoxLayout()
        video_col.addWidget(QLabel("Videos"))
        video_button_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        current_btn = QPushButton("Current video")
        clear_btn = QPushButton("Clear")
        video_button_row.addWidget(select_all_btn)
        video_button_row.addWidget(current_btn)
        video_button_row.addWidget(clear_btn)
        video_col.addLayout(video_button_row)

        list_widget = QListWidget()
        current_video = self.video_path
        for vp in available_videos:
            item = QListWidgetItem(os.path.basename(vp))
            item.setData(Qt.ItemDataRole.UserRole, vp)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            default_checked = len(available_videos) == 1 or vp == current_video
            item.setCheckState(Qt.CheckState.Checked if default_checked else Qt.CheckState.Unchecked)
            list_widget.addItem(item)
        video_col.addWidget(list_widget, stretch=1)
        content_row.addLayout(video_col, stretch=1)

        class_col = QVBoxLayout()
        class_col.addWidget(QLabel("Classes"))
        class_button_row = QHBoxLayout()
        class_all_btn = QPushButton("All")
        class_none_btn = QPushButton("None")
        class_button_row.addWidget(class_all_btn)
        class_button_row.addWidget(class_none_btn)
        class_col.addLayout(class_button_row)

        class_list_widget = QListWidget()
        for cls_idx, cls_name in enumerate(self.classes):
            item = QListWidgetItem(cls_name)
            item.setData(Qt.ItemDataRole.UserRole, cls_idx)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked)
            class_list_widget.addItem(item)
        class_col.addWidget(class_list_widget, stretch=1)
        content_row.addLayout(class_col, stretch=1)

        layout.addLayout(content_row, stretch=1)

        def _set_all_items(state):
            for i in range(list_widget.count()):
                list_widget.item(i).setCheckState(state)

        def _set_all_classes(state):
            for i in range(class_list_widget.count()):
                class_list_widget.item(i).setCheckState(state)

        def _select_current_only():
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                is_current = item.data(Qt.ItemDataRole.UserRole) == current_video
                item.setCheckState(Qt.CheckState.Checked if is_current else Qt.CheckState.Unchecked)

        select_all_btn.clicked.connect(lambda: _set_all_items(Qt.CheckState.Checked))
        clear_btn.clicked.connect(lambda: _set_all_items(Qt.CheckState.Unchecked))
        current_btn.clicked.connect(_select_current_only)
        class_all_btn.clicked.connect(lambda: _set_all_classes(Qt.CheckState.Checked))
        class_none_btn.clicked.connect(lambda: _set_all_classes(Qt.CheckState.Unchecked))

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if not dlg.exec():
            return None

        selected = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.data(Qt.ItemDataRole.UserRole))

        if not selected:
            QMessageBox.information(self, "No videos selected", "Select at least one video to export.")
            return None

        selected_classes = set()
        for i in range(class_list_widget.count()):
            item = class_list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_classes.add(int(item.data(Qt.ItemDataRole.UserRole)))

        if not selected_classes:
            QMessageBox.information(self, "No classes selected", "Select at least one class to export.")
            return None

        return selected, selected_classes

    def _export_current_timeline_to_base_path(self, base_path: str, selected_classes: set[int] | None = None):
        if not self.predictions or not self.video_path:
            raise RuntimeError("No predictions available to export.")

        orig_fps = self._get_video_fps(self.video_path)
        frame_interval = self._get_saved_frame_interval(self.video_path, orig_fps)
        clip_length = self.clip_length_spin.value()
        frame_aggregation_enabled = self.frame_aggregation_check.isChecked()

        csv_path = base_path + "_behaviors.csv"
        import csv

        if frame_aggregation_enabled and self.aggregated_segments:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Behavior', 'Start Time (s)', 'End Time (s)', 'Start Frame', 'End Frame', 'Duration (s)', 'Confidence'])

                seg_source = self.aggregated_segments
                if self._use_ovr and self.aggregated_multiclass_segments:
                    seg_source = self.aggregated_multiclass_segments

                for seg in seg_source:
                    pred_idx = seg['class']
                    if pred_idx < len(self.classes) and (selected_classes is None or pred_idx in selected_classes):
                        behavior = self.classes[pred_idx]
                        start_frame = seg['start']
                        end_frame = seg['end']
                        conf = seg.get('confidence', 1.0)

                        start_time = start_frame / orig_fps
                        end_time = (end_frame + 1) / orig_fps
                        duration = end_time - start_time

                        writer.writerow([
                            behavior,
                            f"{start_time:.3f}",
                            f"{end_time:.3f}",
                            start_frame,
                            end_frame,
                            f"{duration:.3f}",
                            f"{conf:.3f}"
                        ])

            svg_path = base_path + "_timeline.svg"
            self._export_frame_aggregated_svg(svg_path, orig_fps, selected_classes=selected_classes)
            mode_text = "frame-aggregated (precise boundaries)"
        else:
            corrected_preds = self._effective_predictions()

            if self.merge_timeline_check.isChecked():
                display_preds, display_confs, display_starts = self._merge_predictions(
                    corrected_preds, self.confidences, self.clip_starts
                )
            else:
                display_preds = corrected_preds
                display_confs = self.confidences
                display_starts = self.clip_starts

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Behavior', 'Start Time (s)', 'End Time (s)', 'Start Frame', 'End Frame', 'Confidence'])

                for i, (pred_idx, conf, start_frame) in enumerate(zip(display_preds, display_confs, display_starts)):
                    if pred_idx < len(self.classes) and pred_idx >= 0 and (selected_classes is None or pred_idx in selected_classes):
                        behavior = self.classes[pred_idx]
                        if i < len(display_starts) - 1:
                            end_frame = display_starts[i + 1]
                        else:
                            end_frame = start_frame + (clip_length * frame_interval)
                        start_time = start_frame / orig_fps
                        end_time = end_frame / orig_fps
                        writer.writerow([behavior, f"{start_time:.3f}", f"{end_time:.3f}", start_frame, end_frame, f"{conf:.3f}"])

            svg_path = base_path + "_timeline.svg"
            self._export_timeline_svg(
                svg_path,
                display_preds,
                display_confs,
                display_starts,
                orig_fps,
                frame_interval,
                clip_length,
                selected_classes=selected_classes,
            )
            mode_text = "clip-based"

        return csv_path, svg_path, mode_text
    
    def _export_timeline_svg(self, svg_path, display_preds, display_confs, display_starts, orig_fps, frame_interval, clip_length, selected_classes: set[int] | None = None):
        """Export timeline as SVG."""
        num_segments = len(display_preds)
        if num_segments == 0:
            return
        
        # Calculate dimensions
        base_clip_width = 20
        total_clips = len(self.predictions)
        width = max(1200, total_clips * base_clip_width)
        height = 80
        
        colors = self._get_timeline_palette()
        
        with open(svg_path, 'w') as f:
            f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
            f.write(f'<rect width="{width}" height="{height}" fill="white"/>\n')
            
            x_pos = 0
            for seg_idx, (pred_idx, conf) in enumerate(zip(display_preds, display_confs)):
                if seg_idx < len(display_starts) - 1:
                    seg_start_clip = display_starts[seg_idx]
                    seg_end_clip = display_starts[seg_idx + 1]
                    clip_count = sum(1 for orig_start in self.clip_starts if seg_start_clip <= orig_start < seg_end_clip)
                    seg_width = clip_count * base_clip_width
                else:
                    seg_start_clip = display_starts[seg_idx]
                    clip_count = sum(1 for orig_start in self.clip_starts if orig_start >= seg_start_clip)
                    seg_width = clip_count * base_clip_width

                if pred_idx < len(self.classes) and (selected_classes is None or pred_idx in selected_classes):
                    color = colors[pred_idx % len(colors)]
                    r, g, b = color
                    alpha = conf
                    
                    f.write(f'<rect x="{x_pos}" y="0" width="{seg_width}" height="{height}" '
                           f'fill="rgb({r},{g},{b})" opacity="{alpha:.2f}"/>\n')
                    
                    if seg_width >= 30:
                        behavior = self.classes[pred_idx]
                        f.write(f'<text x="{x_pos + 5}" y="{height//2 + 5}" font-family="Arial" font-size="10">{behavior}</text>\n')

                x_pos += seg_width
            
            f.write('</svg>\n')
    
    def _export_frame_aggregated_svg(self, svg_path: str, orig_fps: float, selected_classes: set[int] | None = None):
        """Export frame-aggregated timeline as SVG with precise boundaries."""
        if not self.aggregated_segments:
            return

        use_multiclass_rows = bool(self._use_ovr and self.aggregated_multiclass_segments)
        segments = self.aggregated_multiclass_segments if use_multiclass_rows else self.aggregated_segments
        row_classes = [ci for ci in range(len(self.classes)) if selected_classes is None or ci in selected_classes]
        if not row_classes:
            return
        row_lookup = {cls_idx: row_idx for row_idx, cls_idx in enumerate(row_classes)}
        total_frames = self.total_frames
        if total_frames <= 0 and self.video_path and os.path.exists(self.video_path):
            try:
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                pass
        if total_frames <= 0 and segments:
            total_frames = segments[-1]['end'] + 1
        
        # Calculate dimensions
        min_width = 1200
        max_width = 4000
        pixels_per_frame = max(0.1, min(2.0, max_width / max(1, total_frames)))
        width = max(min_width, int(total_frames * pixels_per_frame))
        if use_multiclass_rows:
            row_h = 20
            height = max(80, row_h * len(row_classes))
        else:
            row_h = 80
            height = 80
        
        colors = self._get_timeline_palette()
        
        with open(svg_path, 'w') as f:
            f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
            f.write(f'<rect width="{width}" height="{height}" fill="white"/>\n')
            
            for seg in segments:
                pred_idx = seg['class']
                conf = seg.get('confidence', 1.0)
                start_frame = seg['start']
                end_frame = seg['end']
                
                if pred_idx >= len(self.classes) or (selected_classes is not None and pred_idx not in selected_classes):
                    continue
                
                x_start = int(start_frame * pixels_per_frame)
                x_end = int((end_frame + 1) * pixels_per_frame)
                seg_width = max(1, x_end - x_start)
                
                color = colors[pred_idx % len(colors)]
                r, g, b = color
                # Normalize confidence for opacity
                alpha = min(1.0, 0.5 + 0.5 * min(1.0, conf))
                
                y0 = (row_lookup[pred_idx] * row_h) if use_multiclass_rows else 0
                h0 = row_h if use_multiclass_rows else height
                f.write(f'<rect x="{x_start}" y="{y0}" width="{seg_width}" height="{h0}" '
                       f'fill="rgb({r},{g},{b})" opacity="{alpha:.2f}"/>\n')
                
                # Draw boundary line
                f.write(f'<line x1="{x_start}" y1="{y0}" x2="{x_start}" y2="{y0 + h0}" '
                       f'stroke="black" stroke-width="0.5" opacity="0.3"/>\n')
                
                if seg_width >= 40:
                    behavior = self.classes[pred_idx]
                    start_time = start_frame / orig_fps
                    end_time = (end_frame + 1) / orig_fps
                    ty = y0 + min(15, h0 - 5)
                    f.write(f'<text x="{x_start + 3}" y="{ty}" font-family="Arial" font-size="10">{behavior}</text>\n')
                    if not use_multiclass_rows and h0 >= 30:
                        f.write(f'<text x="{x_start + 3}" y="30" font-family="Arial" font-size="8" fill="gray">'
                               f'{start_time:.2f}s-{end_time:.2f}s</text>\n')
            
            # Add time axis markers
            duration_sec = total_frames / orig_fps
            tick_interval = max(1, int(duration_sec / 20))  # ~20 ticks
            for t in range(0, int(duration_sec) + 1, tick_interval):
                x = int(t * orig_fps * pixels_per_frame)
                if x < width:
                    f.write(f'<line x1="{x}" y1="{height-10}" x2="{x}" y2="{height}" stroke="black" stroke-width="1"/>\n')
                    f.write(f'<text x="{x}" y="{height-2}" font-family="Arial" font-size="8" text-anchor="middle">{t}s</text>\n')
            
            f.write('</svg>\n')
    
    def _show_clip_popup(self, idx: int, frame_mode: bool = False, ovr_class_idx: int = -1):
        """Show popup dialog with clip video, label, and confidence.
        
        Args:
            idx: Clip index (clip mode) or frame index (frame mode)
            frame_mode: If True, idx is a frame index; find the segment and show video at that frame
            ovr_class_idx: When clicking a per-class OvR row, the class index of the
                clicked row. If >= 0, the popup shows this specific class instead of the
                top-1 class from the aggregated timeline.
        """
        if not self.video_path:
            return
        
        # In frame mode, find the corresponding segment and show info for that frame
        if frame_mode and self.aggregated_segments:
            frame_idx = idx

            # If a specific OvR class row was clicked, build a virtual segment
            # for that class at this frame position so the user can inspect it.
            if ovr_class_idx >= 0 and ovr_class_idx < len(self.classes):
                segment = self._build_ovr_class_segment(frame_idx, ovr_class_idx)
                if segment is not None:
                    self._show_frame_segment_popup(frame_idx, segment)
                    return
            
            # Default: find segment from the main aggregated timeline
            segment = None
            for seg in self.aggregated_segments:
                if seg['start'] <= frame_idx <= seg['end']:
                    segment = seg
                    break
            
            if segment is None:
                return
            
            pred_idx = segment['class']
            conf = segment.get('confidence', 1.0)
            
            if pred_idx >= len(self.classes):
                return
            
            label = self.classes[pred_idx]
            
            # Show frame-specific popup
            self._show_frame_segment_popup(frame_idx, segment)
            return
        
        # Original clip-based mode
        clip_idx = idx
        if clip_idx >= len(self.predictions):
            return
        ClipPopupDialog(self, self, clip_idx)

    def _build_ovr_class_segment(self, frame_idx: int, class_idx: int) -> dict | None:
        """Build a virtual segment for a specific OvR class around frame_idx.

        Finds the contiguous run of active frames for class_idx that contains
        frame_idx, using the stored per-frame scores / active mask.
        Returns a segment dict compatible with FrameSegmentPopupDialog, or None.
        """
        active_mask = getattr(self, "_aggregated_active_mask", None)
        frame_scores = getattr(self, "_aggregated_frame_scores_norm", None)
        if not isinstance(active_mask, np.ndarray) or class_idx >= active_mask.shape[1]:
            return None
        total_frames = active_mask.shape[0]
        if frame_idx < 0 or frame_idx >= total_frames:
            return None

        is_active = bool(active_mask[frame_idx, class_idx])
        # Even if the frame isn't active for this class, still show a
        # single-frame segment so the user can inspect its score.
        if not is_active:
            conf = float(frame_scores[frame_idx, class_idx]) if isinstance(frame_scores, np.ndarray) else 0.0
            return {"class": class_idx, "start": frame_idx, "end": frame_idx, "confidence": conf}

        # Expand outward to find the contiguous active run
        start = frame_idx
        while start > 0 and bool(active_mask[start - 1, class_idx]):
            start -= 1
        end = frame_idx
        while end < total_frames - 1 and bool(active_mask[end + 1, class_idx]):
            end += 1

        conf = float(np.mean(frame_scores[start:end + 1, class_idx])) if isinstance(frame_scores, np.ndarray) else 1.0
        return {"class": class_idx, "start": start, "end": end, "confidence": conf}

    def _show_frame_segment_popup(self, frame_idx: int, segment: dict, segment_idx: int = None):
        """Show popup for a frame-aggregated segment.
        
        Args:
            frame_idx: The specific frame that was clicked
            segment: The segment dict with 'class', 'start', 'end', 'confidence'
            segment_idx: Index of the segment in self.aggregated_segments (for navigation)
        """
        if not self.video_path:
            return
        
        # Find segment index if not provided
        if segment_idx is None:
            for i, seg in enumerate(self.aggregated_segments):
                if seg['start'] == segment['start'] and seg['end'] == segment['end']:
                    segment_idx = i
                    break

        FrameSegmentPopupDialog(self, self, frame_idx, segment, segment_idx)

    def _export_video_with_overlay(self):
        """Export video with configurable overlays (delegates to overlay_export module)."""
        from .overlay_export import run_export_video_with_overlay
        run_export_video_with_overlay(self)

    def _preview_video_with_overlay(self):
        """Open video player to preview the exported video with overlays."""
        from .overlay_export import run_preview_video_with_overlay
        run_preview_video_with_overlay(self)

    def _export_attention_heatmap(self):
        """Export video with spatial attention heatmap overlay."""
        from .attention_export import export_attention_heatmap_video
        export_attention_heatmap_video(self)

    def update_config(self, config: dict):
        """Apply a new configuration (experiment switch)."""
        self.config = config
        self.model = None
        self.classes = []
        self.model_path_edit.clear()
        self.video_path = None
        self.video_path_edit.clear()
        self.video_info_label.setText("No video selected")
        self.run_inference_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.export_timeline_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.results_list.clear()
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.predictions = []
        self.confidences = []
        self.clip_probabilities = []
        self.clip_frame_probabilities = []
        self.clip_starts = []
        self.corrected_labels = {}
        self.corrected_attr_labels = {}
        self.aggregated_segments = []
        self.aggregated_multiclass_segments = []
        self._aggregated_frame_scores_norm = None
        self._aggregated_active_mask = None
        self._aggregated_last_covered_frame = 0
        
        self._interactive_timeline._scene.clear()
        self._interactive_timeline._items.clear()

