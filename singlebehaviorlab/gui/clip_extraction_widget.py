import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QSpinBox, QProgressBar, QFileDialog, QGroupBox, QFormLayout, QMessageBox, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import os
import json

logger = logging.getLogger(__name__)
from singlebehaviorlab.backend.video_utils import extract_clips, get_video_info
from singlebehaviorlab.backend.model import VideoPrismBackbone

BACKBONE_MODELS = [
    "videoprism_public_v1_base",
    "videoprism_public_v1_large",
    "videoprism_public_v1_small",
    "videoprism_public_v1_huge",
    "videoprism_mouse_v1_base",
    "videoprism_mouse_v1_large",
]

class ModelLoadWorker(QThread):
    """Worker thread for loading/downloading VideoPrism model."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            _ = VideoPrismBackbone(model_name=self.model_name)
            self.finished.emit(self.model_name)
        except Exception as e:
            self.error.emit(str(e))



class ClipExtractionWorker(QThread):
    """Worker thread for clip extraction."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(int, str, bool)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, output_dir, target_fps, clip_length, step_frames):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.clip_length = clip_length
        self.step_frames = step_frames
        self.should_stop = False

    def stop(self):
        """Request extraction stop."""
        self.should_stop = True
    
    def run(self):
        try:
            def progress_cb(current, total):
                self.progress.emit(current, total)
            
            num_clips, output_dir = extract_clips(
                self.video_path,
                self.output_dir,
                self.target_fps,
                self.clip_length,
                self.step_frames,
                progress_cb,
                stop_callback=lambda: self.should_stop,
            )
            
            import json
            meta_path = os.path.join(output_dir, "clips_metadata.json")
            try:
                with open(meta_path, 'w') as f:
                    json.dump({
                        "target_fps": self.target_fps,
                        "clip_length": self.clip_length,
                        "step_frames": self.step_frames
                    }, f, indent=2)
            except Exception as e:
                logger.warning("Failed to save clips metadata: %s", e)
                
            self.finished.emit(num_clips, output_dir, self.should_stop)
        except Exception as e:
            self.error.emit(str(e))


class ClipExtractionWidget(QWidget):
    """Widget for extracting 16-frame clips from videos."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.video_path = ""
        self.video_paths = []
        self.worker = None
        self._setup_ui()
    
    def update_config(self, config: dict):
        """Update configuration (called when experiments change)."""
        self.config = config
    
    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()
        
        video_group = QGroupBox("Video selection")
        video_layout = QVBoxLayout()
        
        single_video_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Select single video...")
        self.browse_btn.clicked.connect(self._browse_video)
        single_video_layout.addWidget(QLabel("Single video:"))
        single_video_layout.addWidget(self.video_path_edit)
        single_video_layout.addWidget(self.browse_btn)
        video_layout.addLayout(single_video_layout)
        
        multiple_video_layout = QHBoxLayout()
        self.video_list_label = QLabel("No videos selected")
        self.browse_multiple_btn = QPushButton("Select multiple videos...")
        self.browse_multiple_btn.clicked.connect(self._browse_multiple_videos)
        self.clear_videos_btn = QPushButton("Clear")
        self.clear_videos_btn.clicked.connect(self._clear_videos)
        self.clear_videos_btn.setEnabled(False)
        multiple_video_layout.addWidget(QLabel("Multiple videos:"))
        multiple_video_layout.addWidget(self.video_list_label)
        multiple_video_layout.addWidget(self.browse_multiple_btn)
        multiple_video_layout.addWidget(self.clear_videos_btn)
        video_layout.addLayout(multiple_video_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        info_group = QGroupBox("Video info")
        info_layout = QFormLayout()
        self.info_label = QLabel("No video selected")
        info_layout.addRow("Info:", self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        model_group = QGroupBox("Backbone model")
        model_layout = QHBoxLayout()
        
        model_layout.addWidget(QLabel("Select model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(BACKBONE_MODELS)
        current_model = self.config.get("backbone_model", "videoprism_public_v1_base")
        index = self.model_combo.findText(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        else:
             self.model_combo.addItem(current_model)
             self.model_combo.setCurrentText(current_model)
             
        model_layout.addWidget(self.model_combo)
        
        self.load_model_btn = QPushButton("Load/download model")
        self.load_model_btn.clicked.connect(self._load_backbone)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        params_group = QGroupBox("Extraction parameters")
        params_layout = QFormLayout()
        
        self.target_fps_spin = QSpinBox()
        self.target_fps_spin.setRange(1, 99999)
        self.target_fps_spin.setValue(16)
        params_layout.addRow("Target FPS:", self.target_fps_spin)
        
        self.clip_length_spin = QSpinBox()
        self.clip_length_spin.setRange(1, 64)
        self.clip_length_spin.setValue(16)
        params_layout.addRow("Frames per clip:", self.clip_length_spin)
        
        self.step_frames_spin = QSpinBox()
        self.step_frames_spin.setRange(1, 64)
        self.step_frames_spin.setValue(16)
        params_layout.addRow("Step frames:", self.step_frames_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Auto-generated from video name")
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(QLabel("Output directory:"))
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        button_layout = QHBoxLayout()
        self.extract_btn = QPushButton("Extract clips")
        self.extract_btn.clicked.connect(self._extract_clips)
        self.extract_btn.setEnabled(False)
        button_layout.addWidget(self.extract_btn)

        self.stop_extract_btn = QPushButton("Stop extraction")
        self.stop_extract_btn.clicked.connect(self._stop_extraction)
        self.stop_extract_btn.setEnabled(False)
        button_layout.addWidget(self.stop_extract_btn)
        
        self.extract_multiple_btn = QPushButton("Extract from all selected videos")
        self.extract_multiple_btn.clicked.connect(self._extract_multiple_videos)
        self.extract_multiple_btn.setEnabled(False)
        button_layout.addWidget(self.extract_multiple_btn)
        
        layout.addLayout(button_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def _load_backbone(self):
        """Load/Download selected backbone model."""
        model_name = self.model_combo.currentText()
        
        reply = QMessageBox.question(
            self, 
            "Load Model", 
            f"Load backbone model '{model_name}'?\n\n"
            "If this model is not cached, it will be downloaded (this may take a while and requires internet).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            return
            
        self.load_model_btn.setEnabled(False)
        self.status_label.setText(f"Loading/Downloading {model_name}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        self.model_worker = ModelLoadWorker(model_name)
        self.model_worker.finished.connect(self._on_model_loaded)
        self.model_worker.error.connect(self._on_model_error)
        self.model_worker.start()
        
    def _on_model_loaded(self, model_name: str):
        """Handle successful model load."""
        self.load_model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText(f"Model '{model_name}' loaded successfully")
        
        self.config['backbone_model'] = model_name
        
        QMessageBox.information(self, "Success", f"VideoPrism backbone '{model_name}' is ready to use.")
        
    def _on_model_error(self, error_msg: str):
        """Handle model load error."""
        self.load_model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText(f"Error loading model: {error_msg}")
        QMessageBox.critical(self, "Error", f"Failed to load model:\n{error_msg}")
    
    def _browse_video(self):
        """Browse for video file."""
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos")),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if video_path:
            from .video_utils import ensure_video_in_experiment
            video_path = ensure_video_in_experiment(video_path, self.config, self)
            self.set_video_path(video_path)
    
    def set_video_path(self, video_path: str):
        """Set video path and update UI."""
        self.video_path = video_path
        self.video_path_edit.setText(video_path)
        
        info = get_video_info(video_path)
        if info:
            info_text = f"FPS: {info['fps']:.2f}, Frames: {info['frame_count']}, Size: {info['width']}x{info['height']}"
            self.info_label.setText(info_text)
            # Default target FPS to native video FPS
            native_fps = int(round(info['fps']))
            self.target_fps_spin.setValue(max(1, min(native_fps, self.target_fps_spin.maximum())))
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clips_dir = os.path.join(self.config.get("clips_dir", "data/clips"), video_name)
        self.output_path_edit.setText(clips_dir)
        
        self.extract_btn.setEnabled(True)
    
    def _browse_multiple_videos(self):
        """Browse for multiple video files."""
        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos")),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if video_paths:
            # Ensure videos are in experiment folder (batch operation)
            from .video_utils import ensure_videos_in_experiment
            self.video_paths = ensure_videos_in_experiment(video_paths, self.config, self)
            if len(video_paths) == 1:
                self.video_list_label.setText(f"1 video selected: {os.path.basename(video_paths[0])}")
            else:
                self.video_list_label.setText(f"{len(video_paths)} videos selected")
            self.clear_videos_btn.setEnabled(True)
            self.extract_multiple_btn.setEnabled(True)
            # Default target FPS to first video's native FPS
            first_info = get_video_info(self.video_paths[0])
            if first_info:
                native_fps = int(round(first_info['fps']))
                self.target_fps_spin.setValue(max(1, min(native_fps, self.target_fps_spin.maximum())))
    
    def _clear_videos(self):
        """Clear selected videos."""
        self.video_paths = []
        self.video_list_label.setText("No videos selected")
        self.clear_videos_btn.setEnabled(False)
        self.extract_multiple_btn.setEnabled(False)
    
    def _extract_multiple_videos(self):
        """Extract clips from multiple videos."""
        if not self.video_paths:
            QMessageBox.warning(self, "Error", "Please select at least one video file.")
            return
        
        output_base_dir = self.output_path_edit.text().strip()
        if not output_base_dir:
            output_base_dir = self.config.get("clips_dir", "data/clips")
        
        target_fps = self.target_fps_spin.value()
        clip_length = self.clip_length_spin.value()
        step_frames = self.step_frames_spin.value()
        
        if step_frames > clip_length:
            QMessageBox.warning(self, "Error", "Step frames cannot be greater than clip length.")
            return
        
        self.extract_multiple_btn.setEnabled(False)
        self.extract_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Processing {len(self.video_paths)} videos...")
        
        total_clips = 0
        for i, video_path in enumerate(self.video_paths):
            if not os.path.exists(video_path):
                self.status_label.setText(f"Skipping invalid video: {video_path}")
                continue
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(output_base_dir, video_name)
            
            self.status_label.setText(f"Processing video {i+1}/{len(self.video_paths)}: {video_name}")
            
            try:
                def progress_cb(current, total):
                    overall_progress = int(100 * (i + current / max(total, 1)) / len(self.video_paths))
                    self.progress_bar.setValue(overall_progress)
                
                num_clips, _ = extract_clips(
                    video_path,
                    output_dir,
                    target_fps,
                    clip_length,
                    step_frames,
                    progress_cb
                )
                total_clips += num_clips
            except Exception as e:
                self.status_label.setText(f"Error processing {video_name}: {str(e)}")
                continue
        
        self.extract_multiple_btn.setEnabled(True)
        self.extract_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Extracted {total_clips} clips from {len(self.video_paths)} videos")
        QMessageBox.information(
            self, 
            "Success", 
            f"Extracted {total_clips} clips from {len(self.video_paths)} videos successfully!"
        )
    
    def _browse_output(self):
        """Browse for output directory."""
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.config.get("clips_dir", "data/clips")
        )
        if output_dir:
            self.output_path_edit.setText(output_dir)
    
    def _extract_clips(self):
        """Start clip extraction."""
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Error", "Please select a valid video file.")
            return
        
        output_dir = self.output_path_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please specify output directory.")
            return
        
        target_fps = self.target_fps_spin.value()
        clip_length = self.clip_length_spin.value()
        step_frames = self.step_frames_spin.value()
        
        if step_frames > clip_length:
            QMessageBox.warning(self, "Error", "Step frames cannot be greater than clip length.")
            return
        
        self.extract_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Extracting clips...")
        
        self.worker = ClipExtractionWorker(
            self.video_path,
            output_dir,
            target_fps,
            clip_length,
            step_frames
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self.stop_extract_btn.setEnabled(True)
    
    def _stop_extraction(self):
        """Stop currently running single-video extraction."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.stop_extract_btn.setEnabled(False)
            self.status_label.setText("Stopping extraction...")
    
    def _on_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            progress = int(100 * current / total)
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"Extracted {current} clips...")
    
    def _on_finished(self, num_clips: int, output_dir: str, cancelled: bool):
        """Handle extraction completion."""
        self.extract_btn.setEnabled(True)
        self.stop_extract_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if cancelled:
            self.status_label.setText(f"Stopped. Extracted {num_clips} clips to {output_dir}")
            QMessageBox.information(self, "Stopped", f"Extraction stopped.\n\nSaved {num_clips} clips.")
        else:
            self.status_label.setText(f"Extracted {num_clips} clips to {output_dir}")
            QMessageBox.information(self, "Success", f"Extracted {num_clips} clips successfully!")
    
    def _on_error(self, error_msg: str):
        """Handle extraction error."""
        self.extract_btn.setEnabled(True)
        self.stop_extract_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Extraction failed:\n{error_msg}")

