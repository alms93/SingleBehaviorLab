"""Registration widget for processing videos with masks from segmentation."""
import sys
import os
import glob
# JAX memory: grow on demand, capped at 45% so PyTorch (SAM2) keeps the rest
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import cv2
import numpy as np
import pandas as pd
import torch
import re
# Add parent directory to path for backend imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QComboBox, QProgressBar,
    QTextEdit, QMessageBox, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    QSlider, QApplication, QCheckBox, QProgressDialog
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from singlebehaviorlab.backend.video_processor import process_video, process_video_to_clips, load_segmentation_data
from singlebehaviorlab.backend.model import VideoPrismBackbone


class VideoPlayerDialog(QDialog):
    """Dialog for playing video files using OpenCV with streaming playback (no full caching)."""
    def __init__(self, video_paths: list, start_index: int = 0, parent=None):
        super().__init__(parent)
        self.video_paths = video_paths
        self.current_video_idx = start_index
        
        self.cap = None  # VideoCapture for streaming
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)
        self.slider_pressed = False
        self._current_frame = None  # Keep reference for QImage
        
        self._setup_ui()
        self._load_current_video()
    
    def _setup_ui(self):
        self.setMinimumSize(800, 600)
        layout = QVBoxLayout(self)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous clip")
        self.prev_btn.clicked.connect(self._prev_video)
        nav_layout.addWidget(self.prev_btn)
        
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.title_label)
        
        self.next_btn = QPushButton("Next clip")
        self.next_btn.clicked.connect(self._next_video)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        controls_layout.addWidget(self.stop_btn)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        controls_layout.addWidget(self.frame_label)
        
        controls_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        controls_layout.addWidget(close_btn)
        
        layout.addLayout(controls_layout)
    
    def _load_current_video(self):
        """Load current video for streaming playback."""
        if not self.video_paths:
            return
            
        video_path = self.video_paths[self.current_video_idx]
        self.setWindowTitle(f"Video Player - {os.path.basename(video_path)}")
        self.title_label.setText(f"{self.current_video_idx + 1} / {len(self.video_paths)}: {os.path.basename(video_path)}")
        
        # Update nav buttons
        self.prev_btn.setEnabled(self.current_video_idx > 0)
        self.next_btn.setEnabled(self.current_video_idx < len(self.video_paths) - 1)
        
        # Stop playback and release old capture
        self._stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self._load_video_content(video_path)
    
    def _prev_video(self):
        if self.current_video_idx > 0:
            self.current_video_idx -= 1
            self._load_current_video()
            
    def _next_video(self):
        if self.current_video_idx < len(self.video_paths) - 1:
            self.current_video_idx += 1
            self._load_current_video()
    
    def _load_video_content(self, video_path):
        """Open video for streaming playback (memory efficient)."""
        try:
            self.video_label.setText("Loading video...")
            QApplication.processEvents()
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", f"Could not open video: {video_path}")
                return
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.total_frames <= 0:
                QMessageBox.warning(self, "Error", "Could not determine video length.")
                return
            
            # Setup slider and labels
            self.slider.setRange(0, self.total_frames - 1)
            self.frame_label.setText(f"Frame: 1 / {self.total_frames}")
            
            # Setup timer
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.timer.setInterval(interval)
            
            # Show first frame
            self._display_frame(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
    
    def _display_frame(self, idx):
        """Display frame at index by seeking in video (streaming)."""
        if self.cap is None or not self.cap.isOpened():
            return
        if not (0 <= idx < self.total_frames):
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert BGR to RGB and keep reference to prevent QImage crash
        self._current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self._current_frame.flags['C_CONTIGUOUS']:
            self._current_frame = np.ascontiguousarray(self._current_frame)
        
        h, w, ch = self._current_frame.shape
        bytes_per_line = ch * w
        
        q_image = QImage(self._current_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        self.frame_label.setText(f"Frame: {idx + 1} / {self.total_frames}")
    
    def _advance_frame(self):
        """Advance to next frame."""
        if not self.is_playing or self.total_frames <= 0:
            return
            
        self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self._display_frame(self.current_frame_idx)
    
    def _on_slider_changed(self, value):
        """Handle slider value change."""
        self.current_frame_idx = value
        self._display_frame(value)
    
    def _on_slider_pressed(self):
        """Handle slider press (pause playback)."""
        self.slider_pressed = True
        if self.is_playing:
            self.timer.stop()
    
    def _on_slider_released(self):
        """Handle slider release (resume if was playing)."""
        self.slider_pressed = False
        if self.is_playing:
            self.timer.start()
    
    def _toggle_play(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("Play")
        else:
            self.timer.start()
            self.is_playing = True
            self.play_btn.setText("Pause")
    
    def _stop(self):
        """Stop playback and reset."""
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("Play")
        self.current_frame_idx = 0
        self.slider.setValue(0)
        self._display_frame(0)
    
    def closeEvent(self, event):
        """Clean up."""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._current_frame = None
        super().closeEvent(event)


class ProcessingWorker(QThread):
    """Worker thread for video processing into clips."""
    progress = pyqtSignal(int, int, str)  # current, total, video_name
    finished = pyqtSignal(list)  # List of output clip paths
    error = pyqtSignal(str)
    
    def __init__(self, video_mask_pairs, output_dir, params):
        super().__init__()
        self.video_mask_pairs = video_mask_pairs
        self.output_dir = output_dir
        self.params = params
        self.should_stop = False
        self.output_paths = []
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        try:
            total = len(self.video_mask_pairs)
            for i, (video_path, mask_path) in enumerate(self.video_mask_pairs):
                if self.should_stop:
                    break
                
                video_name = os.path.basename(video_path)
                self.progress.emit(i + 1, total, video_name)
                
                # Generate output directory for clips
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                video_clips_dir = os.path.join(self.output_dir, base_name)
                
                # Progress callback
                def progress_cb(clip_num, total_clips=None, obj_id=None):
                    if self.should_stop:
                        return
                
                # Process video into clips (may return list if multiple objects)
                clip_paths = process_video_to_clips(
                    video_path=video_path,
                    mask_path=mask_path,
                    output_dir=video_clips_dir,
                    box_size=self.params['box_size'],
                    target_size=self.params['target_size'],
                    background_mode=self.params['background_mode'],
                    normalization_method=self.params['normalization_method'],
                    mask_feather_px=self.params['mask_feather_px'],
                    anchor_mode=self.params['anchor_mode'],
                    target_fps=self.params['target_fps'],
                    clip_length_frames=self.params['clip_length_frames'],
                    step_frames=self.params['step_frames'],
                    progress_callback=progress_cb
                )
                
                if clip_paths:
                    # clip_paths is now list of (clip_path, start_frame, end_frame) tuples
                    # Store them for later use in embedding extraction
                    self.output_paths.extend(clip_paths)
                else:
                    self.error.emit(f"Failed to process {video_name} - no clips created")
            
            if not self.should_stop:
                self.finished.emit(self.output_paths)
                
        except Exception as e:
            self.error.emit(str(e))


class EmbeddingExtractionWorker(QThread):
    """Worker thread for extracting VideoPrism embeddings from clips."""
    progress = pyqtSignal(int, int, str)  # current, total, clip_name
    finished = pyqtSignal(str, str)  # feature_matrix_path, metadata_path
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, clip_paths: list, output_dir: str, experiment_name: str = None, model_name: str = 'videoprism_public_v1_base', clip_frame_ranges: dict = None, append_to_existing: bool = False, flip_invariant: bool = False, align_orientation: bool = False, mask_path: str = None):
        super().__init__()
        self.clip_paths = clip_paths
        self.clip_frame_ranges = clip_frame_ranges or {}
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.should_stop = False
        self.append_to_existing = append_to_existing
        self.flip_invariant = flip_invariant
        self.align_orientation = align_orientation
        self.mask_path = mask_path
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        try:
            self.log_message.emit(f"Loading VideoPrism model: {self.model_name}...")
            
            # Clear PyTorch cache to free up VRAM for JAX
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            backbone = VideoPrismBackbone(model_name=self.model_name, log_fn=self.log_message.emit)
            backbone.eval()
            
            embed_dim = backbone.get_embed_dim()
            self.log_message.emit(f"VideoPrism model loaded. Embedding dimension: {embed_dim}")
            if self.flip_invariant:
                self.log_message.emit("Flip-invariant mode: averaging 4 orientations (original, hflip, vflip, both)")
            
            feature_matrix = []
            metadata = []
            
            total = len(self.clip_paths)
            self.log_message.emit(f"Processing {total} clips...")
            
            for i, clip_path in enumerate(self.clip_paths):
                if self.should_stop:
                    break
                
                clip_name = os.path.basename(clip_path)
                self.progress.emit(i + 1, total, clip_name)
                
                # Load clip frames
                frames = self._load_clip_frames(clip_path)
                if frames is None or len(frames) == 0:
                    self.log_message.emit(f"Warning: Could not load frames from {clip_name}, skipping")
                    continue
                
                embedding = self._extract_embedding(backbone, frames)
                del frames
                
                if embedding is None:
                    self.log_message.emit(f"Warning: Could not extract embedding from {clip_name}, skipping")
                    continue
                
                feature_matrix.append(embedding.tolist())
                
                # Free embedding after converting to list
                del embedding
                
                # Periodically clear CUDA cache to prevent GPU memory accumulation
                if (i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # Parse clip name for metadata
                # Format: clip_XXXXXX_objY.mp4 or clip_XXXXXX.mp4
                base_name = os.path.splitext(clip_name)[0]
                video_dir = os.path.dirname(clip_path)
                video_name = os.path.basename(video_dir)
                
                # Extract object ID if present
                obj_match = re.search(r'_obj(\d+)', base_name)
                obj_id = obj_match.group(1) if obj_match else None
                
                # Extract clip index
                clip_match = re.search(r'clip_(\d+)', base_name)
                clip_idx = int(clip_match.group(1)) if clip_match else i
                
                # Get frame range if available
                start_frame = None
                end_frame = None
                if clip_path in self.clip_frame_ranges:
                    start_frame, end_frame = self.clip_frame_ranges[clip_path]
                
                metadata.append({
                    'snippet': f'snippet{i+1}',
                    'group': video_name,
                    'video_id': clip_name,
                    'object_id': obj_id if obj_id else '',
                    'clip_index': clip_idx,
                    'start_frame': start_frame if start_frame is not None else '',
                    'end_frame': end_frame if end_frame is not None else ''
                })
            
            if self.should_stop:
                self.log_message.emit("Extraction stopped by user.")
                return
            
            if not feature_matrix:
                self.error.emit("No embeddings extracted. Please check that clips are valid.")
                return
            
            # Convert to numpy array
            feature_matrix = np.array(feature_matrix, dtype=np.float32)  # downcast to save space
            self.log_message.emit(f"Extracted {len(feature_matrix)} embeddings. Shape: {feature_matrix.shape}")
            
            # Prepare data
            num_snippets = feature_matrix.shape[0]
            num_features = feature_matrix.shape[1]
            metadata_df = pd.DataFrame(metadata)
            
            # Decide whether to append to existing embeddings
            append_used = False
            existing_matrix_path = None
            existing_metadata_path = None
            if self.append_to_existing:
                candidates = sorted(glob.glob(os.path.join(self.output_dir, "*_matrix.npz")), key=os.path.getmtime, reverse=True)
                if candidates:
                    existing_matrix_path = candidates[0]
                    guess_meta = existing_matrix_path.replace("_matrix.npz", "_metadata.npz")
                    if os.path.exists(guess_meta):
                        existing_metadata_path = guess_meta
                if not existing_matrix_path or not existing_metadata_path:
                    self.log_message.emit("Append requested but no existing matrix/metadata found. Creating new files.")
            
            snippet_ids = [f'snippet{i+1}' for i in range(num_snippets)]
            feature_names = [f'behaviorome_embedding_{i}' for i in range(num_features)]
            
            # If appending, load existing and merge
            if existing_matrix_path and existing_metadata_path:
                try:
                    existing_npz = np.load(existing_matrix_path, allow_pickle=True)
                    existing_matrix = existing_npz["matrix"]  # features x snippets
                    existing_feature_names = existing_npz["feature_names"]
                    existing_snippet_ids = existing_npz["snippet_ids"]
                    
                    # Validate feature dimension
                    if existing_matrix.shape[0] != feature_matrix.shape[1]:
                        self.log_message.emit("Append skipped: feature dimension mismatch. Writing new files instead.")
                    else:
                        # Load existing metadata
                        existing_meta_npz = np.load(existing_metadata_path, allow_pickle=True)
                        existing_meta = pd.DataFrame(existing_meta_npz["metadata"], columns=existing_meta_npz["columns"])
                        
                        offset = existing_matrix.shape[1]
                        snippet_ids = [f'snippet{offset + i + 1}' for i in range(num_snippets)]
                        metadata_df['snippet'] = snippet_ids
                        
                        # Combine matrices and metadata
                        feature_matrix = feature_matrix.T  # to features x new_snippets
                        combined_matrix = np.concatenate([existing_matrix, feature_matrix], axis=1)
                        combined_snippet_ids = np.concatenate([existing_snippet_ids, np.array(snippet_ids)])
                        combined_feature_names = existing_feature_names
                        combined_meta = pd.concat([existing_meta, metadata_df], ignore_index=True)
                        
                        # Overwrite base name with existing
                        base_name = os.path.basename(existing_matrix_path).replace("_matrix.npz", "")
                        npz_matrix_path = existing_matrix_path
                        npz_metadata_path = existing_metadata_path
                        
                        feature_matrix = combined_matrix.T  # convert back to snippets x features for downstream naming
                        feature_names = combined_feature_names.tolist()
                        snippet_ids = combined_snippet_ids.tolist()
                        metadata_df = combined_meta
                        append_used = True
                        self.log_message.emit(f"Appending to existing embeddings: {os.path.basename(existing_matrix_path)}")
                except Exception as e:
                    self.log_message.emit(f"Append failed, writing new files instead: {e}")
            
            if not append_used:
                # Generate filename with experiment name and timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Build base filename
                if self.experiment_name:
                    base_name = f"behaviorome_{self.experiment_name}_{timestamp}"
                else:
                    base_name = f"behaviorome_{timestamp}"
                
                # Build paths
                npz_matrix_path = os.path.join(self.output_dir, f'{base_name}_matrix.npz')
                npz_metadata_path = os.path.join(self.output_dir, f'{base_name}_metadata.npz')
            
            # Save as NPZ (fastest, most efficient for large datasets)
            
            try:
                np.savez_compressed(
                    npz_matrix_path,
                    matrix=feature_matrix.T,           # features x snippets
                    feature_names=np.array(feature_names),
                    snippet_ids=np.array(snippet_ids),
                )
                self.log_message.emit(f"Saved feature matrix (NPZ) to {npz_matrix_path}")
            except Exception as e:
                self.log_message.emit(f"NPZ save failed (matrix): {e}")
                npz_matrix_path = None
            
            try:
                # Save metadata as NPZ
                np.savez_compressed(
                    npz_metadata_path,
                    metadata=metadata_df.values,
                    columns=np.array(metadata_df.columns),
                )
                self.log_message.emit(f"Saved metadata (NPZ) to {npz_metadata_path}")
            except Exception as e:
                self.log_message.emit(f"NPZ save failed (metadata): {e}")
                npz_metadata_path = None

            self.finished.emit(npz_matrix_path, npz_metadata_path)
            
        except Exception as e:
            import traceback
            error_msg = f"Error extracting embeddings: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg)
            self.error.emit(str(e))
    
    def _load_clip_frames(self, clip_path: str) -> np.ndarray:
        """Load all frames from a clip video."""
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return None
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return np.array(frames) if frames else None
    
    def _extract_embedding(self, backbone: VideoPrismBackbone, frames: np.ndarray) -> np.ndarray:
        try:
            target_size = 288
            processed_frames = []
            for frame in frames:
                resized = cv2.resize(frame, (target_size, target_size))
                processed_frames.append(resized)
            frames_resized = np.array(processed_frames)
            del processed_frames
            frames_t = np.transpose(frames_resized, (0, 3, 1, 2))
            del frames_resized
            frames_tensor = torch.from_numpy(frames_t).float() / 255.0
            del frames_t
            frames_tensor = frames_tensor.unsqueeze(0)
            
            with torch.no_grad():
                tokens = backbone(frames_tensor)
                embedding = tokens.mean(dim=1).squeeze(0)
                del tokens
                if self.flip_invariant:
                    embs = [embedding.cpu().numpy()]
                    for dims in [[-1], [-2], [-1, -2]]:
                        t_flip = torch.flip(frames_tensor, dims=dims)
                        embs.append(backbone(t_flip).mean(dim=1).squeeze(0).cpu().numpy())
                        del t_flip
                    embedding = torch.from_numpy(np.mean(embs, axis=0))
                    del embs
                del frames_tensor

            result = embedding.cpu().numpy()
            del embedding
            return result
            
        except Exception as e:
            self.log_message.emit(f"Error extracting embedding: {e}")
            return None


class RegistrationWidget(QWidget):
    """Widget for animal registration video processing."""
    
    # Signal emitted when embeddings are extracted (for auto-loading in clustering tab)
    embeddings_extracted = pyqtSignal(str, str)  # matrix_path, metadata_path
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.video_mask_pairs = []  # List of (video_path, mask_path) tuples
        self.worker = None
        self.embedding_worker = None
        self.output_dir = ""  # Initialize output_dir
        self.processed_videos = []  # List of processed video paths
        self.clip_frame_ranges = {}  # Dict mapping clip_path -> (start_frame, end_frame)
        self._setup_ui()
    
    def load_from_segmentation(self, video_path: str, mask_path: str):
        """Load video and mask from segmentation tab."""
        # Clear existing
        self.video_list.clear()
        self.mask_list.clear()
        self.video_mask_pairs = []
        
        # Add video
        if os.path.exists(video_path):
            item = QListWidgetItem(os.path.basename(video_path))
            item.setData(Qt.ItemDataRole.UserRole, video_path)
            self.video_list.addItem(item)
        
        # Add mask
        if os.path.exists(mask_path):
            item = QListWidgetItem(os.path.basename(mask_path))
            item.setData(Qt.ItemDataRole.UserRole, mask_path)
            self.mask_list.addItem(item)
        
        # Auto-match
        self._match_videos_masks()
        
        # Update output directory display
        self._update_pairs_label()
        
        # Check if clips already exist and enable extract embeddings button
        self._check_existing_clips()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Top row: File Selection and Processing Parameters side by side
        top_row_layout = QHBoxLayout()
        
        # File Selection (left side)
        files_group = QGroupBox("File selection")
        files_layout = QVBoxLayout()
        
        # Videos
        video_layout = QHBoxLayout()
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(100)
        video_layout.addWidget(self.video_list)
        
        video_btn_layout = QVBoxLayout()
        self.add_video_btn = QPushButton("Add videos")
        self.add_video_btn.clicked.connect(self._add_videos)
        video_btn_layout.addWidget(self.add_video_btn)
        
        self.remove_video_btn = QPushButton("Remove selected")
        self.remove_video_btn.clicked.connect(self._remove_video)
        video_btn_layout.addWidget(self.remove_video_btn)
        video_btn_layout.addStretch()
        
        video_layout.addLayout(video_btn_layout)
        files_layout.addLayout(video_layout)
        
        # Masks
        mask_layout = QHBoxLayout()
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(100)
        mask_layout.addWidget(self.mask_list)
        
        mask_btn_layout = QVBoxLayout()
        self.add_mask_btn = QPushButton("Add mask files")
        self.add_mask_btn.clicked.connect(self._add_masks)
        mask_btn_layout.addWidget(self.add_mask_btn)
        
        self.remove_mask_btn = QPushButton("Remove selected")
        self.remove_mask_btn.clicked.connect(self._remove_mask)
        mask_btn_layout.addWidget(self.remove_mask_btn)
        mask_btn_layout.addStretch()
        
        mask_layout.addLayout(mask_btn_layout)
        files_layout.addLayout(mask_layout)
        
        # Help button and pairs label
        help_layout = QHBoxLayout()
        self.pairs_label = QLabel("0 video-mask pairs ready")
        help_layout.addWidget(self.pairs_label)
        help_layout.addStretch()
        
        # Help button (small circular button)
        self.help_btn = QPushButton("?")
        self.help_btn.setMaximumSize(25, 25)
        self.help_btn.setToolTip("Click for naming information")
        self.help_btn.clicked.connect(self._show_naming_help)
        help_layout.addWidget(self.help_btn)
        
        files_layout.addLayout(help_layout)
        
        # Load Clips Folder button for extracting embeddings from existing clips
        self.load_clips_btn = QPushButton("Already have clips? Load")
        self.load_clips_btn.clicked.connect(self._load_clips_folder)
        self.load_clips_btn.setToolTip("Load a folder of existing processed clips to extract behaviorome embeddings")
        files_layout.addWidget(self.load_clips_btn)
        
        files_group.setLayout(files_layout)
        top_row_layout.addWidget(files_group)
        
        # Processing Parameters (right side)
        params_container = QVBoxLayout()
        
        # Processing Parameters
        params_group = QGroupBox("Processing parameters")
        params_layout = QFormLayout()
        
        self.box_size_spin = QSpinBox()
        self.box_size_spin.setRange(50, 1000)
        self.box_size_spin.setValue(250)
        params_layout.addRow("Crop Box Size (px):", self.box_size_spin)
        
        self.target_size_spin = QSpinBox()
        self.target_size_spin.setRange(50, 1000)
        self.target_size_spin.setValue(288)
        params_layout.addRow("Output Size (px):", self.target_size_spin)
        
        self.background_combo = QComboBox()
        self.background_combo.addItems(["white", "black", "gray", "blur", "none"])
        self.background_combo.setCurrentText("white")
        params_layout.addRow("Background Mode:", self.background_combo)
        
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["CLAHE", "Histogram Equalization", "Mean-Variance", "None"])
        self.normalization_combo.setCurrentText("CLAHE")
        params_layout.addRow("Normalization:", self.normalization_combo)
        
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 50)
        self.feather_spin.setValue(0)
        params_layout.addRow("Mask Feathering (px):", self.feather_spin)
        
        # Anchor mode checkbox
        self.lock_roi_checkbox = QCheckBox("Lock ROI to first frame of clip")
        self.lock_roi_checkbox.setChecked(False)
        self.lock_roi_checkbox.setToolTip(
            "When checked, the crop box stays fixed at the first frame's centroid,\n"
            "allowing the object to move within the clip (preserves locomotion).\n"
            "When unchecked, the crop box follows the object's centroid each frame."
        )
        params_layout.addRow("", self.lock_roi_checkbox)
        
        params_group.setLayout(params_layout)
        params_container.addWidget(params_group)
        
        # Clip Extraction Parameters
        clip_params_group = QGroupBox("Clip extraction parameters")
        clip_params_layout = QFormLayout()
        
        self.target_fps_spin = QSpinBox()
        self.target_fps_spin.setRange(1, 60)
        self.target_fps_spin.setValue(int(self.config.get("default_target_fps", 12)))
        clip_params_layout.addRow("Target FPS:", self.target_fps_spin)
        
        self.clip_length_spin = QSpinBox()
        self.clip_length_spin.setRange(1, 64)
        self.clip_length_spin.setValue(int(self.config.get("default_clip_length", 8)))
        clip_params_layout.addRow("Frames per clip:", self.clip_length_spin)
        
        self.step_frames_spin = QSpinBox()
        self.step_frames_spin.setRange(1, 64)
        self.step_frames_spin.setValue(int(self.config.get("default_step_frames", 8)))
        clip_params_layout.addRow("Step frames:", self.step_frames_spin)
        
        clip_params_group.setLayout(clip_params_layout)
        params_container.addWidget(clip_params_group)
        
        # Create a widget to hold the params container
        params_widget = QWidget()
        params_widget.setLayout(params_container)
        top_row_layout.addWidget(params_widget)
        
        layout.addLayout(top_row_layout)
        
        # Output (auto-determined from experiment)
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        self.output_dir_label = QLabel("Clips will be saved to experiment folder")
        output_layout.addWidget(self.output_dir_label)

        self.flip_invariant_check = QCheckBox("Flip-invariant embeddings")
        self.flip_invariant_check.setChecked(False)
        self.flip_invariant_check.setToolTip(
            "Run each clip through VideoPrism in 4 orientations (original, hflip,\n"
            "vflip, both) and average the embeddings. Removes sensitivity to the\n"
            "animal's facing direction and vertical orientation. 4x extraction time."
        )
        output_layout.addWidget(self.flip_invariant_check)


        self.append_embeddings_check = QCheckBox("Append to existing embeddings if present")
        self.append_embeddings_check.setChecked(False)
        self.append_embeddings_check.setToolTip("When enabled, if an existing behaviorome matrix/metadata is found in the experiment, new embeddings will be appended instead of creating a new file.")
        output_layout.addWidget(self.append_embeddings_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Process videos")
        self.process_btn.clicked.connect(self._start_processing)
        self.process_btn.setEnabled(False)
        actions_layout.addWidget(self.process_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        actions_layout.addWidget(self.stop_btn)
        
        self.view_videos_btn = QPushButton("View processed videos")
        self.view_videos_btn.clicked.connect(self._view_processed_videos)
        self.view_videos_btn.setEnabled(False)
        actions_layout.addWidget(self.view_videos_btn)
        
        self.extract_embeddings_btn = QPushButton("Extract behaviorome embeddings")
        self.extract_embeddings_btn.clicked.connect(self._extract_embeddings)
        self.extract_embeddings_btn.setEnabled(False)
        actions_layout.addWidget(self.extract_embeddings_btn)
        
        self.export_roi_btn = QPushButton("Export ROI videos (per object)")
        self.export_roi_btn.setToolTip(
            "Export one cropped video per tracked object.\n"
            "Videos are saved in the experiment folder (roi_videos)."
        )
        self.export_roi_btn.clicked.connect(self._export_roi_videos)
        self.export_roi_btn.setEnabled(False)
        actions_layout.addWidget(self.export_roi_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Check for existing clips
        self._check_existing_clips()
    
    def _add_videos(self):
        video_dir = self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos"))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", video_dir, "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if paths:
            # Ensure videos are in experiment folder (batch operation)
            from .video_utils import ensure_videos_in_experiment
            paths = ensure_videos_in_experiment(paths, self.config, self)
            for path in paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.video_list.addItem(item)
            # Auto-match after adding videos
            self._match_videos_masks()
            self._update_pairs_label()
    
    def _remove_video(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        # Re-match after removal
        self._match_videos_masks()
        self._update_pairs_label()
    
    def _add_masks(self):
        # Check experiment folder first, then default location
        experiment_path = self.config.get("experiment_path")
        if experiment_path and os.path.exists(experiment_path):
            masks_dir = os.path.join(experiment_path, "masks")
        else:
            masks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "masks")
        
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Mask Files", masks_dir if os.path.exists(masks_dir) else "", "HDF5 Files (*.h5 *.hdf5)"
        )
        if paths:
            for path in paths:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.mask_list.addItem(item)
            # Auto-match after adding masks
            self._match_videos_masks()
            self._update_pairs_label()
    
    def _remove_mask(self):
        for item in self.mask_list.selectedItems():
            self.mask_list.takeItem(self.mask_list.row(item))
        # Re-match after removal
        self._match_videos_masks()
        self._update_pairs_label()
    
    def _match_videos_masks(self):
        """Match videos to masks based on filename."""
        self.video_mask_pairs = []
        
        videos = []
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            videos.append((item.data(Qt.ItemDataRole.UserRole), item.text()))
        
        masks = []
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            masks.append((item.data(Qt.ItemDataRole.UserRole), item.text()))
        
        # Match by base name
        matched = []
        unmatched_videos = []
        unmatched_masks = []
        
        for video_path, video_name in videos:
            base_video = os.path.splitext(video_name)[0]
            found = False
            for mask_path, mask_name in masks:
                base_mask = os.path.splitext(mask_name)[0]
                # Remove "_masks" suffix from mask name (most common case)
                base_mask_clean = base_mask.replace('_masks', '').replace('_mask', '').replace('_objects', '')
                base_video_clean = base_video.replace('_output', '').replace('_segmented', '')
                
                # Try exact match first
                if base_video == base_mask:
                    matched.append((video_path, mask_path))
                    found = True
                    break
                # Try matching video name with mask name after removing "_masks"
                elif base_video == base_mask_clean:
                    matched.append((video_path, mask_path))
                    found = True
                    break
                # Try matching cleaned versions
                elif base_video_clean == base_mask_clean:
                    matched.append((video_path, mask_path))
                    found = True
                    break
            
            if not found:
                unmatched_videos.append(video_name)
        
        # Check for unmatched masks
        matched_mask_names = {os.path.basename(m) for _, m in matched}
        for mask_path, mask_name in masks:
            if mask_name not in matched_mask_names:
                unmatched_masks.append(mask_name)
        
        self.video_mask_pairs = matched
        # Matching happens automatically, no message box needed
    
    def _update_pairs_label(self):
        count = len(self.video_mask_pairs)
        self.pairs_label.setText(f"{count} video-mask pairs ready")
        # Auto-determine output directory from experiment
        experiment_path = self.config.get("experiment_path")
        if experiment_path and os.path.exists(experiment_path):
            self.output_dir = os.path.join(experiment_path, "registered_clips")
            self.output_dir_label.setText(f"Output: {self.output_dir}")
        else:
            self.output_dir = ""
            self.output_dir_label.setText("No experiment folder - please create an experiment first")
        self.process_btn.setEnabled(count > 0 and bool(self.output_dir))
        self.export_roi_btn.setEnabled(count > 0)
    
    def _show_naming_help(self):
        """Show help dialog about video-mask naming."""
        QMessageBox.information(
            self,
            "Video-Mask Naming",
            "Videos and masks are automatically matched based on their filenames.\n\n"
            "Naming rules:\n"
            "-Video: video_name.mp4\n"
            "-Mask: video_name.h5 (or video_name_masks.h5)\n\n"
            "The mask filename should match the video filename (without extension),\n"
            "or have '_masks' suffix that will be automatically removed.\n\n"
            "Example:\n"
            "-Video: my_video.mp4\n"
            "- Mask: my_video.h5\n"
            "- Mask: my_video_masks.h5"
        )
    
    def _start_processing(self):
        if not self.video_mask_pairs:
            QMessageBox.warning(self, "Error", "Please add videos and masks first.")
            return
        
        # Auto-determine output directory from experiment
        experiment_path = self.config.get("experiment_path")
        if not experiment_path or not os.path.exists(experiment_path):
            QMessageBox.warning(self, "Error", "No experiment folder found. Please create an experiment first.")
            return
        
        output_dir = os.path.join(experiment_path, "registered_clips")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.output_dir_label.setText(f"Output: {output_dir}")
        
        # Get parameters
        params = {
            'box_size': self.box_size_spin.value(),
            'target_size': self.target_size_spin.value(),
            'background_mode': self.background_combo.currentText(),
            'normalization_method': self.normalization_combo.currentText(),
            'mask_feather_px': self.feather_spin.value(),
            'anchor_mode': 'first' if self.lock_roi_checkbox.isChecked() else 'frame',
            'target_fps': self.target_fps_spin.value(),
            'clip_length_frames': self.clip_length_spin.value(),
            'step_frames': self.step_frames_spin.value()
        }
        
        # Create worker
        self.worker = ProcessingWorker(self.video_mask_pairs, self.output_dir, params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.view_videos_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(self.video_mask_pairs))
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.processed_videos = []
        
        self.worker.start()
    
    def _stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.log_text.append("Processing stopped by user.")
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def _on_progress(self, current, total, video_name):
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Processing: {video_name} ({current}/{total})")
    
    def _on_finished(self, output_paths):
        self.log_text.append("=" * 50)
        self.log_text.append("All videos processed successfully!")
        self.log_text.append(f"Output directory: {self.output_dir}")
        self.log_text.append(f"Created {len(output_paths)} clip(s)")
        
        clip_paths_list = []
        self.clip_frame_ranges = {}
        for item in output_paths:
            if isinstance(item, tuple) and len(item) == 3:
                clip_path, start_frame, end_frame = item
                clip_paths_list.append(clip_path)
                self.clip_frame_ranges[clip_path] = (start_frame, end_frame)
            else:
                clip_paths_list.append(item)

        
        # Group clips by video (using extracted paths)
        clips_by_video = {}
        for path in clip_paths_list:
            video_dir = os.path.dirname(path)
            if video_dir not in clips_by_video:
                clips_by_video[video_dir] = []
            clips_by_video[video_dir].append(path)
        
        for video_dir, clips in clips_by_video.items():
            video_name = os.path.basename(video_dir)
            self.log_text.append(f"  {video_name}: {len(clips)} clip(s)")
        
        self.processed_videos = clip_paths_list
        self.status_label.setText("Complete")
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.view_videos_btn.setEnabled(len(clip_paths_list) > 0)
        self.extract_embeddings_btn.setEnabled(len(clip_paths_list) > 0)
        
        msg = f"All videos have been processed into clips.\n\nCreated {len(clip_paths_list)} clip(s) total.\n\nWould you like to view them now?"
        reply = QMessageBox.question(
            self, "Success", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._view_processed_videos()
    
    def _on_error(self, error_msg):
        self.log_text.append(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _view_processed_videos(self):
        """Open video player dialog for processed videos."""
        if not self.processed_videos:
            QMessageBox.information(self, "No Videos", "No processed videos available to view.")
            return
        
        # If multiple videos, let user choose start, otherwise open directly
        if len(self.processed_videos) == 1:
            dialog = VideoPlayerDialog(self.processed_videos, 0, self)
            dialog.exec()
        else:
            # Multiple videos - let user choose which one to start with
            from PyQt6.QtWidgets import QListWidget, QDialogButtonBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Select video to view")
            dialog.setMinimumSize(400, 300)
            layout = QVBoxLayout(dialog)
            
            label = QLabel("Select a video to view:")
            layout.addWidget(label)
            
            list_widget = QListWidget()
            for path in self.processed_videos:
                list_widget.addItem(os.path.basename(path))
            layout.addWidget(list_widget)
            
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_items = list_widget.selectedItems()
                if selected_items:
                    idx = list_widget.row(selected_items[0])
                    player_dialog = VideoPlayerDialog(self.processed_videos, idx, self)
                    player_dialog.exec()
    
    def _load_clips_folder(self):
        """Load a folder of existing clips for embedding extraction."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Clips Folder")
        if not folder_path:
            return
            
        clip_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    clip_paths.append(os.path.join(root, file))
        
        if not clip_paths:
            QMessageBox.warning(self, "No Clips", f"No video clips found in {folder_path}")
            return
            
        self.output_dir = folder_path
        self.processed_videos = sorted(clip_paths)
        
        self.log_text.append("=" * 50)
        self.log_text.append(f"Loaded {len(clip_paths)} clips from: {folder_path}")
        self.status_label.setText(f"Loaded {len(clip_paths)} clips")
        self.extract_embeddings_btn.setEnabled(True)
        self.view_videos_btn.setEnabled(True)
        
        # Ask to extract immediately
        reply = QMessageBox.question(
            self,
            "Extract Embeddings",
            f"Found {len(clip_paths)} clips.\nDo you want to extract embeddings now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._extract_embeddings()

    def _extract_embeddings(self):
        """Extract VideoPrism embeddings from processed clips."""
        if not self.processed_videos:
            QMessageBox.warning(self, "No Clips", "No processed clips available. Please process videos first.")
            return
        
        # Get output directory (where clips are stored)
        if not self.output_dir:
            experiment_path = self.config.get("experiment_path")
            if not experiment_path:
                QMessageBox.warning(self, "No Experiment", "No active experiment. Please create or load an experiment first.")
                return
            self.output_dir = os.path.join(experiment_path, "registered_clips")
        
        if not os.path.exists(self.output_dir):
            QMessageBox.warning(self, "Directory Not Found", f"Clips directory not found: {self.output_dir}")
            return
        
        # Collect all clip files from registered_clips directory
        clip_paths = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    clip_paths.append(os.path.join(root, file))
        
        if not clip_paths:
            QMessageBox.warning(self, "No Clips", f"No video clips found in {self.output_dir}")
            return
        
        # Sort clips for consistent ordering
        clip_paths.sort()
        
        # Get model name from config
        model_name = self.config.get("backbone_model", "videoprism_public_v1_base")
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Extract Embeddings",
            f"Extract VideoPrism embeddings from {len(clip_paths)} clip(s)?\n\n"
            f"Model: {model_name}\n"
            f"Output directory: {self.output_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Get experiment name from config
        experiment_name = self.config.get("experiment_name", None)
        
        # Start extraction worker with frame ranges if available
        mask_path = None
        if self.align_orientation_check.isChecked() and self.video_mask_pairs:
            mask_path = self.video_mask_pairs[0][1] if len(self.video_mask_pairs) > 0 else None
            self.log_text.append(f"Align orientation: mask_path={mask_path}, pairs={len(self.video_mask_pairs)}, frame_ranges={len(self.clip_frame_ranges)}")

        self.embedding_worker = EmbeddingExtractionWorker(
            clip_paths,
            self.output_dir,
            experiment_name=experiment_name,
            model_name=model_name,
            clip_frame_ranges=self.clip_frame_ranges if hasattr(self, 'clip_frame_ranges') else None,
            append_to_existing=self.append_embeddings_check.isChecked(),
            flip_invariant=self.flip_invariant_check.isChecked(),
            align_orientation=self.align_orientation_check.isChecked(),
            mask_path=mask_path,
        )
        self.embedding_worker.progress.connect(self._on_embedding_progress)
        self.embedding_worker.finished.connect(self._on_embedding_finished)
        self.embedding_worker.error.connect(self._on_embedding_error)
        self.embedding_worker.log_message.connect(self._on_embedding_log)
        
        self.extract_embeddings_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(clip_paths))
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_text.append(f"Starting VideoPrism embedding extraction from {len(clip_paths)} clips...")
        
        self.embedding_worker.start()
    
    def _on_embedding_progress(self, current, total, clip_name):
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Extracting embeddings: {clip_name} ({current}/{total})")
    
    def _on_embedding_finished(self, feature_matrix_path, metadata_path):
        self.log_text.append("=" * 50)
        self.log_text.append("Embedding extraction completed successfully!")
        self.log_text.append(f"Feature matrix: {feature_matrix_path}")
        self.log_text.append(f"Metadata: {metadata_path}")
        
        self.status_label.setText("Embedding extraction complete")
        self.extract_embeddings_btn.setEnabled(True)
        
        # Emit signal for auto-loading in clustering tab
        self.embeddings_extracted.emit(feature_matrix_path, metadata_path)
        
        QMessageBox.information(
            self,
            "Success",
            f"Behaviorome embeddings extracted successfully!\n\n"
            f"Feature matrix: {os.path.basename(feature_matrix_path)}\n"
            f"Metadata: {os.path.basename(metadata_path)}\n\n"
            f"Files saved to: {os.path.dirname(feature_matrix_path)}\n\n"
            f"Data will be automatically loaded in the Clustering tab."
        )
    
    def _on_embedding_error(self, error_msg):
        self.log_text.append(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Embedding extraction failed:\n{error_msg}")
        self.extract_embeddings_btn.setEnabled(True)
        self.status_label.setText("Error")
    
    def _on_embedding_log(self, message):
        self.log_text.append(message)
    
    def _export_roi_videos(self):
        """Export one cropped video per tracked object for each video-mask pair."""
        if not self.video_mask_pairs:
            QMessageBox.warning(self, "No Data", "Add video and mask files first.")
            return

        all_exported = []
        for video_path, mask_path in self.video_mask_pairs:
            try:
                mask_data = load_segmentation_data(mask_path)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load mask {os.path.basename(mask_path)}: {e}")
                continue

            frame_objects = mask_data.get("frame_objects", [])
            if not frame_objects:
                QMessageBox.warning(self, "No Frames", f"No mask frames in {os.path.basename(mask_path)}.")
                continue

            start_offset = mask_data.get("start_offset", 0)
            all_obj_ids = set()
            for frame_objs in frame_objects:
                for obj in frame_objs:
                    all_obj_ids.add(obj.get("obj_id", 0))
            all_obj_ids = sorted(all_obj_ids)
            if not all_obj_ids:
                continue

            experiment_path = self.config.get("experiment_path")
            if experiment_path and os.path.exists(experiment_path):
                out_dir = os.path.join(experiment_path, "roi_videos")
            else:
                out_dir = os.path.join(os.path.dirname(video_path), "roi_videos")
            os.makedirs(out_dir, exist_ok=True)
            video_basename = os.path.splitext(os.path.basename(video_path))[0]

            obj_tracks = {oid: {} for oid in all_obj_ids}
            for i, frame_objs in enumerate(frame_objects):
                frame_idx = start_offset + i
                for obj in frame_objs:
                    oid = obj.get("obj_id", 0)
                    bbox = obj.get("bbox")
                    if bbox is not None:
                        obj_tracks[oid][frame_idx] = tuple(int(x) for x in bbox)

            cap = cv2.VideoCapture(video_path)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_fps <= 0:
                vid_fps = 30.0
            cap.release()

            frame_indices = sorted(set(f for track in obj_tracks.values() for f in track))
            if not frame_indices:
                continue
            start_frame = frame_indices[0]
            end_frame = frame_indices[-1]

            crop_padding = 0.25
            writers = {}
            obj_crop_params = {}
            obj_paths = {}
            for obj_id, track in obj_tracks.items():
                if not track:
                    continue
                max_w = max(x2 - x1 for x1, y1, x2, y2 in track.values())
                max_h = max(y2 - y1 for x1, y1, x2, y2 in track.values())
                crop_w = int(max_w * (1 + 2 * crop_padding))
                crop_h = int(max_h * (1 + 2 * crop_padding))
                crop_side = max(crop_w, crop_h, 64)
                obj_crop_params[obj_id] = crop_side
                out_path = os.path.join(out_dir, f"{video_basename}_object{obj_id}.mp4")
                obj_paths[obj_id] = out_path
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers[obj_id] = cv2.VideoWriter(out_path, fourcc, vid_fps, (crop_side, crop_side))

            if not writers:
                continue

            progress = QProgressDialog(
                f"Exporting ROI videos: {os.path.basename(video_path)}...", "Cancel",
                start_frame, end_frame + 1, self
            )
            progress.setWindowTitle("Export ROI Videos")
            progress.setMinimumDuration(0)
            progress.setValue(start_frame)

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            last_bbox = {}
            for fidx in range(start_frame, end_frame + 1):
                if progress.wasCanceled():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                progress.setValue(fidx)
                QApplication.processEvents()
                for obj_id, writer in writers.items():
                    crop_side = obj_crop_params[obj_id]
                    bbox = obj_tracks[obj_id].get(fidx) or last_bbox.get(obj_id)
                    if bbox is None:
                        writer.write(np.zeros((crop_side, crop_side, 3), dtype=np.uint8))
                        continue
                    last_bbox[obj_id] = bbox
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    half = crop_side // 2
                    rx1 = max(0, cx - half)
                    ry1 = max(0, cy - half)
                    rx2 = min(vid_w, rx1 + crop_side)
                    ry2 = min(vid_h, ry1 + crop_side)
                    rx1 = max(0, rx2 - crop_side)
                    ry1 = max(0, ry2 - crop_side)
                    crop = frame[ry1:ry2, rx1:rx2]
                    if crop.shape[0] != crop_side or crop.shape[1] != crop_side:
                        crop = cv2.resize(crop, (crop_side, crop_side), interpolation=cv2.INTER_AREA)
                    writer.write(crop)

            cap.release()
            progress.close()
            for w in writers.values():
                w.release()
            all_exported.extend([(out_dir, list(obj_paths.values()))])

        if not all_exported:
            QMessageBox.warning(self, "No Tracks", "No object tracks with valid bboxes found.")
            return
        summary = []
        for out_dir, paths in all_exported:
            names = [os.path.basename(p) for p in paths]
            summary.append(f"{out_dir}\n  " + "\n  ".join(names))
        QMessageBox.information(
            self, "Export Complete",
            f"Exported ROI videos to:\n\n" + "\n\n".join(summary)
        )

    def _check_existing_clips(self):
        """Check if processed clips exist and enable extract embeddings button."""
        experiment_path = self.config.get("experiment_path")
        if not experiment_path:
            return
        
        clips_dir = os.path.join(experiment_path, "registered_clips")
        if os.path.exists(clips_dir):
            clip_files = []
            for root, dirs, files in os.walk(clips_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        clip_files.append(os.path.join(root, file))
            
            if clip_files:
                self.processed_videos = clip_files
                self.extract_embeddings_btn.setEnabled(True)
                self.output_dir = clips_dir
    
    def update_config(self, config: dict):
        """Update configuration."""
        self.config = config

