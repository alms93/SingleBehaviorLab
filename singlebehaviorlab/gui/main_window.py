import logging
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QMessageBox, QFileDialog, QInputDialog,
    QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QDialogButtonBox,
    QLabel, QGroupBox, QWidget, QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QAction
from .segmentation_tracking_widget import SegmentationTrackingWidget
from .registration_widget import RegistrationWidget
from .clustering_widget import ClusteringWidget
from .labeling_widget import LabelingWidget
from .training_widget import TrainingWidget
from .inference_widget import InferenceWidget
from .analysis_widget import AnalysisWidget
from .review_widget import ReviewWidget
from .tab_tutorial_dialog import show_tab_tutorial
import os
import json
import yaml

logger = logging.getLogger(__name__)


class LabelingSetupDialog(QDialog):
    """Dialog for choosing how to populate the labeling list."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Labeling setup")
        self.setMinimumWidth(500)
        self.result_data = None
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("How would you like to populate the labeling list?"))
        
        self.rb_clustering = QRadioButton("Use representative clips from clustering")
        self.rb_clustering.setChecked(True)
        layout.addWidget(self.rb_clustering)
        
        self.clustering_group = QGroupBox("Clustering options")
        self.clustering_group_layout = QVBoxLayout()
        
        self.rb_segmented = QRadioButton("Use existing segmented clips (ROIs)")
        self.rb_segmented.setChecked(True)
        self.clustering_group_layout.addWidget(self.rb_segmented)
        
        self.rb_raw = QRadioButton("Extract raw clips from original video (Full frame/No mask)")
        self.clustering_group_layout.addWidget(self.rb_raw)
        
        self.clustering_group.setLayout(self.clustering_group_layout)
        layout.addWidget(self.clustering_group)
        
        self.rb_no_clustering = QRadioButton("Annotate raw videos on timeline (integrated in Labeling)")
        layout.addWidget(self.rb_no_clustering)
        
        self.rb_continue = QRadioButton("Continue with existing/manual list")
        layout.addWidget(self.rb_continue)
        
        self.bg_main = QButtonGroup(self)
        self.bg_main.addButton(self.rb_clustering)
        self.bg_main.addButton(self.rb_no_clustering)
        self.bg_main.addButton(self.rb_continue)
        
        self.rb_clustering.toggled.connect(self.clustering_group.setEnabled)
        self.clustering_group.setEnabled(True)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _on_accept(self):
        if self.rb_clustering.isChecked():
            mode = "clustering"
            submode = "segmented" if self.rb_segmented.isChecked() else "raw"
        elif self.rb_no_clustering.isChecked():
            mode = "raw_extraction"
            submode = None
        else:
            mode = "continue"
            submode = None
            
        self.result_data = {"mode": mode, "submode": submode}
        self.accept()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._labeling_setup_prompted_for_clusters = False
        self._update_window_title()
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)
        
        self._setup_menu()
        self._setup_tabs()
        
        self.setCentralWidget(self.tabs)
    
    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Video...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        
        create_exp_action = QAction("Create Experiment...", self)
        create_exp_action.triggered.connect(self._create_experiment)
        file_menu.addAction(create_exp_action)
        
        load_exp_action = QAction("Load Experiment...", self)
        load_exp_action.triggered.connect(self._load_experiment)
        file_menu.addAction(load_exp_action)
        
        save_exp_action = QAction("Save Experiment", self)
        save_exp_action.triggered.connect(self._save_experiment)
        file_menu.addAction(save_exp_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def _setup_tabs(self):
        """Setup tab widget with all tabs."""
        self.tabs = QTabWidget()
        
        self.segmentation_widget = SegmentationTrackingWidget(self.config)
        self.registration_widget = RegistrationWidget(self.config)
        self.clustering_widget = ClusteringWidget(self.config)
        self.labeling_widget = LabelingWidget(self.config)
        self.training_widget = TrainingWidget(self.config)
        self.inference_widget = InferenceWidget(self.config)
        self.analysis_widget = AnalysisWidget(self.config)
        self.review_widget = ReviewWidget(self.config)

        self.tabs.addTab(
            self._wrap_tab_with_help(self.labeling_widget, "labeling"), "Labeling"
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.training_widget, "training"),
            "Training Sequencing Model",
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.inference_widget, "sequencing"), "Sequencing"
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.review_widget, "refine"), "Refine"
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.analysis_widget, "analysis"),
            "Downstream Analysis",
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.segmentation_widget, "segmentation"),
            "Segmentation Tracking",
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.registration_widget, "registration"),
            "Registration",
        )
        self.tabs.addTab(
            self._wrap_tab_with_help(self.clustering_widget, "clustering"),
            "Clustering",
        )

        self.segmentation_widget.tracking_completed.connect(self._on_tracking_completed)
        self.registration_widget.embeddings_extracted.connect(self._on_embeddings_extracted)
        self.inference_widget.review_ready.connect(self._on_review_ready)
        self.review_widget.annotations_updated.connect(self._on_annotations_updated)

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _wrap_tab_with_help(self, inner: QWidget, tab_id: str) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(2)
        top = QHBoxLayout()
        top.addStretch(1)
        help_btn = QPushButton("📖 Tab Guide")
        help_btn.setToolTip("Open a detailed guide for this tab and what to do next (NOR example)")
        help_btn.setStyleSheet(
            "QPushButton {"
            "padding: 6px 12px;"
            "font-weight: 600;"
            "border: 1px solid #8a7b00;"
            "border-radius: 8px;"
            "background-color: #fff3b0;"
            "color: #3d3200;"
            "}"
            "QPushButton:hover {"
            "background-color: #ffe680;"
            "}"
            "QPushButton:pressed {"
            "background-color: #f7d154;"
            "}"
        )
        help_btn.clicked.connect(lambda _=False, tid=tab_id: show_tab_tutorial(self, tid))
        top.addWidget(help_btn)
        layout.addLayout(top)
        layout.addWidget(inner, 1)
        return outer

    def _on_tab_changed(self, index: int):
        """Handle tab change."""
        current_size = self.size()
        tab_name = self.tabs.tabText(index)
        
        if tab_name == "Labeling":
            if self.clustering_widget.clusters is not None and not self._labeling_setup_prompted_for_clusters:
                self._handle_labeling_setup()
                self._labeling_setup_prompted_for_clusters = True
            self.labeling_widget.refresh_clip_list()
        elif tab_name == "Training Sequencing Model":
            self.training_widget._load_current_config()
            self.training_widget.refresh_annotation_info()
        elif tab_name == "Registration":
            pass
        
        self.resize(current_size)
    
    def _handle_labeling_setup(self):
        """Show labeling setup dialog and handle user choice."""
        dialog = LabelingSetupDialog(self)
        if dialog.exec():
            choice = dialog.result_data
            if not choice:
                return
                
            mode = choice.get("mode")
            submode = choice.get("submode")
            
            if mode == "clustering":
                self._prepare_clustering_labeling_data(submode)
            elif mode == "raw_extraction":
                self.tabs.setCurrentWidget(self.labeling_widget)
                self.labeling_widget.open_timeline_import_dialog()
                QMessageBox.information(
                    self,
                    "Timeline labeling",
                    "Use the Timeline Annotation section in Labeling to add raw videos, mark intervals, and generate clips.",
                )

    def _prepare_clustering_labeling_data(self, submode):
        """Prepare labeling data from clustering results."""
        if not self.clustering_widget.clusters is not None:
            QMessageBox.warning(self, "No clusters", "No clustering data available. Please perform clustering first.")
            return
            
        try:
            rep_snippets = self.clustering_widget.get_representative_snippets(n_samples=10)
            if not rep_snippets:
                QMessageBox.warning(self, "No snippets", "Could not identify representative snippets.")
                return
                
            clips_to_add = []
            missing_clips = []
            
            self.clustering_widget._build_snippet_to_clip_map()
            snippet_map = self.clustering_widget.snippet_to_clip_map
            
            experiment_path = self.config.get("experiment_path")
            if not experiment_path:
                QMessageBox.warning(self, "Error", "No experiment loaded.")
                return
                
            labeling_clips_dir = os.path.join(experiment_path, "data", "clips")
            os.makedirs(labeling_clips_dir, exist_ok=True)
            
            import shutil
            
            for cluster_label, snippet_ids in rep_snippets.items():
                label = cluster_label
                
                for snip in snippet_ids:
                    clip_path = snippet_map.get(snip)
                    
                    if submode == "segmented":
                        if clip_path and os.path.exists(clip_path):
                            seg_dir = os.path.join(labeling_clips_dir, "segmented_clips")
                            os.makedirs(seg_dir, exist_ok=True)
                            
                            filename = os.path.basename(clip_path)
                            target_path = os.path.join(seg_dir, filename)
                            
                            if not os.path.exists(target_path):
                                try:
                                    shutil.copy2(clip_path, target_path)
                                except Exception as e:
                                    logger.error("Error copying clip %s: %s", clip_path, e)
                                    continue
                            
                            clips_to_add.append({
                                "path": target_path,
                                "label": label,
                                "snippet_id": snip
                            })
                        else:
                            missing_clips.append(snip)
                            
                    elif submode == "raw":
                        raw_clip_path = self._extract_raw_clip_from_snippet(snip, label)
                        if raw_clip_path and os.path.exists(raw_clip_path):
                            clips_to_add.append({
                                "path": raw_clip_path,
                                "label": label,
                                "snippet_id": snip
                            })
                        else:
                            missing_clips.append(snip)
            
            if submode == "raw" and not clips_to_add:
                QMessageBox.warning(self, "No clips", 
                    "Could not extract raw clips. Make sure video files and mask data are available.")
                return

            if missing_clips:
                logger.warning("Could not find clip files for %d snippets.", len(missing_clips))
            
            if not clips_to_add:
                QMessageBox.warning(self, "No clips", "No valid clips found for labeling data.")
                return
            
            self.labeling_widget.clip_base_dir = labeling_clips_dir
            self.labeling_widget.config["clips_dir"] = labeling_clips_dir
            
            added_count = 0
            am = self.labeling_widget.annotation_manager
            existing_ids = {c.get("id") for c in am.get_all_clips()}
            
            for clip_data in clips_to_add:
                abs_path = clip_data["path"].replace('\\', '/')
                try:
                    rel_path = os.path.relpath(abs_path, labeling_clips_dir).replace('\\', '/')
                except ValueError:
                    rel_path = os.path.basename(abs_path)
                
                if rel_path not in existing_ids:
                    am.add_clip(rel_path, clip_data["label"], meta={"snippet_id": clip_data["snippet_id"]})
                    am.add_class(clip_data["label"])
                    added_count += 1
            
            if added_count > 0:
                annotation_path = self.labeling_widget.annotation_manager.annotation_file
                QMessageBox.information(self, "Data prepared", 
                    f"Added {added_count} representative clips from {len(rep_snippets)} clusters to the labeling list.\n\n"
                    f"Annotations saved to:\n{annotation_path}\n\n"
                    "You can now review and refine labels.")
                self.labeling_widget._update_class_combo()
                self.labeling_widget.refresh_clip_list()
            else:
                QMessageBox.information(self, "Data ready", "All representative clips are already in the labeling list.")
            
        except Exception as e:
            logger.error("Failed to prepare labeling data: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to prepare labeling data: {str(e)}")
    
    def _extract_raw_clip_from_snippet(self, snippet_id: str, label: str) -> str:
        """Extract a raw (unmasked) clip from the original video for a snippet.
        
        Returns the path to the extracted clip, or None if extraction failed.
        """
        import cv2
        from singlebehaviorlab.backend.video_processor import load_segmentation_data
        
        try:
            metadata = self.clustering_widget.metadata
            if metadata is None:
                return None
            
            snippet_col = 'snippet' if 'snippet' in metadata.columns else ('span_id' if 'span_id' in metadata.columns else None)
            if not snippet_col:
                return None
            
            snippet_row = metadata[metadata[snippet_col].astype(str) == str(snippet_id)]
            if len(snippet_row) == 0:
                return None
            
            row = snippet_row.iloc[0]
            start_frame = row.get('start_frame')
            end_frame = row.get('end_frame')
            video_id = row.get('video_id', '')
            group = row.get('group', '')
            
            try:
                start_frame = int(float(start_frame))
                end_frame = int(float(end_frame))
            except (ValueError, TypeError):
                return None
            
            experiment_path = self.config.get("experiment_path")
            if not experiment_path:
                return None
            
            video_name_candidates = []
            
            if group:
                video_name_candidates.append(str(group).strip())
            
            if video_id:
                vid_base = os.path.splitext(os.path.basename(str(video_id)))[0]
                video_name_candidates.append(vid_base)
                import re
                match = re.match(r'^(.+?)_clip_\d+(?:_obj\d+)?$', vid_base)
                if match:
                    video_name_candidates.append(match.group(1))
            
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
            video_path = None
            
            all_videos = []
            for root, dirs, files in os.walk(experiment_path):
                if 'registered_clips' in root or 'raw_clips' in root:
                    continue
                for f in files:
                    if any(f.lower().endswith(ext.lower()) for ext in video_extensions):
                        all_videos.append(os.path.join(root, f))
            
            for video_name in video_name_candidates:
                if video_path:
                    break
                    
                for vp in all_videos:
                    vp_basename = os.path.splitext(os.path.basename(vp))[0]
                    if vp_basename == video_name:
                        video_path = vp
                        break
                
                if not video_path:
                    for vp in all_videos:
                        vp_basename = os.path.splitext(os.path.basename(vp))[0]
                        if video_name in vp_basename or vp_basename in video_name:
                            video_path = vp
                            break
            
            if not video_path:
                logger.info("Could not find video for snippet %s", snippet_id)
                logger.info("  Tried video names: %s", video_name_candidates)
                logger.info("  Available videos: %s", [os.path.basename(v) for v in all_videos[:10]])
                return None
            
            video_name = video_name_candidates[0] if video_name_candidates else ""
            
            mask_path = None
            mask_dirs = [
                os.path.join(experiment_path, "masks"),
                os.path.join(experiment_path, "segmentation_masks"),
                experiment_path
            ]
            
            for mask_dir in mask_dirs:
                if not os.path.exists(mask_dir):
                    continue
                for f in os.listdir(mask_dir):
                    if f.endswith(('.h5', '.hdf5')) and (
                        video_name in f or
                        f.replace('_mask.h5', '').replace('_mask.hdf5', '') in video_name
                    ):
                        mask_path = os.path.join(mask_dir, f)
                        break
                if mask_path:
                    break
            
            centroids = {}
            box_size = 288
            
            if mask_path:
                try:
                    mask_data = load_segmentation_data(mask_path)
                    frame_objects = mask_data.get('frame_objects', [])
                    start_offset = mask_data.get('start_offset', 0)
                    
                    for frame_idx in range(start_frame, end_frame + 1):
                        mask_frame_idx = frame_idx - start_offset
                        if mask_frame_idx < 0 or mask_frame_idx >= len(frame_objects):
                            continue

                        for obj in frame_objects[mask_frame_idx]:
                            mask = obj.get("mask")
                            bbox = obj.get("bbox")
                            if mask is None or bbox is None or not mask.any():
                                continue
                            ys, xs = mask.nonzero()
                            if len(xs) == 0:
                                continue
                            x_min, y_min, _, _ = bbox
                            cx = int(x_min + xs.mean())
                            cy = int(y_min + ys.mean())
                            centroids[frame_idx] = (cx, cy)
                            break
                except Exception as e:
                    logger.error("Error loading mask data: %s", e)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            raw_clips_dir = os.path.join(experiment_path, "data", "clips", "raw_clips")
            os.makedirs(raw_clips_dir, exist_ok=True)
            
            safe_label = re.sub(r'[^\w\-_]', '_', label)
            output_filename = f"{video_name}_{safe_label}_{snippet_id}.mp4"
            output_path = os.path.join(raw_clips_dir, output_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
            
            cap.release()
            out.release()
            
            return output_path
            
        except Exception as e:
            logger.error("Error extracting raw clip from snippet: %s", e, exc_info=True)
            return None

    def _on_review_ready(self, results: dict, classes: list, is_ovr: bool,
                         clip_length: int, target_fps: int):
        """Populate the Review tab when inference finishes and switch to it."""
        self.review_widget.load_from_inference(results, classes, is_ovr, clip_length, target_fps)
        self.tabs.setCurrentWidget(self.review_widget)

    def _on_annotations_updated(self):
        self.labeling_widget.refresh_clip_list()
        self.training_widget.refresh_annotation_info()


    def _on_tracking_completed(self, video_path: str, mask_path: str):
        """Handle tracking completion - switch to registration tab and load data."""
        self.tabs.setCurrentWidget(self.registration_widget)
        
        self.registration_widget.load_from_segmentation(video_path, mask_path)
    
    def _on_embeddings_extracted(self, matrix_path: str, metadata_path: str):
        """Handle embedding extraction completion - auto-load into clustering tab."""
        self.tabs.setCurrentWidget(self.clustering_widget)
        self.clustering_widget.load_from_registration(matrix_path, metadata_path)
        self._labeling_setup_prompted_for_clusters = False
    
    def _open_video(self):
        """Open video file dialog."""
        video_dir = self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos"))
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            video_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if video_path:
            from .video_utils import ensure_video_in_experiment
            video_path = ensure_video_in_experiment(video_path, self.config, self)
            self.labeling_widget.add_source_videos([video_path], select_last=True)
            self.tabs.setCurrentWidget(self.labeling_widget)

    def _create_experiment(self):
        """Create a new experiment directory structure."""
        name, ok = QInputDialog.getText(self, "Create Experiment", "Experiment name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        
        experiments_dir = self.config.get("experiments_dir")
        if not experiments_dir:
            from singlebehaviorlab._paths import get_experiments_dir
            experiments_dir = str(get_experiments_dir())
            self.config["experiments_dir"] = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        
        experiment_path = os.path.abspath(os.path.join(experiments_dir, name))
        if os.path.exists(experiment_path) and os.listdir(experiment_path):
            reply = QMessageBox.question(
                self,
                "Overwrite Experiment",
                f"The experiment folder '{experiment_path}' already exists and is not empty.\n"
                "Do you want to reuse it? Existing files will be kept.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        os.makedirs(experiment_path, exist_ok=True)
        
        data_dir = os.path.join(experiment_path, "data")
        raw_videos_dir = os.path.join(data_dir, "raw_videos")
        clips_dir = os.path.join(data_dir, "clips")
        annotations_dir = os.path.join(data_dir, "annotations")
        models_dir = os.path.join(experiment_path, "models", "behavior_heads")
        
        for path in [data_dir, raw_videos_dir, clips_dir, annotations_dir, models_dir]:
            os.makedirs(path, exist_ok=True)
        
        annotation_file = os.path.join(annotations_dir, "annotations.json")
        if not os.path.exists(annotation_file):
            default_annotations = {"classes": [], "clips": []}
            with open(annotation_file, "w", encoding="utf-8") as f:
                json.dump(default_annotations, f, indent=2)

        profiles_file = os.path.join(experiment_path, "training_profiles.json")
        if not os.path.exists(profiles_file):
            from singlebehaviorlab._paths import get_training_profiles_path
            src = get_training_profiles_path()
            if src and os.path.exists(str(src)):
                import shutil
                shutil.copy2(str(src), profiles_file)
            else:
                with open(profiles_file, "w", encoding="utf-8") as f:
                    json.dump({"Default": {}}, f, indent=2)

        experiment_config_path = os.path.join(experiment_path, "config.yaml")
        
        new_config = dict(self.config)
        new_config.update({
            "experiment_name": name,
            "experiment_path": experiment_path,
            "data_dir": data_dir,
            "raw_videos_dir": raw_videos_dir,
            "clips_dir": clips_dir,
            "annotations_dir": annotations_dir,
            "annotation_file": annotation_file,
            "training_clips_dir": clips_dir,
            "training_annotation_file": annotation_file,
            "models_dir": models_dir,
            "config_path": experiment_config_path,
            # Use a conservative default that works well for small labeled datasets.
            "default_weight_decay": 0.001,
        })
        
        self._update_config(new_config)
        self._save_experiment_config()
        self._apply_config_to_widgets()
        self._labeling_setup_prompted_for_clusters = False
        try:
            with open(profiles_file, "r", encoding="utf-8") as f:
                profiles_data = json.load(f) or {}
            if isinstance(profiles_data, dict) and "LowInputData" in profiles_data:
                self.training_widget.apply_training_config(profiles_data["LowInputData"])
                self.training_widget.current_profile_name = "LowInputData"
        except Exception:
            pass
        self._update_window_title()
        QMessageBox.information(self, "Experiment created", f"Experiment '{name}' created at:\n{experiment_path}")

    def _load_experiment(self):
        """Load an existing experiment configuration."""
        start_dir = self.config.get("experiments_dir", os.getcwd())
        config_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Experiment",
            start_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not config_path:
            return
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load experiment config:\n{exc}")
            return
        
        experiment_path = loaded.get("experiment_path") or os.path.dirname(os.path.abspath(config_path))
        loaded["experiment_path"] = experiment_path
        loaded["config_path"] = config_path
        loaded.setdefault("experiments_dir", self.config.get("experiments_dir"))
        
        # Resolve experiment paths relative to the loaded config when needed.
        def _abs_path(key, default_subpath):
            if key not in loaded or not loaded[key]:
                loaded[key] = os.path.join(experiment_path, default_subpath)
            elif not os.path.isabs(loaded[key]):
                loaded[key] = os.path.join(experiment_path, loaded[key])
        
        _abs_path("data_dir", "data")
        _abs_path("raw_videos_dir", os.path.join("data", "raw_videos"))
        _abs_path("clips_dir", os.path.join("data", "clips"))
        _abs_path("annotations_dir", os.path.join("data", "annotations"))
        _abs_path("models_dir", os.path.join("models", "behavior_heads"))
        _abs_path("annotation_file", os.path.join("data", "annotations", "annotations.json"))
        
        self._update_config(loaded)
        self._apply_config_to_widgets()
        self._labeling_setup_prompted_for_clusters = False
        self._update_window_title()
        QMessageBox.information(self, "Experiment loaded", f"Loaded experiment from:\n{config_path}")

    def _save_experiment(self):
        """Save current experiment configuration."""
        config_path = self.config.get("config_path")
        if not config_path:
            QMessageBox.warning(self, "Save experiment", "No experiment is currently loaded.")
            return
        self._save_experiment_config()
        QMessageBox.information(self, "Experiment saved", f"Experiment saved to:\n{config_path}")

    def _save_experiment_config(self):
        """Write the current config to disk."""
        config_path = self.config.get("config_path")
        if not config_path:
            return
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(self.config), f, sort_keys=False)

    def _update_config(self, new_values: dict):
        """Update the shared config dictionary in place."""
        self.config.clear()
        self.config.update(new_values)

    def _apply_config_to_widgets(self):
        """Propagate config changes to all widgets."""
        self.segmentation_widget.update_config(self.config)
        self.registration_widget.update_config(self.config)
        self.clustering_widget.update_config(self.config)
        self.labeling_widget.update_config(self.config)
        self.training_widget.update_config(self.config)
        self.inference_widget.update_config(self.config)
        self.review_widget.update_config(self.config)
        self.analysis_widget.update_config(self.config)

    def _update_window_title(self):
        """Update window title with experiment name."""
        name = self.config.get("experiment_name")
        base_title = "SingleBehavior Lab"
        if name:
            self.setWindowTitle(f"{base_title} - {name}")
        else:
            self.setWindowTitle(base_title)


