"""
Clustering Widget for SingleBehavior Lab.
Integrates preprocessing and clustering (UMAP, Leiden, HBSCAN) of behaviorome embeddings.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)
import numpy as np
import umap
import leidenalg as la
import igraph as ig
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import pickle

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QCheckBox, QGroupBox, QScrollArea, QSplitter,
    QMessageBox, QListWidget, QTextEdit, QFileDialog, QProgressBar, QDialog,
    QSizePolicy, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from .plot_integration import PlotlyWidget
from .qt_helpers import create_status_label, update_status_label, create_section

class ClusteringWorker(QThread):
    """Worker thread for clustering computation"""
    finished = pyqtSignal(str, object)  # status message, figure
    
    def __init__(self, clustering_widget, params, data):
        super().__init__()
        self.clustering_widget = clustering_widget
        self.params = params
        self.data = data
    
    def run(self):
        """Run clustering in background thread"""
        try:
            status, fig = self.clustering_widget.perform_clustering(self.data, **self.params)
            self.finished.emit(status, fig)
        except Exception as e:
            logger.error("Error running clustering: %s", e, exc_info=True)
            self.finished.emit(f"Error: {str(e)}", None)


class LoadDataWorker(QThread):
    """Background loader for large embedding matrices."""
    loaded = pyqtSignal(object, object, str)  # matrix_df, metadata_df, metadata_path
    error = pyqtSignal(str)
    
    def __init__(self, matrix_path: str, metadata_path: str | None):
        super().__init__()
        self.matrix_path = matrix_path
        self.metadata_path = metadata_path
    
    def run(self):
        try:
            matrix_df = self._load_matrix(self.matrix_path)
            metadata_df = None
            meta_path = self.metadata_path
            if self.metadata_path and os.path.exists(self.metadata_path):
                metadata_df = self._load_metadata(self.metadata_path)
            else:
                meta_path = None
            self.loaded.emit(matrix_df, metadata_df, meta_path)
        except Exception as e:
            logger.error("Error loading data: %s", e, exc_info=True)
            self.error.emit(str(e))
    
    def _load_matrix(self, path: str) -> pd.DataFrame:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as data:
                matrix = data["matrix"]
                feature_names = data["feature_names"]
                snippet_ids = data["snippet_ids"] if "snippet_ids" in data else data.get("span_ids", None)  # Backward compatibility
                if snippet_ids is None:
                    # Fallback: generate snippet IDs
                    snippet_ids = np.array([f'snippet{i+1}' for i in range(matrix.shape[1])])
                return pd.DataFrame(matrix, index=feature_names, columns=snippet_ids)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        # default csv
        return pd.read_csv(path, index_col=0)
    
    def _load_metadata(self, path: str) -> pd.DataFrame:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as data:
                metadata_values = data["metadata"]
                columns = list(data["columns"])
                return pd.DataFrame(metadata_values, columns=columns)
        elif path.endswith(".parquet"):
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)


class ClusterExportDialog(QDialog):
    """Dialog shown after cluster export with option to load the new dataset."""
    
    def __init__(self, parent, message: str, matrix_path: str, metadata_path: str):
        super().__init__(parent)
        self.matrix_path = matrix_path
        self.metadata_path = metadata_path
        self.load_requested = False
        
        self.setWindowTitle("Cluster export complete")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Message label
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._on_load_clicked)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(load_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
    
    def _on_load_clicked(self):
        """Mark that user wants to load the dataset."""
        self.load_requested = True
        self.accept()


class ClusteringWidget(QWidget):
    """Widget for clustering behaviorome embeddings."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.matrix_data = None
        self.metadata = None
        self.metadata_file_path = None  # Store path to metadata file for updates
        self.processed_data = None
        self.embedding = None
        self.clusters = None
        self.current_fig = None
        self.current_df = None
        self.snippet_to_clip_map = {}  # Map snippet_id -> clip_path
        
        # Preprocessing state
        self.selected_features = None
        
        self._setup_ui()
        
    def update_config(self, config: dict):
        """Update configuration."""
        self.config = config
        
    def _setup_ui(self):
        """Setup UI components."""
        self.main_layout = QVBoxLayout(self)
        
        # Splitter: Settings on left, Plot on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Settings (Scrollable)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(300)
        settings_scroll.setMaximumWidth(350)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(5, 5, 5, 5)
        settings_layout.setSpacing(5)
        
        # 1. Data Loading Section
        data_group = QGroupBox("Data loading")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(5)
        
        self.load_status_label = QLabel("No data loaded")
        self.load_status_label.setWordWrap(True)
        self.load_status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.load_status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Set size policy to prevent expansion
        self.load_status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        data_layout.addWidget(self.load_status_label)
        
        self.load_btn = QPushButton("Load from registration")
        self.load_btn.clicked.connect(self.load_data)
        data_layout.addWidget(self.load_btn)
        
        self.load_file_btn = QPushButton("Load external matrix...")
        self.load_file_btn.clicked.connect(self.load_external_data)
        data_layout.addWidget(self.load_file_btn)

        self.load_progress = QProgressBar()
        self.load_progress.setVisible(False)
        self.load_progress.setRange(0, 0)  # indeterminate
        data_layout.addWidget(self.load_progress)
        
        data_group.setLayout(data_layout)
        settings_layout.addWidget(data_group)
        
        # 2. Preprocessing Section (Collapsible-ish via GroupBox)
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout()
        
        # Normalization
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalization:"))
        self.normalization_method = QComboBox()
        self.normalization_method.addItems(["none", "l2", "standard", "minmax"])
        self.normalization_method.setCurrentText("none") # Default for embeddings
        self.normalization_method.setToolTip(
            "None: Embeddings usually normalized\n"
            "L2: Unit norm (good for embeddings)\n"
            "Standard: Zero mean, unit var\n"
            "MinMax: [0,1] range"
        )
        norm_row.addWidget(self.normalization_method)
        preprocess_layout.addLayout(norm_row)

        self.subtract_video_mean_check = QCheckBox("Subtract per-video mean")
        self.subtract_video_mean_check.setToolTip(
            "Remove the average embedding of each video/group before clustering.\n"
            "Reduces sensitivity to camera setup, lighting, and background\n"
            "while preserving within-video behavior differences."
        )
        preprocess_layout.addWidget(self.subtract_video_mean_check)


        self.preprocess_btn = QPushButton("Apply preprocessing")
        self.preprocess_btn.clicked.connect(self.apply_preprocessing)
        preprocess_layout.addWidget(self.preprocess_btn)
        
        self.preprocess_status = QLabel("Ready")
        self.preprocess_status.setWordWrap(True)
        self.preprocess_status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        preprocess_layout.addWidget(self.preprocess_status)
        
        preprocess_group.setLayout(preprocess_layout)
        settings_layout.addWidget(preprocess_group)
        
        # 3. Clustering Parameters
        cluster_params_group = QGroupBox("Clustering & projection")
        cluster_params_layout = QVBoxLayout()
        
        cluster_params_layout.addWidget(QLabel("<b>Dimensionality Reduction (UMAP)</b>"))
        
        self.n_neighbors, self.n_neighbors_lbl = self._create_slider(
            "Neighbors:", 2, 200, 30, 1
        )
        cluster_params_layout.addWidget(self._slider_widget(self.n_neighbors, self.n_neighbors_lbl))
        
        self.min_dist, self.min_dist_lbl = self._create_slider(
            "Min Dist:", 0.0, 0.99, 0.1, 0.01, is_float=True
        )
        cluster_params_layout.addWidget(self._slider_widget(self.min_dist, self.min_dist_lbl))
        
        self.n_components, self.n_components_lbl = self._create_slider(
            "Components:", 2, 3, 2, 1
        )
        cluster_params_layout.addWidget(self._slider_widget(self.n_components, self.n_components_lbl))
        
        # Clustering Method
        cluster_params_layout.addWidget(QLabel("<b>Clustering Method</b>"))
        self.clustering_method = QComboBox()
        self.clustering_method.addItems(['leiden', 'hdbscan']) # Only these two + UMAP visualization
        self.clustering_method.currentIndexChanged.connect(self._toggle_clustering_params)
        cluster_params_layout.addWidget(self.clustering_method)
        
        # Leiden Params
        self.leiden_container = QWidget()
        leiden_layout = QVBoxLayout(self.leiden_container)
        leiden_layout.setContentsMargins(0,0,0,0)
        
        self.leiden_resolution, self.leiden_res_lbl = self._create_slider(
            "Resolution:", 0.1, 5.0, 1.0, 0.1, is_float=True
        )
        leiden_layout.addWidget(self._slider_widget(self.leiden_resolution, self.leiden_res_lbl))
        
        self.leiden_k, self.leiden_k_lbl = self._create_slider(
            "K-Neighbors:", 2, 100, 15, 1
        )
        leiden_layout.addWidget(self._slider_widget(self.leiden_k, self.leiden_k_lbl))
        
        cluster_params_layout.addWidget(self.leiden_container)
        
        # HDBSCAN Params
        self.hdbscan_container = QWidget()
        hdbscan_layout = QVBoxLayout(self.hdbscan_container)
        hdbscan_layout.setContentsMargins(0,0,0,0)
        
        self.min_cluster_size, self.min_cluster_size_lbl = self._create_slider(
            "Min Cluster Size:", 2, 100, 5, 1
        )
        hdbscan_layout.addWidget(self._slider_widget(self.min_cluster_size, self.min_cluster_size_lbl))
        
        self.min_samples, self.min_samples_lbl = self._create_slider(
            "Min Samples:", 1, 50, 1, 1
        )
        hdbscan_layout.addWidget(self._slider_widget(self.min_samples, self.min_samples_lbl))
        
        self.hdbscan_epsilon, self.hdbscan_eps_lbl = self._create_slider(
            "Epsilon:", 0.0, 5.0, 0.0, 0.1, is_float=True
        )
        hdbscan_layout.addWidget(self._slider_widget(self.hdbscan_epsilon, self.hdbscan_eps_lbl))
        
        cluster_params_layout.addWidget(self.hdbscan_container)
        self.hdbscan_container.hide() # Hide initially
        
        # Run Button
        self.run_btn = QPushButton("Run analysis")
        self.run_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 5px;")
        self.run_btn.clicked.connect(self.run_clustering)
        cluster_params_layout.addWidget(self.run_btn)
        
        cluster_params_group.setLayout(cluster_params_layout)
        settings_layout.addWidget(cluster_params_group)
        
        # 4. Metadata Management
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout()
        self.manage_metadata_btn = QPushButton("Manage metadata")
        self.manage_metadata_btn.clicked.connect(self.open_metadata_manager)
        metadata_layout.addWidget(self.manage_metadata_btn)
        metadata_group.setLayout(metadata_layout)
        settings_layout.addWidget(metadata_group)
        
        # 5. Cluster Export
        cluster_export_group = QGroupBox("Cluster export")
        cluster_export_layout = QVBoxLayout()
        
        # Help button with explanation
        help_row = QHBoxLayout()
        help_row.addWidget(QLabel("Select clusters:"))
        self.cluster_export_help_btn = QPushButton("?")
        self.cluster_export_help_btn.setMaximumWidth(30)
        self.cluster_export_help_btn.setToolTip("Click for help")
        self.cluster_export_help_btn.clicked.connect(self._show_cluster_export_help)
        help_row.addWidget(self.cluster_export_help_btn)
        help_row.addStretch()
        cluster_export_layout.addLayout(help_row)
        
        self.cluster_export_list = QListWidget()
        self.cluster_export_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        cluster_export_layout.addWidget(self.cluster_export_list)
        
        self.use_raw_data_checkbox = QCheckBox("Use raw data")
        self.use_raw_data_checkbox.setToolTip("If checked, exports raw (unnormalized) data. Otherwise exports preprocessed data.")
        cluster_export_layout.addWidget(self.use_raw_data_checkbox)
        
        self.extract_cluster_btn = QPushButton("Extract selected cluster")
        self.extract_cluster_btn.clicked.connect(self._extract_cluster)
        self.extract_cluster_btn.setEnabled(False)
        cluster_export_layout.addWidget(self.extract_cluster_btn)
        
        self.exclude_clusters_btn = QPushButton("Exclude selected clusters")
        self.exclude_clusters_btn.clicked.connect(self._exclude_clusters)
        self.exclude_clusters_btn.setEnabled(False)
        cluster_export_layout.addWidget(self.exclude_clusters_btn)
        
        cluster_export_group.setLayout(cluster_export_layout)
        settings_layout.addWidget(cluster_export_group)
        
        settings_layout.addStretch()
        settings_widget.setLayout(settings_layout)
        settings_scroll.setWidget(settings_widget)
        
        splitter.addWidget(settings_scroll)
        
        # Middle Panel: Plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = PlotlyWidget()
        # Set up click callback for snippet selection
        self.plot_widget.set_click_callback(self._on_umap_point_clicked)
        plot_layout.addWidget(self.plot_widget)
        
        # Status label (minimal, at bottom)
        self.status_label = QLabel("Ready")
        self.status_label.setMaximumHeight(20)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        plot_layout.addWidget(self.status_label)
        
        splitter.addWidget(plot_container)
        
        # Right Panel: Plot Settings
        plot_settings_scroll = QScrollArea()
        plot_settings_scroll.setWidgetResizable(True)
        plot_settings_scroll.setMinimumWidth(300)
        plot_settings_scroll.setMaximumWidth(350)
        plot_settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        plot_settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        plot_settings_widget = QWidget()
        plot_settings_layout = QVBoxLayout(plot_settings_widget)
        plot_settings_layout.setContentsMargins(5, 5, 5, 5)
        plot_settings_layout.setSpacing(5)
        
        # Plot Settings Group
        plot_settings_group = QGroupBox("Plot settings")
        plot_settings_group_layout = QVBoxLayout()
        plot_settings_group_layout.setSpacing(5)
        
        # Color Theme Selector
        plot_settings_group_layout.addWidget(QLabel("<b>Color Theme:</b>"))
        self.color_theme_combo = QComboBox()
        self.color_theme_combo.addItems(["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"])
        self.color_theme_combo.setCurrentText("simple_white")
        self.color_theme_combo.currentIndexChanged.connect(self._update_plots_by_metadata)
        plot_settings_group_layout.addWidget(self.color_theme_combo)
        
        # Metadata Column Selector
        plot_settings_group_layout.addWidget(QLabel("<b>Group by metadata column:</b>"))
        self.metadata_column_combo = QComboBox()
        self.metadata_column_combo.addItem("None (Show all clusters)")
        self.metadata_column_combo.currentTextChanged.connect(self._update_plots_by_metadata)
        plot_settings_group_layout.addWidget(self.metadata_column_combo)
        
        # Point Size for grouped plots
        point_size_label = QLabel("Point size:")
        self.plot_point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.plot_point_size_slider.setMinimum(1)
        self.plot_point_size_slider.setMaximum(20)
        self.plot_point_size_slider.setValue(5)
        self.plot_point_size_label = QLabel("5")
        self.plot_point_size_slider.valueChanged.connect(
            lambda v: (self.plot_point_size_label.setText(str(v)), self._update_plots_by_metadata())
        )
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(point_size_label)
        point_size_layout.addWidget(self.plot_point_size_slider)
        point_size_layout.addWidget(self.plot_point_size_label)
        plot_settings_group_layout.addLayout(point_size_layout)
        
        # Plot Type Selector
        plot_settings_group_layout.addWidget(QLabel("<b>Plot type:</b>"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["UMAP", "Cluster Proportions", "Single Cluster Analysis", "Spatial cluster distribution"])
        self.plot_type_combo.currentTextChanged.connect(self._update_plot_type)
        plot_settings_group_layout.addWidget(self.plot_type_combo)
        
        # Single Cluster Analysis (only show when relevant)
        self.single_cluster_label = QLabel("<b>Select cluster:</b>")
        self.single_cluster_combo = QComboBox()
        self.single_cluster_combo.addItem("None")
        self.single_cluster_combo.currentTextChanged.connect(self._update_plots_by_metadata)
        plot_settings_group_layout.addWidget(self.single_cluster_label)
        plot_settings_group_layout.addWidget(self.single_cluster_combo)
        self.single_cluster_label.setVisible(False)
        self.single_cluster_combo.setVisible(False)
        
        # Spatial distribution: Video and Object selectors
        self.spatial_video_label = QLabel("<b>Video:</b>")
        self.spatial_video_combo = QComboBox()
        self.spatial_video_combo.addItem("All")
        self.spatial_video_combo.currentTextChanged.connect(self._on_spatial_video_changed)
        plot_settings_group_layout.addWidget(self.spatial_video_label)
        plot_settings_group_layout.addWidget(self.spatial_video_combo)
        self.spatial_video_label.setVisible(False)
        self.spatial_video_combo.setVisible(False)
        
        self.spatial_object_label = QLabel("<b>Object:</b>")
        self.spatial_object_combo = QComboBox()
        self.spatial_object_combo.addItem("All")
        self.spatial_object_combo.currentTextChanged.connect(self._update_plots_by_metadata)
        plot_settings_group_layout.addWidget(self.spatial_object_label)
        plot_settings_group_layout.addWidget(self.spatial_object_combo)
        self.spatial_object_label.setVisible(False)
        self.spatial_object_combo.setVisible(False)
        
        plot_settings_group.setLayout(plot_settings_group_layout)
        plot_settings_layout.addWidget(plot_settings_group)
        
        # Export Group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_plot_btn = QPushButton("Export plot (PDF/SVG)")
        self.export_plot_btn.clicked.connect(self._export_plot)
        self.export_plot_btn.setEnabled(False)
        export_layout.addWidget(self.export_plot_btn)
        
        self.export_csv_btn = QPushButton("Export results (CSV)")
        self.export_csv_btn.clicked.connect(self.export_results)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)
        
        export_group.setLayout(export_layout)
        plot_settings_layout.addWidget(export_group)
        
        # Analysis State Group
        state_group = QGroupBox("Analysis state")
        state_layout = QVBoxLayout()
        
        self.save_state_btn = QPushButton("Save full analysis")
        self.save_state_btn.clicked.connect(self._save_analysis_state)
        state_layout.addWidget(self.save_state_btn)
        
        self.load_state_btn = QPushButton("Load full analysis")
        self.load_state_btn.clicked.connect(self._load_analysis_state)
        state_layout.addWidget(self.load_state_btn)
        
        self.file_info_label = QLabel("No analysis loaded")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("color: gray; font-style: italic;")
        state_layout.addWidget(self.file_info_label)
        
        state_group.setLayout(state_layout)
        plot_settings_layout.addWidget(state_group)
        
        plot_settings_layout.addStretch()
        plot_settings_widget.setLayout(plot_settings_layout)
        plot_settings_scroll.setWidget(plot_settings_widget)
        
        splitter.addWidget(plot_settings_scroll)
        splitter.setStretchFactor(1, 3)  # Plot takes most space
        
        self.main_layout.addWidget(splitter)
        
    def _create_slider(self, label_text, min_val, max_val, default, step, is_float=False):
        slider = QSlider(Qt.Orientation.Horizontal)
        if is_float:
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
            slider.setSingleStep(int(step * 100))
            value_label = QLabel(f"{default:.2f}")
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v/100:.2f}"))
        else:
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            slider.setSingleStep(step)
            value_label = QLabel(str(default))
            slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.label_text = label_text
        return slider, value_label

    def _slider_widget(self, slider, label):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(slider.label_text))
        layout.addWidget(slider)
        layout.addWidget(label)
        widget.setLayout(layout)
        return widget
        
    def _get_slider_value(self, slider, is_float=False):
        if is_float:
            return slider.value() / 100.0
        return slider.value()

    def _toggle_clustering_params(self):
        method = self.clustering_method.currentText()
        if method == 'leiden':
            self.leiden_container.show()
            self.hdbscan_container.hide()
        else:
            self.leiden_container.hide()
            self.hdbscan_container.show()

    def _find_latest_behaviorome(self, directory: str):
        """Pick the latest behaviorome matrix with preferred extensions."""
        exts = ["npz", "parquet", "csv"]
        candidates = []
        for fname in os.listdir(directory):
            for ext in exts:
                if fname.startswith("behaviorome_") and fname.endswith(f"_matrix.{ext}"):
                    candidates.append(os.path.join(directory, fname))
        if not candidates:
            return None, None
        candidates.sort(reverse=True)
        matrix_path = candidates[0]
        base, ext = os.path.splitext(matrix_path)
        # ext includes .npz etc; build metadata preference (NPZ first, then Parquet, then CSV)
        metadata_path = None
        for meta_ext in ["npz", "parquet", "csv"]:
            candidate_meta = base.replace("_matrix", "_metadata") + f".{meta_ext}"
            if os.path.exists(candidate_meta):
                metadata_path = candidate_meta
                break
        return matrix_path, metadata_path

    def _load_files_async(self, matrix_path: str, metadata_path: str | None):
        """Start background load with progress bar."""
        self.load_progress.setVisible(True)
        self.load_status_label.setText("Loading data...")
        self.load_status_label.setStyleSheet("color: black;")
        self.load_progress.setRange(0, 0)
        self.load_worker = LoadDataWorker(matrix_path, metadata_path)
        self.load_worker.loaded.connect(self._on_loaded_data)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.start()

    def _on_loaded_data(self, matrix_df: pd.DataFrame, metadata_df: pd.DataFrame | None, metadata_path: str | None):
        self.load_progress.setVisible(False)
        # Downcast to float32 to reduce memory for large matrices
        try:
            self.matrix_data = matrix_df.astype(np.float32, copy=False)
        except Exception:
            self.matrix_data = matrix_df
        self.metadata = metadata_df
        self.metadata_file_path = metadata_path
        self.processed_data = None
        self.preprocess_status.setText("Raw data loaded")
        shape = (self.matrix_data.shape[0], self.matrix_data.shape[1]) if self.matrix_data is not None else (0, 0)
        
        # Format status text to fit in container (truncate long filenames)
        max_filename_len = 35
        status_text = f"Matrix: {shape[0]} x {shape[1]}"
        if metadata_path:
            meta_filename = os.path.basename(metadata_path)
            if len(meta_filename) > max_filename_len:
                meta_filename = meta_filename[:max_filename_len-3] + "..."
            status_text += f"\nMeta: {meta_filename}"
        
        self.load_status_label.setText(status_text)
        self.load_status_label.setStyleSheet("color: green;")
        self.apply_preprocessing()

    def _on_load_error(self, msg: str):
        self.load_progress.setVisible(False)
        QMessageBox.critical(self, "Load Error", f"Failed to load data: {msg}")

    def load_data(self):
        """Load data from experiment folder."""
        experiment_path = self.config.get("experiment_path")
        if not experiment_path:
            QMessageBox.warning(self, "No Experiment", "Please create or load an experiment first.")
            return
            
        registered_clips_dir = os.path.join(experiment_path, "registered_clips")
        if not os.path.exists(registered_clips_dir):
            QMessageBox.warning(self, "No Data", "No registered clips directory found.")
            return
            
        matrix_path, metadata_path = self._find_latest_behaviorome(registered_clips_dir)
        if not matrix_path:
            QMessageBox.warning(self, "No Data", "No behaviorome matrix files found (npz/parquet/csv).")
            return
        self._load_files_async(matrix_path, metadata_path)

    def load_external_data(self):
        """Load external data (npz/parquet/csv)."""
        matrix_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Feature Matrix",
            "",
            "Matrices (*.npz *.parquet *.csv);;All Files (*)"
        )
        if not matrix_path:
            return
            
        metadata_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Metadata",
            os.path.dirname(matrix_path),
            "Metadata (*.parquet *.csv);;All Files (*)"
        )
        self._load_files_async(matrix_path, metadata_path if metadata_path else None)

    def load_from_registration(self, matrix_path: str, metadata_path: str):
        """Load data from registration tab (NPZ/Parquet paths)."""
        self._load_data_files(matrix_path, metadata_path)
    
    def _load_csvs(self, matrix_path, metadata_path):
        """Load from CSV files (legacy support)."""
        self._load_data_files(matrix_path, metadata_path)
    
    def _load_data_files(self, matrix_path: str, metadata_path: str = None):
        """Load data from matrix and metadata files (supports NPZ, Parquet, CSV)."""
        try:
            # Load matrix
            if matrix_path.endswith('.npz'):
                npz_data = np.load(matrix_path, allow_pickle=True)  # Need allow_pickle for string arrays
                matrix = npz_data['matrix']  # features x snippets
                feature_names = npz_data['feature_names']
                snippet_ids = npz_data['snippet_ids'] if 'snippet_ids' in npz_data else npz_data.get('span_ids', None)  # Backward compatibility
                if snippet_ids is None:
                    # Fallback: generate snippet IDs
                    snippet_ids = np.array([f'snippet{i+1}' for i in range(matrix.shape[1])])
                self.matrix_data = pd.DataFrame(matrix, index=feature_names, columns=snippet_ids)
            elif matrix_path.endswith('.parquet'):
                self.matrix_data = pd.read_parquet(matrix_path, engine='pyarrow')
            else:  # CSV
                self.matrix_data = pd.read_csv(matrix_path, index_col=0)
            
            # Load metadata
            if metadata_path:
                if metadata_path.endswith('.npz'):
                    npz_meta = np.load(metadata_path, allow_pickle=True)  # Need allow_pickle for string arrays
                    metadata_array = npz_meta['metadata']
                    columns = npz_meta['columns']
                    self.metadata = pd.DataFrame(metadata_array, columns=columns)
                elif metadata_path.endswith('.parquet'):
                    self.metadata = pd.read_parquet(metadata_path, engine='pyarrow')
                else:  # CSV
                    self.metadata = pd.read_csv(metadata_path)
                self.metadata_file_path = metadata_path
            else:
                self.metadata_file_path = None
                self.metadata = None
            
            # Reset processing
            self.processed_data = None
            self.preprocess_status.setText("Raw data loaded")
            
            # Format status text to fit in container
            filename = os.path.basename(matrix_path)
            max_filename_len = 35
            if len(filename) > max_filename_len:
                filename = filename[:max_filename_len-3] + "..."
            
            status_text = f"Matrix: {self.matrix_data.shape[0]} x {self.matrix_data.shape[1]}\nFile: {filename}"
            if metadata_path:
                meta_filename = os.path.basename(metadata_path)
                if len(meta_filename) > max_filename_len:
                    meta_filename = meta_filename[:max_filename_len-3] + "..."
                status_text += f"\nMeta: {meta_filename}"
            
            self.load_status_label.setText(status_text)
            self.load_status_label.setStyleSheet("color: green;")
            
            # Auto-apply default preprocessing
            self.apply_preprocessing()
            
            # Refresh metadata columns in plot settings
            self._refresh_metadata_columns()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data: {e}")

    def apply_preprocessing(self):
        """Apply normalization."""
        if self.matrix_data is None:
            return
            
        try:
            data = self.matrix_data.copy()
            
            # Transpose for sklearn (samples as rows)
            # Matrix format: Rows=Features, Cols=Samples usually? 
            # Check registration widget format: 
            # feature_matrix.shape = (n_samples, embed_dim) in extraction worker list, then transposed?
            # Registration widget: pd.DataFrame(feature_matrix.T, index=feature_names, columns=snippet_ids)
            # So Rows are Features (dimensions), Columns are Samples (snippets).
            # sklearn expects Samples as Rows. So we transpose.
            
            X = data.T
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            steps = []

            if self.subtract_video_mean_check.isChecked() and self.metadata is not None:
                group_col = None
                for col in ("group", "video_id"):
                    if col in self.metadata.columns:
                        group_col = col
                        break
                if group_col is not None:
                    snippet_col = "snippet" if "snippet" in self.metadata.columns else None
                    if snippet_col:
                        for grp in self.metadata[group_col].unique():
                            grp_snippets = self.metadata.loc[
                                self.metadata[group_col] == grp, snippet_col
                            ].values
                            mask = X.index.isin(grp_snippets)
                            if mask.sum() > 1:
                                X.loc[mask] -= X.loc[mask].mean(axis=0)
                        steps.append("video-mean-sub")


            norm_method = self.normalization_method.currentText()
            if norm_method == 'standard':
                X_norm = StandardScaler().fit_transform(X)
            elif norm_method == 'minmax':
                X_norm = MinMaxScaler().fit_transform(X)
            elif norm_method == 'l2':
                X_norm = Normalizer(norm='l2').fit_transform(X)
            else:
                X_norm = X.values if hasattr(X, 'values') else X
            if norm_method != 'none':
                steps.append(norm_method)

            self.processed_data = pd.DataFrame(X_norm, index=X.index, columns=range(X_norm.shape[1]))

            self.preprocess_status.setText(f"Preprocessed: {' → '.join(steps) or 'none'}")
            self.preprocess_status.setStyleSheet("color: green;")
            
        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Error", f"Error: {e}")

    def run_clustering(self):
        """Start clustering worker."""
        if self.processed_data is None:
            QMessageBox.warning(self, "No Data", "Please load and preprocess data first.")
            return
            
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.status_label.setText("Computing clustering...")
        
        params = {
            'n_neighbors': self._get_slider_value(self.n_neighbors),
            'min_dist': self._get_slider_value(self.min_dist, is_float=True),
            'n_components': self._get_slider_value(self.n_components),
            'method': self.clustering_method.currentText(),
            'leiden_resolution': self._get_slider_value(self.leiden_resolution, is_float=True),
            'leiden_k': self._get_slider_value(self.leiden_k),
            'min_cluster_size': self._get_slider_value(self.min_cluster_size),
            'min_samples': self._get_slider_value(self.min_samples),
            'hdbscan_epsilon': self._get_slider_value(self.hdbscan_epsilon, is_float=True)
        }
        
        self.worker = ClusteringWorker(self, params, self.processed_data)
        self.worker.finished.connect(self.on_clustering_finished)
        self.worker.start()

    def perform_clustering(self, data, **params):
        """Execute clustering logic (runs in thread)."""
        # UMAP Embedding
        reducer = umap.UMAP(
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            n_components=params['n_components'],
            random_state=42
        )
        embedding = reducer.fit_transform(data)
        self.embedding = embedding # Store for export
        
        # Clustering
        method = params['method']
        if method == 'leiden':
            # Construct k-NN graph
            knn_graph = kneighbors_graph(
                data, 
                n_neighbors=params['leiden_k'], 
                mode='connectivity', 
                include_self=False
            )
            sources, targets = knn_graph.nonzero()
            edges = list(zip(sources.tolist(), targets.tolist()))
            g = ig.Graph(n=data.shape[0], edges=edges, directed=False)
            
            partition = la.find_partition(
                g, 
                la.RBConfigurationVertexPartition, 
                resolution_parameter=params['leiden_resolution']
            )
            clusters = np.array(partition.membership)
            
        elif method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                cluster_selection_epsilon=params['hdbscan_epsilon']
            )
            clusters = clusterer.fit_predict(data)
            
        self.clusters = clusters
        
        # Plotting
        df_plot = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'Cluster': [f'Cluster_{c}' if c >= 0 else 'Noise' for c in clusters],
            'Sample': data.index
        })
        
        # Add customdata with snippet IDs for click handling
        snippet_ids = data.index.tolist()
        
        if params['n_components'] == 3:
            df_plot['UMAP3'] = embedding[:, 2]
            fig = px.scatter_3d(
                df_plot, x='UMAP1', y='UMAP2', z='UMAP3',
                color='Cluster', hover_data=['Sample'],
                title=f"UMAP + {method.title()} Clustering",
                custom_data=[snippet_ids]  # Add snippet IDs for click handling
            )
        else:
            fig = px.scatter(
                df_plot, x='UMAP1', y='UMAP2',
                color='Cluster', hover_data=['Sample'],
                title=f"UMAP + {method.title()} Clustering",
                custom_data=[snippet_ids]  # Add snippet IDs for click handling
            )
            
        theme = self._get_plot_theme()
        fig.update_layout(template=theme if theme else None)
        return "Clustering Complete", fig

    def on_clustering_finished(self, status, fig):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run analysis")
        self.status_label.setText(status)
        
        if fig:
            point_size = self.plot_point_size_slider.value() if hasattr(self, 'plot_point_size_slider') else 5
            fig.update_traces(marker=dict(size=point_size))
            self.current_fig = fig
            self.plot_widget.update_plot(fig)
            
            # Enable export buttons
            if hasattr(self, 'export_plot_btn'):
                self.export_plot_btn.setEnabled(True)
            if hasattr(self, 'export_csv_btn'):
                self.export_csv_btn.setEnabled(True)
            
            # Build snippet-to-clip mapping after clustering
            self._build_snippet_to_clip_map()
            
            # Immediately update metadata with cluster assignments
            self._update_metadata_with_clusters()
            
            # Refresh metadata columns and cluster list after clustering
            self._refresh_metadata_columns()
            self._refresh_cluster_list()
            self._refresh_cluster_export_list()
            
            # Update plots if metadata column is selected
            if hasattr(self, 'metadata_column_combo') and self.metadata_column_combo.currentText() != "None (Show all clusters)":
                self._update_plots_by_metadata()
        else:
            QMessageBox.warning(self, "Error", status)
    
    def _update_metadata_with_clusters(self):
        """Update metadata CSV file with cluster assignments."""
        if self.metadata is None or self.clusters is None or self.processed_data is None:
            return
            
        try:
            # Map clusters to snippet_ids
            # processed_data.index contains snippet_ids (from matrix columns)
            cluster_series = pd.Series(self.clusters, index=self.processed_data.index, name='Cluster')
            
            # Convert cluster numbers to cluster labels
            cluster_labels = cluster_series.map(lambda c: f'Cluster_{c}' if c >= 0 else 'Noise')
            
            # Update metadata
            # Check for both 'snippet' and 'span_id' (backward compatibility)
            snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
            if snippet_col:
                # Merge cluster assignments by snippet/span_id
                cluster_df = pd.DataFrame({
                    snippet_col: self.processed_data.index,
                    'Cluster': cluster_labels
                })
                
                # Update or add Cluster column
                if 'Cluster' in self.metadata.columns:
                    # Remove old Cluster column
                    self.metadata = self.metadata.drop(columns=['Cluster'])
                
                # Merge new cluster assignments
                self.metadata = self.metadata.merge(cluster_df, on=snippet_col, how='left')
                
                # Save updated metadata back to file
                if self.metadata_file_path:
                    # Use the original metadata file path, respecting format
                    self._save_metadata_to_file(self.metadata, self.metadata_file_path)
                    current_text = self.status_label.text()
                    self.status_label.setText(f"{current_text}\nMetadata updated with cluster assignments")
                else:
                    # Try to find metadata file in experiment folder
                    experiment_path = self.config.get("experiment_path")
                    if experiment_path:
                        registered_clips_dir = os.path.join(experiment_path, "registered_clips")
                        if os.path.exists(registered_clips_dir):
                            # Find the latest metadata file (prefer parquet, then npz, then csv)
                            meta_files = [f for f in os.listdir(registered_clips_dir) 
                                        if f.startswith("behaviorome_") and "_metadata" in f]
                            if meta_files:
                                # Prefer parquet > npz > csv
                                parquet_files = [f for f in meta_files if f.endswith(".parquet")]
                                npz_files = [f for f in meta_files if f.endswith(".npz")]
                                csv_files = [f for f in meta_files if f.endswith(".csv")]
                                
                                chosen_file = None
                                if parquet_files:
                                    parquet_files.sort(reverse=True)
                                    chosen_file = parquet_files[0]
                                elif csv_files:
                                    csv_files.sort(reverse=True)
                                    chosen_file = csv_files[0]
                                # Skip npz for saving to avoid format issues
                                
                                if chosen_file:
                                    metadata_file = os.path.join(registered_clips_dir, chosen_file)
                                    self._save_metadata_to_file(self.metadata, metadata_file)
                                    self.metadata_file_path = metadata_file
                                    current_text = self.status_label.text()
                                    self.status_label.setText(f"{current_text}\nMetadata updated with cluster assignments")
                            
        except Exception as e:
            logger.error("Could not update metadata file: %s", e, exc_info=True)
            QMessageBox.warning(self, "Metadata Update Warning",
                              f"Could not update metadata file:\n{str(e)}\n\nCluster assignments are still available for export.")

    def export_results(self):
        """Export current plot data to CSV."""
        if self.processed_data is None or self.clusters is None:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return
        
        # Determine plot type and get appropriate data
        plot_type = self.plot_type_combo.currentText() if hasattr(self, 'plot_type_combo') else "UMAP"
        group_name = self.metadata_column_combo.currentText() if hasattr(self, 'metadata_column_combo') else None
        
        experiment_path = self.config.get("experiment_path")
        if not experiment_path:
            return
            
        output_dir = os.path.join(experiment_path, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base dataframe with UMAP coordinates and clusters
        df_export = pd.DataFrame({
            'UMAP1': self.embedding[:, 0],
            'UMAP2': self.embedding[:, 1],
            'Cluster': [f'Cluster_{c}' if c >= 0 else 'Noise' for c in self.clusters],
            'Sample': self.processed_data.index
        })
        
        if self.embedding.shape[1] > 2:
            df_export['UMAP3'] = self.embedding[:, 2]
        
        # Add metadata if available
        if self.metadata is not None:
            snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
            if snippet_col and snippet_col in self.metadata.columns:
                df_export = df_export.merge(
                    self.metadata, 
                    left_on='Sample', 
                    right_on=snippet_col, 
                    how='left'
                )
        
        # Add plot-specific data based on plot type
        if plot_type == "Cluster Proportions" and group_name and group_name != "None (Show all clusters)":
            if group_name in df_export.columns:
                filename = os.path.join(output_dir, f"cluster_proportions_{group_name}_{timestamp}.csv")
            else:
                filename = os.path.join(output_dir, f"cluster_proportions_{timestamp}.csv")
        elif plot_type == "Single Cluster Analysis" and group_name and group_name != "None (Show all clusters)":
            selected_cluster = self.single_cluster_combo.currentText() if hasattr(self, 'single_cluster_combo') and self.single_cluster_combo.currentText() != "None" else None
            if selected_cluster and group_name in df_export.columns:
                filename = os.path.join(output_dir, f"single_cluster_{selected_cluster}_{group_name}_{timestamp}.csv")
            else:
                filename = os.path.join(output_dir, f"single_cluster_analysis_{timestamp}.csv")
        else:
            filename = os.path.join(output_dir, f"clustering_results_{timestamp}.csv")
        
        df_export.to_csv(filename, index=False)
        QMessageBox.information(self, "Export", f"Results saved to:\n{filename}")
    
    def open_metadata_manager(self):
        """Open metadata management dialog."""
        if self.metadata is None:
            QMessageBox.warning(self, "No Data", "Please load metadata first.")
            return
            
        # Import here to avoid circular imports
        from .metadata_management_widget import MetadataManagementDialog
        
        dialog = MetadataManagementDialog(self.metadata.copy(), self.metadata_file_path, self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update metadata from dialog
            self.metadata = dialog.get_metadata()
            self.metadata_file_path = dialog.get_metadata_path()
            
            # Reload data to reflect changes
            if self.matrix_data is not None:
                # Reapply preprocessing if needed
                self.apply_preprocessing()
            
            # Refresh metadata columns
            self._refresh_metadata_columns()
            
            # Update plots if metadata column is selected
            if hasattr(self, 'metadata_column_combo') and self.metadata_column_combo.currentText() != "None (Show all clusters)":
                self._update_plots_by_metadata()
                
            QMessageBox.information(self, "Success", "Metadata updated successfully.")
    
    def _refresh_metadata_columns(self):
        """Refresh metadata columns in combo box."""
        if not hasattr(self, 'metadata_column_combo'):
            return
        
        # Block signals to prevent recursion
        self.metadata_column_combo.blockSignals(True)
        
        try:
            current_selection = self.metadata_column_combo.currentText()
            self.metadata_column_combo.clear()
            self.metadata_column_combo.addItem("None (Show all clusters)")
            
            if self.metadata is not None:
                # Include all columns from metadata (user can choose which to use)
                available_cols = list(self.metadata.columns)
                self.metadata_column_combo.addItems(available_cols)
                
                # Restore previous selection if still available
                if current_selection in available_cols:
                    idx = self.metadata_column_combo.findText(current_selection)
                    if idx >= 0:
                        self.metadata_column_combo.setCurrentIndex(idx)
        finally:
            self.metadata_column_combo.blockSignals(False)
    
    def _refresh_cluster_list(self):
        """Refresh cluster list in single cluster combo."""
        if not hasattr(self, 'single_cluster_combo') or self.clusters is None:
            return
        
        # Block signals to prevent recursion
        self.single_cluster_combo.blockSignals(True)
        
        try:
            current_selection = self.single_cluster_combo.currentText()
            self.single_cluster_combo.clear()
            self.single_cluster_combo.addItem("None")
            
            # Normalize cluster identifiers to string labels: Cluster_X or Noise
            labels = set()
            for cid in set(self.clusters):
                label = None
                if isinstance(cid, str):
                    lc = cid.lower()
                    if lc.startswith("cluster_"):
                        label = cid
                    elif lc == "noise":
                        label = "Noise"
                    else:
                        # Try to parse numeric from string
                        try:
                            num = int(cid)
                            label = f"Cluster_{num}" if num >= 0 else "Noise"
                        except Exception:
                            label = None
                else:
                    # numeric cluster id
                    try:
                        num = int(cid)
                        label = f"Cluster_{num}" if num >= 0 else "Noise"
                    except Exception:
                        label = None
                
                if label:
                    labels.add(label)
            
            # Sort labels numerically, keep Noise last if present
            def _label_key(lbl):
                if lbl == "Noise":
                    return (1e9,)
                if lbl.lower().startswith("cluster_"):
                    try:
                        return (int(lbl.split("_", 1)[1]),)
                    except Exception:
                        return (1e8,)
                return (1e8,)
            
            for lbl in sorted(labels, key=_label_key):
                self.single_cluster_combo.addItem(lbl)
            
            # Restore previous selection if still available
            if current_selection and current_selection != "None":
                idx = self.single_cluster_combo.findText(current_selection)
                if idx >= 0:
                    self.single_cluster_combo.setCurrentIndex(idx)
        finally:
            # Always unblock signals
            self.single_cluster_combo.blockSignals(False)
    
    def _refresh_spatial_selectors(self):
        """Refresh video and object selectors for spatial distribution plot."""
        if self.metadata is None:
            return
        
        # Block signals to prevent recursion
        self.spatial_video_combo.blockSignals(True)
        self.spatial_object_combo.blockSignals(True)
        
        try:
            # Get current selections
            current_video = self.spatial_video_combo.currentText()
            current_object = self.spatial_object_combo.currentText()
            
            # Get unique videos from metadata
            # Prefer 'group' column (contains original video/animal name)
            # Otherwise extract from 'video_id' (clip filenames)
            videos = ["All"]
            video_names = set()
            
            if 'group' in self.metadata.columns:
                # Use group column which has original video names
                for v in self.metadata['group'].dropna().unique():
                    v_str = str(v).strip()
                    if v_str:
                        video_names.add(v_str)
            elif 'video_id' in self.metadata.columns:
                # Extract video name from clip filenames
                # Clip format: {video_name}_clip_{clip_idx:06d}_obj{obj_id}.mp4
                import re
                for v in self.metadata['video_id'].dropna().unique():
                    v_str = str(v)
                    base = os.path.splitext(os.path.basename(v_str))[0]
                    # Remove _clip_XXXXXX and _objX suffixes
                    match = re.match(r'^(.+?)_clip_\d+(?:_obj\d+)?$', base)
                    if match:
                        video_name = match.group(1)
                        if video_name:
                            video_names.add(video_name)
                    else:
                        # Fallback: use base name if pattern doesn't match
                        if base:
                            video_names.add(base)
            
            for v in sorted(video_names):
                videos.append(v)
            
            self.spatial_video_combo.clear()
            for v in videos:
                self.spatial_video_combo.addItem(v)
            
            # Restore video selection
            if current_video in videos:
                idx = self.spatial_video_combo.findText(current_video)
                if idx >= 0:
                    self.spatial_video_combo.setCurrentIndex(idx)
            
            # Get unique objects from metadata
            object_col = 'object_id' if 'object_id' in self.metadata.columns else None
            objects = ["All"]
            if object_col:
                unique_objects = self.metadata[object_col].dropna().unique()
                for o in sorted(set(str(obj) for obj in unique_objects if str(obj).strip())):
                    if o not in objects:
                        objects.append(o)
            
            self.spatial_object_combo.clear()
            for o in objects:
                self.spatial_object_combo.addItem(o)
            
            # Restore object selection
            if current_object in objects:
                idx = self.spatial_object_combo.findText(current_object)
                if idx >= 0:
                    self.spatial_object_combo.setCurrentIndex(idx)
                    
        finally:
            self.spatial_video_combo.blockSignals(False)
            self.spatial_object_combo.blockSignals(False)
    
    def _on_spatial_video_changed(self):
        """Handle video selection change - refresh object list and update plot."""
        # Refresh object list based on selected video
        if self.metadata is None:
            return
        
        selected_video = self.spatial_video_combo.currentText()
        object_col = 'object_id' if 'object_id' in self.metadata.columns else None
        video_col = 'video_id' if 'video_id' in self.metadata.columns else None
        
        self.spatial_object_combo.blockSignals(True)
        try:
            current_object = self.spatial_object_combo.currentText()
            self.spatial_object_combo.clear()
            self.spatial_object_combo.addItem("All")
            
            if object_col:
                if selected_video == "All":
                    # Show all objects across all videos
                    unique_objects = self.metadata[object_col].dropna().unique()
                else:
                    # Filter objects by selected video
                    if 'group' in self.metadata.columns:
                        # Use group column for matching
                        mask = self.metadata['group'].apply(lambda x: str(x).strip() == selected_video)
                    elif video_col:
                        # Extract video name from clip filenames and match
                        import re
                        def extract_video_name(clip_name):
                            base = os.path.splitext(os.path.basename(str(clip_name)))[0]
                            match = re.match(r'^(.+?)_clip_\d+(?:_obj\d+)?$', base)
                            return match.group(1) if match else base
                        mask = self.metadata[video_col].apply(lambda x: extract_video_name(x) == selected_video)
                    else:
                        mask = pd.Series([True] * len(self.metadata), index=self.metadata.index)
                    unique_objects = self.metadata.loc[mask, object_col].dropna().unique()
                
                for o in sorted(set(str(obj) for obj in unique_objects if str(obj).strip())):
                    self.spatial_object_combo.addItem(o)
            
            # Try to restore previous selection
            idx = self.spatial_object_combo.findText(current_object)
            if idx >= 0:
                self.spatial_object_combo.setCurrentIndex(idx)
        finally:
            self.spatial_object_combo.blockSignals(False)
        
        # Update plot
        self._update_plots_by_metadata()
    
    def _refresh_cluster_export_list(self):
        """Refresh cluster list in export list widget."""
        if not hasattr(self, 'cluster_export_list') or self.clusters is None:
            return
        
        self.cluster_export_list.clear()
        
        # Get unique cluster labels
        labels = set()
        for cid in set(self.clusters):
            label = None
            if isinstance(cid, str):
                lc = cid.lower()
                if lc.startswith("cluster_"):
                    label = cid
                elif lc == "noise":
                    label = "Noise"
                else:
                    try:
                        num = int(cid)
                        label = f"Cluster_{num}" if num >= 0 else "Noise"
                    except Exception:
                        label = None
            else:
                try:
                    num = int(cid)
                    label = f"Cluster_{num}" if num >= 0 else "Noise"
                except Exception:
                    label = None
            
            if label:
                labels.add(label)
        
        # Sort labels numerically, keep Noise last if present
        def _label_key(lbl):
            if lbl == "Noise":
                return (1e9,)
            if lbl.lower().startswith("cluster_"):
                try:
                    num = int(lbl.split('_')[1])
                    return (0, num)
                except Exception:
                    return (2, lbl)
            return (2, lbl)
        
        sorted_labels = sorted(labels, key=_label_key)
        
        for label in sorted_labels:
            self.cluster_export_list.addItem(label)
        
        # Enable buttons if clusters are available
        if hasattr(self, 'extract_cluster_btn'):
            self.extract_cluster_btn.setEnabled(len(sorted_labels) > 0)
        if hasattr(self, 'exclude_clusters_btn'):
            self.exclude_clusters_btn.setEnabled(len(sorted_labels) > 0)
    
    def _extract_cluster(self):
        """Extract data for a specific cluster and save as NPZ files."""
        if self.clusters is None:
            QMessageBox.warning(self, "No Data", "No clustering data available. Please perform clustering first.")
            return
        
        selected_items = self.cluster_export_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a cluster to extract.")
            return
        
        if len(selected_items) > 1:
            QMessageBox.warning(self, "Multiple Selection", "Please select only one cluster for extraction. Use 'Exclude Selected Clusters' for multiple clusters.")
            return
        
        selected_cluster = selected_items[0].text()
        use_raw_data = self.use_raw_data_checkbox.isChecked()
        
        try:
            # Get cluster number from string (e.g., "Cluster_0" -> 0, "Noise" -> -1)
            if selected_cluster.lower() == "noise":
                cluster_num = -1
            else:
                cluster_num = int(selected_cluster.split('_')[-1])
            
            # Get indices of samples in the selected cluster
            cluster_indices = [i for i, c in enumerate(self.clusters) if c == cluster_num]
            
            if not cluster_indices:
                QMessageBox.warning(self, "No Data", f"No samples found in {selected_cluster}")
                return
            
            # Choose data source based on user preference
            if use_raw_data:
                if self.matrix_data is None:
                    QMessageBox.warning(self, "No Data", "Raw data not available.")
                    return
                data = self.matrix_data
                data_type = "raw"
            else:
                if self.processed_data is None:
                    QMessageBox.warning(self, "No Data", "Processed data not available. Please apply preprocessing first.")
                    return
                data = self.processed_data.T  # Transpose back to features x samples format
                data_type = "processed"
            
            metadata = self.metadata
            
            if data is None or metadata is None:
                QMessageBox.warning(self, "No Data", "Original data not available. Please ensure data is loaded.")
                return
            
            # Extract subset of data (columns are samples/snippets)
            subset_data = data.iloc[:, cluster_indices]
            
            # Get snippet IDs for the selected samples
            snippet_ids = subset_data.columns.tolist()
            
            # Extract corresponding metadata
            snippet_col = 'snippet' if 'snippet' in metadata.columns else ('span_id' if 'span_id' in metadata.columns else None)
            if snippet_col:
                subset_metadata = metadata[metadata[snippet_col].isin(snippet_ids)].copy()
            else:
                # Fallback: align by index if snippet column not found
                subset_metadata = metadata.iloc[cluster_indices].copy() if len(metadata) == len(self.clusters) else metadata.copy()
            
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine output directory
            experiment_path = self.config.get("experiment_path")
            if experiment_path:
                output_dir = os.path.join(experiment_path, "analysis_results")
            else:
                output_dir = os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            
            # Save matrix as NPZ
            matrix_filename = f"matrix_{selected_cluster}_{data_type}_{timestamp}.npz"
            matrix_path = os.path.join(output_dir, matrix_filename)
            
            feature_names = subset_data.index.tolist()
            matrix_array = subset_data.values  # features x samples
            
            np.savez_compressed(
                matrix_path,
                matrix=matrix_array,
                feature_names=np.array(feature_names, dtype=object),
                snippet_ids=np.array(snippet_ids, dtype=object),
            )
            
            # Save metadata as NPZ
            metadata_filename = f"metadata_{selected_cluster}_{timestamp}.npz"
            metadata_path = os.path.join(output_dir, metadata_filename)
            
            np.savez_compressed(
                metadata_path,
                metadata=subset_metadata.values,
                columns=np.array(subset_metadata.columns, dtype=object),
            )
            
            msg = (f"Successfully extracted {len(cluster_indices)} samples from {selected_cluster}.\n"
                  f"Data type: {data_type}\n"
                  f"Saved as:\n- {matrix_filename} (shape: {subset_data.shape})\n"
                  f"- {metadata_filename} (shape: {subset_metadata.shape})\n\n"
                  f"Click 'Load Dataset' to load this data now, or 'OK' to continue with current data.")
            
            dialog = ClusterExportDialog(self, msg, matrix_path, metadata_path)
            dialog.exec()
            
            # Load dataset if user clicked "Load Dataset"
            if dialog.load_requested:
                self._load_files_async(matrix_path, metadata_path)
            
        except Exception as e:
            error_msg = f"Error extracting cluster data: {str(e)}"
            logger.error("Error extracting cluster data: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
    
    def _exclude_clusters(self):
        """Export data excluding specific clusters as NPZ files."""
        if self.clusters is None:
            QMessageBox.warning(self, "No Data", "No clustering data available. Please perform clustering first.")
            return
        
        selected_items = self.cluster_export_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one cluster to exclude.")
            return
        
        clusters_to_exclude = [item.text() for item in selected_items]
        use_raw_data = self.use_raw_data_checkbox.isChecked()
        
        try:
            # Get cluster numbers from strings
            exclude_nums = []
            for cluster_str in clusters_to_exclude:
                if cluster_str.lower() == "noise":
                    exclude_nums.append(-1)
                else:
                    exclude_nums.append(int(cluster_str.split('_')[-1]))
            
            # Get indices of samples NOT in the excluded clusters
            keep_indices = [i for i, c in enumerate(self.clusters) if c not in exclude_nums]
            
            if not keep_indices:
                QMessageBox.warning(self, "Error", "Excluding these clusters would remove all data!")
                return
            
            # Choose data source based on user preference
            if use_raw_data:
                if self.matrix_data is None:
                    QMessageBox.warning(self, "No Data", "Raw data not available.")
                    return
                data = self.matrix_data
                data_type = "raw"
            else:
                if self.processed_data is None:
                    QMessageBox.warning(self, "No Data", "Processed data not available. Please apply preprocessing first.")
                    return
                data = self.processed_data.T  # Transpose back to features x samples format
                data_type = "processed"
            
            metadata = self.metadata
            
            if data is None or metadata is None:
                QMessageBox.warning(self, "No Data", "Original data not available. Please ensure data is loaded.")
                return
            
            # Extract subset of data (excluding the selected clusters)
            subset_data = data.iloc[:, keep_indices]
            
            # Get snippet IDs for the kept samples
            snippet_ids = subset_data.columns.tolist()
            
            # Extract corresponding metadata
            snippet_col = 'snippet' if 'snippet' in metadata.columns else ('span_id' if 'span_id' in metadata.columns else None)
            if snippet_col:
                subset_metadata = metadata[metadata[snippet_col].isin(snippet_ids)].copy()
            else:
                # Fallback: align by index if snippet column not found
                subset_metadata = metadata.iloc[keep_indices].copy() if len(metadata) == len(self.clusters) else metadata.copy()
            
            # Create timestamp and descriptive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excluded_str = "_".join([str(num) for num in sorted(exclude_nums)])
            
            # Determine output directory
            experiment_path = self.config.get("experiment_path")
            if experiment_path:
                output_dir = os.path.join(experiment_path, "analysis_results")
            else:
                output_dir = os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            
            # Save matrix as NPZ
            matrix_filename = f"matrix_excluding_clusters_{excluded_str}_{data_type}_{timestamp}.npz"
            matrix_path = os.path.join(output_dir, matrix_filename)
            
            feature_names = subset_data.index.tolist()
            matrix_array = subset_data.values  # features x samples
            
            np.savez_compressed(
                matrix_path,
                matrix=matrix_array,
                feature_names=np.array(feature_names, dtype=object),
                snippet_ids=np.array(snippet_ids, dtype=object),
            )
            
            # Save metadata as NPZ
            metadata_filename = f"metadata_excluding_clusters_{excluded_str}_{timestamp}.npz"
            metadata_path = os.path.join(output_dir, metadata_filename)
            
            np.savez_compressed(
                metadata_path,
                metadata=subset_metadata.values,
                columns=np.array(subset_metadata.columns, dtype=object),
            )
            
            # Calculate statistics
            original_count = len(self.clusters)
            remaining_count = len(keep_indices)
            excluded_count = original_count - remaining_count
            
            # Get remaining clusters
            remaining_clusters = sorted(set(self.clusters[i] for i in keep_indices))
            remaining_labels = []
            for cid in remaining_clusters:
                if isinstance(cid, str):
                    remaining_labels.append(cid)
                else:
                    remaining_labels.append(f"Cluster_{cid}" if cid >= 0 else "Noise")
            
            msg = (f"Successfully excluded {len(clusters_to_exclude)} clusters.\n"
                  f"Data type: {data_type}\n"
                  f"Removed {excluded_count} samples, kept {remaining_count} samples.\n"
                  f"Remaining clusters: {remaining_labels}\n\n"
                  f"Saved as:\n- {matrix_filename} (shape: {subset_data.shape})\n"
                  f"- {metadata_filename} (shape: {subset_metadata.shape})\n\n"
                  f"Click 'Load Dataset' to load this data now, or 'OK' to continue with current data.")
            
            dialog = ClusterExportDialog(self, msg, matrix_path, metadata_path)
            dialog.exec()
            
            # Load dataset if user clicked "Load Dataset"
            if dialog.load_requested:
                self._load_files_async(matrix_path, metadata_path)
            
        except Exception as e:
            error_msg = f"Error excluding clusters: {str(e)}"
            logger.error("Error excluding clusters: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
    
    def _show_cluster_export_help(self):
        """Show help dialog for cluster export."""
        QMessageBox.information(
            self,
            "Cluster Export Help",
            "Cluster Export allows you to:\n\n"
            "-Extract Selected Cluster: Export a single cluster for subclustering analysis. "
            "This creates a new dataset containing only samples from the selected cluster, "
            "which you can then load and analyze separately.\n\n"
            "-Exclude Selected Clusters: Export data excluding specific clusters. "
            "This is useful for removing noise or artifacts from your analysis. "
            "You can select multiple clusters to exclude at once.\n\n"
            "-Use raw data: If checked, exports the original (unnormalized) data. "
            "Otherwise, exports the preprocessed data.\n\n"
            "All exports are saved as NPZ files in the analysis_results folder."
        )
    
    def _get_plot_theme(self):
        """Get the currently selected plot theme."""
        if hasattr(self, 'color_theme_combo'):
            theme = self.color_theme_combo.currentText()
            return theme if theme != "none" else None
        return "simple_white"  # Default
    
    def _export_plot(self):
        """Export current plot as PDF or SVG."""
        if self.current_fig is None:
            QMessageBox.warning(self, "No Plot", "No plot available to export. Please run clustering first.")
            return
        
        # Determine default directory
        experiment_path = self.config.get("experiment_path")
        if experiment_path:
            default_dir = os.path.join(experiment_path, "analysis_results")
            os.makedirs(default_dir, exist_ok=True)
        else:
            default_dir = os.getcwd()
        
        # Create timestamp for default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = os.path.join(default_dir, f"clustering_plot_{timestamp}.pdf")
        
        # Open file dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            default_filename,
            "PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if not file_path:
            return
        
        try:
            # Determine format from file extension
            if file_path.lower().endswith('.svg'):
                format_type = 'svg'
            elif file_path.lower().endswith('.pdf'):
                format_type = 'pdf'
            else:
                # Default to PDF if extension not recognized
                format_type = 'pdf'
                if not file_path.endswith('.pdf'):
                    file_path += '.pdf'
            
            # Check if kaleido is available (required for PDF/SVG export)
            try:
                import kaleido
            except ImportError:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    "The 'kaleido' package is required for PDF/SVG export.\n\n"
                    "Please install it with: pip install kaleido"
                )
                return
            
            # Get figure dimensions (use on-screen canvas size when possible)
            fig_width = 1200
            fig_height = 800
            
            # Prefer the plot widget's rendered size to avoid stretching
            if hasattr(self, 'plot_widget') and self.plot_widget is not None:
                try:
                    w = self.plot_widget.width()
                    h = self.plot_widget.height()
                    if w and h:
                        fig_width = max(int(w), 400)
                        fig_height = max(int(h), 300)
                except Exception as e:
                    logger.debug("Could not read plot widget dimensions: %s", e)
            
            # If explicit layout sizes are set on the figure, prefer them
            if hasattr(self.current_fig, 'layout') and self.current_fig.layout:
                if getattr(self.current_fig.layout, 'width', None):
                    fig_width = self.current_fig.layout.width
                if getattr(self.current_fig.layout, 'height', None):
                    fig_height = self.current_fig.layout.height
            
            # Export using plotly.io.write_image
            pio.write_image(
                self.current_fig,
                file_path,
                format=format_type,
                width=fig_width,
                height=fig_height
            )
            
            QMessageBox.information(self, "Success", f"Plot exported successfully to:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Error exporting plot: {str(e)}"
            logger.error("Error exporting plot: %s", e, exc_info=True)
            QMessageBox.critical(self, "Export Error", error_msg)
    
    def _update_plot_type(self):
        """Update plot type based on selection."""
        self._update_plots_by_metadata()
    
    def _update_plots_by_metadata(self):
        """Update plots based on selected metadata column and plot type."""
        if self.embedding is None or self.clusters is None or self.processed_data is None:
            return
        
        plot_type = self.plot_type_combo.currentText()
        
        # Show/hide single cluster selector based on plot type
        if plot_type in ["Single Cluster Analysis", "Spatial cluster distribution"]:
            self.single_cluster_label.setVisible(True)
            self.single_cluster_combo.setVisible(True)
            # Make sure cluster list is populated
            if self.clusters is not None:
                self._refresh_cluster_list()
        else:
            self.single_cluster_label.setVisible(False)
            self.single_cluster_combo.setVisible(False)
        
        # Show/hide video and object selectors for spatial distribution
        if plot_type == "Spatial cluster distribution":
            self.spatial_video_label.setVisible(True)
            self.spatial_video_combo.setVisible(True)
            self.spatial_object_label.setVisible(True)
            self.spatial_object_combo.setVisible(True)
            # Populate video/object lists
            self._refresh_spatial_selectors()
        else:
            self.spatial_video_label.setVisible(False)
            self.spatial_video_combo.setVisible(False)
            self.spatial_object_label.setVisible(False)
            self.spatial_object_combo.setVisible(False)
        
        # Handle "UMAP" plot type
        if plot_type == "UMAP":
            group_name = self.metadata_column_combo.currentText()
            
            # Handle empty or "None" selection
            if not group_name or group_name.strip() == "" or group_name == "None (Show all clusters)":
                # Regenerate default clustering plot to show all clusters
                self._regenerate_default_plot()
                if hasattr(self, 'export_plot_btn'):
                    self.export_plot_btn.setEnabled(True)
                if hasattr(self, 'export_csv_btn'):
                    self.export_csv_btn.setEnabled(True)
                return
            
            if self.metadata is None:
                QMessageBox.warning(self, "No Metadata", "Please load metadata first.")
                return
            
            # Validate column exists
            if group_name not in self.metadata.columns:
                QMessageBox.warning(self, "Invalid Column", f"Column '{group_name}' not found in metadata.\n\nAvailable columns: {', '.join(self.metadata.columns)}")
                # Reset to "None" selection
                self.metadata_column_combo.setCurrentIndex(0)
                return
            
            try:
                point_size = self.plot_point_size_slider.value()
                umap_fig, props_fig, single_fig = self._create_grouped_plots(group_name, point_size, None)
                
                if umap_fig:
                    self.current_fig = umap_fig
                    self.plot_widget.update_plot(umap_fig)
                    if hasattr(self, 'export_plot_btn'):
                        self.export_plot_btn.setEnabled(True)
                    if hasattr(self, 'export_csv_btn'):
                        self.export_csv_btn.setEnabled(True)
                    
            except Exception as e:
                logger.error("Error creating plots: %s", e, exc_info=True)
                QMessageBox.critical(self, "Plot Error", f"Error creating plots: {e}")

        # Handle "Cluster Proportions" plot type
        elif plot_type == "Cluster Proportions":
            group_name = self.metadata_column_combo.currentText()
            
            if not group_name or group_name.strip() == "" or group_name == "None (Show all clusters)":
                QMessageBox.warning(self, "No Group Selected", "Please select a metadata column to show proportions.")
                return
            
            if self.metadata is None:
                QMessageBox.warning(self, "No Metadata", "Please load metadata first.")
                return
            
            if group_name not in self.metadata.columns:
                QMessageBox.warning(self, "Invalid Column", f"Column '{group_name}' not found in metadata.")
                return
            
            try:
                point_size = self.plot_point_size_slider.value()
                umap_fig, props_fig, single_fig = self._create_grouped_plots(group_name, point_size, None)
                
                if props_fig:
                    self.current_fig = props_fig
                    self.plot_widget.update_plot(props_fig)
                    if hasattr(self, 'export_plot_btn'):
                        self.export_plot_btn.setEnabled(True)
                    if hasattr(self, 'export_csv_btn'):
                        self.export_csv_btn.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Plot Error", "Could not create proportions plot.")
                    
            except Exception as e:
                logger.error("Error creating plots: %s", e, exc_info=True)
                QMessageBox.critical(self, "Plot Error", f"Error creating plots: {e}")

        # Handle "Single Cluster Analysis" plot type
        elif plot_type == "Single Cluster Analysis":
            # Check if clusters are available
            if self.clusters is None:
                QMessageBox.warning(self, "No Clusters", "Please run clustering first.")
                return
            
            # Make sure cluster list is populated
            self._refresh_cluster_list()
            
            group_name = self.metadata_column_combo.currentText()
            selected_cluster = self.single_cluster_combo.currentText() if self.single_cluster_combo.currentText() != "None" else None
            
            if not selected_cluster:
                # Don't show warning immediately - let user select first
                # Just clear the plot or show a message
                self.status_label.setText("Please select a cluster from the dropdown above.")
                return
            
            if not group_name or group_name.strip() == "" or group_name == "None (Show all clusters)":
                QMessageBox.warning(self, "No Group Selected", "Please select a metadata column to show single cluster proportions.")
                return
            
            if self.metadata is None:
                QMessageBox.warning(self, "No Metadata", "Please load metadata first.")
                return
            
            if group_name not in self.metadata.columns:
                QMessageBox.warning(self, "Invalid Column", f"Column '{group_name}' not found in metadata.")
                return
            
            try:
                point_size = self.plot_point_size_slider.value()
                umap_fig, props_fig, single_fig = self._create_grouped_plots(group_name, point_size, selected_cluster)
                
                if single_fig:
                    self.current_fig = single_fig
                    self.plot_widget.update_plot(single_fig)
                    if hasattr(self, 'export_plot_btn'):
                        self.export_plot_btn.setEnabled(True)
                    if hasattr(self, 'export_csv_btn'):
                        self.export_csv_btn.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Plot Error", "Could not create single cluster plot.")
                    
            except Exception as e:
                logger.error("Error creating plots: %s", e, exc_info=True)
                QMessageBox.critical(self, "Plot Error", f"Error creating plots: {e}")

        # Handle "Spatial cluster distribution" plot type
        elif plot_type == "Spatial cluster distribution":
            # Check if clusters are available
            if self.clusters is None:
                self.status_label.setText("Please run clustering first.")
                return
            
            # Make sure cluster list is populated
            if hasattr(self, 'single_cluster_combo'):
                self._refresh_cluster_list()
                selected_cluster = self.single_cluster_combo.currentText() if self.single_cluster_combo.currentText() != "None" else None
            else:
                selected_cluster = None
            
            if not selected_cluster:
                self.status_label.setText("Please select a cluster from the dropdown above to visualize its spatial distribution.")
                return
            
            try:
                spatial_fig = self._create_spatial_distribution_plot(selected_cluster)
                
                if spatial_fig:
                    self.current_fig = spatial_fig
                    self.plot_widget.update_plot(spatial_fig)
                    if hasattr(self, 'export_plot_btn'):
                        self.export_plot_btn.setEnabled(True)
                    if hasattr(self, 'export_csv_btn'):
                        self.export_csv_btn.setEnabled(True)
                    self.status_label.setText(f"Spatial distribution plot for {selected_cluster} displayed.")
                else:
                    self.status_label.setText("Could not create spatial distribution plot. Make sure mask data is available in the experiment folder.")
                    
            except Exception as e:
                error_msg = f"Error creating spatial distribution plot: {str(e)}"
                self.status_label.setText(error_msg)
                logger.error("Error creating spatial distribution plot: %s", e, exc_info=True)
                QMessageBox.critical(self, "Plot Error", error_msg)
    
    def _create_grouped_plots(self, group_name, point_size, selected_cluster=None):
        """Create UMAP subplots grouped by metadata column, proportion plot, and single cluster plot."""
        # Create base dataframe with UMAP coordinates and clusters
        df_plot = pd.DataFrame({
            'UMAP1': self.embedding[:, 0],
            'UMAP2': self.embedding[:, 1],
            'Cluster': [f'Cluster_{c}' if c >= 0 else 'Noise' for c in self.clusters],
            'Sample': self.processed_data.index
        })
        
        # Validate group_name exists in metadata
        if group_name not in self.metadata.columns:
            return None, None, None
        
        # Merge metadata with processed data
        snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
        
        # Prepare columns to merge (only what we need)
        cols_to_merge = [group_name]
        if snippet_col and snippet_col in self.metadata.columns:
            cols_to_merge.append(snippet_col)
        
        if snippet_col and snippet_col in self.metadata.columns:
            # Merge using snippet column
            df_plot = df_plot.merge(
                self.metadata[cols_to_merge], 
                left_on='Sample', 
                right_on=snippet_col, 
                how='left'
            )
        else:
            # Try to align by index - reset index if needed
            metadata_aligned = self.metadata.copy()
            if len(metadata_aligned) == len(df_plot):
                # Align by position
                metadata_aligned.index = df_plot.index
                df_plot = pd.concat([df_plot, metadata_aligned[[group_name]]], axis=1)
            else:
                # Try to merge by resetting index
                metadata_reset = self.metadata.reset_index()
                if 'index' in metadata_reset.columns:
                    df_plot = df_plot.merge(metadata_reset[cols_to_merge + ['index']], left_on='Sample', right_on='index', how='left')
                else:
                    # Last resort: align by position if lengths match
                    if len(metadata_aligned) == len(df_plot):
                        df_plot[group_name] = metadata_aligned[group_name].values
                    else:
                        return None, None, None
        
        # Ensure Cluster column is still present (merge might have dropped it)
        if 'Cluster' not in df_plot.columns:
            df_plot['Cluster'] = [f'Cluster_{c}' if c >= 0 else 'Noise' for c in self.clusters]
        
        if group_name not in df_plot.columns or df_plot[group_name].isnull().all():
            return None, None, None
        
        # Filter out NaN values
        df_plot = df_plot.dropna(subset=[group_name])
        
        # Generate cluster colors
        unique_clusters = sorted(df_plot['Cluster'].unique())
        n_clusters = len(unique_clusters)
        cluster_colors = self._generate_cluster_colors(n_clusters)
        cluster_color_map = {f'Cluster_{i}': cluster_colors[i] for i in range(n_clusters) if f'Cluster_{i}' in unique_clusters}
        if 'Noise' in unique_clusters:
            cluster_color_map['Noise'] = '#808080'
        
        # Get unique values in the group
        unique_values = sorted(df_plot[group_name].dropna().unique())
        n_groups = len(unique_values)
        
        if n_groups == 0:
            return None, None, None
        
        # Create UMAP subplots
        n_cols = 2
        n_rows = (n_groups + 1 + n_cols - 1) // n_cols  # Add 1 for combined view
        
        subplot_titles = [f'{group_name}: {val}' for val in unique_values]
        subplot_titles.append(f'All {group_name} Groups Combined')
        
        umap_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.08
        )
        
        # Calculate global axis ranges
        umap1_min, umap1_max = df_plot['UMAP1'].min(), df_plot['UMAP1'].max()
        umap2_min, umap2_max = df_plot['UMAP2'].min(), df_plot['UMAP2'].max()
        range1 = umap1_max - umap1_min
        range2 = umap2_max - umap2_min
        max_range = max(range1, range2)
        center1 = (umap1_min + umap1_max) / 2
        center2 = (umap2_min + umap2_max) / 2
        padding = max_range * 0.1
        axis_min1 = center1 - (max_range / 2) - padding
        axis_max1 = center1 + (max_range / 2) + padding
        axis_min2 = center2 - (max_range / 2) - padding
        axis_max2 = center2 + (max_range / 2) + padding
        
        # Plot individual groups
        for idx, value in enumerate(unique_values):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            mask = df_plot[group_name] == value
            df_subset = df_plot[mask]
            
            for cluster in unique_clusters:
                cluster_data = df_subset[df_subset['Cluster'] == cluster]
                if len(cluster_data) == 0:
                    continue
                
                # Add snippet IDs as customdata for click handling
                snippet_ids = cluster_data['Sample'].tolist()
                
                umap_fig.add_trace(
                    go.Scatter(
                        x=cluster_data['UMAP1'],
                        y=cluster_data['UMAP2'],
                        mode='markers',
                        marker=dict(size=point_size, color=cluster_color_map.get(cluster, '#CCCCCC')),
                        name=cluster,
                        showlegend=(idx == 0),
                        legendgroup=cluster,
                        customdata=[[sid] for sid in snippet_ids],  # Wrap in list for each point
                        hovertemplate='<b>%{hovertext}</b><br>Cluster: ' + cluster + '<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                        hovertext=snippet_ids
                    ),
                    row=row, col=col
                )
            
            umap_fig.update_xaxes(range=[axis_min1, axis_max1], row=row, col=col)
            umap_fig.update_yaxes(range=[axis_min2, axis_max2], row=row, col=col)
        
        # Combined view
        combined_row = (n_groups // n_cols) + 1
        combined_col = (n_groups % n_cols) + 1
        
        # Generate group colors for combined view
        group_colors = self._generate_pastel_palette(n_groups)
        group_color_map = {val: group_colors[i] for i, val in enumerate(unique_values)}
        
        for value in unique_values:
            mask = df_plot[group_name] == value
            df_subset = df_plot[mask]
            
            # Add snippet IDs as customdata for click handling
            snippet_ids_combined = df_subset['Sample'].tolist()
            
            umap_fig.add_trace(
                go.Scatter(
                    x=df_subset['UMAP1'],
                    y=df_subset['UMAP2'],
                    mode='markers',
                    marker=dict(size=point_size, color=group_color_map[value]),
                    name=f'{group_name}: {value}',
                    showlegend=True,
                    legendgroup=f'group_{value}',
                    customdata=[[sid] for sid in snippet_ids_combined],  # Wrap in list for each point
                    hovertemplate='<b>%{hovertext}</b><br>' + group_name + ': ' + str(value) + '<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    hovertext=snippet_ids_combined
                ),
                row=combined_row, col=combined_col
            )
        
        umap_fig.update_xaxes(range=[axis_min1, axis_max1], row=combined_row, col=combined_col)
        umap_fig.update_yaxes(range=[axis_min2, axis_max2], row=combined_row, col=combined_col)
        
        theme = self._get_plot_theme()
        umap_fig.update_layout(
            title=f'UMAP + Leiden Clustering by {group_name}',
            template=theme if theme else None,
            height=300 * n_rows
        )
        
        # Create proportion plot
        props_fig = self._create_proportion_plot(df_plot, group_name, cluster_color_map)
        
        # Create single cluster plot if selected and exists in data
        single_fig = None
        if selected_cluster:
            if selected_cluster in df_plot['Cluster'].unique():
                single_fig = self._create_single_cluster_plot(df_plot, group_name, selected_cluster, cluster_color_map)
            else:
                single_fig = None
        
        return umap_fig, props_fig, single_fig
    
    def _create_proportion_plot(self, df_plot, group_name, cluster_color_map):
        """Create cluster proportion bar plot."""
        # Calculate proportions
        props = []
        for group_val in sorted(df_plot[group_name].dropna().unique()):
            group_data = df_plot[df_plot[group_name] == group_val]
            total = len(group_data)
            cluster_props = group_data['Cluster'].value_counts() / total * 100
            for cluster in cluster_props.index:
                props.append({
                    group_name: group_val,
                    'Cluster': cluster,
                    'Proportion (%)': cluster_props[cluster]
                })
        
        df_props = pd.DataFrame(props)
        
        # Create bar plot
        unique_groups = sorted(df_props[group_name].unique())
        unique_clusters = sorted(df_props['Cluster'].unique())
        
        fig = go.Figure()
        
        n_groups = len(unique_groups)
        n_clusters = len(unique_clusters)
        bar_width = 0.8 / n_groups
        
        for i, group in enumerate(unique_groups):
            group_data = df_props[df_props[group_name] == group]
            
            x_positions = []
            proportions = []
            
            for j, cluster in enumerate(unique_clusters):
                cluster_row = group_data[group_data['Cluster'] == cluster]
                prop = cluster_row['Proportion (%)'].iloc[0] if len(cluster_row) > 0 else 0
                x_pos = j + (i - n_groups/2 + 0.5) * bar_width
                x_positions.append(x_pos)
                proportions.append(prop)
            
            fig.add_trace(go.Bar(
                x=x_positions,
                y=proportions,
                name=f"{group_name}: {group}",
                width=bar_width
            ))
        
        theme = self._get_plot_theme()
        fig.update_layout(
            title=f'Cluster Proportions by {group_name}',
            xaxis_title='Cluster',
            yaxis_title='Proportion (%)',
            height=400,
            template=theme if theme else None,
            barmode='group',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(n_clusters)),
                ticktext=[str(c) for c in unique_clusters]
            )
        )
        
        return fig
    
    def _create_single_cluster_plot(self, df_plot, group_name, selected_cluster, cluster_color_map):
        """Create single cluster proportion bar plot."""
        # Extract cluster number from "Cluster_X" format
        cluster_num = selected_cluster.replace('Cluster_', '')
        
        props = []
        for group_val in sorted(df_plot[group_name].dropna().unique()):
            group_data = df_plot[df_plot[group_name] == group_val]
            total = len(group_data)
            cluster_count = len(group_data[group_data['Cluster'] == selected_cluster])
            proportion = (cluster_count / total) * 100 if total > 0 else 0
            props.append({
                group_name: group_val,
                'Proportion (%)': proportion
            })
        
        df_props = pd.DataFrame(props)
        
        color = cluster_color_map.get(selected_cluster, '#CCCCCC')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_props[group_name],
            y=df_props['Proportion (%)'],
            marker_color=color,
            name=selected_cluster,
            text=df_props['Proportion (%)'].round(1).astype(str) + '%',
            textposition='auto'
        ))
        
        theme = self._get_plot_theme()
        fig.update_layout(
            title=f'Proportion of {selected_cluster} Across {group_name}',
            xaxis_title=group_name,
            yaxis_title='Proportion (%)',
            height=400,
            template=theme if theme else None
        )
        
        return fig
    
    def _create_spatial_distribution_plot(self, selected_cluster):
        """Create spatial distribution plot showing trajectory with cluster clip segments overlaid."""
        try:
            if self.metadata is None or self.clusters is None or self.processed_data is None:
                return None
            
            # Get selected video and object filters
            selected_video = self.spatial_video_combo.currentText() if hasattr(self, 'spatial_video_combo') else "All"
            selected_object = self.spatial_object_combo.currentText() if hasattr(self, 'spatial_object_combo') else "All"
            
            # Get full trajectory and clip trajectories from mask data
            trajectory_data = self._get_trajectory_data(selected_video, selected_object)
            if trajectory_data is None:
                return None
            
            all_trajectories = trajectory_data.get('all_trajectories', [])
            clip_trajectories = trajectory_data.get('clip_trajectories', {})
            
            if not all_trajectories:
                return None
            
            # Get theme
            theme = self._get_plot_theme()
            
            # Create figure
            fig = go.Figure()
            
            # Plot all trajectories as base layer (grey lines)
            for traj_idx, traj_info in enumerate(all_trajectories):
                traj = traj_info['trajectory']
                traj_name = traj_info.get('name', f'Trajectory {traj_idx + 1}')
                
                if len(traj) > 0:
                    traj_x = [p[0] for p in traj]
                    traj_y = [p[1] for p in traj]
                    
                    fig.add_trace(go.Scatter(
                        x=traj_x,
                        y=traj_y,
                        mode='lines',
                        line=dict(
                            color='lightgrey',
                            width=1
                        ),
                        name=traj_name if traj_idx == 0 else None,
                        showlegend=(traj_idx == 0),
                        legendgroup='trajectories',
                        hoverinfo='skip'
                    ))
            
            # Get cluster color
            unique_clusters = sorted(set(self.clusters))
            cluster_colors = self._generate_cluster_colors(len(unique_clusters))
            cluster_to_color = {f'Cluster_{c}' if c >= 0 else 'Noise': cluster_colors[i] 
                               for i, c in enumerate(unique_clusters)}
            cluster_color = cluster_to_color.get(selected_cluster, '#e74c3c')
            
            # Get snippets belonging to selected cluster (filtered by video/object)
            snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
            video_col = 'video_id' if 'video_id' in self.metadata.columns else None
            object_col = 'object_id' if 'object_id' in self.metadata.columns else None
            
            if snippet_col is None:
                return None
            
            # Build filter mask for metadata
            filter_mask = pd.Series([True] * len(self.metadata), index=self.metadata.index)
            if selected_video != "All":
                if 'group' in self.metadata.columns:
                    # Use group column for matching
                    filter_mask &= self.metadata['group'].apply(lambda x: str(x).strip() == selected_video)
                elif video_col:
                    # Extract video name from clip filenames and match
                    import re
                    def extract_video_name(clip_name):
                        base = os.path.splitext(os.path.basename(str(clip_name)))[0]
                        match = re.match(r'^(.+?)_clip_\d+(?:_obj\d+)?$', base)
                        return match.group(1) if match else base
                    filter_mask &= self.metadata[video_col].apply(lambda x: extract_video_name(x) == selected_video)
            if object_col and selected_object != "All":
                filter_mask &= self.metadata[object_col].apply(lambda x: str(x) == selected_object)
            
            filtered_snippets = set(self.metadata.loc[filter_mask, snippet_col].values)
            
            # Map snippets to clusters
            cluster_snippets = []
            for i, snip in enumerate(self.processed_data.index):
                if snip not in filtered_snippets:
                    continue
                cluster_label = f'Cluster_{self.clusters[i]}' if self.clusters[i] >= 0 else 'Noise'
                if cluster_label == selected_cluster:
                    cluster_snippets.append(snip)
            
            # Plot clip trajectory segments for selected cluster
            segment_count = 0
            for snippet_id in cluster_snippets:
                if snippet_id in clip_trajectories:
                    clip_traj = clip_trajectories[snippet_id]
                    if len(clip_traj) >= 1:
                        clip_x = [p[0] for p in clip_traj]
                        clip_y = [p[1] for p in clip_traj]
                        
                        fig.add_trace(go.Scatter(
                            x=clip_x,
                            y=clip_y,
                            mode='lines+markers',
                            line=dict(
                                color=cluster_color,
                                width=3
                            ),
                            marker=dict(
                                size=4,
                                color=cluster_color
                            ),
                            name=selected_cluster if segment_count == 0 else None,
                            showlegend=(segment_count == 0),
                            legendgroup='cluster',
                            hovertext=f'{snippet_id}',
                            hoverinfo='text'
                        ))
                        segment_count += 1
            
            # Build title with filter info
            filter_info = []
            if selected_video != "All":
                filter_info.append(f"Video: {selected_video}")
            if selected_object != "All":
                filter_info.append(f"Object: {selected_object}")
            filter_str = f" | {', '.join(filter_info)}" if filter_info else ""
            
            # Update layout
            fig.update_layout(
                title=f'Spatial Distribution: {selected_cluster} ({segment_count} clips){filter_str}',
                xaxis_title='X Position (pixels)',
                yaxis_title='Y Position (pixels)',
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(autorange='reversed'),
                height=600,
                template=theme if theme else None,
                hovermode='closest'
            )
            
            return fig
        except Exception as e:
            logger.error("Error creating spatial distribution plot: %s", e, exc_info=True)
            return None

    def get_representative_snippets(self, n_samples=10):
        """Find representative snippets for each cluster (closest to centroid).
        
        Returns:
            dict: {cluster_label: [snippet_id1, snippet_id2, ...]}
        """
        if self.processed_data is None or self.clusters is None:
            return {}
        
        from sklearn.metrics import pairwise_distances_argmin_min
        
        representative_snippets = {}
        unique_clusters = sorted(set(self.clusters))
        
        for cluster_id in unique_clusters:
            label = f'Cluster_{cluster_id}' if cluster_id >= 0 else 'Noise'
            if label == 'Noise':
                continue
                
            # Get indices of samples in this cluster
            indices = [i for i, c in enumerate(self.clusters) if c == cluster_id]
            if not indices:
                continue
            
            # Get features for these samples
            cluster_features = self.processed_data.iloc[indices].values
            snippet_ids = self.processed_data.iloc[indices].index.tolist()
            
            # Compute centroid
            centroid = np.mean(cluster_features, axis=0).reshape(1, -1)
            
            # Calculate distances to centroid
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(cluster_features, centroid).flatten()
            
            # Get closest samples
            n = min(n_samples, len(indices))
            closest_indices = np.argsort(distances)[:n]
            
            representative_snippets[label] = [snippet_ids[i] for i in closest_indices]
            
        return representative_snippets

    def _get_trajectory_data(self, selected_video="All", selected_object="All"):
        """Get full trajectory and clip trajectory segments from mask data.
        
        Args:
            selected_video: Filter by video name, or "All" for all videos
            selected_object: Filter by object ID, or "All" for all objects
        """
        try:
            if self.metadata is None:
                return None
            
            # Get experiment path
            experiment_path = self.config.get("experiment_path")
            if not experiment_path or not os.path.exists(experiment_path):
                return None
            
            # Try to find mask files in common locations
            possible_mask_dirs = [
                os.path.join(experiment_path, "masks"),
                os.path.join(experiment_path, "segmentation_masks"),
                experiment_path
            ]
            
            mask_files_found = []
            for mask_dir in possible_mask_dirs:
                if os.path.exists(mask_dir):
                    try:
                        for f in os.listdir(mask_dir):
                            if f.endswith(('.h5', '.hdf5')):
                                mask_files_found.append((mask_dir, f))
                    except (OSError, PermissionError):
                        continue
            
            if not mask_files_found:
                return None
            
            # Get video names from metadata for matching
            metadata_video_names = set()
            if 'group' in self.metadata.columns:
                for v in self.metadata['group'].dropna().unique():
                    metadata_video_names.add(str(v).strip())
            elif 'video_id' in self.metadata.columns:
                import re
                for v in self.metadata['video_id'].dropna().unique():
                    base = os.path.splitext(os.path.basename(str(v)))[0]
                    match = re.match(r'^(.+?)_clip_\d+(?:_obj\d+)?$', base)
                    if match:
                        metadata_video_names.add(match.group(1))
                    else:
                        metadata_video_names.add(base)
            
            from singlebehaviorlab.backend.video_processor import load_segmentation_data
            
            all_trajectories = []  # List of {name, trajectory}
            all_frame_centroids = {}  # (video_name, obj_id, frame_idx) -> (cx, cy)
            
            # Load trajectories from each mask file
            for mask_dir, mask_file in mask_files_found:
                mask_path = os.path.join(mask_dir, mask_file)
                
                # Extract video name from mask filename
                mask_base = os.path.splitext(mask_file)[0]
                mask_video_name = mask_base.replace('_mask', '').replace('_objects', '').replace('_segmentation', '')
                
                # Find matching video name from metadata
                matched_video_name = None
                for meta_video_name in metadata_video_names:
                    # Check if mask filename contains metadata video name or vice versa
                    if meta_video_name in mask_video_name or mask_video_name in meta_video_name:
                        matched_video_name = meta_video_name
                        break
                
                # If no match found, use mask filename as video name
                if matched_video_name is None:
                    matched_video_name = mask_video_name
                
                # Filter by video if not "All"
                if selected_video != "All" and selected_video != matched_video_name:
                    continue
                
                try:
                    mask_data = load_segmentation_data(mask_path)
                    frame_objects = mask_data.get('frame_objects', [])
                    
                    if not frame_objects:
                        continue
                    
                    video_height = mask_data.get('height', 1080)
                    video_width = mask_data.get('width', 1920)
                    
                    # Get unique object IDs in this mask file
                    obj_ids = set()
                    for frame_objs in frame_objects:
                        for obj in frame_objs:
                            obj_id = obj.get('obj_id', 0)
                            obj_ids.add(str(obj_id))
                    
                    # Process each object
                    for obj_id_str in obj_ids:
                        # Filter by object if not "All"
                        if selected_object != "All" and selected_object != obj_id_str:
                            continue
                        
                        trajectory = []
                        
                        for frame_idx, frame_objs in enumerate(frame_objects):
                            for obj in frame_objs:
                                if str(obj.get('obj_id', 0)) == obj_id_str:
                                    bbox = obj.get('bbox', (0, 0, video_width, video_height))
                                    x_min, y_min, x_max, y_max = bbox
                                    cx = (x_min + x_max) / 2.0
                                    cy = (y_min + y_max) / 2.0
                                    trajectory.append((cx, cy))
                                    # Store for clip trajectory lookup using matched video name
                                    all_frame_centroids[(matched_video_name, obj_id_str, frame_idx)] = (cx, cy)
                                    break
                        
                        if len(trajectory) > 0:
                            traj_name = f"{matched_video_name} obj{obj_id_str}" if obj_id_str != "0" else matched_video_name
                            all_trajectories.append({
                                'name': traj_name,
                                'trajectory': trajectory,
                                'video': matched_video_name,
                                'object_id': obj_id_str
                            })
                except Exception as e:
                    logger.debug("Could not process trajectory for video: %s", e)
                    continue
            
            if not all_trajectories:
                return None
            
            # Build clip trajectories for each snippet
            clip_trajectories = {}
            snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
            video_col = 'video_id' if 'video_id' in self.metadata.columns else None
            object_col = 'object_id' if 'object_id' in self.metadata.columns else None
            
            if snippet_col is None:
                return {'all_trajectories': all_trajectories, 'clip_trajectories': {}}
            
            for idx, row in self.metadata.iterrows():
                try:
                    snippet_id = row[snippet_col]
                    start_frame = row.get('start_frame')
                    end_frame = row.get('end_frame')
                    
                    # Get video name for this snippet (from group or extract from video_id)
                    snippet_video_name = None
                    if 'group' in self.metadata.columns:
                        snippet_video_name = str(row.get('group', '')).strip()
                    elif video_col:
                        import re
                        snippet_video = str(row.get(video_col, ''))
                        base = os.path.splitext(os.path.basename(snippet_video))[0]
                        match = re.match(r'^(.+?)_clip_\d+(?:_obj\d+)?$', base)
                        snippet_video_name = match.group(1) if match else base
                    
                    snippet_obj = str(row.get(object_col, '0')) if object_col else '0'
                    
                    if pd.notna(start_frame) and pd.notna(end_frame):
                        try:
                            start = int(float(start_frame))
                            end = int(float(end_frame))
                        except (ValueError, TypeError):
                            continue
                    else:
                        clip_idx = row.get('clip_index')
                        if pd.notna(clip_idx):
                            try:
                                start = int(clip_idx) * 16
                                end = start + 15
                            except (ValueError, TypeError):
                                continue
                        else:
                            continue
                    
                    # Get trajectory segment for this clip
                    clip_traj = []
                    if snippet_video_name:
                        for f in range(start, end + 1):
                            # Try with object ID first, then without
                            key = (snippet_video_name, snippet_obj, f)
                            if key in all_frame_centroids:
                                clip_traj.append(all_frame_centroids[key])
                            else:
                                key = (snippet_video_name, '0', f)
                                if key in all_frame_centroids:
                                    clip_traj.append(all_frame_centroids[key])
                    
                    if len(clip_traj) >= 1:
                        clip_trajectories[snippet_id] = clip_traj

                except Exception as e:
                    logger.debug("Could not build clip trajectory for snippet: %s", e)
                    continue
            
            return {
                'all_trajectories': all_trajectories,
                'clip_trajectories': clip_trajectories
            }
            
        except Exception as e:
            logger.error("Error creating spatial distribution plot: %s", e, exc_info=True)
            return None

    def _save_analysis_state(self):
        """Save full analysis state to a pickle file."""
        if self.matrix_data is None:
            QMessageBox.warning(self, "No Data", "No data to save. Please load data first.")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"clustering_analysis_{timestamp}.pkl"
            
            # Determine output directory
            experiment_path = self.config.get("experiment_path")
            initial_dir = ""
            if experiment_path:
                initial_dir = os.path.join(experiment_path, "analysis_results")
                os.makedirs(initial_dir, exist_ok=True)
                
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Analysis State", os.path.join(initial_dir, default_name), "Pickle Files (*.pkl)"
            )
            
            if not path:
                return
                
            state = {
                'matrix_data': self.matrix_data,
                'metadata': self.metadata,
                'processed_data': self.processed_data,
                'embedding': self.embedding,
                'clusters': self.clusters,
                'selected_features': self.selected_features,
                'snippet_to_clip_map': self.snippet_to_clip_map,
                'metadata_file_path': self.metadata_file_path,
                'timestamp': timestamp,
                'version': '1.0'
            }
            
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            QMessageBox.information(self, "Success", f"Analysis saved to:\n{path}")
            
        except Exception as e:
            logger.error("Failed to save analysis: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save analysis: {str(e)}")

    def _load_analysis_state(self):
        """Load full analysis state from a pickle file."""
        try:
            experiment_path = self.config.get("experiment_path")
            initial_dir = ""
            if experiment_path:
                initial_dir = os.path.join(experiment_path, "analysis_results")
                
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Analysis State", initial_dir, "Pickle Files (*.pkl)"
            )
            
            if not path:
                return
                
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.matrix_data = state.get('matrix_data')
            self.metadata = state.get('metadata')
            self.processed_data = state.get('processed_data')
            self.embedding = state.get('embedding')
            self.clusters = state.get('clusters')
            self.selected_features = state.get('selected_features')
            self.snippet_to_clip_map = state.get('snippet_to_clip_map', {})
            self.metadata_file_path = state.get('metadata_file_path')
            
            # Refresh UI
            self.status_label.setText("Analysis state loaded.")
            self.file_info_label.setText(f"Loaded analysis from: {os.path.basename(path)}")
            
            # Re-build clip map if missing
            if not self.snippet_to_clip_map:
                self._build_snippet_to_clip_map()
                
            # Update UI elements
            self._refresh_metadata_columns()
            self._refresh_cluster_list()
            self._refresh_cluster_export_list()
            self._refresh_spatial_selectors()
            
            # Enable buttons
            has_data = self.matrix_data is not None
            self.run_btn.setEnabled(has_data)
            
            # Update plot if embedding exists
            if self.embedding is not None and self.clusters is not None:
                # Regenerate default UMAP plot from loaded data
                self._regenerate_default_plot()
                
                # Trigger plot update via plot type
                self._update_plots_by_metadata()
                
                # Enable export buttons
                if hasattr(self, 'export_plot_btn'):
                    self.export_plot_btn.setEnabled(True)
                if hasattr(self, 'export_csv_btn'):
                    self.export_csv_btn.setEnabled(True)
            
            QMessageBox.information(self, "Success", "Analysis state loaded successfully.")
            
        except Exception as e:
            logger.error("Failed to load analysis: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load analysis: {str(e)}")

    def _regenerate_default_plot(self):
        """Regenerate the default UMAP plot from loaded embedding and clusters."""
        if self.embedding is None or self.clusters is None:
            return
            
        try:
            # Create sample index
            if self.processed_data is not None:
                sample_index = self.processed_data.index.tolist()
            else:
                sample_index = [f"snippet{i}" for i in range(len(self.clusters))]
            
            df_plot = pd.DataFrame({
                'UMAP1': self.embedding[:, 0],
                'UMAP2': self.embedding[:, 1],
                'Cluster': [f'Cluster_{c}' if c >= 0 else 'Noise' for c in self.clusters],
                'Sample': sample_index
            })
            
            if self.embedding.shape[1] >= 3:
                df_plot['UMAP3'] = self.embedding[:, 2]
                fig = px.scatter_3d(
                    df_plot, x='UMAP1', y='UMAP2', z='UMAP3',
                    color='Cluster', hover_data=['Sample'],
                    title="UMAP Clustering (Loaded)",
                    custom_data=[sample_index]
                )
            else:
                fig = px.scatter(
                    df_plot, x='UMAP1', y='UMAP2',
                    color='Cluster', hover_data=['Sample'],
                    title="UMAP Clustering (Loaded)",
                    custom_data=[sample_index]
                )
            
            theme = self._get_plot_theme()
            fig.update_layout(template=theme if theme else None)
            point_size = self.plot_point_size_slider.value() if hasattr(self, 'plot_point_size_slider') else 5
            fig.update_traces(marker=dict(size=point_size))
            self.current_fig = fig
            self.plot_widget.update_plot(fig)
            
        except Exception as e:
            logger.error("Error regenerating plot: %s", e, exc_info=True)
    
    def _generate_cluster_colors(self, n_clusters):
        """Generate colors for clusters."""
        base_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        if n_clusters <= len(base_colors):
            return base_colors[:n_clusters]
        # Generate more colors if needed
        import colorsys
        colors = []
        for i in range(n_clusters):
            hue = i / n_clusters
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
        return colors
    
    def _generate_pastel_palette(self, n_colors):
        """Generate pastel color palette."""
        base_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA500', '#FF1493',
            '#32CD32', '#9B59B6', '#FF8C00', '#00CED1', '#DA70D6'
        ]
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        # Cycle through colors
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]
    
    def _save_metadata_to_file(self, df: pd.DataFrame, path: str):
        """Save metadata DataFrame to file, respecting the file format based on extension."""
        if path.endswith(".npz"):
            # Save as NPZ format
            np.savez_compressed(
                path,
                metadata=df.values,
                columns=np.array(df.columns, dtype=object),
            )
        elif path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            # Default to CSV
            df.to_csv(path, index=False)
    
    def _build_snippet_to_clip_map(self):
        """Build mapping from snippet IDs to clip file paths."""
        self.snippet_to_clip_map = {}
        
        if self.metadata is None:
            return
        
        # Get experiment path and registered_clips directory
        experiment_path = self.config.get("experiment_path")
        if not experiment_path:
            return
        
        registered_clips_dir = os.path.join(experiment_path, "registered_clips")
        if not os.path.exists(registered_clips_dir):
            return
        
        # Get snippet column name
        snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
        if snippet_col is None:
            return
        
        # Find all clip files recursively (clips are in subdirectories: registered_clips/video_name/clip_XXXXXX.mp4)
        import glob
        clip_files = glob.glob(os.path.join(registered_clips_dir, "**", "*.avi"), recursive=True)
        # Also find legacy .mp4 clips for backwards compatibility
        clip_files += glob.glob(os.path.join(registered_clips_dir, "**", "*.mp4"), recursive=True)
        clip_name_to_path = {}
        for clip_file in clip_files:
            clip_name = os.path.basename(clip_file)
            clip_name_to_path[clip_name] = clip_file
            clip_name_to_path[clip_name.lower()] = clip_file
        
        # Map snippets to clips using video_id from metadata
        # The metadata has 'video_id' which contains the clip filename (e.g., 'clip_000000_obj1.mp4')
        for idx, row in self.metadata.iterrows():
            snippet_id = str(row.get(snippet_col, ''))
            if not snippet_id:
                continue
            
            # Get video_id (clip filename) from metadata
            video_id = str(row.get('video_id', ''))
            
            # Try exact match first
            if video_id and video_id in clip_name_to_path:
                self.snippet_to_clip_map[snippet_id] = clip_name_to_path[video_id]
                continue
            
            # Try case-insensitive match
            if video_id and video_id.lower() in clip_name_to_path:
                self.snippet_to_clip_map[snippet_id] = clip_name_to_path[video_id.lower()]
                continue
            
            # Fallback: try to match by clip filename pattern
            clip_index = row.get('clip_index', None)
            object_id = str(row.get('object_id', '')) if pd.notna(row.get('object_id')) and row.get('object_id') else None
            
            if clip_index is not None:
                clip_idx_str = f"{int(clip_index):06d}"
                for clip_name, clip_path in clip_name_to_path.items():
                    if clip_name.islower():
                        continue
                    if clip_idx_str in clip_name:
                        if object_id:
                            if f"_obj{object_id}" in clip_name:
                                self.snippet_to_clip_map[snippet_id] = clip_path
                                break
                        else:
                            if "_obj" not in clip_name:
                                self.snippet_to_clip_map[snippet_id] = clip_path
                                break
            
            # If still not found, try matching by position in metadata
            if snippet_id not in self.snippet_to_clip_map:
                try:
                    snippet_num = int(snippet_id.replace('snippet', '')) - 1
                    if 0 <= snippet_num < len(clip_files):
                        sorted_clips = sorted(clip_files)
                        self.snippet_to_clip_map[snippet_id] = sorted_clips[snippet_num]
                except (ValueError, IndexError):
                    pass
    
    def _on_umap_point_clicked(self, snippet_id: str):
        """Handle click on UMAP point - open video popup for corresponding clip."""
        if not snippet_id:
            return
        
        # Build snippet-to-clip mapping if not already done
        if not self.snippet_to_clip_map:
            self._build_snippet_to_clip_map()
        
        # Find clip file
        clip_path = self.snippet_to_clip_map.get(snippet_id)
        if not clip_path or not os.path.exists(clip_path):
            QMessageBox.warning(self, "Clip Not Found", 
                f"Could not find clip file for snippet: {snippet_id}\n\n"
                f"Please ensure clips are extracted in the Registration tab.")
            return
        
        # Get metadata for this snippet
        clip_metadata = None
        if self.metadata is not None:
            snippet_col = 'snippet' if 'snippet' in self.metadata.columns else ('span_id' if 'span_id' in self.metadata.columns else None)
            if snippet_col:
                snippet_row = self.metadata[self.metadata[snippet_col].astype(str) == str(snippet_id)]
                if len(snippet_row) > 0:
                    row = snippet_row.iloc[0]
                    start_frame = row.get('start_frame')
                    end_frame = row.get('end_frame')
                    video_id = row.get('video_id', '')
                    
                    # Get FPS from video file
                    fps = None
                    try:
                        import cv2
                        cap = cv2.VideoCapture(clip_path)
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                    except Exception as e:
                        logger.debug("Could not read clip FPS: %s", e)
                    
                    # Only set clip metadata if frame indices are valid numbers
                    try:
                        # Check if values are not NaN/None and not empty strings
                        start_valid = (pd.notna(start_frame) and str(start_frame).strip() != '')
                        end_valid = (pd.notna(end_frame) and str(end_frame).strip() != '')
                        
                        if start_valid and end_valid:
                            clip_metadata = {
                                'start_frame': int(float(start_frame)),
                                'end_frame': int(float(end_frame)),
                                'context_frames': 30,  # Default context frames
                                'fps': fps if fps else 30.0
                            }
                    except (ValueError, TypeError):
                        # If conversion fails, skip metadata
                        clip_metadata = None
        
        # Open video popup
        self._open_video_popup(clip_path, snippet_id, clip_metadata)
    
    def _open_video_popup(self, file_path: str, label: str = None, clip_metadata: dict = None):
        """Open video in popup dialog with timeline indicator (from clustering_behavior)"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtMultimediaWidgets import QVideoWidget
        from PyQt6.QtCore import QUrl
        from .plot_integration import TimelineWidget
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Video: {label if label else os.path.basename(file_path)}")
        dialog.setMinimumSize(800, 650)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(5)
        
        # Video container with timeline
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        # Expanded video widget
        expanded_video = QVideoWidget()
        expanded_video.setMinimumSize(800, 600)
        
        # Calculate clip boundaries in milliseconds (for clip-only playback)
        clip_start_ms = None
        clip_end_ms = None
        if clip_metadata:
            start_frame = clip_metadata.get('start_frame')
            end_frame = clip_metadata.get('end_frame')
            context_frames = clip_metadata.get('context_frames', 30)
            fps = clip_metadata.get('fps')
            
            if start_frame is not None and end_frame is not None and fps:
                # Calculate clip boundaries in milliseconds
                # The extracted video starts at context_start = max(0, start_frame - context_frames)
                # So in the extracted video, the clip starts at (start_frame - context_start)
                context_start = max(0, start_frame - context_frames)
                clip_start_in_extracted = start_frame - context_start
                clip_end_in_extracted = end_frame - context_start
                clip_start_ms = int((clip_start_in_extracted / fps) * 1000)
                clip_end_ms = int((clip_end_in_extracted / fps) * 1000)
        
        # Timeline widget
        timeline_widget = TimelineWidget(clip_metadata=clip_metadata)
        timeline_widget.setFixedHeight(30)
        timeline_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-top: 1px solid #555;
            }
        """)
        
        # Expanded player
        expanded_player = QMediaPlayer()
        expanded_audio = QAudioOutput()
        expanded_player.setAudioOutput(expanded_audio)
        expanded_player.setVideoOutput(expanded_video)
        expanded_player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))
        
        # Clip-only mode state
        clip_only_mode = [False]
        
        # Enhanced position tracking with clip boundary enforcement
        def update_timeline_position_with_clip(position):
            try:
                if timeline_widget and timeline_widget.isVisible():
                    timeline_widget.set_current_position(position, expanded_player.duration())
                
                # Enforce clip boundaries when clip-only mode is enabled
                if clip_only_mode[0] and clip_start_ms is not None and clip_end_ms is not None:
                    if position < clip_start_ms:
                        # Jump to clip start if before clip
                        expanded_player.setPosition(clip_start_ms)
                    elif position > clip_end_ms:
                        # If looping is enabled, jump back to clip start
                        if loop_enabled[0]:
                            expanded_player.setPosition(clip_start_ms)
                        else:
                            # Otherwise pause at clip end
                            expanded_player.pause()
                            play_btn.setText("▶ Play")
            except RuntimeError:
                # Widget was deleted, ignore
                pass
        
        def update_timeline_duration(duration):
            try:
                if timeline_widget and timeline_widget.isVisible():
                    timeline_widget.set_duration(duration)
            except RuntimeError:
                # Widget was deleted, ignore
                pass
        
        expanded_player.positionChanged.connect(update_timeline_position_with_clip)
        expanded_player.durationChanged.connect(update_timeline_duration)
        
        # Loop state
        loop_enabled = [False]
        
        def on_playback_state_changed(state):
            if loop_enabled[0] and state == QMediaPlayer.PlaybackState.StoppedState:
                if clip_only_mode[0] and clip_start_ms is not None:
                    expanded_player.setPosition(clip_start_ms)
                else:
                    expanded_player.setPosition(0)
                expanded_player.play()
        
        def on_media_status_changed(status):
            if loop_enabled[0] and status == QMediaPlayer.MediaStatus.EndOfMedia:
                if clip_only_mode[0] and clip_start_ms is not None:
                    expanded_player.setPosition(clip_start_ms)
                else:
                    expanded_player.setPosition(0)
                expanded_player.play()
        
        expanded_player.playbackStateChanged.connect(on_playback_state_changed)
        expanded_player.mediaStatusChanged.connect(on_media_status_changed)
        
        video_layout.addWidget(expanded_video)
        video_layout.addWidget(timeline_widget)
        
        # Controls
        controls = QHBoxLayout()
        
        def toggle_play():
            if expanded_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                expanded_player.pause()
                play_btn.setText("▶ Play")
            else:
                expanded_player.play()
                play_btn.setText("⏸ Pause")
        
        play_btn = QPushButton("▶ Play")
        play_btn.clicked.connect(toggle_play)
        
        def toggle_loop(checked):
            loop_enabled[0] = checked
            if checked:
                loop_btn.setText("🔁 Loop ON")
            else:
                loop_btn.setText("🔁 Loop")
        
        loop_btn = QPushButton("🔁 Loop")
        loop_btn.setCheckable(True)
        loop_btn.toggled.connect(toggle_loop)
        
        # Clip-only mode checkbox (only show if clip metadata is available)
        clip_only_chk = None
        if clip_start_ms is not None and clip_end_ms is not None:
            clip_only_chk = QCheckBox("Clip only")
            clip_only_chk.setToolTip("When checked, play and loop only the clip area (without context frames)")
            
            def toggle_clip_only(checked):
                clip_only_mode[0] = checked
                if checked:
                    # Jump to clip start when enabling clip-only mode
                    expanded_player.setPosition(clip_start_ms)
                    # Auto-enable loop for better UX in clip-only mode
                    if not loop_enabled[0]:
                        loop_btn.setChecked(True)
                # If unchecked, allow playback from current position (full video)
            
            clip_only_chk.toggled.connect(toggle_clip_only)
        
        # Playback speed controls
        speed_label = QLabel("Speed:")
        speed_1x_btn = QPushButton("1x")
        speed_1x_btn.setCheckable(True)
        speed_1x_btn.setChecked(True)
        speed_05x_btn = QPushButton("0.5x")
        speed_05x_btn.setCheckable(True)
        speed_025x_btn = QPushButton("0.25x")
        speed_025x_btn.setCheckable(True)
        speed_0166x_btn = QPushButton("0.166x")
        speed_0166x_btn.setCheckable(True)
        
        # Speed button group
        speed_buttons = [speed_1x_btn, speed_05x_btn, speed_025x_btn, speed_0166x_btn]
        current_speed = [1.0]  # Use list to allow modification in nested functions
        
        def set_speed(speed, button):
            current_speed[0] = speed
            expanded_player.setPlaybackRate(speed)
            # Update button states
            for btn in speed_buttons:
                btn.setChecked(btn == button)
        
        speed_1x_btn.clicked.connect(lambda: set_speed(1.0, speed_1x_btn))
        speed_05x_btn.clicked.connect(lambda: set_speed(0.5, speed_05x_btn))
        speed_025x_btn.clicked.connect(lambda: set_speed(0.25, speed_025x_btn))
        speed_0166x_btn.clicked.connect(lambda: set_speed(1.0/6.0, speed_0166x_btn))
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        
        controls.addWidget(play_btn)
        controls.addWidget(loop_btn)
        if clip_only_chk is not None:
            controls.addWidget(clip_only_chk)
        controls.addWidget(speed_label)
        controls.addWidget(speed_1x_btn)
        controls.addWidget(speed_05x_btn)
        controls.addWidget(speed_025x_btn)
        controls.addWidget(speed_0166x_btn)
        controls.addStretch()
        controls.addWidget(close_btn)
        
        layout.addWidget(video_container)
        layout.addLayout(controls)
        
        # Auto-play when opened (if clip-only mode is enabled, start at clip start)
        if clip_only_mode[0] and clip_start_ms is not None:
            expanded_player.setPosition(clip_start_ms)
        expanded_player.play()
        play_btn.setText("⏸ Pause")
        
        dialog.exec()
        
        # Cleanup - disconnect signals before stopping to prevent errors
        try:
            expanded_player.positionChanged.disconnect()
            expanded_player.durationChanged.disconnect()
            expanded_player.playbackStateChanged.disconnect()
            expanded_player.mediaStatusChanged.disconnect()
        except (RuntimeError, TypeError):
            # Signals already disconnected or widget deleted
            pass
        
        expanded_player.stop()
