import json
import logging
import os
import numpy as np
import cv2
import yaml

logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
    QFormLayout, QLineEdit, QMessageBox, QSplitter, QCheckBox, QFileDialog,
    QSizePolicy, QScrollArea, QTabWidget, QListWidget, QListWidgetItem,
    QTextEdit, QAbstractItemView, QSlider, QToolButton, QInputDialog,
    QButtonGroup
)
from PyQt6.QtCore import Qt, QUrl, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF, QPainterPath

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class AnalysisWidget(QWidget):
    """Widget for downstream analysis of behavior data."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.results = {}
        self.groups = {} # {group_name: [video_path, ...]}
        self.video_groups = {} # {video_path: group_name}
        self.spatial_regions = [] # [{name, type, vertices}, ...]
        self.merged_data = [] # List of dicts
        self.all_behaviors = []
        self.selected_behaviors = set()
        self.visible_groups = set()
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Global Controls (Load, Manage Groups)
        global_controls = QGroupBox("Data controls")
        global_controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        global_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load results")
        self.load_btn.clicked.connect(self._load_results)
        global_layout.addWidget(self.load_btn)
        
        self.manage_groups_btn = QPushButton("Manage groups")
        self.manage_groups_btn.clicked.connect(self._manage_groups)
        self.manage_groups_btn.setEnabled(False)
        global_layout.addWidget(self.manage_groups_btn)
        
        self.filter_behaviors_btn = QPushButton("Filter behaviors")
        self.filter_behaviors_btn.clicked.connect(self._filter_behaviors)
        self.filter_behaviors_btn.setEnabled(False)
        global_layout.addWidget(self.filter_behaviors_btn)
        
        global_layout.addStretch()
        global_controls.setLayout(global_layout)
        layout.addWidget(global_controls)
        
        # Tab Widget
        self.tabs = QTabWidget()
        
        # Overview Tab
        self.overview_tab = self._create_overview_tab()
        self.tabs.addTab(self.overview_tab, "Overview")
        
        # Behavior Transitions Tab
        self.transitions_tab = self._create_transitions_tab()
        self.tabs.addTab(self.transitions_tab, "Behavior Transitions")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def _create_overview_tab(self):
        """Create the Overview tab with split layout."""
        tab = QWidget()
        main_layout = QHBoxLayout()
        
        # Left: Plot area (70%)
        if HAS_WEBENGINE:
            self.webview = QWebEngineView()
            self.webview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            main_layout.addWidget(self.webview, 70)
        else:
            lbl = QLabel("PyQt6.QtWebEngineWidgets not installed. Plots will open in default browser.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(lbl, 70)
        
        # Right: Controls & Analysis (30%)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(15)
        
        # 1. Plot Settings
        plot_group = QGroupBox("Plot settings")
        plot_layout = QFormLayout()
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Occurrences (Count)", "Average Bout Duration (s)", "Total Duration (s)", "Percent Time (%)"])
        self.metric_combo.currentIndexChanged.connect(self._update_plots)
        plot_layout.addRow("Metric:", self.metric_combo)
        
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["General Overview", "Group Comparison"])
        self.plot_mode_combo.currentIndexChanged.connect(self._update_plots)
        plot_layout.addRow("Analysis Mode:", self.plot_mode_combo)
        
        plot_group.setLayout(plot_layout)
        sidebar_layout.addWidget(plot_group)

        # 1b. Graph Appearance
        app_group = QGroupBox("Graph appearance")
        app_layout = QFormLayout()
        
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["Auto", "Bar Chart", "Box Plot", "Violin Plot", "Strip Plot", "Line Plot"])
        self.graph_type_combo.currentIndexChanged.connect(self._update_plots)
        app_layout.addRow("Graph Type:", self.graph_type_combo)
        
        self.color_theme_combo = QComboBox()
        self.color_theme_combo.addItems(["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"])
        self.color_theme_combo.setCurrentText("simple_white")
        self.color_theme_combo.currentIndexChanged.connect(self._update_plots)
        app_layout.addRow("Color Theme:", self.color_theme_combo)
        
        app_group.setLayout(app_layout)
        sidebar_layout.addWidget(app_group)

        # 1c. Spatial Distribution (uses localization data)
        spatial_group = QGroupBox("Spatial distribution")
        spatial_layout = QVBoxLayout()

        self.spatial_behavior_combo = QComboBox()
        self.spatial_behavior_combo.addItem("All behaviors")
        self.spatial_behavior_combo.currentIndexChanged.connect(self._update_spatial_plot)
        spatial_form = QFormLayout()
        spatial_form.addRow("Behavior:", self.spatial_behavior_combo)

        self.spatial_video_combo = QComboBox()
        self.spatial_video_combo.addItem("All")
        self.spatial_video_combo.currentIndexChanged.connect(self._update_spatial_plot)
        spatial_form.addRow("Video:", self.spatial_video_combo)

        self.spatial_dot_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.spatial_dot_size_slider.setMinimum(1)
        self.spatial_dot_size_slider.setMaximum(25)
        self.spatial_dot_size_slider.setValue(7)
        self.spatial_dot_size_slider.valueChanged.connect(self._update_spatial_plot)
        spatial_form.addRow("Dot size:", self.spatial_dot_size_slider)

        spatial_layout.addLayout(spatial_form)

        self.spatial_show_btn = QPushButton("Show spatial distribution")
        self.spatial_show_btn.clicked.connect(self._update_spatial_plot)
        self.spatial_show_btn.setEnabled(False)
        spatial_layout.addWidget(self.spatial_show_btn)

        self.spatial_save_btn = QPushButton("Save spatial plot (PDF/SVG)")
        self.spatial_save_btn.clicked.connect(self._save_spatial_plot)
        spatial_layout.addWidget(self.spatial_save_btn)

        self.spatial_info_label = QLabel("Load results with localization data to use.")
        self.spatial_info_label.setWordWrap(True)
        self.spatial_info_label.setStyleSheet("color: grey; font-size: 11px;")
        spatial_layout.addWidget(self.spatial_info_label)

        spatial_group.setToolTip(
            "One point per inference clip (step-frame grid). Not based on aggregated bout boundaries."
        )

        self.manage_regions_btn = QPushButton("Manage spatial regions")
        self.manage_regions_btn.setToolTip("Draw named regions on the spatial map, then filter analysis by region.")
        self.manage_regions_btn.clicked.connect(self._manage_spatial_regions)
        self.manage_regions_btn.setEnabled(False)
        spatial_layout.addWidget(self.manage_regions_btn)

        self.region_filter_combo = QComboBox()
        self.region_filter_combo.addItem("All Regions")
        self.region_filter_combo.currentIndexChanged.connect(self._on_region_filter_changed)
        spatial_form.addRow("Region:", self.region_filter_combo)

        spatial_group.setLayout(spatial_layout)
        sidebar_layout.addWidget(spatial_group)

        # 2. Group Selection
        group_group = QGroupBox("Groups")
        group_layout = QVBoxLayout()
        self.group_list_widget = QListWidget()
        self.group_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection) # Checkboxes only
        self.group_list_widget.setFixedHeight(150)
        self.group_list_widget.itemChanged.connect(self._on_group_selection_changed)
        group_layout.addWidget(self.group_list_widget)
        group_group.setLayout(group_layout)
        sidebar_layout.addWidget(group_group)
        
        # 3. Statistics
        stats_group = QGroupBox("Statistics (Group comp.)")
        stats_layout = QVBoxLayout()
        
        self.stats_test_combo = QComboBox()
        self.stats_test_combo.addItems(["T-Test / Mann-Whitney (2 groups)", "ANOVA / Kruskal-Wallis (>2 groups)"])
        stats_layout.addWidget(QLabel("Test type:"))
        stats_layout.addWidget(self.stats_test_combo)
        
        self.run_stats_btn = QPushButton("Run statistics")
        self.run_stats_btn.clicked.connect(self._run_statistics)
        stats_layout.addWidget(self.run_stats_btn)
        
        self.stats_output = QTextEdit()
        self.stats_output.setReadOnly(True)
        self.stats_output.setPlaceholderText("Results will appear here...")
        self.stats_output.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_output)
        
        stats_group.setLayout(stats_layout)
        sidebar_layout.addWidget(stats_group)
        
        # 4. Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.save_graph_btn = QPushButton("Save graph")
        self.save_graph_btn.clicked.connect(self._save_graph)
        export_layout.addWidget(self.save_graph_btn)
        
        self.save_csv_btn = QPushButton("Save data (.csv)")
        self.save_csv_btn.clicked.connect(self._save_csv)
        export_layout.addWidget(self.save_csv_btn)
        
        export_group.setLayout(export_layout)
        sidebar_layout.addWidget(export_group)
        
        sidebar_layout.addStretch()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_scroll.setWidget(sidebar_widget)
        
        main_layout.addWidget(sidebar_scroll, 30)
        
        tab.setLayout(main_layout)
        return tab

    def _update_sidebar_groups(self):
        """Update group list in sidebar based on current data."""
        self.group_list_widget.blockSignals(True)
        self.group_list_widget.clear()
        
        # Get all unique groups from data or metadata
        groups = sorted(list(self.groups.keys()))
        if not groups and self.merged_data:
             # Fallback if groups dict not fully populated but data exists
             df = pd.DataFrame(self.merged_data)
             if "Group" in df.columns:
                 groups = sorted(df["Group"].dropna().unique().tolist())
        
        if not groups:
            groups = ["Unassigned"]

        for grp in groups:
            item = QListWidgetItem(grp)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.group_list_widget.addItem(item)
            
        self.visible_groups = set(groups)
        self.group_list_widget.blockSignals(False)

    def _on_group_selection_changed(self, item):
        """Handle group checkbox toggles."""
        self.visible_groups = set()
        for i in range(self.group_list_widget.count()):
            it = self.group_list_widget.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                self.visible_groups.add(it.text())
        self._update_plots()

    def _run_statistics(self):
        if not HAS_SCIPY:
            QMessageBox.warning(self, "Error", "Scipy is not installed. Cannot run statistics.")
            return
            
        if not self.merged_data:
            return

        df = pd.DataFrame(self.merged_data)
        
        # Apply filters
        if self.selected_behaviors:
            df = df[df["Behavior"].isin(self.selected_behaviors)]
        if self.visible_groups:
            df = df[df["Group"].isin(self.visible_groups)]
        sel_region = self.region_filter_combo.currentText()
        if sel_region != "All Regions" and "Region" in df.columns:
            df = df[df["Region"] == sel_region]
            
        if df.empty:
            self.stats_output.setText("No data available for selected filters.")
            return
            
        metric = self.metric_combo.currentText()
        test_type = self.stats_test_combo.currentText()
        
        # Prepare data for stats
        if "Occurrences" in metric:
            agg = df.groupby(["Video", "Group", "Behavior"]).size().reset_index(name="Value")
        elif "Average" in metric:
            agg = df.groupby(["Video", "Group", "Behavior"])["Duration"].mean().reset_index(name="Value")
        elif "Total" in metric:
            agg = df.groupby(["Video", "Group", "Behavior"])["Duration"].sum().reset_index(name="Value")
        elif "Percent" in metric:
            total_times = df.groupby("Video")["Duration"].sum().to_dict()
            agg = df.groupby(["Video", "Group", "Behavior"])["Duration"].sum().reset_index(name="Value")
            agg["Value"] = agg.apply(lambda x: (x["Value"] / total_times.get(x["Video"], 1)) * 100, axis=1)
            
        output = []
        output.append(f"Metric: {metric}")
        output.append(f"Test: {test_type}\n")
        
        unique_behaviors = sorted(agg["Behavior"].unique())
        unique_groups = sorted(agg["Group"].unique())
        
        if len(unique_groups) < 2:
            self.stats_output.setText("Need at least 2 groups for comparison.")
            return

        for beh in unique_behaviors:
            beh_data = agg[agg["Behavior"] == beh]
            groups_data = [beh_data[beh_data["Group"] == g]["Value"].values for g in unique_groups]
            
            # Filter valid data
            valid_groups_data = []
            valid_group_names = []
            for g, d in zip(unique_groups, groups_data):
                if len(d) > 0:
                    valid_groups_data.append(d)
                    valid_group_names.append(g)
            
            if len(valid_groups_data) < 2:
                output.append(f"{beh}: Not enough data")
                continue
                
            try:
                if "2 groups" in test_type:
                    # Pairwise Mann-Whitney
                    import itertools
                    for g1, g2 in itertools.combinations(valid_group_names, 2):
                        d1 = valid_groups_data[valid_group_names.index(g1)]
                        d2 = valid_groups_data[valid_group_names.index(g2)]
                        stat, p = stats.mannwhitneyu(d1, d2)
                        output.append(f"{beh} ({g1} vs {g2}): p={p:.4f}")
                        
                else:
                    # Kruskal-Wallis
                    stat, p = stats.kruskal(*valid_groups_data)
                    output.append(f"{beh} (Kruskal-Wallis): p={p:.4f}")
                    
            except Exception as e:
                output.append(f"{beh}: Error ({str(e)})")
                
        self.stats_output.setText("\n".join(output))

    def _save_graph(self):
        if not hasattr(self, 'last_fig') or self.last_fig is None:
             QMessageBox.warning(self, "Error", "No plot to save.")
             return
             
        path, filter_ = QFileDialog.getSaveFileName(
            self, "Save Graph", 
            self.config.get("experiment_path", ""),
            "PDF Files (*.pdf);;SVG Files (*.svg);;PNG Files (*.png);;HTML Files (*.html)"
        )
        
        if not path:
            return
            
        try:
            if path.lower().endswith(".html"):
                self.last_fig.write_html(path)
            else:
                self.last_fig.write_image(path)
            QMessageBox.information(self, "Success", f"Graph saved to {path}")
        except Exception as e:
             if "kaleido" in str(e).lower() or "executable" in str(e).lower():
                 QMessageBox.warning(self, "Error", "Saving as static image requires 'kaleido'.\nPlease install it (pip install kaleido) or save as HTML.")
             else:
                 QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")

    def _save_csv(self):
        if not self.merged_data:
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", 
            self.config.get("experiment_path", ""),
            "CSV Files (*.csv)"
        )
        
        if not path:
            return
            
        df = pd.DataFrame(self.merged_data)
        if self.selected_behaviors:
            df = df[df["Behavior"].isin(self.selected_behaviors)]
        if self.visible_groups:
            df = df[df["Group"].isin(self.visible_groups)]
        sel_region = self.region_filter_combo.currentText()
        if sel_region != "All Regions" and "Region" in df.columns:
            df = df[df["Region"] == sel_region]
            
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Success", f"Data saved to {path}")
    
    def _create_transitions_tab(self):
        """Create the Behavior Transitions tab (Markov chain analysis)."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox("Transition analysis controls")
        controls_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Analysis type:"))
        self.transition_type_combo = QComboBox()
        self.transition_type_combo.addItems(["Individual Video", "Group Comparison"])
        self.transition_type_combo.currentIndexChanged.connect(self._on_transition_type_changed)
        controls_layout.addWidget(self.transition_type_combo)
        
        controls_layout.addWidget(QLabel("Select:"))
        self.transition_select_combo = QComboBox()
        controls_layout.addWidget(self.transition_select_combo)
        
        # New Controls for Layout and Filtering
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Circular Layout", "Network Layout"])
        controls_layout.addWidget(QLabel("Layout:"))
        controls_layout.addWidget(self.layout_combo)
        
        self.sig_filter_check = QCheckBox("Significant only")
        self.sig_filter_check.setToolTip("Only show transitions that occur significantly more than expected by chance (Z-score > 1.96)")
        controls_layout.addWidget(self.sig_filter_check)
        
        self.compute_transition_btn = QPushButton("Compute transitions")
        self.compute_transition_btn.clicked.connect(self._compute_transitions)
        self.compute_transition_btn.setEnabled(False)
        controls_layout.addWidget(self.compute_transition_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Transition Matrix Display
        matrix_group = QGroupBox("Transition matrix")
        matrix_layout = QVBoxLayout()
        self.transition_matrix_table = QTableWidget()
        self.transition_matrix_table.setMaximumHeight(250)
        matrix_layout.addWidget(self.transition_matrix_table)
        matrix_group.setLayout(matrix_layout)
        layout.addWidget(matrix_group)
        
        # Transition Graph
        if HAS_WEBENGINE:
            self.transition_webview = QWebEngineView()
            self.transition_webview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self.transition_webview, 1)
        else:
            layout.addWidget(QLabel("PyQt6.QtWebEngineWidgets not installed."))
        
        tab.setLayout(layout)
        return tab
        
    def update_config(self, config: dict):
        self.config = config
        self._load_groups_from_config()
        # Try to auto-load
        self._auto_load_results()

    def _load_groups_from_config(self):
        groups = self.config.get("analysis_groups", {})
        video_groups = self.config.get("analysis_video_groups", {})
        if isinstance(groups, dict):
            self.groups = {
                str(group_name): [str(path) for path in (paths or [])]
                for group_name, paths in groups.items()
            }
        if isinstance(video_groups, dict):
            self.video_groups = {
                str(video_path): str(group_name)
                for video_path, group_name in video_groups.items()
            }

    def _save_groups_to_config(self):
        self.config["analysis_groups"] = {
            str(group_name): [str(path) for path in sorted(set(paths or []))]
            for group_name, paths in self.groups.items()
        }
        self.config["analysis_video_groups"] = {
            str(video_path): str(group_name)
            for video_path, group_name in self.video_groups.items()
        }
        config_path = self.config.get("config_path")
        if not config_path:
            return
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dict(self.config), f, sort_keys=False)
        except Exception as e:
            logger.error("Error saving group config: %s", e)

    def _auto_load_results(self):
        exp_path = self.config.get("experiment_path")
        if exp_path:
            results_path = os.path.join(exp_path, "results", "inference_results.json")
            if os.path.exists(results_path):
                self._load_results_from_file(results_path)

    def _load_results(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Inference Results",
            self.config.get("experiment_path", ""),
            "JSON Files (*.json)"
        )
        if path:
            self._load_results_from_file(path)

    def _load_results_from_file(self, path):
        try:
            with open(path, 'r') as f:
                self.results = json.load(f)
            
            self.manage_groups_btn.setEnabled(True)
            
            # Load metadata if exists
            meta_path = path.replace("inference_results.json", "analysis_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    if not self.groups:
                        self.groups = meta.get("groups", {})
                    if not self.video_groups:
                        self.video_groups = meta.get("video_groups", {})
                    raw_regions = meta.get("spatial_regions", [])
                    self.spatial_regions = []
                    for r in raw_regions:
                        self.spatial_regions.append({
                            "name": r["name"],
                            "type": r["type"],
                            "vertices": [tuple(v) for v in r["vertices"]],
                        })
            
            self._update_region_filter_combo()
            self.manage_regions_btn.setEnabled(True)
            self._process_data()
            self._update_plots()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load results: {e}")

    @staticmethod
    def _point_in_region(cx, cy, region):
        verts = region["vertices"]
        if region["type"] == "rect":
            return verts[0][0] <= cx <= verts[1][0] and verts[0][1] <= cy <= verts[1][1]
        from matplotlib.path import Path as MplPath
        return MplPath(verts).contains_point((cx, cy))

    def _region_for_point(self, cx, cy):
        """Return region name for a centroid, or 'Outside'."""
        for r in self.spatial_regions:
            if self._point_in_region(cx, cy, r):
                return r["name"]
        return "Outside"

    def _build_clip_centroids(self, v_data):
        """Return list of (cx, cy) per clip index, or None for clips without bbox."""
        loc_bboxes = v_data.get("localization_bboxes", [])
        centroids = []
        for raw in loc_bboxes:
            cx, cy = self._bbox_centroid(raw)
            centroids.append((cx, cy) if cx is not None else None)
        return centroids

    def _process_data(self):
        """Merge timelines and prepare data structures."""
        self.merged_data = []
        has_regions = bool(self.spatial_regions)
        
        data_container = self.results
        if "results" in data_container:
            results_dict = data_container["results"]
            classes = data_container.get("classes", [])
            params = data_container.get("parameters", {})
            target_fps = params.get("target_fps", 30)
            clip_length = params.get("clip_length", 16)
            step_frames = params.get("step_frames", 16)
        else:
            results_dict = data_container
            classes = [] 
            params = {}
            target_fps = 30
            clip_length = 16
            step_frames = 16
            
        frame_agg_enabled = params.get("frame_aggregation_enabled", False)
        use_ovr_param = params.get("use_ovr", None)

        for video_path, v_data in results_dict.items():
            agg_segments = v_data.get("aggregated_segments", [])
            agg_multiclass = v_data.get("aggregated_multiclass_segments", [])
            use_ovr = bool(use_ovr_param) if use_ovr_param is not None else bool(agg_multiclass)
            clip_starts = v_data.get("clip_starts", [])
            total_frames = int(v_data.get("total_frames", 0) or 0)

            # Determine orig_fps: prefer saved per-video metadata.
            orig_fps = v_data.get("orig_fps", 0)
            if orig_fps <= 0:
                if os.path.exists(video_path):
                    try:
                        cap = cv2.VideoCapture(video_path)
                        orig_fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                    except Exception:
                        orig_fps = 0
                if orig_fps <= 0 and len(clip_starts) >= 2 and step_frames > 0:
                    actual_step = clip_starts[1] - clip_starts[0]
                    frame_interval = actual_step / step_frames
                    orig_fps = frame_interval * target_fps
                if orig_fps <= 0:
                    orig_fps = 30.0

            # Prefer stored inference-time interval when available.
            frame_interval = v_data.get("frame_interval", 0)
            try:
                frame_interval = int(frame_interval)
            except Exception:
                frame_interval = 0
            if frame_interval <= 0:
                if len(clip_starts) >= 2 and step_frames > 0:
                    inferred = int(round((clip_starts[1] - clip_starts[0]) / step_frames))
                    frame_interval = max(1, inferred)
                else:
                    frame_interval = max(1, int(round(orig_fps / max(1e-6, float(target_fps)))))

            # Build per-clip centroid lookup for spatial region assignment
            clip_centroids = self._build_clip_centroids(v_data) if has_regions else []

            def _region_for_frame_range(start_f, end_f):
                """Find region for a segment by checking the clip whose start is nearest the midpoint."""
                if not has_regions or not clip_centroids or not clip_starts:
                    return "All"
                mid = (start_f + end_f) / 2.0
                best_ci = 0
                best_dist = float("inf")
                for ci, cs in enumerate(clip_starts):
                    d = abs(cs - mid)
                    if d < best_dist:
                        best_dist = d
                        best_ci = ci
                if best_ci < len(clip_centroids) and clip_centroids[best_ci] is not None:
                    return self._region_for_point(*clip_centroids[best_ci])
                return "Outside"

            def _region_for_clip_range(bout_start, bout_end):
                """Find region for a clip-based bout by majority vote of clip centroids."""
                if not has_regions or not clip_centroids:
                    return "All"
                region_votes = {}
                for ci in range(bout_start, min(bout_end + 1, len(clip_centroids))):
                    c = clip_centroids[ci]
                    if c is not None:
                        rn = self._region_for_point(*c)
                        region_votes[rn] = region_votes.get(rn, 0) + 1
                if not region_votes:
                    return "Outside"
                return max(region_votes, key=region_votes.get)

            # Use frame-level aggregated segments when available (precise boundaries).
            # For OvR, prefer per-class multiclass segments so downstream analysis
            # sees overlapping labels as independent behavior bouts.
            #
            # IMPORTANT: saved aggregated_segments have per-class ignore thresholds
            # applied (frames below threshold are labelled class=-1 / "Filtered").
            # When the threshold for a class is set higher than the model's actual
            # confidence range for that class, the entire class disappears from the
            # saved segments even though the model predicted it as dominant.
            # To avoid this, we rebuild segments from the raw frame probabilities
            # (argmax, no threshold) when they are available so that analysis always
            # reflects true model predictions.
            agg_probs = v_data.get("aggregated_frame_probs")

            def _relabel_filtered_segs(segs, probs_list):
                """Relabel class=-1 (Filtered) segments using the argmax of the
                underlying frame probabilities over that segment's frame range.

                The saved aggregated_segments already encode the correct temporal
                structure (temporal smoothing, merge-gap, min-segment all applied).
                The only problem is that the per-class ignore threshold may have
                labelled some segments as Filtered (-1) even though the model
                predicted a real class with high confidence. This function restores
                the true model prediction for those segments without touching the
                segment boundaries.
                """
                if not probs_list or not segs:
                    return segs
                n_probs = len(probs_list)
                result = []
                for seg in segs:
                    if seg.get("class", -1) >= 0:
                        result.append(seg)
                        continue
                    # Filtered segment: determine label from mean probs over its range
                    s = max(0, int(seg["start"]))
                    e = min(n_probs - 1, int(seg["end"]))
                    if s > e:
                        result.append(seg)
                        continue
                    # Sum probabilities over the frame range and pick argmax
                    n_cls = len(probs_list[s])
                    totals = [0.0] * n_cls
                    for fi in range(s, e + 1):
                        for ci, p in enumerate(probs_list[fi]):
                            totals[ci] += p
                    best = int(max(range(n_cls), key=lambda ci: totals[ci]))
                    new_seg = dict(seg)
                    new_seg["class"] = best
                    result.append(new_seg)
                return result

            if use_ovr and agg_multiclass:
                seg_source = agg_multiclass
            elif frame_agg_enabled and agg_probs:
                # Use saved segments (correct temporal structure: temporal smoothing,
                # merge-gap, min-segment all already applied) but relabel any
                # Filtered (-1) segments with the true argmax from raw probabilities.
                seg_source = _relabel_filtered_segs(agg_segments, agg_probs)
            else:
                seg_source = agg_segments

            if frame_agg_enabled and seg_source:
                covered_frames = 0
                for seg in seg_source:
                    pred_idx = seg["class"]
                    start_frame = seg["start"]
                    end_frame = seg["end"]
                    covered_frames += max(0, (end_frame - start_frame + 1))

                    if pred_idx < 0:
                        label_name = "Filtered"
                    elif classes and pred_idx < len(classes):
                        label_name = classes[pred_idx]
                    else:
                        label_name = f"Class {pred_idx}"

                    duration_sec = (end_frame - start_frame + 1) / orig_fps

                    self.merged_data.append({
                        "Video": os.path.basename(video_path),
                        "VideoPath": video_path,
                        "Group": self.video_groups.get(video_path, "Unassigned"),
                        "Behavior": label_name,
                        "Duration": duration_sec,
                        "Region": _region_for_frame_range(start_frame, end_frame),
                    })

                # Keep totals comparable across videos by accounting for uncovered tail.
                if total_frames > 0 and covered_frames < total_frames:
                    self.merged_data.append({
                        "Video": os.path.basename(video_path),
                        "VideoPath": video_path,
                        "Group": self.video_groups.get(video_path, "Unassigned"),
                        "Behavior": "Uncovered",
                        "Duration": (total_frames - covered_frames) / orig_fps,
                        "Region": "All",
                    })
                continue

            # Fallback: clip-based bout detection
            preds = v_data.get("predictions", [])
            corrections = v_data.get("corrected_labels", {})
            confs = v_data.get("confidences", [])
            
            if not preds:
                continue
            
            # Reconstruct ignore threshold from saved parameters
            apply_ignore = params.get("use_ignore_threshold", False)
            ignore_thr = float(params.get("ignore_threshold", 0.5))

            # Apply corrections and threshold filtering
            final_preds = []
            for i, p in enumerate(preds):
                if str(i) in corrections:
                    pred = corrections[str(i)]
                elif i in corrections:
                    pred = corrections[i]
                else:
                    pred = p
                if apply_ignore and i < len(confs) and float(confs[i]) < ignore_thr:
                    pred = -1
                final_preds.append(pred)
            
            if not final_preds:
                continue

            # Derive bouts by clip index, then convert to seconds in original timeline.
            bout_start_idx = 0
            current_label = final_preds[0]
            covered_frames = 0

            for i in range(1, len(final_preds) + 1):
                boundary = (i == len(final_preds)) or (final_preds[i] != current_label)
                if not boundary:
                    continue

                bout_end_idx = i - 1
                if clip_starts and len(clip_starts) == len(final_preds):
                    start_frame = int(clip_starts[bout_start_idx])
                    if i < len(clip_starts):
                        end_frame_exclusive = int(clip_starts[i])
                    else:
                        end_frame_exclusive = start_frame + (clip_length - 1) * frame_interval + 1
                        if total_frames > 0:
                            end_frame_exclusive = min(end_frame_exclusive, total_frames)
                    duration_frames = max(0, end_frame_exclusive - start_frame)
                    covered_frames += duration_frames
                    duration_sec = duration_frames / orig_fps
                else:
                    # Legacy fallback when clip starts are missing.
                    clip_count = bout_end_idx - bout_start_idx + 1
                    duration_subsampled = ((clip_count - 1) * step_frames + clip_length)
                    duration_sec = duration_subsampled / max(1e-6, float(target_fps))

                if current_label < 0:
                    label_name = "Filtered"
                elif classes and current_label < len(classes):
                    label_name = classes[current_label]
                else:
                    label_name = f"Class {current_label}"

                self.merged_data.append({
                    "Video": os.path.basename(video_path),
                    "VideoPath": video_path,
                    "Group": self.video_groups.get(video_path, "Unassigned"),
                    "Behavior": label_name,
                    "Duration": duration_sec,
                    "Region": _region_for_clip_range(bout_start_idx, bout_end_idx),
                })

                if i < len(final_preds):
                    current_label = final_preds[i]
                    bout_start_idx = i

            if total_frames > 0 and covered_frames < total_frames:
                self.merged_data.append({
                    "Video": os.path.basename(video_path),
                    "VideoPath": video_path,
                    "Group": self.video_groups.get(video_path, "Unassigned"),
                    "Behavior": "Uncovered",
                    "Duration": (total_frames - covered_frames) / orig_fps,
                    "Region": "All",
                })

        # Update available behaviors and selection (include all model classes so behaviors
        # with zero or few segments still appear in filters and plots)
        if self.merged_data:
            df = pd.DataFrame(self.merged_data)
            from_data = set(df["Behavior"].unique().tolist())
            classes = []
            if isinstance(self.results, dict):
                classes = self.results.get("classes", [])
            self.all_behaviors = sorted(from_data | set(classes))
            
            # Initialize selection if empty (first load)
            if not self.selected_behaviors:
                self.selected_behaviors = set(self.all_behaviors)
            else:
                # Clean up stale behaviors
                self.selected_behaviors = self.selected_behaviors.intersection(set(self.all_behaviors))
                # If nothing selected (e.g. all prev behaviors gone), select all new ones
                if not self.selected_behaviors:
                    self.selected_behaviors = set(self.all_behaviors)
            
            self.filter_behaviors_btn.setEnabled(True)
            
            # Update transition tab
            self._on_transition_type_changed()
            self._update_sidebar_groups()

            # Populate spatial distribution combos
            self._populate_spatial_combos()
        else:
            self.filter_behaviors_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Spatial distribution (localization-based)
    # ------------------------------------------------------------------

    def _populate_spatial_combos(self):
        """Populate behavior and video combos for spatial distribution."""
        data_container = self.results
        results_dict = data_container.get("results", data_container)
        classes = data_container.get("classes", [])

        # Check if any video has localization data
        has_loc = any(
            "localization_bboxes" in v_data
            for v_data in (results_dict.values() if isinstance(results_dict, dict) else [])
        )

        self.spatial_behavior_combo.blockSignals(True)
        self.spatial_behavior_combo.clear()
        self.spatial_behavior_combo.addItem("All behaviors")
        for b in self.all_behaviors:
            self.spatial_behavior_combo.addItem(b)
        self.spatial_behavior_combo.blockSignals(False)

        self.spatial_video_combo.blockSignals(True)
        self.spatial_video_combo.clear()
        self.spatial_video_combo.addItem("All")
        if isinstance(results_dict, dict):
            for vp in results_dict:
                self.spatial_video_combo.addItem(os.path.basename(vp))
        self.spatial_video_combo.blockSignals(False)

        self.spatial_show_btn.setEnabled(has_loc)
        self.manage_regions_btn.setEnabled(has_loc)
        if has_loc:
            self.spatial_info_label.setText("Localization data available.")
        else:
            self.spatial_info_label.setText("No localization data found in results.")

    def _extract_spatial_data(self):
        """Extract per-clip centroids grouped by behavior from loaded results.

        Returns dict:
            {
                video_basename: {
                    "all_centroids": [(cx, cy), ...],
                    "behavior_centroids": {behavior_name: [(cx, cy), ...], ...}
                }
            }
        """
        data_container = self.results
        results_dict = data_container.get("results", data_container)
        classes = data_container.get("classes", [])
        params = data_container.get("parameters", {})

        spatial_data = {}

        for video_path, v_data in (results_dict.items() if isinstance(results_dict, dict) else []):
            loc_bboxes = v_data.get("localization_bboxes", [])
            if not loc_bboxes:
                continue

            preds = v_data.get("predictions", [])
            corrections = v_data.get("corrected_labels", {})
            video_name = os.path.basename(video_path)

            all_centroids = []
            behavior_centroids = {}

            for clip_idx, raw in enumerate(loc_bboxes):
                # Compute centroid from bbox
                cx, cy = self._bbox_centroid(raw)
                if cx is None:
                    continue

                all_centroids.append((cx, cy))

                # Determine behavior label for this clip
                if clip_idx < len(preds):
                    pred_idx = preds[clip_idx]
                    # Apply correction if exists
                    if str(clip_idx) in corrections:
                        pred_idx = corrections[str(clip_idx)]
                    elif clip_idx in corrections:
                        pred_idx = corrections[clip_idx]

                    if classes and pred_idx < len(classes):
                        label = classes[pred_idx]
                    else:
                        label = f"Class {pred_idx}"

                    behavior_centroids.setdefault(label, []).append((cx, cy))

            if all_centroids:
                spatial_data[video_name] = {
                    "all_centroids": all_centroids,
                    "behavior_centroids": behavior_centroids,
                }

        return spatial_data

    @staticmethod
    def _bbox_centroid(raw):
        """Return (cx, cy) from a localization bbox entry, or (None, None)."""
        try:
            if not isinstance(raw, (list, tuple)) or len(raw) == 0:
                return None, None
            # Single bbox [x1, y1, x2, y2]
            if len(raw) == 4 and all(not isinstance(v, (list, tuple)) for v in raw):
                x1, y1, x2, y2 = [float(v) for v in raw]
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # Per-frame bboxes [[x1,y1,x2,y2], ...] — use middle frame
            if isinstance(raw[0], (list, tuple)):
                mid = len(raw) // 2
                box = raw[mid]
                x1, y1, x2, y2 = [float(v) for v in box]
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        except Exception as e:
            logger.debug("Could not parse localization bbox center: %s", e)
        return None, None

    def _update_spatial_plot(self):
        """Build and display the spatial distribution plot."""
        if not HAS_PLOTLY:
            return

        spatial_data = self._extract_spatial_data()
        if not spatial_data:
            self.spatial_info_label.setText("No localization centroids could be extracted.")
            return

        selected_behavior = self.spatial_behavior_combo.currentText()
        selected_video = self.spatial_video_combo.currentText()
        theme = self.color_theme_combo.currentText()
        if theme == "none":
            theme = None

        fig = go.Figure()

        # Collect centroids based on video filter
        all_cx, all_cy = [], []
        beh_centroids = {}  # {behavior: [(cx, cy), ...]}

        for video_name, vdata in spatial_data.items():
            if selected_video != "All" and video_name != selected_video:
                continue
            for cx, cy in vdata["all_centroids"]:
                all_cx.append(cx)
                all_cy.append(cy)
            for beh, pts in vdata["behavior_centroids"].items():
                beh_centroids.setdefault(beh, []).extend(pts)

        if not all_cx:
            self.spatial_info_label.setText("No centroids for the selected filter.")
            return

        dot_size = self.spatial_dot_size_slider.value()
        base_size = max(1, dot_size - 3)

        # Base layer: all centroids as light grey
        fig.add_trace(go.Scatter(
            x=all_cx,
            y=all_cy,
            mode='markers',
            marker=dict(color='lightgrey', size=base_size, opacity=0.4),
            name='All clips',
            showlegend=True,
            legendgroup='all',
            hoverinfo='skip',
        ))

        # Use same palette as Bar Plot (General Overview) so behavior colors match
        behaviors = sorted(beh_centroids.keys())
        color_palette = px.colors.qualitative.Plotly

        if selected_behavior == "All behaviors":
            for i, beh in enumerate(behaviors):
                pts = beh_centroids[beh]
                bx = [p[0] for p in pts]
                by = [p[1] for p in pts]
                color = color_palette[i % len(color_palette)]
                fig.add_trace(go.Scatter(
                    x=bx, y=by,
                    mode='markers',
                    marker=dict(color=color, size=dot_size, opacity=0.8),
                    name=beh,
                    showlegend=True,
                    legendgroup=beh,
                    hovertext=[beh] * len(bx),
                    hoverinfo='text',
                ))
            title = "Spatial Distribution: All Behaviors"
        else:
            pts = beh_centroids.get(selected_behavior, [])
            bx = [p[0] for p in pts]
            by = [p[1] for p in pts]
            color = color_palette[behaviors.index(selected_behavior) % len(color_palette)] if selected_behavior in behaviors else color_palette[0]
            fig.add_trace(go.Scatter(
                x=bx, y=by,
                mode='markers',
                marker=dict(color=color, size=dot_size, opacity=0.8),
                name=selected_behavior,
                showlegend=True,
                legendgroup='behavior',
                hovertext=[selected_behavior] * len(bx),
                hoverinfo='text',
            ))
            title = f"Spatial Distribution: {selected_behavior} ({len(pts)} clips)"

        # Overlay defined spatial regions as semi-transparent fills
        region_colors = px.colors.qualitative.Set2
        for ri, region in enumerate(self.spatial_regions):
            verts = region["vertices"]
            rc = region_colors[ri % len(region_colors)]
            if region["type"] == "rect" and len(verts) == 2:
                rx = [verts[0][0], verts[1][0], verts[1][0], verts[0][0], verts[0][0]]
                ry = [verts[0][1], verts[0][1], verts[1][1], verts[1][1], verts[0][1]]
            else:
                rx = [v[0] for v in verts] + [verts[0][0]]
                ry = [v[1] for v in verts] + [verts[0][1]]
            fig.add_trace(go.Scatter(
                x=rx, y=ry, fill="toself",
                fillcolor=rc, opacity=0.15,
                line=dict(color=rc, width=2),
                name=region["name"],
                showlegend=True,
                legendgroup=f"region_{ri}",
                hoverinfo="name",
            ))

        filter_parts = []
        if selected_video != "All":
            filter_parts.append(f"Video: {selected_video}")
        filter_str = f" | {', '.join(filter_parts)}" if filter_parts else ""

        fig.update_layout(
            title=f'{title}{filter_str}',
            xaxis_title='X (normalized)',
            yaxis_title='Y (normalized)',
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[0, 1]),
            yaxis=dict(autorange='reversed', range=[0, 1]),
            height=600,
            template=theme,
            hovermode='closest',
        )

        self.last_spatial_fig = fig

        # Render to same webview
        import tempfile
        temp_dir = os.path.join(self.config.get("data_dir", "."), "temp_plots")
        os.makedirs(temp_dir, exist_ok=True)
        plot_path = os.path.join(temp_dir, "spatial_distribution.html")

        with open(plot_path, 'w', encoding="utf-8") as f:
            f.write(fig.to_html(include_plotlyjs=True))

        if HAS_WEBENGINE:
            self.webview.setUrl(QUrl.fromLocalFile(os.path.abspath(plot_path)))

        self.spatial_info_label.setText(f"Showing {len(all_cx)} total centroids.")

    def _save_spatial_plot(self):
        """Save the current spatial distribution plot as PDF or SVG."""
        if not HAS_PLOTLY or not getattr(self, "last_spatial_fig", None):
            QMessageBox.warning(self, "Save spatial plot", "No spatial distribution plot to save. Show the plot first.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save spatial plot",
            self.config.get("experiment_path", ""),
            "PDF Files (*.pdf);;SVG Files (*.svg);;PNG Files (*.png);;HTML Files (*.html)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".html"):
                self.last_spatial_fig.write_html(path)
            else:
                self.last_spatial_fig.write_image(path)
            QMessageBox.information(self, "Success", f"Spatial plot saved to {path}")
        except Exception as e:
            if "kaleido" in str(e).lower() or "executable" in str(e).lower():
                QMessageBox.warning(self, "Error", "Saving as PDF/SVG requires 'kaleido'. Install with: pip install kaleido")
            else:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _filter_behaviors(self):
        """Open dialog to filter behaviors."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select behaviors")
        dialog.resize(300, 400)
        layout = QVBoxLayout()
        
        # Buttons
        btn_layout = QHBoxLayout()
        all_btn = QPushButton("Select all")
        none_btn = QPushButton("Deselect all")
        btn_layout.addWidget(all_btn)
        btn_layout.addWidget(none_btn)
        layout.addLayout(btn_layout)
        
        # Scrollable Checkbox List
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        vbox = QVBoxLayout()
        
        checkboxes = []
        for beh in self.all_behaviors:
            cb = QCheckBox(beh)
            if beh in self.selected_behaviors:
                cb.setChecked(True)
            checkboxes.append((beh, cb))
            vbox.addWidget(cb)
        
        vbox.addStretch()
        widget.setLayout(vbox)
        scroll.setWidget(widget)
        layout.addWidget(scroll)
        
        # Connect All/None
        def select_all():
            for _, cb in checkboxes: cb.setChecked(True)
        def select_none():
            for _, cb in checkboxes: cb.setChecked(False)
            
        all_btn.clicked.connect(select_all)
        none_btn.clicked.connect(select_none)
        
        # OK Button
        ok_btn = QPushButton("Update plots")
        ok_btn.clicked.connect(dialog.accept)
        layout.addWidget(ok_btn)
        
        dialog.setLayout(layout)
        
        if dialog.exec():
            self.selected_behaviors = {beh for beh, cb in checkboxes if cb.isChecked()}
            self._update_plots()

    def _manage_groups(self):
        # Get video paths from results
        if "results" in self.results:
            video_paths = list(self.results["results"].keys())
        else:
            video_paths = list(self.results.keys())
            
        dialog = GroupManagementDialog(video_paths, self.groups, self.video_groups, self)
        if dialog.exec():
            self.groups = dialog.groups
            self.video_groups = dialog.video_groups
            self._save_groups_to_config()
            self._save_metadata()
            
            # Re-process to update groups in merged_data
            self._process_data()
            self._update_plots()

    def _save_metadata(self):
        exp_path = self.config.get("experiment_path")
        if exp_path:
            meta_path = os.path.join(exp_path, "results", "analysis_metadata.json")
            try:
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                serializable_regions = []
                for r in self.spatial_regions:
                    serializable_regions.append({
                        "name": r["name"],
                        "type": r["type"],
                        "vertices": [list(v) for v in r["vertices"]],
                    })
                with open(meta_path, 'w') as f:
                    json.dump({
                        "groups": self.groups,
                        "video_groups": self.video_groups,
                        "spatial_regions": serializable_regions,
                    }, f, indent=2)
            except Exception as e:
                logger.error("Error saving metadata: %s", e)

    def _on_region_filter_changed(self):
        if self.merged_data:
            self._update_plots()

    def _update_region_filter_combo(self):
        self.region_filter_combo.blockSignals(True)
        prev = self.region_filter_combo.currentText()
        self.region_filter_combo.clear()
        self.region_filter_combo.addItem("All Regions")
        for r in self.spatial_regions:
            self.region_filter_combo.addItem(r["name"])
        self.region_filter_combo.addItem("Outside")
        idx = self.region_filter_combo.findText(prev)
        if idx >= 0:
            self.region_filter_combo.setCurrentIndex(idx)
        self.region_filter_combo.blockSignals(False)

    def _manage_spatial_regions(self):
        spatial_data = self._extract_spatial_data()
        all_centroids = []
        for vdata in spatial_data.values():
            all_centroids.extend(vdata["all_centroids"])
        dialog = SpatialRegionEditor(all_centroids, self.spatial_regions, self)
        if dialog.exec():
            self.spatial_regions = dialog.regions
            self._save_metadata()
            self._update_region_filter_combo()
            self._process_data()
            self._update_plots()

    def _update_plots(self):
        if not HAS_PLOTLY or not self.merged_data:
            if not HAS_PLOTLY:
                QMessageBox.warning(self, "Error", "Plotly is not installed.")
            return
            
        df = pd.DataFrame(self.merged_data)
        
        # Filter behaviors
        if self.selected_behaviors:
            df = df[df["Behavior"].isin(self.selected_behaviors)]
        
        # Filter groups
        if self.visible_groups:
            df = df[df["Group"].isin(self.visible_groups)]

        # Filter by spatial region
        selected_region = self.region_filter_combo.currentText()
        if selected_region != "All Regions" and "Region" in df.columns:
            df = df[df["Region"] == selected_region]
            
        if df.empty:
            self.last_fig = None
            return

        metric = self.metric_combo.currentText()
        analysis_mode = self.plot_mode_combo.currentText()
        graph_type = self.graph_type_combo.currentText()
        theme = self.color_theme_combo.currentText()
        
        fig = None
        
        # Aggregation
        if "Occurrences" in metric:
            agg = df.groupby(["Video", "Group", "Behavior"]).size().reset_index(name="Value")
            y_label = "Count"
        elif "Average" in metric:
            agg = df.groupby(["Video", "Group", "Behavior"])["Duration"].mean().reset_index(name="Value")
            y_label = "Avg Duration (s)"
        elif "Total" in metric:
            agg = df.groupby(["Video", "Group", "Behavior"])["Duration"].sum().reset_index(name="Value")
            y_label = "Total Duration (s)"
        elif "Percent" in metric:
            # Calculate total time per video
            total_times = df.groupby("Video")["Duration"].sum().to_dict()
            agg = df.groupby(["Video", "Group", "Behavior"])["Duration"].sum().reset_index(name="Value")
            agg["Value"] = agg.apply(lambda x: (x["Value"] / total_times.get(x["Video"], 1)) * 100, axis=1)
            y_label = "Percent Time (%)"
            
        # Auto-determine graph type if needed
        if graph_type == "Auto":
            if analysis_mode == "General Overview":
                if "Average" in metric: graph_type = "Box Plot"
                else: graph_type = "Bar Chart"
            else: # Group Comparison
                graph_type = "Box Plot"
        
        # Common Plot Settings
        template = theme if theme != "none" else None
        title = f"{analysis_mode}: {metric}"
        labels = {"Value": y_label, "Duration": "Bout Duration (s)", "Video": "Video", "Group": "Group"}
        plot_args = {"template": template, "title": title, "labels": labels}

        if analysis_mode == "General Overview":
            # X=Video, Color=Behavior
            
            # Decide data source: Bout-level (df) or Video-level (agg)
            if "Average" in metric and graph_type in ["Box Plot", "Violin Plot", "Strip Plot"]:
                data = df
                y_col = "Duration"
            else:
                data = agg
                y_col = "Value"
                
            if graph_type == "Bar Chart":
                fig = px.bar(data, x="Video", y=y_col, color="Behavior", **plot_args)
            elif graph_type == "Box Plot":
                fig = px.box(data, x="Video", y=y_col, color="Behavior", points="all", **plot_args)
            elif graph_type == "Violin Plot":
                fig = px.violin(data, x="Video", y=y_col, color="Behavior", points="all", box=True, **plot_args)
            elif graph_type == "Strip Plot":
                fig = px.strip(data, x="Video", y=y_col, color="Behavior", **plot_args)
            elif graph_type == "Line Plot":
                fig = px.line(agg, x="Video", y="Value", color="Behavior", markers=True, **plot_args)
            
        elif analysis_mode == "Group Comparison":
            # X=Behavior, Color=Group, Data=Video Aggregates (agg)
            data = agg
            
            if graph_type == "Bar Chart":
                # Create bar chart with mean, SEM error bars, and individual points
                fig = go.Figure()
                
                # Get unique behaviors and groups
                behaviors = sorted(data["Behavior"].unique())
                groups = sorted(data["Group"].unique())
                num_groups = len(groups)
                
                # Color palette
                colors = px.colors.qualitative.Plotly
                
                # Layout settings
                group_width = 0.8
                bar_width = group_width / num_groups
                
                for i, group in enumerate(groups):
                    group_data = data[data["Group"] == group]
                    
                    means = []
                    sems = []
                    x_positions_bar = []
                    x_positions_points = []
                    y_points = []
                    
                    # Calculate offset for this group
                    # We map behaviors to integers 0..N-1
                    # Center of bar i is at: index - 0.4 + (i + 0.5) * bar_width
                    offset = -0.4 + bar_width * (i + 0.5)
                    
                    for j, behavior in enumerate(behaviors):
                        beh_data = group_data[group_data["Behavior"] == behavior]["Value"]
                        
                        # Stats
                        if len(beh_data) > 0:
                            means.append(beh_data.mean())
                            sems.append(beh_data.sem() if len(beh_data) > 1 else 0)
                        else:
                            means.append(0)
                            sems.append(0)
                        
                        # Bar Position
                        x_positions_bar.append(j + offset)
                        
                        # Points Position
                        if len(beh_data) > 0:
                            x_positions_points.extend([j + offset] * len(beh_data))
                            y_points.extend(beh_data.values)
                    
                    # Add bar trace with error bars
                    fig.add_trace(go.Bar(
                        name=group,
                        x=x_positions_bar,
                        y=means,
                        error_y=dict(type='data', array=sems, visible=True),
                        marker_color=colors[i % len(colors)],
                        width=bar_width,
                        showlegend=True
                    ))
                    
                    # Add individual points as scatter (black)
                    if x_positions_points:
                        fig.add_trace(go.Scatter(
                            x=x_positions_points,
                            y=y_points,
                            mode='markers',
                            marker=dict(
                                color='black',
                                size=5,
                                opacity=0.7,
                            ),
                            showlegend=False,
                            hovertemplate=f'{group}<br>%{{y:.2f}}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Behavior",
                    yaxis_title=y_label,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(behaviors))),
                        ticktext=behaviors
                    ),
                    template=template,
                    hovermode='closest'
                )
                
            elif graph_type == "Box Plot":
                fig = px.box(data, x="Behavior", y="Value", color="Group", points="all", **plot_args)
            elif graph_type == "Violin Plot":
                fig = px.violin(data, x="Behavior", y="Value", color="Group", points="all", box=True, **plot_args)
            elif graph_type == "Strip Plot":
                fig = px.strip(data, x="Behavior", y="Value", color="Group", **plot_args)
            elif graph_type == "Line Plot":
                fig = px.line(data, x="Behavior", y="Value", color="Group", markers=True, **plot_args)
            
        self.last_fig = fig
        
        if fig:
            # Save to temp html
            import tempfile
            import shutil
            
            # Use a fixed temp file in app dir to avoid permission issues sometimes
            temp_dir = os.path.join(self.config.get("data_dir", "."), "temp_plots")
            os.makedirs(temp_dir, exist_ok=True)
            plot_path = os.path.join(temp_dir, "current_plot.html")
            
            with open(plot_path, 'w', encoding="utf-8") as f:
                # include_plotlyjs=True embeds the ~3MB library directly in the HTML
                # preventing 'Plotly is not defined' errors if CDN fails
                f.write(fig.to_html(include_plotlyjs=True))
            
            if HAS_WEBENGINE:
                self.webview.setUrl(QUrl.fromLocalFile(os.path.abspath(plot_path)))
    
    def _on_transition_type_changed(self):
        """Update transition select combo when type changes."""
        self.transition_select_combo.clear()
        
        if not self.merged_data:
            return
        
        analysis_type = self.transition_type_combo.currentText()
        
        if analysis_type == "Individual Video":
            # Populate with video names
            df = pd.DataFrame(self.merged_data)
            videos = sorted(df["Video"].unique().tolist())
            self.transition_select_combo.addItems(videos)
        else:  # Group Comparison
            # Populate with group names
            groups = sorted([g for g in self.groups.keys() if g])
            if not groups:
                groups = ["Unassigned"]
            self.transition_select_combo.addItems(groups)
        
        self.compute_transition_btn.setEnabled(len(self.transition_select_combo) > 0)
    
    def _compute_transitions(self):
        """Compute transition matrix and plot transition graph."""
        if not HAS_PLOTLY:
            QMessageBox.warning(self, "Error", "Plotly is required for transition analysis.")
            return
        
        analysis_type = self.transition_type_combo.currentText()
        selection = self.transition_select_combo.currentText()
        
        if not selection:
            return
        
        df = pd.DataFrame(self.merged_data)
        
        # Filter data based on selection
        if analysis_type == "Individual Video":
            df_filtered = df[df["Video"] == selection]
            title_suffix = f"Video: {selection}"
        else:  # Group Comparison
            df_filtered = df[df["Group"] == selection]
            title_suffix = f"Group: {selection}"
        
        if df_filtered.empty:
            QMessageBox.warning(self, "No Data", f"No data found for {selection}")
            return
        
        # Build sequence of behaviors for this selection
        # Group by video, sort by implicit order (row order in merged_data represents time)
        sequences = []
        
        if analysis_type == "Individual Video":
            # Single video - one sequence
            sequence = df_filtered["Behavior"].tolist()
            sequences.append(sequence)
        else:
            # Group - multiple videos, analyze separately then aggregate
            for video in df_filtered["Video"].unique():
                video_df = df_filtered[df_filtered["Video"] == video]
                sequence = video_df["Behavior"].tolist()
                sequences.append(sequence)
        
        # Compute transition matrix
        behaviors = sorted(self.all_behaviors) if self.all_behaviors else sorted(df["Behavior"].unique().tolist())
        transition_counts = {b: {b2: 0 for b2 in behaviors} for b in behaviors}
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                from_beh = sequence[i]
                to_beh = sequence[i + 1]
                if from_beh in transition_counts and to_beh in transition_counts[from_beh]:
                    transition_counts[from_beh][to_beh] += 1
        
        # Normalize to probabilities and calculate residuals
        total_transitions = sum(sum(row.values()) for row in transition_counts.values())
        
        # Row and Column totals for Expected values
        row_totals = {b: sum(transition_counts[b].values()) for b in behaviors}
        col_totals = {b: sum(transition_counts[row][b] for row in behaviors) for b in behaviors}
        
        transition_matrix = {}
        residuals_matrix = {}
        
        for from_beh in behaviors:
            transition_matrix[from_beh] = {}
            residuals_matrix[from_beh] = {}
            row_sum = row_totals[from_beh]
            
            for to_beh in behaviors:
                # Probability
                if row_sum > 0:
                    transition_matrix[from_beh][to_beh] = transition_counts[from_beh][to_beh] / row_sum
                else:
                    transition_matrix[from_beh][to_beh] = 0.0
                
                # Residual (Observed - Expected) / sqrt(Expected)
                # Expected = (RowTotal * ColTotal) / GrandTotal
                if total_transitions > 0:
                    expected = (row_totals[from_beh] * col_totals[to_beh]) / total_transitions
                else:
                    expected = 0
                
                observed = transition_counts[from_beh][to_beh]
                
                if expected > 0:
                    # Standardized Residual
                    z_score = (observed - expected) / np.sqrt(expected)
                else:
                    z_score = 0.0
                
                residuals_matrix[from_beh][to_beh] = z_score
        
        # Display matrix in table
        self._display_transition_matrix(behaviors, transition_matrix)
        
        # Plot transition graph
        self._plot_transition_graph(behaviors, transition_matrix, residuals_matrix, title_suffix)
    
    def _display_transition_matrix(self, behaviors, transition_matrix):
        """Display transition matrix in QTableWidget."""
        self.transition_matrix_table.clear()
        self.transition_matrix_table.setRowCount(len(behaviors))
        self.transition_matrix_table.setColumnCount(len(behaviors) + 1)
        
        # Headers
        self.transition_matrix_table.setHorizontalHeaderLabels(["From \\ To"] + behaviors)
        
        for i, from_beh in enumerate(behaviors):
            # Row label
            self.transition_matrix_table.setItem(i, 0, QTableWidgetItem(from_beh))
            
            # Probabilities
            for j, to_beh in enumerate(behaviors):
                prob = transition_matrix[from_beh][to_beh]
                item = QTableWidgetItem(f"{prob:.3f}")
                self.transition_matrix_table.setItem(i, j + 1, item)
        
        self.transition_matrix_table.resizeColumnsToContents()
    
    def _plot_transition_graph(self, behaviors, transition_matrix, residuals_matrix, title_suffix):
        """Plot transition graph using Plotly network graph with circular layout."""
        if not HAS_PLOTLY:
            return
        
        use_sig_filter = self.sig_filter_check.isChecked()
        layout_mode = self.layout_combo.currentText()
        
        # Build edges
        threshold = 0.05
        edges = []
        edge_weights = []
        edge_texts = []
        
        for from_beh in behaviors:
            for to_beh in behaviors:
                prob = transition_matrix[from_beh][to_beh]
                resid = residuals_matrix[from_beh][to_beh]
                
                # Filtering condition
                include = False
                if use_sig_filter:
                    # Significant positive deviation (Z > 1.96 corresponds to p < 0.05)
                    if resid > 1.96: 
                        include = True
                else:
                    # Probability threshold
                    if prob > threshold:
                        include = True
                
                if include:
                    edges.append((from_beh, to_beh))
                    edge_weights.append(prob) # Always visualize probability width
                    
                    # Tooltip text
                    txt = f"{from_beh} → {to_beh}<br>Prob: {prob:.2%}"
                    if resid != 0:
                        txt += f"<br>Z-score: {resid:.2f}"
                    edge_texts.append(txt)
        
        if not edges:
            msg = "No significant transitions found (Z > 1.96)" if use_sig_filter else "No significant transitions found (prob > 0.05)"
            QMessageBox.information(self, "No Transitions", msg)
            return
        
        # Calculate Layout
        import math
        n = len(behaviors)
        node_positions = {}
        
        if layout_mode == "Network Layout":
            try:
                import networkx as nx
                # Create graph for layout calculation
                G = nx.DiGraph()
                G.add_nodes_from(behaviors)
                # Add edges with weights (inverse of probability for 'distance')
                for (u, v), w in zip(edges, edge_weights):
                    if w > 0:
                        G.add_edge(u, v, weight=w)
                
                # Spring layout
                pos = nx.spring_layout(G, k=2.0/math.sqrt(n) if n > 0 else 1, seed=42, iterations=50)
                # Scale to fit roughly in range
                for node, p in pos.items():
                    node_positions[node] = (p[0] * 2, p[1] * 2)
            except ImportError:
                logger.warning("NetworkX not installed, falling back to Circular Layout")
                layout_mode = "Circular Layout" # Fallback
        
        if layout_mode == "Circular Layout":
            radius = 1.5
            for i, beh in enumerate(behaviors):
                angle = 2 * math.pi * i / n - math.pi / 2
                node_positions[beh] = (radius * math.cos(angle), radius * math.sin(angle))
        
        # Generate distinct colors for nodes
        import plotly.colors as pc
        if n <= 10:
            node_colors = pc.qualitative.Set3[:n]
        else:
            node_colors = pc.sample_colorscale("hsv", [i/n for i in range(n)])
        
        behavior_to_color = {beh: node_colors[i] for i, beh in enumerate(behaviors)}
        
        # Build Plotly figure with curved edges
        edge_traces = []
        for (from_beh, to_beh), weight, txt in zip(edges, edge_weights, edge_texts):
            x0, y0 = node_positions[from_beh]
            x1, y1 = node_positions[to_beh]
            
            # Curve control points
            if layout_mode == "Circular Layout":
                # Control point slightly outside circle
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                dist = math.sqrt(mid_x**2 + mid_y**2)
                if dist > 0:
                    offset = 0.2
                    ctrl_x = mid_x + offset * mid_x / dist
                    ctrl_y = mid_y + offset * mid_y / dist
                else:
                    ctrl_x, ctrl_y = mid_x, mid_y
            else:
                # Network layout: simple quadratic curve to avoid overlap
                # Perpendicular offset
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                perp_x, perp_y = -dy, dx
                norm = math.sqrt(perp_x**2 + perp_y**2)
                if norm > 0:
                    offset = 0.2  # Fixed offset amount
                    ctrl_x = mid_x + offset * perp_x / norm
                    ctrl_y = mid_y + offset * perp_y / norm
                else:
                    ctrl_x, ctrl_y = mid_x + 0.2, mid_y + 0.2 # Loop
            
            # Approximate curve
            curve_x = [x0, ctrl_x, x1, None]
            curve_y = [y0, ctrl_y, y1, None]
            
            # Style
            edge_color = behavior_to_color[from_beh]
            opacity = 0.3 + 0.5 * weight
            
            edge_trace = go.Scatter(
                x=curve_x,
                y=curve_y,
                mode='lines',
                line=dict(
                    width=max(1, weight * 10),
                    color=edge_color.replace('rgb', 'rgba').replace(')', f',{opacity})'),
                    shape='spline'
                ),
                hoverinfo='text',
                text=txt,
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Node trace
        node_x = [node_positions[beh][0] for beh in behaviors]
        node_y = [node_positions[beh][1] for beh in behaviors]
        node_colors_list = [behavior_to_color[beh] for beh in behaviors]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=behaviors,
            textposition='middle center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            marker=dict(
                size=50, 
                color=node_colors_list,
                line=dict(width=3, color='white'),
                opacity=0.9
            ),
            hoverinfo='text',
            hovertext=behaviors,
            showlegend=False
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=dict(
                text=f"Behavior Transition Graph - {title_suffix}" + (" (Significant Only)" if use_sig_filter else "") + "<br><span style='font-size:12px;color:grey'>Edge color matches SOURCE behavior (From -> To)</span>",
                font=dict(size=18)
            ),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
            plot_bgcolor='white',
            height=700,
            width=700
        )
        
        # Save and display
        temp_dir = os.path.join(self.config.get("data_dir", "."), "temp_plots")
        os.makedirs(temp_dir, exist_ok=True)
        plot_path = os.path.join(temp_dir, "transition_graph.html")
        
        with open(plot_path, 'w', encoding="utf-8") as f:
            f.write(fig.to_html(include_plotlyjs=True))
        
        if HAS_WEBENGINE:
            self.transition_webview.setUrl(QUrl.fromLocalFile(os.path.abspath(plot_path)))

class SpatialCanvas(QWidget):
    """Interactive canvas for drawing spatial regions over centroid scatter."""

    MARGIN = 30
    REGION_COLORS = [
        QColor(102, 194, 165, 60), QColor(252, 141, 98, 60),
        QColor(141, 160, 203, 60), QColor(231, 138, 195, 60),
        QColor(166, 216, 84, 60), QColor(255, 217, 47, 60),
    ]
    REGION_BORDER_COLORS = [
        QColor(102, 194, 165), QColor(252, 141, 98),
        QColor(141, 160, 203), QColor(231, 138, 195),
        QColor(166, 216, 84), QColor(255, 217, 47),
    ]

    def __init__(self, centroids, regions, parent=None):
        super().__init__(parent)
        self.centroids = centroids  # [(cx, cy), ...]
        self.regions = regions      # list of region dicts (mutable reference)
        self.setMinimumSize(500, 500)
        self.setMouseTracking(True)

        self.draw_mode = None       # "polygon" or "rect"
        self._poly_points = []      # in-progress polygon vertices (normalized)
        self._rect_start = None     # in-progress rect start (normalized)
        self._rect_end = None
        self._mouse_norm = None     # current mouse in normalized coords
        self._pending_name = None   # name for the region being drawn

    def _norm_to_pixel(self, nx, ny):
        m = self.MARGIN
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        return int(m + nx * w), int(m + ny * h)

    def _pixel_to_norm(self, px, py):
        m = self.MARGIN
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        nx = max(0.0, min(1.0, (px - m) / max(1, w)))
        ny = max(0.0, min(1.0, (py - m) / max(1, h)))
        return nx, ny

    def start_drawing(self, mode, name):
        self.draw_mode = mode
        self._pending_name = name
        self._poly_points = []
        self._rect_start = None
        self._rect_end = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_drawing(self):
        self.draw_mode = None
        self._poly_points = []
        self._rect_start = None
        self._rect_end = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mousePressEvent(self, event):
        if not self.draw_mode:
            return
        pos = event.position()
        nx, ny = self._pixel_to_norm(int(pos.x()), int(pos.y()))
        if self.draw_mode == "polygon":
            self._poly_points.append((nx, ny))
            self.update()
        elif self.draw_mode == "rect":
            self._rect_start = (nx, ny)
            self._rect_end = (nx, ny)
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        self._mouse_norm = self._pixel_to_norm(int(pos.x()), int(pos.y()))
        if self.draw_mode == "rect" and self._rect_start:
            self._rect_end = self._mouse_norm
        self.update()

    def mouseReleaseEvent(self, event):
        if self.draw_mode == "rect" and self._rect_start and self._rect_end:
            x0, y0 = self._rect_start
            x1, y1 = self._rect_end
            if abs(x1 - x0) > 0.005 and abs(y1 - y0) > 0.005:
                self.regions.append({
                    "name": self._pending_name or f"Region {len(self.regions)+1}",
                    "type": "rect",
                    "vertices": [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))],
                })
                self.draw_mode = None
                self._rect_start = None
                self._rect_end = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                # Signal parent to refresh list
                parent = self.parent()
                while parent and not isinstance(parent, SpatialRegionEditor):
                    parent = parent.parent()
                if parent:
                    parent._refresh_region_list()
            self.update()

    def mouseDoubleClickEvent(self, event):
        if self.draw_mode == "polygon" and len(self._poly_points) >= 3:
            self.regions.append({
                "name": self._pending_name or f"Region {len(self.regions)+1}",
                "type": "polygon",
                "vertices": list(self._poly_points),
            })
            self._poly_points = []
            self.draw_mode = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            parent = self.parent()
            while parent and not isinstance(parent, SpatialRegionEditor):
                parent = parent.parent()
            if parent:
                parent._refresh_region_list()
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.MARGIN
        draw_w = self.width() - 2 * m
        draw_h = self.height() - 2 * m

        # Background
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(m, m, draw_w, draw_h)

        # Centroids
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(180, 180, 180, 120)))
        for cx, cy in self.centroids:
            px, py = self._norm_to_pixel(cx, cy)
            p.drawEllipse(QPointF(px, py), 3, 3)

        # Saved regions
        for ri, region in enumerate(self.regions):
            col = self.REGION_COLORS[ri % len(self.REGION_COLORS)]
            border = self.REGION_BORDER_COLORS[ri % len(self.REGION_BORDER_COLORS)]
            p.setBrush(QBrush(col))
            p.setPen(QPen(border, 2))
            verts = region["vertices"]
            if region["type"] == "rect" and len(verts) == 2:
                px0, py0 = self._norm_to_pixel(*verts[0])
                px1, py1 = self._norm_to_pixel(*verts[1])
                p.drawRect(QRectF(QPointF(px0, py0), QPointF(px1, py1)))
            else:
                poly = QPolygonF([QPointF(*self._norm_to_pixel(*v)) for v in verts])
                p.drawPolygon(poly)
            # Label
            if verts:
                cx_avg = sum(v[0] for v in verts) / len(verts)
                cy_avg = sum(v[1] for v in verts) / len(verts)
                lx, ly = self._norm_to_pixel(cx_avg, cy_avg)
                p.setPen(QPen(border.darker(130), 1))
                p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                p.drawText(lx - 30, ly - 5, 60, 20, Qt.AlignmentFlag.AlignCenter, region["name"])

        # In-progress polygon
        if self.draw_mode == "polygon" and self._poly_points:
            p.setPen(QPen(QColor(255, 80, 80), 2, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(QColor(255, 80, 80, 40)))
            pts = [QPointF(*self._norm_to_pixel(*v)) for v in self._poly_points]
            if self._mouse_norm:
                pts.append(QPointF(*self._norm_to_pixel(*self._mouse_norm)))
            if len(pts) >= 3:
                p.drawPolygon(QPolygonF(pts))
            elif len(pts) == 2:
                p.drawLine(pts[0], pts[1])
            for pt in pts[:-1]:
                p.setBrush(QBrush(QColor(255, 80, 80)))
                p.drawEllipse(pt, 4, 4)
                p.setBrush(QBrush(QColor(255, 80, 80, 40)))

        # In-progress rectangle
        if self.draw_mode == "rect" and self._rect_start and self._rect_end:
            p.setPen(QPen(QColor(80, 80, 255), 2, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(QColor(80, 80, 255, 40)))
            px0, py0 = self._norm_to_pixel(*self._rect_start)
            px1, py1 = self._norm_to_pixel(*self._rect_end)
            p.drawRect(QRectF(QPointF(px0, py0), QPointF(px1, py1)))

        # Axis labels
        p.setPen(QColor(100, 100, 100))
        p.setFont(QFont("Arial", 8))
        p.drawText(m, m - 5, "0,0")
        p.drawText(m + draw_w - 20, m - 5, "1,0")
        p.drawText(m, m + draw_h + 14, "0,1")

        p.end()


class SpatialRegionEditor(QDialog):
    """Dialog for drawing and managing named spatial regions."""

    def __init__(self, centroids, existing_regions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Region Editor")
        self.resize(850, 600)
        # Deep copy to allow cancel
        self.regions = [
            {"name": r["name"], "type": r["type"], "vertices": list(r["vertices"])}
            for r in existing_regions
        ]
        self.centroids = centroids
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Toolbar
        toolbar = QHBoxLayout()
        self.polygon_btn = QToolButton()
        self.polygon_btn.setText("Draw Polygon")
        self.polygon_btn.setToolTip("Click vertices, double-click to finish")
        self.polygon_btn.clicked.connect(self._start_polygon)

        self.rect_btn = QToolButton()
        self.rect_btn.setText("Draw Rectangle")
        self.rect_btn.setToolTip("Click and drag to define rectangle")
        self.rect_btn.clicked.connect(self._start_rect)

        self.cancel_draw_btn = QPushButton("Cancel drawing")
        self.cancel_draw_btn.clicked.connect(self._cancel_draw)
        self.cancel_draw_btn.setEnabled(False)

        toolbar.addWidget(self.polygon_btn)
        toolbar.addWidget(self.rect_btn)
        toolbar.addWidget(self.cancel_draw_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Main splitter: canvas (left) + region list (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.canvas = SpatialCanvas(self.centroids, self.regions)
        splitter.addWidget(self.canvas)

        right = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Defined Regions"))
        self.region_list = QListWidget()
        self.region_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        right_layout.addWidget(self.region_list)

        btn_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._rename_region)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_region)
        btn_row.addWidget(self.rename_btn)
        btn_row.addWidget(self.delete_btn)
        right_layout.addLayout(btn_row)
        right.setLayout(right_layout)
        splitter.addWidget(right)
        splitter.setSizes([600, 250])
        layout.addWidget(splitter, stretch=1)

        # OK/Cancel
        btns = QHBoxLayout()
        ok_btn = QPushButton("Done")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        self.setLayout(layout)
        self._refresh_region_list()

    def _start_polygon(self):
        name, ok = QInputDialog.getText(self, "Region Name", "Name for the new polygon region:")
        if not ok or not name.strip():
            return
        self.cancel_draw_btn.setEnabled(True)
        self.canvas.start_drawing("polygon", name.strip())

    def _start_rect(self):
        name, ok = QInputDialog.getText(self, "Region Name", "Name for the new rectangle region:")
        if not ok or not name.strip():
            return
        self.cancel_draw_btn.setEnabled(True)
        self.canvas.start_drawing("rect", name.strip())

    def _cancel_draw(self):
        self.canvas.cancel_drawing()
        self.cancel_draw_btn.setEnabled(False)

    def _refresh_region_list(self):
        self.cancel_draw_btn.setEnabled(False)
        self.region_list.clear()
        for r in self.regions:
            n_verts = len(r["vertices"])
            self.region_list.addItem(f"{r['name']}  ({r['type']}, {n_verts} pts)")
        self.canvas.update()

    def _rename_region(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            return
        old = self.regions[idx]["name"]
        name, ok = QInputDialog.getText(self, "Rename Region", "New name:", text=old)
        if ok and name.strip():
            self.regions[idx]["name"] = name.strip()
            self._refresh_region_list()

    def _delete_region(self):
        idx = self.region_list.currentRow()
        if idx < 0 or idx >= len(self.regions):
            return
        self.regions.pop(idx)
        self._refresh_region_list()


class GroupManagementDialog(QDialog):
    def __init__(self, video_paths, groups, video_groups, parent=None):
        super().__init__(parent)
        self.video_paths = sorted(list(video_paths))
        self.groups = groups.copy()
        self.video_groups = video_groups.copy()
        self.setWindowTitle("Manage groups")
        self.resize(800, 600)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # Top: Group creation
        group_input_layout = QHBoxLayout()
        self.group_name_edit = QLineEdit()
        self.group_name_edit.setPlaceholderText("New Group Name")
        self.add_group_btn = QPushButton("Add group")
        self.add_group_btn.clicked.connect(self._add_group)
        group_input_layout.addWidget(self.group_name_edit)
        group_input_layout.addWidget(self.add_group_btn)
        layout.addLayout(group_input_layout)
        
        # Main: Splitter with Videos (left) and Groups (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Videos List (Table to show current group)
        video_widget = QWidget()
        video_layout = QVBoxLayout()
        video_layout.addWidget(QLabel("Videos"))
        self.video_table = QTableWidget()
        self.video_table.setColumnCount(2)
        self.video_table.setHorizontalHeaderLabels(["Video", "Assigned Group"])
        self.video_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.video_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.video_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        video_layout.addWidget(self.video_table)
        video_widget.setLayout(video_layout)
        splitter.addWidget(video_widget)
        
        # Groups List
        group_widget = QWidget()
        group_layout = QVBoxLayout()
        group_layout.addWidget(QLabel("Groups (Select to assign)"))
        self.group_list = QTableWidget()
        self.group_list.setColumnCount(1)
        self.group_list.setHorizontalHeaderLabels(["Group Name"])
        self.group_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.group_list.itemSelectionChanged.connect(self._assign_group)
        group_layout.addWidget(self.group_list)
        group_widget.setLayout(group_layout)
        splitter.addWidget(group_widget)
        
        layout.addWidget(splitter)
        
        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)
        
        self.setLayout(layout)
        self._refresh_lists()
        
    def _refresh_lists(self):
        # Videos
        self.video_table.setRowCount(len(self.video_paths))
        for i, path in enumerate(self.video_paths):
            name = os.path.basename(path)
            self.video_table.setItem(i, 0, QTableWidgetItem(name))
            group = self.video_groups.get(path, "Unassigned")
            self.video_table.setItem(i, 1, QTableWidgetItem(group))
            
        # Groups
        self.group_list.setRowCount(len(self.groups))
        for i, group in enumerate(sorted(self.groups.keys())):
            self.group_list.setItem(i, 0, QTableWidgetItem(group))
            
    def _add_group(self):
        name = self.group_name_edit.text().strip()
        if name and name not in self.groups:
            self.groups[name] = []
            self._refresh_lists()
            self.group_name_edit.clear()
            
    def _assign_group(self):
        selected_group_items = self.group_list.selectedItems()
        if not selected_group_items:
            return
        
        group_name = selected_group_items[0].text()
        
        selected_video_rows = set(idx.row() for idx in self.video_table.selectedIndexes())
        
        for row in selected_video_rows:
            # Using index from video_paths since table order matches
            full_path = self.video_paths[row]
            
            # Remove from old group
            old_group = self.video_groups.get(full_path)
            if old_group and old_group in self.groups and full_path in self.groups[old_group]:
                self.groups[old_group].remove(full_path)
                
            # Assign new
            self.video_groups[full_path] = group_name
            if full_path not in self.groups[group_name]:
                self.groups[group_name].append(full_path)
        
        self._refresh_lists()
