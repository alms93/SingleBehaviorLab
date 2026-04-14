"""
Plot integration utilities for PyQt6.
Handles both matplotlib and plotly plots.
"""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import QRect
import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.io as pio
import io
from PIL import Image

logger = logging.getLogger(__name__)


class TimelineWidget(QWidget):
    """Timeline widget showing context (grey) and clip (green) sections"""
    
    def __init__(self, parent=None, clip_metadata: dict = None):
        super().__init__(parent)
        self.clip_metadata = clip_metadata or {}
        self.duration_ms = 0
        self.current_position_ms = 0
        self.context_start_ms = 0
        self.clip_start_ms = 0
        self.clip_end_ms = 0
        self.context_end_ms = 0

        if clip_metadata:
            self._calculate_segments()
    
    def _calculate_segments(self):
        """Calculate timeline segment positions based on clip metadata"""
        if not self.clip_metadata or self.duration_ms == 0:
            return
        
        fps = self.clip_metadata.get('fps', 30)
        start_frame = self.clip_metadata.get('start_frame', 0)
        end_frame = self.clip_metadata.get('end_frame', 0)
        context_frames = self.clip_metadata.get('context_frames', 30)

        # The extracted video is laid out as: context_before + clip + context_after.
        total_frames_in_video = (end_frame - start_frame + 1) + (2 * context_frames)

        if total_frames_in_video == 0:
            return

        context_before_frames = context_frames
        clip_frames = end_frame - start_frame + 1
        context_after_frames = context_frames

        frame_duration_ms = 1000.0 / fps if fps > 0 else 33.33
        
        self.context_start_ms = 0
        self.clip_start_ms = context_before_frames * frame_duration_ms
        self.clip_end_ms = (context_before_frames + clip_frames) * frame_duration_ms
        self.context_end_ms = self.duration_ms
    
    def set_duration(self, duration_ms: int):
        """Set total video duration"""
        self.duration_ms = duration_ms
        if self.clip_metadata:
            self._calculate_segments()
        self.update()
    
    def set_current_position(self, position_ms: int, duration_ms: int):
        """Update current playback position"""
        if duration_ms > 0:
            self.duration_ms = duration_ms
            if self.clip_metadata:
                self._calculate_segments()
        self.current_position_ms = position_ms
        self.update()
    
    def paintEvent(self, event):
        """Draw timeline with context and clip sections"""
        if self.duration_ms == 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw timeline background
        painter.fillRect(0, 0, width, height, QColor(40, 40, 40))
        
        if not self.clip_metadata or self.clip_start_ms == 0:
            # No metadata, just draw a simple timeline
            painter.fillRect(0, 0, width, height, QColor(60, 60, 60))
            # Draw current position indicator
            if self.current_position_ms > 0:
                pos_x = int((self.current_position_ms / self.duration_ms) * width)
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawLine(pos_x, 0, pos_x, height)
            return
        
        # Draw segments
        # Context before (grey)
        context_before_width = int((self.clip_start_ms / self.duration_ms) * width)
        painter.fillRect(0, 0, context_before_width, height, QColor(100, 100, 100))
        
        # Clip section (green)
        clip_start_x = context_before_width
        clip_width = int(((self.clip_end_ms - self.clip_start_ms) / self.duration_ms) * width)
        painter.fillRect(clip_start_x, 0, clip_width, height, QColor(0, 200, 0))
        
        # Context after (grey)
        context_after_start_x = clip_start_x + clip_width
        context_after_width = width - context_after_start_x
        painter.fillRect(context_after_start_x, 0, context_after_width, height, QColor(100, 100, 100))
        
        # Draw current position indicator (white line)
        if self.current_position_ms > 0:
            pos_x = int((self.current_position_ms / self.duration_ms) * width)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(pos_x, 0, pos_x, height)
        
        # Draw labels
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Label for clip section (green)
        if clip_width > 100:  # Only draw if wide enough
            label_rect = QRect(clip_start_x + 5, 2, clip_width - 10, height - 4)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Clip to evaluate")
        
        # Label for context before (grey)
        if context_before_width > 50:
            context_label_rect = QRect(5, 2, context_before_width - 10, height - 4)
            painter.drawText(context_label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Context")
        
        # Label for context after (grey)
        if context_after_width > 50:
            context_after_label_rect = QRect(context_after_start_x + 5, 2, context_after_width - 10, height - 4)
            painter.drawText(context_after_label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Context")


class MatplotlibWidget(QWidget):
    """Widget for displaying matplotlib figures"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.original_figure = None  # Store original figure for saving
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def update_plot(self, fig):
        """Update the plot with a new figure"""
        # Store the original figure for saving
        self.original_figure = fig
        
        # Clear existing figure
        self.figure.clear()
        
        # Matplotlib artists (especially collections from seaborn heatmaps) cannot be
        # moved between figures. The safest approach is to save the figure as an image
        # and display it. This avoids all "artist in more than one figure" errors.
        try:
            import io
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.image import imread
            
            # Match the figure size
            if hasattr(fig, 'get_size_inches'):
                self.figure.set_size_inches(fig.get_size_inches())
            
            # Save the input figure to a buffer as PNG
            buf = io.BytesIO()
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            
            # Load the image and display it
            img = imread(buf)
            buf.close()
            
            # Display the image in our figure
            ax = self.figure.add_subplot(111)
            ax.imshow(img, aspect='auto')
            ax.axis('off')
            
        except Exception as e:
            logger.error("Error updating plot: %s", e, exc_info=True)
            # On error, at least try to show something
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error displaying plot:\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes)
        
        self.canvas.draw()
    
    def clear(self):
        """Clear the plot"""
        self.figure.clear()
        self.canvas.draw()
    
    def get_figure(self):
        """Get the matplotlib figure"""
        return self.figure


class ScrollablePlotContainer(QWidget):
    """Container widget with scrollable plot and save button"""
    
    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.current_figure = None  # Store current figure for saving
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(plot_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(scroll_area)
        container.setLayout(container_layout)

        self.save_btn = QPushButton("Save Plot")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.save_btn.clicked.connect(self._save_plot)
        self.save_btn.setFixedSize(120, 35)

        main_widget = QWidget()
        main_widget_layout = QVBoxLayout()
        main_widget_layout.setContentsMargins(0, 0, 0, 0)
        main_widget_layout.addWidget(scroll_area)
        main_widget.setLayout(main_widget_layout)

        # Floats the save button over the top-right corner of the scroll area.
        class OverlayWidget(QWidget):
            def __init__(self, parent, button):
                super().__init__(parent)
                self.button = button
                layout = QVBoxLayout()
                layout.setContentsMargins(10, 10, 10, 10)
                layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
                layout.addStretch()
                self.setLayout(layout)
                self.setStyleSheet("background-color: transparent;")
            
            def resizeEvent(self, event):
                super().resizeEvent(event)
                self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        
        overlay = OverlayWidget(main_widget, self.save_btn)
        overlay.raise_()
        
        main_layout.addWidget(main_widget)
        self.setLayout(main_layout)
    
    def _save_plot(self):
        """Save the current plot as PNG or PDF"""
        import os
        
        if self.current_figure is None:
            QMessageBox.warning(self, "No Plot", "No plot to save. Please generate a plot first.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Plot", "plot", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if not file_path:
            return
        
        try:
            # Determine format from extension or filter
            if selected_filter.startswith("PNG") or file_path.endswith('.png'):
                format = 'png'
            elif selected_filter.startswith("PDF") or file_path.endswith('.pdf'):
                format = 'pdf'
            elif selected_filter.startswith("SVG") or file_path.endswith('.svg'):
                format = 'svg'
            else:
                format = 'png'
            
            # Save based on widget type
            if isinstance(self.plot_widget, PlotlyWidget):
                # Save Plotly figure
                import plotly.io as pio
                if format == 'png':
                    pio.write_image(self.current_figure, file_path, format='png', width=1200, height=800, scale=2)
                elif format == 'pdf':
                    pio.write_image(self.current_figure, file_path, format='pdf', width=1200, height=800)
                elif format == 'svg':
                    pio.write_image(self.current_figure, file_path, format='svg', width=1200, height=800)
            elif isinstance(self.plot_widget, MatplotlibWidget):
                # Save Matplotlib figure
                # Use the original figure stored in the widget
                if hasattr(self.plot_widget, 'original_figure') and self.plot_widget.original_figure is not None:
                    self.plot_widget.original_figure.savefig(file_path, format=format, dpi=300, bbox_inches='tight')
                elif hasattr(self.current_figure, 'savefig'):
                    self.current_figure.savefig(file_path, format=format, dpi=300, bbox_inches='tight')
                else:
                    # Fallback: save the widget's figure
                    self.plot_widget.figure.savefig(file_path, format=format, dpi=300, bbox_inches='tight')
            
            QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
        except Exception as e:
            logger.error("Error saving plot: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Error saving plot:\n{str(e)}")
    
    def update_plot(self, fig):
        """Update the plot and store the figure"""
        self.current_figure = fig
        if hasattr(self.plot_widget, 'update_plot'):
            self.plot_widget.update_plot(fig)
    
    def clear(self):
        """Clear the plot"""
        self.current_figure = None
        if hasattr(self.plot_widget, 'clear'):
            self.plot_widget.clear()


class PlotlyWidget(QWidget):
    """Widget for displaying plotly figures using HTML export with full interactivity"""
    
    # Signal emitted when a point is clicked (snippet_id)
    point_clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._click_callback = None  # Callback for point clicks
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            from PyQt6.QtCore import QUrl
            from PyQt6.QtWebEngineCore import QWebEngineSettings
            
            self.web_view = QWebEngineView()
            
            # Configure settings for maximum interactivity
            settings = self.web_view.settings()
            # Enable JavaScript (should be enabled by default, but ensure it)
            try:
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.ErrorPageEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
            except AttributeError:
                pass

            # Set up QWebChannel for JavaScript-Python communication
            try:
                from PyQt6.QtWebChannel import QWebChannel
                from PyQt6.QtCore import QObject, pyqtSlot
                
                class ClickBridge(QObject):
                    def __init__(self, callback):
                        super().__init__()
                        self.callback = callback
                    
                    @pyqtSlot(str)
                    def on_click(self, snippet_id):
                        if self.callback:
                            self.callback(snippet_id)
                
                self.click_bridge = ClickBridge(self._handle_snippet_click)
                self.web_channel = QWebChannel()
                self.web_channel.registerObject('bridge', self.click_bridge)
                self.web_view.page().setWebChannel(self.web_channel)
            except ImportError:
                # QWebChannel not available, fall back to URL scheme
                self.click_bridge = None
                self.web_channel = None
            
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.web_view)
            self.setLayout(layout)
            self.use_webview = True
            
            # Store a temporary file path for HTML (optional, for better compatibility)
            import tempfile
            self.temp_dir = tempfile.gettempdir()
            
        except ImportError as e:
            logger.warning("QWebEngineWidgets not available: %s. Plotly plots will be static images.", e)
            # Fallback to static image if WebEngine not available
            from PyQt6.QtWidgets import QLabel
            from PyQt6.QtGui import QPixmap
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setText("Plotly interactive plots require PyQt6.QtWebEngineWidgets.\nPlease install: pip install PyQt6-WebEngine")
            layout = QVBoxLayout()
            layout.addWidget(self.image_label)
            self.setLayout(layout)
            self.use_webview = False
    
    def update_plot(self, fig):
        """Update the plot with a plotly figure"""
        if self.use_webview:
            try:
                from PyQt6.QtCore import QUrl
                import tempfile
                import os
                
                # Ensure figure has responsive layout and full interactivity
                if not hasattr(fig, 'layout') or fig.layout is None:
                    fig.update_layout(template='plotly_white')
                
                # Make layout responsive and ensure interactivity
                fig.update_layout(
                    autosize=True,
                    hovermode='closest',
                    dragmode='pan'  # Allow panning by default
                )
                
                # Create a temporary HTML file for better compatibility with QWebEngineView
                # This ensures all JavaScript and resources load properly
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, dir=self.temp_dir)
                temp_path = temp_file.name
                temp_file.close()

                # Embedding plotly.js inline is more reliable than the CDN path
                # when QtWebEngine loads from a local file URL.
                html = pio.to_html(
                    fig,
                    include_plotlyjs='inline',
                    div_id='plotly-div',
                    config={
                        'displayModeBar': True,  # Show toolbar
                        'displaylogo': False,    # Hide plotly logo
                        'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'plot',
                            'height': None,
                            'width': None,
                            'scale': 1
                        },
                        'responsive': True,  # Enable responsive behavior
                        'staticPlot': False,  # Ensure interactivity is enabled
                        'doubleClick': 'reset',  # Double-click to reset zoom
                        'showTips': True,  # Show interaction tips
                        'showLink': False  # Hide "Edit chart" link
                    }
                )
                
                if hasattr(self, '_click_callback') and self._click_callback:
                    html = self._inject_click_handler(html)

                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(html)

                # Loading from a file URL is more reliable than setHtml when
                # injected JavaScript (click handlers) must run.
                file_url = QUrl.fromLocalFile(temp_path)

                if not self.web_view.isVisible():
                    self.web_view.show()

                if self.web_view.width() < 100 or self.web_view.height() < 100:
                    self.web_view.setMinimumSize(400, 300)

                self.web_view.setUrl(file_url)

                if hasattr(self, '_last_temp_file') and os.path.exists(self._last_temp_file):
                    try:
                        os.unlink(self._last_temp_file)
                    except:
                        pass
                
                self._last_temp_file = temp_path
                
            except Exception as e:
                logger.error("Error updating plotly plot: %s", e, exc_info=True)
                try:
                    html = pio.to_html(fig, include_plotlyjs='inline')
                    self.web_view.setHtml(html)
                except Exception as e2:
                    logger.error("Error with setHtml fallback: %s", e2)
        else:
            # Static image fallback when QtWebEngine is not available.
            try:
                img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
                from PyQt6.QtGui import QPixmap
                pixmap = QPixmap()
                pixmap.loadFromData(img_bytes)
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                self.image_label.setText(f"Error rendering plot: {str(e)}")
    
    def set_click_callback(self, callback):
        """Set callback function for point clicks. Callback receives snippet_id (str)."""
        self._click_callback = callback
        if self.use_webview and hasattr(self, 'click_bridge') and self.click_bridge:
            self.click_bridge.callback = callback
    
    def _handle_snippet_click(self, snippet_id):
        """Handle snippet:// URL clicks"""
        if self._click_callback:
            self._click_callback(snippet_id)
    
    def _inject_click_handler(self, html):
        """Inject JavaScript to handle plotly_click events."""
        use_webchannel = hasattr(self, 'web_channel') and self.web_channel is not None
        
        if use_webchannel:
            js_injection = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
            var bridge = null;
            new QWebChannel(qt.webChannelTransport, function(channel) {
                bridge = channel.objects.bridge;
            });
            
            // Wait for Plotly to be loaded and plot to be ready
            function setupClickHandler() {
                if (typeof Plotly === 'undefined') {
                    setTimeout(setupClickHandler, 100);
                    return;
                }
                
                var checkPlot = setInterval(function() {
                    var plotDivs = document.getElementsByClassName('plotly-graph-div');
                    if (plotDivs.length > 0) {
                        var plotDiv = plotDivs[0];
                        if (plotDiv && (plotDiv.data || plotDiv._fullLayout)) {
                            clearInterval(checkPlot);
                            
                            // Attach click handler using Plotly's event system
                            plotDiv.on('plotly_click', function(data) {
                                if (data && data.points && data.points.length > 0) {
                                    var point = data.points[0];
                                    // Get snippet_id from customdata
                                    var snippet_id = null;
                                    if (point.customdata !== undefined && point.customdata !== null) {
                                        if (Array.isArray(point.customdata) && point.customdata.length > 0) {
                                            snippet_id = point.customdata[0];
                                        } else if (Array.isArray(point.customdata[0]) && point.customdata[0].length > 0) {
                                            snippet_id = point.customdata[0][0];
                                        } else {
                                            snippet_id = point.customdata;
                                        }
                                    }
                                    
                                    if (bridge && snippet_id) {
                                        bridge.on_click(String(snippet_id));
                                    }
                                }
                            });
                        }
                    }
                }, 100);
                
                setTimeout(function() {
                    clearInterval(checkPlot);
                }, 10000);
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupClickHandler);
            } else {
                setupClickHandler();
            }
        </script>
        </body>
        """
        else:
            # Fallback: use URL scheme (less reliable but works without QWebChannel)
            js_injection = """
        <script>
            // Wait for Plotly to be loaded and plot to be ready
            function setupClickHandler() {
                if (typeof Plotly === 'undefined') {
                    setTimeout(setupClickHandler, 100);
                    return;
                }
                
                var checkPlot = setInterval(function() {
                    var plotDivs = document.getElementsByClassName('plotly-graph-div');
                    if (plotDivs.length > 0) {
                        var plotDiv = plotDivs[0];
                        if (plotDiv && (plotDiv.data || plotDiv._fullLayout)) {
                            clearInterval(checkPlot);
                            
                            // Attach click handler using Plotly's event system
                            plotDiv.on('plotly_click', function(data) {
                                if (data && data.points && data.points.length > 0) {
                                    var point = data.points[0];
                                    var snippet_id = null;
                                    if (point.customdata !== undefined && point.customdata !== null) {
                                        if (Array.isArray(point.customdata) && point.customdata.length > 0) {
                                            snippet_id = point.customdata[0];
                                        } else if (Array.isArray(point.customdata[0]) && point.customdata[0].length > 0) {
                                            snippet_id = point.customdata[0][0];
                                        } else {
                                            snippet_id = point.customdata;
                                        }
                                    }
                                    
                                    if (snippet_id) {
                                        // Use window.location to trigger navigation
                                        window.location.href = 'snippet://' + encodeURIComponent(String(snippet_id));
                                    }
                                }
                            });
                        }
                    }
                }, 100);
                
                setTimeout(function() {
                    clearInterval(checkPlot);
                }, 10000);
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupClickHandler);
            } else {
                setupClickHandler();
            }
        </script>
        </body>
        """
        return html.replace('</body>', js_injection)
    
    def clear(self):
        """Clear the plot"""
        if self.use_webview:
            self.web_view.setHtml("")
        else:
            self.image_label.clear()
    
    def __del__(self):
        """Cleanup temporary files when widget is destroyed"""
        if hasattr(self, '_last_temp_file'):
            import os
            try:
                if os.path.exists(self._last_temp_file):
                    os.unlink(self._last_temp_file)
            except:
                pass


class ScrollablePlotWidget(QWidget):
    """Scrollable container for plots (useful for large plots)"""
    
    def __init__(self, plot_widget: QWidget, parent=None):
        super().__init__(parent)
        scroll = QScrollArea()
        scroll.setWidget(plot_widget)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setLayout(layout)
        
        self.plot_widget = plot_widget
    
    def update_plot(self, fig):
        """Update the contained plot"""
        if hasattr(self.plot_widget, 'update_plot'):
            self.plot_widget.update_plot(fig)
    
    def clear(self):
        """Clear the contained plot"""
        if hasattr(self.plot_widget, 'clear'):
            self.plot_widget.clear()


def create_plot_widget(plot_type='matplotlib', width=8, height=6, scrollable=False):
    """
    Factory function to create appropriate plot widget.
    
    Args:
        plot_type: 'matplotlib' or 'plotly'
        width: Figure width (for matplotlib)
        height: Figure height (for matplotlib)
        scrollable: Whether to wrap in scrollable container
    
    Returns:
        Plot widget instance
    """
    if plot_type == 'matplotlib':
        widget = MatplotlibWidget(width=width, height=height)
    elif plot_type == 'plotly':
        widget = PlotlyWidget()
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    if scrollable:
        return ScrollablePlotWidget(widget)
    return widget

