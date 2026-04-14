"""Smoke tests for SAM2 integration (no model weights or GPU required)."""

import inspect
import pytest


class TestSAM2Import:
    def test_segmentation_tracking_widget_importable(self):
        """The module should import without requiring SAM2 itself."""
        try:
            from singlebehaviorlab.gui import segmentation_tracking_widget
            assert hasattr(segmentation_tracking_widget, "SegmentationTrackingWidget")
            assert hasattr(segmentation_tracking_widget, "TrackingWorker")
            assert hasattr(segmentation_tracking_widget, "SAM2SetupWorker")
            assert hasattr(segmentation_tracking_widget, "CheckpointDownloadWorker")
        except ImportError as e:
            if "PyQt6" in str(e):
                pytest.skip("PyQt6 not available in headless env")
            raise

    def test_tracking_worker_accepts_expected_args(self):
        """TrackingWorker.__init__ should accept predictor, video_path, user_points, etc."""
        try:
            from singlebehaviorlab.gui.segmentation_tracking_widget import TrackingWorker
        except ImportError:
            pytest.skip("PyQt6 not available")
        sig = inspect.signature(TrackingWorker.__init__)
        params = list(sig.parameters.keys())
        for expected in ["predictor", "video_path", "user_points", "start_frame", "end_frame"]:
            assert expected in params, f"Missing parameter: {expected}"

    def test_checkpoint_download_worker_signature(self):
        try:
            from singlebehaviorlab.gui.segmentation_tracking_widget import CheckpointDownloadWorker
        except ImportError:
            pytest.skip("PyQt6 not available")
        sig = inspect.signature(CheckpointDownloadWorker.__init__)
        params = list(sig.parameters.keys())
        for expected in ["checkpoint_name", "checkpoint_path", "checkpoint_url"]:
            assert expected in params

    def test_sam2_setup_worker_signature(self):
        try:
            from singlebehaviorlab.gui.segmentation_tracking_widget import SAM2SetupWorker
        except ImportError:
            pytest.skip("PyQt6 not available")
        sig = inspect.signature(SAM2SetupWorker.__init__)
        params = list(sig.parameters.keys())
        assert "sam2_backend_dir" in params
        assert "sam2_checkpoints_dir" in params

    def test_sam2_paths_resolve(self):
        """The path resolver should return a sam2 backend directory path."""
        from singlebehaviorlab._paths import get_sam2_backend_dir, get_sam2_checkpoints_dir
        backend_dir = get_sam2_backend_dir()
        checkpoints_dir = get_sam2_checkpoints_dir()
        assert backend_dir is not None
        assert checkpoints_dir is not None
