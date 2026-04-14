"""Tests for motion tracking utilities (singlebehaviorlab.gui.motion_tracking)."""

import numpy as np
import pytest

from singlebehaviorlab.gui.motion_tracking import (
    mask_to_bbox,
    compute_iou,
    compute_mask_score,
)


class TestMaskToBbox:
    def test_simple_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1
        bbox = mask_to_bbox(mask)
        assert bbox is not None
        assert list(bbox) == [30, 20, 60, 40]

    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert mask_to_bbox(mask) is None

    def test_none_mask_returns_none(self):
        assert mask_to_bbox(None) is None


class TestComputeIoU:
    def test_identical_boxes(self):
        bbox = [10, 10, 50, 50]
        assert compute_iou(bbox, bbox) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        assert compute_iou(bbox1, bbox2) == 0.0

    def test_partial_overlap(self):
        bbox1 = [0, 0, 20, 20]
        bbox2 = [10, 10, 30, 30]
        iou = compute_iou(bbox1, bbox2)
        assert 0.0 < iou < 1.0

    def test_none_input(self):
        assert compute_iou(None, [0, 0, 10, 10]) == 0.0


class TestComputeMaskScore:
    def test_known_score_range(self):
        mask_logit = np.ones((100, 100), dtype=np.float32) * 5.0
        pred_bbox = np.array([10, 10, 90, 90], dtype=np.float32)
        actual_bbox = np.array([10, 10, 90, 90], dtype=np.float32)
        score, obj_score, motion_iou = compute_mask_score(
            mask_logit, pred_bbox, actual_bbox
        )
        assert 0.0 <= score <= 1.0
        assert 0.0 <= obj_score <= 1.0
        assert 0.0 <= motion_iou <= 1.0


class TestKalmanBoxTracker:
    @pytest.mark.slow
    def test_predict_update_cycle(self):
        """Kalman tracker predict/update tracks a rightward-moving box."""
        try:
            from filterpy.kalman import KalmanFilter  # noqa: F401
        except ImportError:
            pytest.skip("filterpy not installed")

        from singlebehaviorlab.gui.motion_tracking import KalmanBoxTracker
        bbox = [100, 100, 150, 150]
        tracker = KalmanBoxTracker(bbox)

        for step in range(5):
            predicted = tracker.predict()
            assert predicted is not None
            assert len(predicted) == 4
            new_bbox = [100 + step * 10, 100, 150 + step * 10, 150]
            tracker.update(new_bbox)

        state = tracker.get_state()
        assert state[2] > state[0]  # x2 > x1
        assert state[3] > state[1]  # y2 > y1
