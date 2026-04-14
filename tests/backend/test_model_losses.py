"""Tests for model loss functions and pos embed interpolation (singlebehaviorlab.backend.model)."""

import torch
import numpy as np
import pytest

from singlebehaviorlab.backend.model import (
    frame_classification_loss,
    boundary_detection_loss,
    temporal_smoothness_loss,
    localization_bbox_loss,
    interpolate_pos_embed_2d,
)


class TestFrameClassificationLoss:
    def test_correct_prediction_low_loss(self):
        B, T, C = 2, 4, 3
        logits = torch.zeros(B, T, C)
        logits[:, :, 0] = 10.0  # strongly predict class 0
        labels = torch.zeros(B, T, dtype=torch.long)  # true class 0
        loss = frame_classification_loss(logits, labels)
        assert loss.item() < 0.01

    def test_wrong_prediction_positive_loss(self):
        B, T, C = 2, 4, 3
        logits = torch.zeros(B, T, C)
        logits[:, :, 2] = 10.0  # predict class 2
        labels = torch.zeros(B, T, dtype=torch.long)  # true class 0
        loss = frame_classification_loss(logits, labels)
        assert loss.item() > 1.0

    def test_all_ignored_returns_zero(self):
        logits = torch.randn(2, 4, 3)
        labels = torch.full((2, 4), -1, dtype=torch.long)
        loss = frame_classification_loss(logits, labels)
        assert loss.item() == 0.0


class TestBoundaryDetectionLoss:
    def test_perfect_boundary_detection(self):
        B, T = 1, 8
        logits = torch.full((B, T, 1), -10.0)
        logits[0, 3, 0] = 10.0  # predict boundary at frame 3
        labels = torch.zeros(B, T)
        labels[0, 3] = 1.0  # true boundary at frame 3
        loss = boundary_detection_loss(logits, labels)
        assert loss.item() < 0.5

    def test_all_ignored_returns_zero(self):
        logits = torch.randn(1, 8, 1)
        labels = torch.full((1, 8), -1.0)
        loss = boundary_detection_loss(logits, labels)
        assert loss.item() == 0.0


class TestTemporalSmoothnessLoss:
    def test_constant_predictions_near_zero(self):
        B, T, C = 1, 8, 3
        logits = torch.ones(B, T, C) * 5.0
        labels = torch.zeros(B, T, dtype=torch.long)
        loss = temporal_smoothness_loss(logits, labels)
        assert loss.item() < 1e-6

    def test_jagged_predictions_high_loss(self):
        B, T, C = 1, 8, 3
        logits = torch.zeros(B, T, C)
        for t in range(T):
            logits[0, t, t % C] = 10.0  # oscillate
        labels = torch.zeros(B, T, dtype=torch.long)
        loss = temporal_smoothness_loss(logits, labels)
        assert loss.item() > 1.0

    def test_single_frame_returns_zero(self):
        logits = torch.randn(1, 1, 3)
        labels = torch.zeros(1, 1, dtype=torch.long)
        loss = temporal_smoothness_loss(logits, labels)
        assert loss.item() == 0.0


class TestLocalizationBboxLoss:
    def test_perfect_match_near_zero(self):
        pred = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        target = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        mask = torch.tensor([1.0])
        loss = localization_bbox_loss(pred, target, mask)
        assert loss.item() < 0.01

    def test_mismatch_positive_loss(self):
        pred = torch.tensor([[0.0, 0.0, 0.3, 0.3]])
        target = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        mask = torch.tensor([1.0])
        loss = localization_bbox_loss(pred, target, mask)
        assert loss.item() > 0.1

    def test_no_valid_returns_zero(self):
        pred = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        target = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        mask = torch.tensor([0.0])
        loss = localization_bbox_loss(pred, target, mask)
        assert loss.item() == 0.0


class TestInterpolatePosEmbed2D:
    def test_output_shape(self):
        orig_grid = 4
        new_grid = 6
        D = 16
        pos = np.random.randn(orig_grid * orig_grid, D).astype(np.float32)
        result = interpolate_pos_embed_2d(pos, orig_grid, new_grid)
        assert result.shape == (new_grid * new_grid, D)

    def test_same_grid_identity(self):
        grid = 4
        D = 8
        pos = np.random.randn(grid * grid, D).astype(np.float32)
        result = interpolate_pos_embed_2d(pos, grid, grid)
        np.testing.assert_array_equal(result, pos)
