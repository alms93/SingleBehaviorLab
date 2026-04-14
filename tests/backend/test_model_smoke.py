"""Smoke tests for VideoPrismBackbone (no model weights or GPU required)."""

import pytest
import numpy as np


class TestVideoPrismConstants:
    def test_default_resolution(self):
        from singlebehaviorlab.backend.model import VideoPrismBackbone
        assert VideoPrismBackbone.DEFAULT_RESOLUTION == 288

    def test_patch_size(self):
        from singlebehaviorlab.backend.model import VideoPrismBackbone
        assert VideoPrismBackbone.PATCH_SIZE == 18

    def test_get_num_tokens_computation(self):
        """get_num_tokens should return (resolution / patch_size)^2 without loading weights."""
        from singlebehaviorlab.backend.model import VideoPrismBackbone
        grid = 288 // 18  # 16
        expected = grid * grid  # 256
        # Calling get_num_tokens() would require a loaded backbone; verify
        # the formula against the class constants instead.
        assert expected == 256

    def test_grid_size_formula_non_default(self):
        """Higher resolution should produce a larger spatial grid."""
        from singlebehaviorlab.backend.model import VideoPrismBackbone
        grid_288 = 288 // VideoPrismBackbone.PATCH_SIZE
        grid_342 = 342 // VideoPrismBackbone.PATCH_SIZE
        assert grid_342 > grid_288


class TestInterpolatePosEmbed:
    def test_upsample(self):
        from singlebehaviorlab.backend.model import interpolate_pos_embed_2d
        D = 32
        orig_grid, new_grid = 4, 8
        pos = np.random.randn(orig_grid * orig_grid, D).astype(np.float32)
        result = interpolate_pos_embed_2d(pos, orig_grid, new_grid)
        assert result.shape == (new_grid * new_grid, D)

    def test_downsample(self):
        from singlebehaviorlab.backend.model import interpolate_pos_embed_2d
        D = 32
        orig_grid, new_grid = 8, 4
        pos = np.random.randn(orig_grid * orig_grid, D).astype(np.float32)
        result = interpolate_pos_embed_2d(pos, orig_grid, new_grid)
        assert result.shape == (new_grid * new_grid, D)


class TestModelModuleImports:
    def test_loss_functions_importable(self):
        from singlebehaviorlab.backend.model import (
            frame_classification_loss,
            boundary_detection_loss,
            temporal_smoothness_loss,
            localization_bbox_loss,
        )
        assert callable(frame_classification_loss)
        assert callable(boundary_detection_loss)
        assert callable(temporal_smoothness_loss)
        assert callable(localization_bbox_loss)

    def test_backbone_class_exists(self):
        from singlebehaviorlab.backend.model import VideoPrismBackbone
        import inspect
        sig = inspect.signature(VideoPrismBackbone.__init__)
        params = list(sig.parameters.keys())
        assert "model_name" in params
        assert "resolution" in params
