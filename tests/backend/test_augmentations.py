"""Tests for clip augmentation (singlebehaviorlab.backend.augmentations)."""

import torch
import pytest

from singlebehaviorlab.backend.augmentations import ClipAugment


def _make_clip(T=8, C=3, H=64, W=64):
    """Random clip tensor in [0,1]."""
    return torch.rand(T, C, H, W)


class TestClipAugment:
    def test_output_shape_matches_input(self):
        aug = ClipAugment(use_horizontal_flip=True, use_color_jitter=True)
        clip = _make_clip()
        out = aug(clip)
        assert out.shape == clip.shape

    def test_deterministic_with_seed(self):
        aug = ClipAugment(use_horizontal_flip=True, use_color_jitter=True)
        clip = _make_clip()
        torch.manual_seed(0)
        out1 = aug(clip.clone())
        torch.manual_seed(0)
        out2 = aug(clip.clone())
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_no_augmentation_passthrough(self):
        aug = ClipAugment(
            use_horizontal_flip=False,
            use_vertical_flip=False,
            use_color_jitter=False,
            use_gaussian_blur=False,
            use_random_noise=False,
            use_small_rotation=False,
            use_speed_perturb=False,
            use_random_shapes=False,
            use_grayscale=False,
            use_lighting_robustness=False,
        )
        clip = _make_clip()
        out = aug(clip.clone())
        assert torch.allclose(out, clip, atol=1e-5)

    def test_augment_with_params_reproducible(self):
        aug = ClipAugment(use_horizontal_flip=True, use_color_jitter=True)
        clip = _make_clip()
        out1, params = aug.augment_with_params(clip.clone())
        out2 = aug._apply_with_params(clip.clone(), params)
        assert torch.allclose(out1, out2, atol=1e-5)
