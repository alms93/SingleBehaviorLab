"""Tests for video utilities (singlebehaviorlab.backend.video_utils)."""

import os
import pytest
import cv2

from singlebehaviorlab.backend.video_utils import (
    get_video_info,
    load_clip_frames,
    extract_clips,
    save_clip,
)


class TestGetVideoInfo:
    def test_returns_metadata(self, synthetic_video):
        info = get_video_info(synthetic_video)
        assert info["fps"] == pytest.approx(30.0, abs=1.0)
        assert info["frame_count"] == 60
        assert info["width"] == 320
        assert info["height"] == 240

    def test_invalid_path_returns_empty(self, tmp_path):
        info = get_video_info(str(tmp_path / "nonexistent.mp4"))
        assert info == {}


class TestLoadClipFrames:
    def test_loads_correct_frame_count(self, synthetic_video):
        frames = load_clip_frames(synthetic_video)
        assert len(frames) == 60

    def test_resize_frames(self, synthetic_video):
        frames = load_clip_frames(synthetic_video, target_size=(160, 120))
        assert frames[0].shape == (120, 160, 3)


class TestExtractClips:
    @pytest.mark.slow
    def test_correct_clip_count(self, synthetic_video, tmp_path):
        output_dir = str(tmp_path / "clips")
        n_clips, _ = extract_clips(
            synthetic_video, output_dir,
            target_fps=30, clip_length_frames=8, step_frames=8,
        )
        expected = 60 // 8  # 7 clips (no subsampling since target==source fps)
        assert n_clips == expected

    @pytest.mark.slow
    def test_save_clip_roundtrip(self, tmp_path):
        import numpy as np
        frames = [
            (np.ones((120, 160, 3), dtype=np.uint8) * i)
            for i in range(8)
        ]
        clip_path = str(tmp_path / "roundtrip.mp4")
        save_clip(frames, clip_path, fps=12.0)
        assert os.path.exists(clip_path)
        loaded = load_clip_frames(clip_path)
        assert len(loaded) == 8
        assert loaded[0].shape == (120, 160, 3)
