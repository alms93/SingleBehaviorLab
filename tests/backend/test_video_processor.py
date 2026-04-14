"""Tests for video_processor utilities (singlebehaviorlab.backend.video_processor)."""

import numpy as np
import pytest

from singlebehaviorlab.backend.video_processor import (
    convert_old_format_to_objects,
    process_frame_with_mask,
)


class TestConvertOldFormat:
    def test_single_object_converted(self):
        data = np.zeros((4, 1, 100, 100), dtype=np.uint8)
        data[0, 0, 20:40, 30:60] = 1  # mask in frame 0
        objects = convert_old_format_to_objects(data)
        assert len(objects) == 4
        assert len(objects[0]) == 1
        bbox = objects[0][0]["bbox"]
        assert bbox == (30, 20, 59, 39)

    def test_empty_frames(self):
        data = np.zeros((3, 1, 50, 50), dtype=np.uint8)
        objects = convert_old_format_to_objects(data)
        assert all(len(frame_objs) == 0 for frame_objs in objects)


class TestProcessFrameWithMask:
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        data = np.zeros((1, 1, 240, 320), dtype=np.uint8)
        data[0, 0, 80:160, 100:220] = 1
        frame_objects = convert_old_format_to_objects(data)
        result = process_frame_with_mask(
            frame, frame_objects, frame_idx=0,
            target_size=288, background_mode="white",
        )
        assert result.shape == (288, 288, 3)
        assert result.dtype == np.float32

    def test_missing_mask_returns_zeros(self):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame_objects = [[]]  # no objects
        result = process_frame_with_mask(frame, frame_objects, frame_idx=0, target_size=288)
        assert result.shape == (288, 288, 3)
        assert result.max() == 0.0
