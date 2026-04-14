"""Shared fixtures for SingleBehaviorLab tests."""

import json
import os
import pytest
import numpy as np
import cv2


@pytest.fixture
def tmp_annotation_file(tmp_path):
    """Path (str) for an annotation JSON that doesn't exist yet."""
    return str(tmp_path / "annotations.json")


@pytest.fixture
def sample_annotations(tmp_path):
    """Pre-populated annotation file with 2 classes and 4 clips."""
    path = tmp_path / "annotations.json"
    data = {
        "classes": ["grooming", "rearing"],
        "clips": [
            {"id": "clip_000000.mp4", "label": "grooming", "labels": ["grooming"], "meta": {}},
            {"id": "clip_000001.mp4", "label": "rearing", "labels": ["rearing"], "meta": {}},
            {"id": "clip_000002.mp4", "label": "grooming", "labels": ["grooming"], "meta": {}},
            {"id": "clip_000003.mp4", "label": "", "labels": [], "meta": {}},
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    return str(path)


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a 2-second 320x240 @30fps synthetic .mp4 and return its path."""
    path = str(tmp_path / "test_video.mp4")
    fps = 30.0
    w, h = 320, 240
    total_frames = 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(total_frames):
        frame = np.full((h, w, 3), fill_value=(i * 4) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
