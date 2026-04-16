"""Tests for the SegmentsModel editing operations."""

import numpy as np
import pytest

from singlebehaviorlab.backend.segments import Segment, SegmentsModel


@pytest.fixture
def model():
    segs = [
        {"class": 0, "start": 10, "end": 30, "confidence": 0.9},
        {"class": 1, "start": 40, "end": 60, "confidence": 0.8},
        {"class": 2, "start": 70, "end": 90, "confidence": 0.7},
    ]
    return SegmentsModel(segs, classes=["walk", "run", "rest"], total_frames=100, orig_fps=25.0)


def test_init_sorts(model):
    assert [s.start for s in model.segments] == [10, 40, 70]


def test_resize_left(model):
    model.resize(1, "left", -5)
    assert model[1].start == 35
    assert model[1].end == 60


def test_resize_left_clamps_to_prev(model):
    model.resize(1, "left", -20)
    assert model[1].start == 30


def test_resize_right(model):
    model.resize(0, "right", 5)
    assert model[0].end == 35


def test_resize_right_clamps_to_next(model):
    model.resize(0, "right", 20)
    assert model[0].end == 40


def test_move(model):
    model.move(1, -5)
    assert model[1].start == 35
    assert model[1].end == 55


def test_move_clamps_to_neighbors(model):
    model.move(1, -30)
    assert model[1].start == 30
    assert model[1].end == 50


def test_reclass(model):
    model.reclass(0, 2)
    assert model[0].class_idx == 2


def test_reclass_invalid(model):
    assert not model.reclass(0, 99)


def test_delete(model):
    model.delete(1)
    assert len(model) == 2
    assert model[0].end == 30
    assert model[1].start == 70


def test_split(model):
    model.split(0, 20)
    assert len(model) == 4
    assert model[0].start == 10
    assert model[0].end == 20
    assert model[1].start == 20
    assert model[1].end == 30


def test_split_at_boundary_rejected(model):
    assert not model.split(0, 10)
    assert not model.split(0, 30)
    assert len(model) == 3


def test_merge(model):
    model.split(0, 20)
    model.merge_with_next(0)
    assert len(model) == 3
    assert model[0].start == 10
    assert model[0].end == 30


def test_undo_redo(model):
    original_start = model[1].start
    model.move(1, -5)
    assert model[1].start == original_start - 5
    model.undo()
    assert model[1].start == original_start
    model.redo()
    assert model[1].start == original_start - 5


def test_undo_empty(model):
    assert not model.can_undo
    assert not model.undo()


def test_to_frame_labels(model):
    labels = model.to_frame_labels()
    assert labels.shape == (100,)
    assert labels[15] == 0
    assert labels[50] == 1
    assert labels[80] == 2
    assert labels[5] == -1
    assert labels[35] == -1


def test_to_csv_rows(model):
    rows = model.to_csv_rows()
    assert len(rows) == 3
    assert rows[0]["Behavior"] == "walk"
    assert rows[1]["Behavior"] == "run"
    assert rows[2]["Behavior"] == "rest"
    assert rows[0]["Start Frame"] == 10
    assert rows[0]["End Frame"] == 30


def test_from_frame_labels():
    labels = np.array([0, 0, 0, 1, 1, -1, -1, 2, 2, 2])
    model = SegmentsModel.from_frame_labels(
        labels, classes=["a", "b", "c"], total_frames=10
    )
    assert len(model) == 3
    assert model[0].class_idx == 0
    assert model[0].start == 0
    assert model[0].end == 3
    assert model[1].class_idx == 1
    assert model[1].start == 3
    assert model[1].end == 5
    assert model[2].class_idx == 2
    assert model[2].start == 7
    assert model[2].end == 10
