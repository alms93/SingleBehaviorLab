"""Tests for path resolution (singlebehaviorlab._paths)."""

import pytest
from pathlib import Path

from singlebehaviorlab._paths import (
    _first_existing,
    get_default_config_path,
    get_training_profiles_path,
    get_package_dir,
)


class TestFirstExisting:
    def test_returns_first_match(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.touch()
        b.touch()
        assert _first_existing(a, b) == a

    def test_fallback_when_none_exist(self, tmp_path):
        a = tmp_path / "no_a.txt"
        b = tmp_path / "no_b.txt"
        result = _first_existing(a, b)
        assert result == a  # returns first candidate regardless

    def test_skips_nonexistent_to_existing(self, tmp_path):
        a = tmp_path / "missing.txt"
        b = tmp_path / "exists.txt"
        b.touch()
        assert _first_existing(a, b) == b


class TestPublicPaths:
    def test_get_default_config_path_exists(self):
        p = get_default_config_path()
        assert p.exists(), f"Config path {p} does not exist"

    def test_get_training_profiles_path_exists(self):
        p = get_training_profiles_path()
        assert p.exists(), f"Profiles path {p} does not exist"

    def test_get_package_dir_is_directory(self):
        p = get_package_dir()
        assert p.is_dir()
