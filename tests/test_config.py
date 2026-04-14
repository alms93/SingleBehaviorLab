"""Tests for config loading (singlebehaviorlab.__main__.load_config)."""

import os
import yaml
import pytest

from singlebehaviorlab.__main__ import load_config


class TestLoadConfig:
    def test_defaults_populated(self, tmp_path):
        cfg_file = str(tmp_path / "config.yaml")
        with open(cfg_file, "w") as f:
            yaml.safe_dump({}, f)
        config = load_config(cfg_file)
        assert config.get("data_dir")
        assert config.get("models_dir")
        assert config.get("experiments_dir")

    def test_blank_experiments_dir_overridden(self, tmp_path):
        cfg_file = str(tmp_path / "config.yaml")
        with open(cfg_file, "w") as f:
            yaml.safe_dump({"experiments_dir": None}, f)
        config = load_config(cfg_file)
        assert config["experiments_dir"] is not None
        assert len(config["experiments_dir"]) > 0

    def test_user_yaml_overrides_template(self, tmp_path):
        cfg_file = str(tmp_path / "config.yaml")
        custom_dir = str(tmp_path / "my_data")
        with open(cfg_file, "w") as f:
            yaml.safe_dump({"data_dir": custom_dir}, f)
        config = load_config(cfg_file)
        assert config["data_dir"] == custom_dir

    def test_nonexistent_config_still_returns_defaults(self, tmp_path):
        config = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(config, dict)
        assert "data_dir" in config
