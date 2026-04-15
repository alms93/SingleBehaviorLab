"""Configuration loading shared by the GUI and the CLI."""

import os
import yaml

from singlebehaviorlab._paths import (
    get_default_config_path,
    get_experiments_dir,
    get_package_dir,
)


def load_config(config_path: str = None) -> dict:
    """Load configuration from a YAML file, filling in default paths."""
    if config_path is None:
        config_path = str(get_default_config_path())

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    base_dir = str(get_package_dir().parent)

    defaults = {
        "data_dir":        os.path.join(base_dir, "data"),
        "raw_videos_dir":  os.path.join(base_dir, "data", "raw_videos"),
        "clips_dir":       os.path.join(base_dir, "data", "clips"),
        "annotations_dir": os.path.join(base_dir, "data", "annotations"),
        "models_dir":      os.path.join(base_dir, "models", "behavior_heads"),
        "backbone_dir":    os.path.join(base_dir, "models", "videoprism_backbone"),
        "annotation_file": os.path.join(base_dir, "data", "annotations", "annotations.json"),
    }

    for key, value in defaults.items():
        if not config.get(key):
            config[key] = value
        elif not os.path.isabs(config[key]):
            config[key] = os.path.join(base_dir, config[key])

    experiments_dir = str(get_experiments_dir())
    if not config.get("experiments_dir"):
        config["experiments_dir"] = experiments_dir
    os.makedirs(config["experiments_dir"], exist_ok=True)

    config["config_path"] = config_path
    config.setdefault("experiment_name", None)
    config.setdefault("experiment_path", None)

    return config
