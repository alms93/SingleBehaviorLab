"""
Central path resolver for SingleBehaviorLab.

Handles two install modes transparently:

  Source / zip distribution
    SingleBehaviorLab/
      singlebehaviorlab/   ← this file lives here
      sam2_backend/
      sam2_checkpoints/
      experiments/

  pip install (site-packages)
    site-packages/singlebehaviorlab/  ← this file lives here
    ~/SingleBehaviorLab/              ← user data lives here
"""

from pathlib import Path

# Directory where this file (and the package) lives.
_PKG_DIR = Path(__file__).parent

# One level up: the SingleBehaviorLab root when running from source/zip,
# or site-packages when pip-installed.
_PKG_PARENT = _PKG_DIR.parent

# Standard user data directory — always writable.
USER_DATA_DIR = Path.home() / "SingleBehaviorLab"


def _first_existing(*candidates: Path) -> Path:
    """Return the first candidate path that exists, otherwise the first candidate."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


# Public helpers.

def get_package_dir() -> Path:
    """The installed singlebehaviorlab package directory."""
    return _PKG_DIR


def get_sam2_backend_dir() -> Path:
    """Return the directory containing the sam2 Python package."""
    try:
        import sam2  # type: ignore
        sam2_pkg = Path(sam2.__file__).resolve().parent
        if (sam2_pkg / "configs").exists():
            return sam2_pkg.parent
    except Exception:
        pass
    return _first_existing(
        _PKG_PARENT / "sam2_backend",
        _PKG_DIR / "sam2_backend",
        USER_DATA_DIR / "sam2_backend",
    )


def get_sam2_checkpoints_dir() -> Path:
    """Return the directory where SAM2 checkpoint files live, creating it if absent."""
    source_dir = _PKG_PARENT / "sam2_checkpoints"
    if source_dir.exists():
        return source_dir
    user_dir = USER_DATA_DIR / "sam2_checkpoints"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_training_profiles_path() -> Path:
    """Locate training_profiles.json."""
    return _first_existing(
        _PKG_DIR / "data" / "training_profiles.json",          # package data
        USER_DATA_DIR / "training_profiles.json",              # user copy
    )


def get_default_config_path() -> Path:
    """Locate the default (template) config.yaml."""
    return _first_existing(
        _PKG_PARENT / "config" / "config.yaml",                # source / zip install
        _PKG_DIR / "data" / "config" / "config.yaml",          # package data
    )


def get_experiments_dir() -> Path:
    """
    Default experiments root directory.

    - Source / zip install: SingleBehaviorLab/experiments/
    - pip install: ~/SingleBehaviorLab/experiments/
    """
    local = _PKG_PARENT / "experiments"
    if local.exists():
        return local
    user_exp = USER_DATA_DIR / "experiments"
    user_exp.mkdir(parents=True, exist_ok=True)
    return user_exp
