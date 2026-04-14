#!/bin/bash
# Launch script for SingleBehaviorLab

set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="singlebehaviorlab"
VENV_DIR="$APP_DIR/.venv_singlebehaviorlab"
CONDA_EXE_PATH=""

if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_EXE_PATH="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_EXE_PATH="$(command -v conda)"
elif [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE_PATH="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE_PATH="$HOME/miniconda3/bin/conda"
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE_PATH="$HOME/anaconda3/bin/conda"
fi

if [ -n "$CONDA_EXE_PATH" ]; then
    CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
elif [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "No Conda installation or local venv found."
    echo "Run 'bash install.sh' first."
    exit 1
fi

# Suppress CuDNN / TensorFlow log noise
export TF_CPP_MIN_LOG_LEVEL=2

python - <<'PY'
from importlib import metadata
def parse_version(value):
    parts = []
    for token in value.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])
try:
    cudnn_version = metadata.version("nvidia-cudnn-cu12")
    if parse_version(cudnn_version) < (9, 8):
        print("WARNING: nvidia-cudnn-cu12 is older than 9.8 and may break VideoPrism/JAX GPU loading.")
        print("Run:")
        print("  pip install --upgrade torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
        print("  pip install --upgrade \"jax[cuda12]==0.6.2\" flax==0.10.7")
        print("  pip install --upgrade \"nvidia-cudnn-cu12==9.20.0.48\"")
except metadata.PackageNotFoundError:
    pass
PY

# Change to app directory so source-install path resolution works correctly
cd "$APP_DIR"

# Launch via the installed entry point (works after `pip install .`)
# Falls back to python -m if entry point is not on PATH yet
if command -v singlebehaviorlab &>/dev/null; then
    singlebehaviorlab
else
    python -m singlebehaviorlab
fi
