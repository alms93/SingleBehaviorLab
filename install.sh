#!/bin/bash
# =============================================================================
# SingleBehaviorLab — Installation Script
# Creates the 'singlebehaviorlab' conda environment and installs all
# dependencies including PyTorch, JAX, VideoPrism, and SAM2.
#
# Usage:
#   bash install.sh
#
# Requires an NVIDIA GPU with a CUDA 12-compatible driver.
# =============================================================================

set -e

if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1"
  echo "Usage: bash install.sh"
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="singlebehaviorlab"
VENV_DIR="$APP_DIR/.venv_singlebehaviorlab"
ENV_TYPE="conda"
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

echo "============================================================"
echo "  SingleBehaviorLab Installer"
echo "  App directory : $APP_DIR"
echo "  Conda env     : $ENV_NAME"
echo "  Mode          : GPU (CUDA)"
echo "============================================================"
echo ""

# Step 1 — create the conda environment.
echo "[1/6] Preparing Python environment..."
if [ -n "$CONDA_EXE_PATH" ]; then
  CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"

  # Load conda into this non-interactive shell before using `conda activate`.
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"

  if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Conda environment '$ENV_NAME' already exists — skipping creation."
    echo "  To recreate it, run: conda env remove -n $ENV_NAME"
  else
    conda create -y -n "$ENV_NAME" python=3.10 git cudnn -c conda-forge
    echo "  Conda environment created."
  fi

  conda activate "$ENV_NAME"
else
  ENV_TYPE="venv"
  echo "  Conda not found. Falling back to Python venv at:"
  echo "  $VENV_DIR"

  if [ ! -d "$VENV_DIR" ]; then
    if command -v python3.10 >/dev/null 2>&1; then
      python3.10 -m venv "$VENV_DIR"
    else
      python3 -m venv "$VENV_DIR"
    fi
    echo "  venv created."
  else
    echo "  venv already exists — skipping creation."
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

echo "[1/6] Done."

# Step 2 — upgrade pip.
echo "[2/6] Upgrading pip..."
pip install --upgrade pip
echo "[2/6] Done."

# Step 3 — install PyTorch (CUDA 12.4 wheel).
echo "[3/6] Installing PyTorch..."
pip install --upgrade torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
echo "[3/6] Done."

# Step 4 — install JAX with its bundled CUDA 12 runtime.
echo "[4/6] Installing JAX..."
pip install --upgrade "jax[cuda12]==0.6.2"
pip install --upgrade "nvidia-cudnn-cu12==9.20.0.48"
pip install --upgrade flax==0.10.7
echo "[4/6] Done."

# Step 5 — install remaining Python dependencies.
echo "[5/6] Installing Python dependencies..."
pip install --upgrade \
  PyQt6==6.11.0 PyQt6-WebEngine==6.11.0 PyYAML==6.0.3 \
  numpy==2.2.6 h5py==3.14.0 opencv-python==4.13.0.92 Pillow==12.1.1 scipy==1.15.3 eva-decord==0.6.1 \
  scikit-learn==1.7.2 pandas==2.3.3 \
  umap-learn==0.5.11 leidenalg==0.11.0 python-igraph==1.0.0 hdbscan==0.8.42 \
  plotly==6.6.0 matplotlib==3.10.8

echo "  Installing VideoPrism from GitHub..."
pip install --upgrade git+https://github.com/google-deepmind/videoprism.git

echo "[5/6] Done."

# Step 6 — install the bundled SAM2 backend.
echo "[6/7] Installing SAM2 (local)..."
SAM2_DIR="$APP_DIR/sam2_backend"
if [ -d "$SAM2_DIR" ]; then
  pip install -e "$SAM2_DIR" --no-build-isolation
  echo "  SAM2 installed from $SAM2_DIR"
else
  echo "  WARNING: sam2_backend/ not found — skipping SAM2 install."
  echo "  The Segmentation & Tracking tab will not be available."
fi
echo "[6/7] Done."

# Step 7 — install the SingleBehaviorLab package and register the entry point.
echo "[7/7] Installing SingleBehaviorLab package..."
pip install -e "$APP_DIR" --no-deps
echo "  SingleBehaviorLab installed. You can now run: singlebehaviorlab"
echo "[7/7] Done."

# Verify.
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"

python -c "from PyQt6.QtWidgets import QApplication; print('  PyQt6          OK')"
python -c "import torch; print(f'  PyTorch        OK  ({torch.__version__})')"
python -c "
import jax
devs = jax.devices()
kinds = set(d.platform for d in devs)
gpu_ok = 'gpu' in kinds
print(f'  JAX            OK  ({jax.__version__}, {\"GPU\" if gpu_ok else \"NO GPU DETECTED\"})')
if not gpu_ok:
    print('  JAX did not find a GPU. SingleBehaviorLab requires an NVIDIA GPU.')
    print('  Re-run: pip install -U \"jax[cuda12]==0.6.2\"')
"
python -c "
from importlib import metadata
try:
    v = metadata.version('nvidia-cudnn-cu12')
    print(f'  CuDNN runtime  OK  ({v})')
    parts = []
    for token in v.split('.'):
        digits = ''.join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts[:2]) < (9, 8):
        print('  CuDNN runtime is older than 9.8 and may break VideoPrism/JAX GPU loading.')
except metadata.PackageNotFoundError:
    print('  CuDNN runtime  OK  (managed outside pip)')
"
python -c "from videoprism import models as vp; print('  VideoPrism     OK')"
python -c "import decord; print('  eva-decord     OK')"
python -c "import sam2; print('  SAM2           OK')" 2>/dev/null || \
  echo "  SAM2           SKIPPED (install sam2_backend manually if needed)"
python -c "import singlebehaviorlab; print(f'  SingleBehaviorLab OK  (v{singlebehaviorlab.__version__})')"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  To launch SingleBehaviorLab:"
if [ "$ENV_TYPE" = "conda" ]; then
  echo "    conda activate $ENV_NAME"
else
  echo "    source \"$VENV_DIR/bin/activate\""
fi
echo "    cd \"$APP_DIR\""
echo "    python main.py"
echo ""
echo "  Or use the launch script:"
echo "    bash run.sh"
echo "============================================================"
