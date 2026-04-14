# SingleBehaviorLab — Installation Guide

## Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Linux (Ubuntu 20.04+) |
| **Conda** | Miniconda or Anaconda — [download here](https://docs.conda.io/en/latest/miniconda.html) |
| **GPU** | NVIDIA GPU with a recent CUDA 12-compatible driver **required** (8 GB VRAM minimum, 12 GB+ recommended). |
| **Disk space** | ~10 GB free (SAM2 checkpoints ~3.5 GB, VideoPrism backbone ~1 GB on first run) |
| **Internet** | Required on first launch only (VideoPrism backbone auto-downloads) |

Check your CUDA version before starting:
```bash
nvidia-smi   # CUDA version shown in top-right corner
```

---

## Option A — Bundled install from zip/folder (recommended)

This is the standard path when you have downloaded the full SingleBehaviorLab package,
which ships with SAM2.1 checkpoints.

```bash
cd SingleBehaviorLab
bash install.sh
```

The script creates the `singlebehaviorlab` conda environment and runs all steps automatically.
At the end you should see:

```
  PyQt6             OK
  PyTorch           OK  (2.x.x+cu12x)
  JAX               OK  (0.x.x)
  VideoPrism        OK
  SAM2              OK
  SingleBehaviorLab OK  (v1.3.2)
```

Then launch the app:

```bash
# Option 1 — entry point command (works from any directory)
conda activate singlebehaviorlab
singlebehaviorlab

# Option 2 — launch script (activates env for you)
bash run.sh

# Option 3 — Python module (from the SingleBehaviorLab repo root)
conda activate singlebehaviorlab
cd /path/to/SingleBehaviorLab
python -m singlebehaviorlab

# Option 4 — main.py in a source checkout (equivalent to Option 3)
conda activate singlebehaviorlab
cd /path/to/SingleBehaviorLab
python main.py
```

---

## Option B — pip install (code only, no bundled weights)

Use this if you want to install from PyPI or from a cloned GitHub repository,
without the bundled SAM2 checkpoints.

### Step 1 — Create the conda environment

```bash
conda create -n singlebehaviorlab python=3.10 git cudnn -c conda-forge
conda activate singlebehaviorlab
pip install --upgrade pip
```

### Step 2 — Install PyTorch

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### Step 3 — Install JAX

```bash
pip install "jax[cuda12]==0.6.2"
pip install flax==0.10.7
```

### Step 4 — Install SingleBehaviorLab

```bash
# From PyPI (once published):
pip install singlebehaviorlab

# Or from a local folder / cloned repo:
pip install /path/to/SingleBehaviorLab
```

This installs all remaining Python dependencies automatically and registers
the `singlebehaviorlab` command.

### Step 5 — Install VideoPrism and video reader support

```bash
pip install git+https://github.com/google-deepmind/videoprism.git
pip install eva-decord
```

### Step 6 — Install SAM2

```bash
# From the bundled directory (if you have it):
pip install -e /path/to/SingleBehaviorLab/sam2_backend --no-build-isolation

# Or from GitHub:
pip install git+https://github.com/facebookresearch/sam2.git
```

### Step 7 — Download SAM2 checkpoints

If you are not using the bundled zip (which already includes checkpoints), download them
into `~/SingleBehaviorLab/sam2_checkpoints/checkpoints/`:

```bash
mkdir -p ~/SingleBehaviorLab/sam2_checkpoints/checkpoints
cd ~/SingleBehaviorLab/sam2_checkpoints/checkpoints

# SAM2.1 models (choose the ones you need)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Step 8 — Verify and launch

```bash
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import jax; print('JAX', jax.__version__, '| devices:', jax.devices())"
python -c "from videoprism import models; print('VideoPrism OK')"
python -c "import sam2; print('SAM2 OK')"
python -c "import singlebehaviorlab; print('SingleBehaviorLab', singlebehaviorlab.__version__)"

singlebehaviorlab
```

---

## GPU Memory Notes

SingleBehaviorLab runs JAX (VideoPrism feature extraction) and PyTorch (classifier training)
on the same GPU simultaneously:

- **JAX** is capped at 45% of GPU VRAM and grows memory on demand
- **PyTorch** uses the remaining ~55%

| GPU VRAM | Experience |
|---|---|
| 8 GB | Feature extraction and inference work well. Reduce batch size for training. |
| 12 GB | All operations including batch training run comfortably. |
| 16 GB+ | No constraints. |

If you get out-of-memory errors during training, lower the **Batch Size** in the Training tab.

---

## Troubleshooting

### `singlebehaviorlab: command not found`
The package is not installed or the conda environment is not active:
```bash
conda activate singlebehaviorlab
pip install -e /path/to/SingleBehaviorLab   # re-install if needed
```

### CuDNN version mismatch
```bash
conda activate singlebehaviorlab
pip install --upgrade torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade "jax[cuda12]==0.6.2" flax==0.10.7
pip install --upgrade "nvidia-cudnn-cu12==9.20.0.48"
```

### VideoPrism fails to download on first launch
Requires internet on first run. If behind a firewall, set proxy variables first:
```bash
export https_proxy=http://your-proxy:port
singlebehaviorlab
```

### SAM2 import error
```bash
pip install -e /path/to/SingleBehaviorLab/sam2_backend --no-build-isolation --force-reinstall
```

### `decord not found` in the Segmentation tab
```bash
conda activate singlebehaviorlab
pip install eva-decord
```

### Wrong environment / import errors
```bash
conda activate singlebehaviorlab
which python        # should point inside the singlebehaviorlab env
pip list | grep -E "PyQt6|torch|jax|videoprism|sam2|decord|singlebehaviorlab"
```

### App window doesn't open (remote server / no display)
SingleBehaviorLab requires a graphical desktop. Use X11 forwarding for remote access:
```bash
ssh -X user@your-server
singlebehaviorlab
```
