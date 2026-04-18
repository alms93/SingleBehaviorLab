# SingleBehaviorLab

**SingleBehaviorLab (SBL)** is a tool for behavior action localization in animal video. It supports lightweight few-shot training of behavior classifiers, referred to here as *behavior sequencing*, along with unsupervised behavior discovery for exploring unlabeled recordings, and a full GUI pipeline for downstream analysis: ethograms, behavior clustering, transition state analysis, and more comprehesive tools focused on postprocessing.

<table align="center">
  <tr>
    <td><img src="docs/behavior_seq.png" alt="Behavior Sequencing GUI" width="420"></td>
    <td><img src="docs/behavior_clustering.png" alt="Unsupervised Discovery GUI" width="420"></td>
  </tr>
  <tr>
    <td align="center"><em>Behavior Sequencing GUI</em></td>
    <td align="center"><em>Unsupervised Discovery</em></td>
  </tr>
</table>
<p align="center"><em>Figure 1 — The two main modules of SBL.</em></p>

**SBL demo videos:**

<table align="center">
  <tr>
    <td><a href="https://www.youtube.com/watch?v=Ov9rxxtYbXk"><img src="https://img.youtube.com/vi/Ov9rxxtYbXk/hqdefault.jpg" height="200" alt="SBL walkthrough demo"></a></td>
    <td><a href="https://www.youtube.com/shorts/GEKvee0-Vvc"><img src="https://i.ytimg.com/vi/GEKvee0-Vvc/oar1.jpg" height="200" alt="SBL Shorts demo"></a></td>
    <td><a href="https://www.youtube.com/shorts/2IZIfpOn6xo"><img src="https://i.ytimg.com/vi/2IZIfpOn6xo/oar1.jpg" height="200" alt="SBL Shorts demo"></a></td>
  </tr>
</table>

---

## Table of Contents

1. [What You Need Before Starting](#1-what-you-need-before-starting)
2. [Installation](#2-installation)
3. [Launching the App](#3-launching-the-app)
4. [First Launch](#4-first-launch)
5. [How to Use](#5-how-to-use)
6. [Workflow Overview](#6-workflow-overview)
7. [Tabs Reference](#7-tabs-reference)
8. [SAM2 Models](#8-sam2-models)
9. [VideoPrism Backbone](#9-videoprism-backbone)
10. [GPU Memory Notes](#10-gpu-memory-notes)
11. [Keyboard Shortcuts](#11-keyboard-shortcuts)
12. [Directory Structure](#12-directory-structure)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What You Need Before Starting

| Requirement | Notes |
|---|---|
| **Operating System** | Linux (Ubuntu 20.04 or later recommended) |
| **GPU** | NVIDIA GPU with a CUDA 12-compatible driver **required**. 8 GB VRAM minimum; 12 GB+ comfortable for training. |
| **Python** | 3.10 or later |
| **Disk space** | ~10 GB free (SAM2 weights: ~3.5 GB, VideoPrism backbone: ~1 GB downloaded on first run, plus your experiment data) |
| **Internet** | Required on first launch only (to auto-download the VideoPrism backbone). After that the app works offline. |

To check your CUDA version:
```bash
nvidia-smi
```
Look for `CUDA Version: XX.X` in the top-right corner.

---

## 2. Installation

Install into a fresh virtual environment:

```bash
python -m venv sbl_env
source sbl_env/bin/activate
pip install singlebehaviorlab
```

This pulls in PyTorch (CUDA 12 build), JAX/Flax (CUDA 12), the vendored SAM2 fork, VideoPrism, and all other dependencies from PyPI in a single step. Installation takes 5–15 minutes and requires an NVIDIA GPU with a CUDA 12-compatible driver.

### Development install

Contributors working from a source checkout can use an editable install instead:

```bash
git clone https://github.com/alms93/SingleBehaviorLab
cd SingleBehaviorLab
pip install -e .
```

---

## 3. Launching the App

Activate the environment and run:

```bash
source sbl_env/bin/activate
singlebehaviorlab
```

Equivalent module form: `python -m singlebehaviorlab`.

### Headless / server use (CLI)

The same `singlebehaviorlab` command also runs as a headless CLI when a subcommand is supplied. Use it to run the GPU-heavy pipeline steps on a remote machine without opening the GUI:

```bash
# Train a classifier from an experiment directory
singlebehaviorlab train --experiment /path/to/my_experiment --profile balanced

# Run a trained model on a long video
singlebehaviorlab infer --experiment /path/to/my_experiment \
    --model /path/to/my_experiment/models/behavior_heads/model.pt \
    --video /data/recording.mp4 \
    --out /data/recording_inference.json

# Extract VideoPrism embeddings from a video + mask
singlebehaviorlab register --experiment /path/to/my_experiment \
    --video /data/recording.mp4 \
    --mask /data/recording_masks.h5 \
    --out /data/recording_matrix.npz

# Cluster an embedding matrix (loadable via the GUI's "Load Analysis State")
singlebehaviorlab cluster --matrix /data/recording_matrix.npz \
    --metadata /data/recording_matrix_metadata.npz \
    --out /data/recording_clusters.pkl

# Run SAM2 tracking from a prompts JSON exported in the Segmentation tab
singlebehaviorlab segment --video /data/recording.mp4 \
    --prompts /data/prompts.json \
    --out /data/recording_masks.h5
```

Run `singlebehaviorlab <command> --help` for the full flag list on each subcommand. The GUI-only steps (labeling, refinement review, cluster inspection) still require the graphical interface; the CLI covers the batch-processing steps where no human input is needed.

> **Full CLI reference:** [**CLI.md**](CLI.md) — detailed per-command docs, file-format reference, Python API, and troubleshooting.
>
> **Notebook demos:** [**demo/**](demo/) — two Jupyter notebooks walking through behavior sequencing and segmentation/clustering end-to-end. Drop your own demo video + prompts into `demo/data/` and step through the cells.

---

## 4. First Launch

On first launch, two things happen automatically:

1. **The VideoPrism backbone downloads** (~1 GB from Google DeepMind). This happens once and is cached for all future sessions. You need an internet connection for this step.

2. **A startup dialog appears** asking you to choose:
   - **Create New Experiment** — opens a dialog to name your experiment and choose a folder. The app creates a clean project directory with all required subfolders.
   - **Load Existing Experiment** — opens a file browser to load a `config.yaml` from a previous experiment.

Each experiment stores everything in one self-contained folder:
```
your_experiment/
├── config.yaml              # Experiment settings
├── data/
│   ├── raw_videos/          # Your input videos go here
│   ├── clips/               # Auto-extracted short clips
│   └── annotations/         # Label files (annotations.json)
└── models/
    └── behavior_heads/      # Your trained model files (.pt)
```

---

## 5. How to Use

Start with raw videos, extract short clips, label behaviors, train a model, run inference on new recordings, and iteratively refine with active learning.

> **▶ Full user guide:** [**HOWTOUSE.md**](HOWTOUSE.md) — step-by-step walkthrough (mouse-in-cage example), every tab explained in detail, unbiased discovery path, and practical tips.

---

## 6. Workflow Overview

SingleBehaviorLab has two complementary pipelines: a supervised pipeline that takes you from raw video to a trained behavior classifier, and an unsupervised discovery pipeline that uses segmentation, registration, and clustering to surface structure in your data. The two pipelines can be used independently, or clustering results can be fed back into labeling and training to refine the supervised model. Each step corresponds to a tab in the app.

### Supervised pipeline

```
Your raw video(s)
        │
        ▼
┌─────────────────────────────┐
│  1. Labeling                │  ← Assign behavior labels to clips
│     → outputs: annotations  │    (keyboard shortcuts 1–9)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. Training                │  ← Train a classifier on your labeled clips
│     → outputs: model (.pt)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. Inference               │  ← Run your model on new videos
│     → outputs: behavior     │    Generates the per-frame behavior sequence
│       sequence / timeline   │    (ethogram)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  4. Refinement              │  ← Review uncertain predictions, correct
│     → outputs: more labels  │    mistakes, and retrain for better accuracy
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5. Analysis                │  ← Ethograms, statistics, export videos
└─────────────────────────────┘
```

### Unsupervised discovery pipeline

Use this pipeline when you don't yet know what behaviors to label, or when you want to surface rare or unknown behaviors before training. It can be run stand-alone, or its outputs can be fed back into **Labeling → Training** to refine the supervised model.

```
Your raw video(s)
        │
        ▼
┌─────────────────────────────┐
│  A. Segmentation & Tracking │  ← SAM2 segments and tracks animals
│     → outputs: mask files   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  B. Registration            │  ← Crops around the animal, normalizes,
│     → outputs: embeddings   │    extracts VideoPrism features
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  C. Clustering              │  ← Groups embeddings by similarity
│     → outputs: clip groups  │    (UMAP + Leiden/HDBSCAN)
└────────────┬────────────────┘
             │
             │  feedback loop: pick representative clips from each
             │  cluster and feed them into Labeling → Training
             ▼
     Labeling (step 1 of the supervised pipeline)
```

---

## 7. Tabs Reference

*Supervised pipeline:*

### Labeling
- Browse clips in the left panel; click to preview in the video player
- Press **1–9** to assign a behavior class, or use the class buttons
- Press **Ctrl+S** to save; move to the next clip
- Frame-level labels can be drawn on the timeline for clips with mixed behaviors

### Training
- Select the behavior classes to train
- Choose a preset training profile or configure hyperparameters manually
- Click **Start Training** and monitor loss/accuracy in real time
- The best model checkpoint is saved automatically to `models/behavior_heads/`

### Inference
- Load a trained model (`.pt` file)
- Select a video to run predictions on
- The app outputs a color-coded per-frame behavior sequence (ethogram)
- Clips are ranked by prediction uncertainty for efficient review

### Refinement
- Review clips flagged as uncertain by the model
- Accept correct predictions or reassign labels
- Corrected clips are added to your annotation set
- Retrain from the Training tab to improve accuracy

### Analysis
- Generate ethograms (behavior-over-time plots)
- Compute per-class duration, frequency, and transitions
- Export annotated video overlays
- Export data tables (CSV) for statistical analysis

*Unsupervised discovery pipeline:*

### Segmentation & Tracking
- Load a raw video
- Click points on the animal(s) you want to track — SAM2 segments them automatically
- Adjust the segmentation mask if needed, then click **Track** to propagate across all frames
- Exports `.h5` mask files used by the Registration tab

### Registration
- Load your video + its mask file
- Configure cropping (box size) and normalization (CLAHE recommended)
- Click **Extract Embeddings** — VideoPrism processes each frame and saves feature vectors
- These embeddings feed directly into Clustering

### Clustering
- Load embeddings from Registration
- Run UMAP to reduce to 2D, then cluster with Leiden or HDBSCAN
- Visualize clusters interactively — each point is a short clip
- Select representative clips from each cluster and send them back to **Labeling → Training** to bootstrap or refine your supervised model

---

## 8. SAM2 Models

SingleBehaviorLab ships a locally modified build of [SAM2](https://github.com/facebookresearch/sam2) vendored inside the package. The modifications adjust the video predictor's memory handling so that long recordings (thousands of frames) can be segmented without exhausting GPU memory; the model weights and architecture are unchanged. All credit for the underlying model and checkpoints goes to the original SAM2 authors at Meta AI — see the upstream repository for the model cards and license.

SAM2 checkpoints download automatically on first use.

| Model | Size | Speed | Quality | Recommended for |
|---|---|---|---|---|
| SAM2.1 Tiny | 39M | Fastest | Good | Quick exploration, limited GPU memory |
| SAM2.1 Small | 46M | Fast | Better | General use |
| SAM2.1 Base+ | 80M | Moderate | High | Standard choice |
| SAM2.1 Large | 224M | Slowest | Best | High-quality tracking, multi-animal |

---

## 9. VideoPrism Backbone

[VideoPrism](https://github.com/google-deepmind/videoprism) (`videoprism_public_v1_base`) is the frozen video feature extractor at the core of SingleBehaviorLab.

- **Downloaded automatically on first launch** from Google DeepMind (~1 GB). Internet required once.
- Cached in `models/videoprism_backbone/` after first download.
- You do **not** need to interact with it directly — the app handles loading and inference.

---

## 10. GPU Memory Notes

SingleBehaviorLab runs two GPU frameworks simultaneously:

| Framework | Use | Memory allocation |
|---|---|---|
| JAX | VideoPrism backbone (feature extraction) | Capped at 45% of GPU VRAM |
| PyTorch | Classification head training and inference | Uses remaining ~55% |

- JAX grows memory on demand (no pre-allocation) to coexist with PyTorch.
- For a GPU with 8 GB VRAM: extraction and inference work comfortably. Training with large batch sizes may need batch size reduction.
- For 12 GB+ VRAM: all operations including batch training run without issues.
- If you get out-of-memory errors during training, reduce the batch size in the Training tab.

---

## 11. Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` – `9` | Assign behavior class (by position in class list) |
| `Space` | Play / pause video |
| `Ctrl+S` | Save current label |
| `Ctrl+O` | Open video file |
| `Ctrl+Q` | Quit application |

---

## 12. Directory Structure

After `pip install`, the application code lives in your Python environment and experiments are created in a folder of your choice. The source repository layout (useful for contributors):

```
SingleBehaviorLab/
├── pyproject.toml                   # Package metadata and dependencies
├── README.md
├── HOWTOUSE.md
│
├── singlebehaviorlab/               # Main package (all app code)
│   ├── __main__.py                  # Entry point for `singlebehaviorlab`
│   ├── backend/                     # Core ML and data processing
│   │   ├── model.py                 # VideoPrism + BehaviorClassifier head
│   │   ├── train.py                 # Training loop
│   │   ├── data_store.py            # Annotation file manager
│   │   ├── video_processor.py       # Mask-based video processing
│   │   ├── augmentations.py         # Data augmentation
│   │   ├── uncertainty.py           # Active-learning uncertainty scoring
│   │   └── video_utils.py           # Video I/O helpers
│   ├── gui/                         # PyQt6 interface
│   │   ├── main_window.py           # Main tabbed window
│   │   ├── labeling_widget.py       # Clip labeling
│   │   ├── training_widget.py       # Training UI
│   │   ├── inference_widget.py      # Inference and timeline
│   │   ├── review_widget.py         # Active-learning review
│   │   ├── analysis_widget.py       # Analysis and export
│   │   ├── segmentation_tracking_widget.py
│   │   ├── registration_widget.py
│   │   ├── clustering_widget.py
│   │   └── ...                      # Supporting widgets and helpers
│   ├── data/                        # Bundled config template and presets
│   └── licenses/                    # Third-party license notices (SAM2, VideoPrism)
│
├── third_party/                     # Vendored upstream code
│   ├── sam2_backend/                # Memory-optimized SAM2 fork
│   │   └── sam2/                    # Shipped as the `sam2` package
│   │                                # (upstream: facebookresearch/sam2)
│   └── videoprism_backend/
│       └── videoprism/              # Shipped as the `videoprism` package
│                                    # (upstream: google-deepmind/videoprism)
│
└── tests/
```

A typical experiment directory (created and managed by the app):
```
my_experiment/
├── config.yaml
├── data/
│   ├── raw_videos/                  # Your input videos
│   ├── clips/                       # Auto-extracted short clips
│   └── annotations/
└── models/
    └── behavior_heads/              # Trained classifier checkpoints
```

---

## 13. Troubleshooting

- **Out of memory during training.** Reduce **Batch Size** in the Training tab (try 8 or 4), or reduce **Clip Length**.
- **App window doesn't open (no display).** SingleBehaviorLab requires a graphical desktop and cannot run headless. For remote servers, use X11 forwarding: `ssh -X user@host` then `singlebehaviorlab`.
- **`nvidia-smi` shows a CUDA 11 driver.** The application requires a CUDA 12-compatible driver. Update the NVIDIA driver before installing.
- **PyPI resolution conflict between `torch` and `jax`.** The wheel pins `torch>=2.8` so its bundled cuDNN matches JAX's ABI. If an older `torch` is already installed in the environment, upgrade it: `pip install -U "torch>=2.8"`.
