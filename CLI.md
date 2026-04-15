# CLI reference

SingleBehaviorLab ships with a command-line interface for the GPU-heavy pipeline steps. The interactive steps (labeling, refinement review, cluster inspection) stay in the GUI; everything else can run headlessly on a server without a display.

Running `singlebehaviorlab` with no subcommand still launches the graphical interface, so the CLI is additive — it never replaces the GUI, it complements it.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Workflow overview](#2-workflow-overview)
3. [Commands](#3-commands)
   - [train](#31-train)
   - [infer](#32-infer)
   - [register](#33-register)
   - [segment](#34-segment)
   - [cluster](#35-cluster)
4. [Common flags (every subcommand)](#4-common-flags-every-subcommand)
5. [Full pipeline example](#5-full-pipeline-example)
6. [Python API](#6-python-api)
7. [File formats](#7-file-formats)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick start

Install into a fresh virtual environment:

```bash
python -m venv sbl_env
source sbl_env/bin/activate
pip install singlebehaviorlab
```

List available commands:

```bash
singlebehaviorlab --help
```

Get detailed help for a specific command:

```bash
singlebehaviorlab train --help
singlebehaviorlab infer --help
singlebehaviorlab register --help
singlebehaviorlab segment --help
singlebehaviorlab cluster --help
```

---

## 2. Workflow overview

The CLI mirrors the GUI's pipeline, minus the interactive steps:

```
                  GUI (interactive, laptop)        CLI (batch, server)
  ┌──────────────────────────────────────┐  ┌──────────────────────────────┐
  │  Labeling          (label clips)     │  │  train     (train a model)   │
  │  Refinement review (correct clips)   │  │  infer     (ethogram JSON)   │
  │  Cluster review    (inspect points)  │  │  segment   (SAM2 masks)      │
  │  SAM2 point prompts (click objects)  │  │  register  (embeddings)      │
  │                                      │  │  cluster   (UMAP + Leiden)   │
  └──────────────────────────────────────┘  └──────────────────────────────┘
```

A typical hybrid workflow:

1. **Locally**, in the GUI, you label a small sample of clips, click the initial SAM2 prompts in the Segmentation tab, and export the prompts JSON via the **Export prompts** button.
2. **On the server**, you run `train` to train a model, `segment` to propagate SAM2 masks across hours of footage, `register` to extract VideoPrism embeddings, and `cluster` to compute UMAP + clustering.
3. **Back on the laptop**, you load the outputs in the GUI to review ethograms, inspect cluster points, refine annotations, and iterate.

Every CLI output format matches the corresponding GUI "Load …" action, so round-tripping between the two is seamless.

---

## 3. Commands

### 3.1 `train`

Train a behavior classifier from an experiment directory.

```bash
singlebehaviorlab train --experiment /path/to/experiment [options]
```

**Required inputs in the experiment directory:**
- `config.yaml` — experiment configuration
- `data/annotations/annotations.json` — labeled clips
- `data/clips/` — the clip video files referenced by the annotations

**Output:** a `.pt` checkpoint written to `models/behavior_heads/<output-name>.pt` with a sibling `<output-name>.pt.meta.json` metadata file.

**Key flags:**

| Flag | Meaning |
|---|---|
| `--experiment DIR` | Experiment directory (required). |
| `--config PATH` | Alternate `config.yaml` to override the one in the experiment directory. |
| `--profile NAME` | Training profile from `data/training_profiles.json` (e.g. `balanced`, `quick`, `precise`). |
| `--epochs N` | Number of training epochs (overrides the profile/config value). |
| `--batch-size N` | Mini-batch size. |
| `--lr X` | Classification head learning rate. |
| `--output-name NAME` | Basename for the saved `.pt` file (default: `model`). |
| `--resume CHECKPOINT` | Warm-start from an existing `.pt` checkpoint. |

**Example:**

```bash
singlebehaviorlab train \
    --experiment ~/experiments/mouse_openfield \
    --profile balanced \
    --epochs 50 \
    --output-name openfield_v1
```

**Scope note:** the CLI trainer covers the GUI's "train once with these hyperparameters" path. Auto-tune hyperparameter search and multi-run sweeps stay in the Training tab.

---

### 3.2 `infer`

Run a trained classifier on a single video and write a GUI-loadable results JSON.

```bash
singlebehaviorlab infer \
    --experiment /path/to/experiment \
    --model /path/to/model.pt \
    --video /path/to/video.mp4 \
    --out /path/to/results.json
```

**Output:** a JSON file containing `classes`, `parameters`, and `results[video_path]` with `predictions`, `confidences`, `clip_probabilities`, `clip_starts`, `total_frames`, `aggregated_frame_probs`, and `clip_frame_probabilities`.

Open the file in the GUI via **Inference tab → Load results** to apply interactive smoothing, Viterbi decoding, gap filling, and segment merging. The CLI writes raw model output only — those post-processing steps are still driven from the GUI.

**Key flags:**

| Flag | Meaning |
|---|---|
| `--experiment DIR` | Experiment directory, used to find `annotations.json` if the model lacks class metadata. |
| `--model PATH` | Trained `.pt` checkpoint (with sibling `.meta.json` when available). |
| `--video PATH` | Input video to run inference on. |
| `--out PATH` | Destination JSON. |
| `--target-fps N` | Target FPS for clip extraction (default: from model metadata). |
| `--clip-length N` | Frames per clip (default: from model metadata). |
| `--batch-size N` | Inference batch size (default: auto-scaled to free GPU memory). |
| `--save-arrays` | Also write a `.arrays.npz` sidecar with raw per-frame probability arrays. |

**Example:**

```bash
singlebehaviorlab infer \
    --experiment ~/experiments/mouse_openfield \
    --model ~/experiments/mouse_openfield/models/behavior_heads/openfield_v1.pt \
    --video ~/data/session_042.mp4 \
    --out ~/data/session_042_results.json
```

**Scope note:** the happy-path pipeline assumes softmax output without one-vs-rest calibration, localization sub-heads, or attention collection. Models that rely on those GUI-only features should be run from the Inference tab.

---

### 3.3 `register`

Extract VideoPrism embeddings from a video and its SAM2 mask.

```bash
singlebehaviorlab register \
    --video /path/to/video.mp4 \
    --mask /path/to/masks.h5 \
    --out /path/to/features_matrix.npz
```

**Output:**
- `features_matrix.npz` — feature matrix shaped `(features, snippets)`
- `features_metadata.npz` — companion metadata (written as a sibling file)
- `features_matrix_clips/` — intermediate clip videos; safe to delete after clustering

Both NPZ files can be loaded directly in the Clustering tab via **Load matrix/metadata**, or passed to `singlebehaviorlab cluster` on the same machine.

**Key flags:**

| Flag | Meaning |
|---|---|
| `--video PATH` | Input video. |
| `--mask PATH` | SAM2 mask HDF5 file. |
| `--out PATH` | Output matrix `.npz`. |
| `--backbone NAME` | VideoPrism backbone name (default: `videoprism_public_v1_base`). |
| `--clip-length N` | Frames per extracted clip (default: 16). |
| `--step-frames N` | Stride between consecutive clips (default: `clip-length / 2`, giving 50% overlap). |
| `--target-fps N` | Subsampling FPS (default: 12). |
| `--clahe` / `--no-clahe` | Toggle CLAHE contrast normalization on the extracted clips (default: on). |

**Metadata columns written:** `snippet`, `group`, `video_id`, `object_id`, `clip_index`, `start_frame`, `end_frame`, `clip_path`. The `clip_path` column is used by `cluster` to pre-populate the snippet→clip map so clicking a UMAP point in the GUI opens the underlying clip video.

**Example:**

```bash
singlebehaviorlab register \
    --video ~/data/session_042.mp4 \
    --mask ~/data/session_042_masks.h5 \
    --out ~/data/session_042_matrix.npz \
    --clip-length 8 \
    --target-fps 25
```

---

### 3.4 `segment`

Propagate SAM2 segmentation across a video using a prompts JSON.

```bash
singlebehaviorlab segment \
    --video /path/to/video.mp4 \
    --prompts /path/to/prompts.json \
    --out /path/to/masks.h5
```

Clicks can't happen in a terminal, so the CLI needs a prompts JSON produced by the GUI. In the Segmentation tab, add the point prompts interactively and click **Export prompts**. The resulting file encodes every `(frame_idx, obj_id, x, y, label)` tuple and is ready to ship to a server.

Long videos are processed in fixed-size 200-frame chunks so the memory-optimised SAM2 fork bundled with this project never has to hold the entire video in RAM.

**Key flags:**

| Flag | Meaning |
|---|---|
| `--video PATH` | Input video. |
| `--prompts PATH` | Point/box prompts JSON exported from the GUI. |
| `--out PATH` | Output HDF5 mask file. |
| `--model FILE` | SAM2 checkpoint filename: `sam2.1_hiera_tiny.pt`, `…small.pt`, `…base_plus.pt`, `…large.pt` (default: `large`). |
| `--start-frame N` | First frame to track (default: 0). |
| `--end-frame N` | Last frame exclusive (default: end of video). |

**Prompts JSON schema:**

```json
{
  "video_path": "original_video.mp4",
  "prompts": [
    {"frame_idx": 0, "obj_id": 1, "x": 476.0, "y": 332.0, "label": 1},
    {"frame_idx": 0, "obj_id": 2, "x": 120.0, "y": 240.0, "label": 1}
  ]
}
```

`label` follows SAM2 convention: `1` = foreground click, `0` = negative click. Multiple entries sharing the same `(frame_idx, obj_id)` are grouped into a single SAM2 prompt call.

**Example:**

```bash
singlebehaviorlab segment \
    --video ~/data/session_042.mp4 \
    --prompts ~/data/session_042_prompts.json \
    --out ~/data/session_042_masks.h5 \
    --end-frame 1000
```

---

### 3.5 `cluster`

Run UMAP dimensionality reduction followed by Leiden or HDBSCAN clustering, and save a file the Clustering tab can load directly.

```bash
singlebehaviorlab cluster \
    --matrix /path/to/features_matrix.npz \
    --metadata /path/to/features_metadata.npz \
    --out /path/to/analysis.pkl
```

**Output:** a `.pkl` that matches the Clustering tab's **Save Analysis State** format. Open it via **Clustering tab → Load Analysis State** and the UMAP scatter, clusters, metadata, and clip links appear immediately — no re-clustering on the laptop.

**Key flags:**

| Flag | Meaning |
|---|---|
| `--matrix PATH` | Feature matrix produced by `register`. |
| `--metadata PATH` | Companion metadata `.npz`. Enables the snippet→clip map for click-to-inspect in the GUI. |
| `--out PATH` | Destination `.pkl`. |
| `--method {leiden,hdbscan}` | Clustering algorithm (default: leiden). |
| `--n-components {2,3}` | UMAP output dimensionality (default: 2). |
| `--umap-neighbors N` | UMAP `n_neighbors` and Leiden `k` (default: 15). |
| `--umap-min-dist X` | UMAP `min_dist` parameter (default: 0.1). |
| `--leiden-resolution X` | Leiden resolution; lower values yield fewer, larger clusters (default: 1.0). |
| `--hdbscan-min-cluster-size N` | HDBSCAN `min_cluster_size` (default: 10). |
| `--plot-save PATH` | Render the UMAP scatter and save it. Format inferred from the extension (`.pdf`, `.png`, `.svg`). |
| `--plot-show` | Open the UMAP scatter in an interactive window (requires a display). |

**Example:**

```bash
singlebehaviorlab cluster \
    --matrix ~/data/session_042_matrix.npz \
    --metadata ~/data/session_042_metadata.npz \
    --out ~/data/session_042_analysis.pkl \
    --plot-save ~/data/session_042_umap.pdf \
    --leiden-resolution 0.7
```

---

## 4. Common flags (every subcommand)

Every CLI subcommand accepts these runtime flags:

| Flag | Default | Meaning |
|---|---|---|
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | `INFO` | Verbosity of the log stream written to stderr. |
| `--log-file PATH` | — | Duplicate log output to a file in addition to stderr. |
| `--no-progress` | off | Disable tqdm progress bars. Useful when piping logs on a server where carriage returns clutter the output. |

Exit codes: `0` on success, `1` for argument errors, `2` for runtime errors, `130` for SIGINT (Ctrl+C).

---

## 5. Full pipeline example

Running segment → register → cluster end-to-end as a shell script:

```bash
#!/bin/bash
set -euo pipefail

VIDEO=~/data/session_042.mp4
PROMPTS=~/data/session_042_prompts.json
PREFIX=~/data/session_042

singlebehaviorlab segment \
    --video "$VIDEO" \
    --prompts "$PROMPTS" \
    --out "${PREFIX}_masks.h5"

singlebehaviorlab register \
    --video "$VIDEO" \
    --mask "${PREFIX}_masks.h5" \
    --out "${PREFIX}_matrix.npz"

singlebehaviorlab cluster \
    --matrix "${PREFIX}_matrix.npz" \
    --metadata "${PREFIX}_metadata.npz" \
    --out "${PREFIX}_analysis.pkl" \
    --plot-save "${PREFIX}_umap.pdf"
```

The clustering result can then be opened in the GUI via **Clustering tab → Load Analysis State** for interactive inspection.

---

## 6. Python API

Every CLI command is a thin wrapper around a backend function, so the same pipeline is available from Python notebooks and scripts.

```python
from singlebehaviorlab.backend.training_runner import run_training_session
from singlebehaviorlab.backend.inference import run_inference_on_video
from singlebehaviorlab.backend.registration import run_registration, RegistrationParams
from singlebehaviorlab.backend.segmentation import (
    run_sam2_segmentation,
    load_prompts_json,
    save_prompts_json,
)
from singlebehaviorlab.backend.clustering import (
    run_clustering,
    ClusteringParams,
    plot_umap_clusters,
)
```

### Plotting a saved analysis file

```python
from singlebehaviorlab.backend.clustering import plot_umap_clusters

# Pop up an interactive window
plot_umap_clusters("session_042_analysis.pkl", show=True)

# Save a PDF with a custom title
plot_umap_clusters(
    "session_042_analysis.pkl",
    save="session_042_umap.pdf",
    title="Session 042 — open field",
    point_size=8,
)

# Or pass an in-memory state dict
import pickle
with open("session_042_analysis.pkl", "rb") as f:
    state = pickle.load(f)
fig = plot_umap_clusters(state, show=True, save="session_042_umap.pdf")
```

`plot_umap_clusters` returns the `matplotlib.figure.Figure` so callers can customise it further or embed it in a notebook. When `show=False` and no display is available, the function forces a non-interactive matplotlib backend, so it is safe to call over SSH without X forwarding.

---

## 7. File formats

| File | Produced by | Consumed by | Notes |
|---|---|---|---|
| `model.pt` + `model.pt.meta.json` | `train` | `infer`, GUI Inference tab | Classifier head weights + metadata (classes, clip length, resolution). |
| `results.json` + `results.arrays.npz` | `infer` | GUI Inference tab → Load results | Raw per-frame probabilities. Smoothing/Viterbi are applied in the GUI. |
| `masks.h5` | `segment`, GUI Segmentation tab | `register`, GUI Registration tab | SAM2 masks with per-object bounding boxes. |
| `matrix.npz` + `metadata.npz` | `register`, GUI Registration tab | `cluster`, GUI Clustering tab → Load matrix/metadata | VideoPrism feature matrix plus snippet metadata. |
| `analysis.pkl` | `cluster`, GUI Clustering tab → Save Analysis State | GUI Clustering tab → Load Analysis State | Matrix, embedding, cluster labels, snippet→clip map. |
| `prompts.json` | GUI Segmentation tab → Export prompts | `segment`, GUI Segmentation tab → Import prompts | Point/box clicks for SAM2 initialisation. |

---

## 8. Troubleshooting

**`Could not find clip file for snippet`** when clicking a UMAP point in the GUI.
Re-run `register` — earlier releases did not record absolute clip paths in the metadata, so the clustering pickle has an empty snippet→clip map. The current release writes a `clip_path` column on every metadata row and pre-populates the map automatically.

**`RuntimeError: ... Cannot allocate memory: ... allocate 94371840000 bytes`** during `segment`.
This comes from trying to load every video frame into a single tensor. The CLI now processes in 200-frame chunks to avoid this. If you still see the error, upgrade to the latest release and verify the chunked code path is running.

**`FileNotFoundError: SAM2 checkpoint 'sam2.1_hiera_large.pt' was not found`**.
Launch the GUI once to trigger the automatic checkpoint download, or place the file manually in `~/SingleBehaviorLab/sam2_checkpoints/`.

**UMAP plot window does not open on a server.**
`--plot-show` needs an X display. Use `--plot-save output.pdf` instead when running headlessly; the rendered PDF is identical to the popup window.

**`torch` and `jax` cuDNN ABI mismatch.**
The packaged dependency set pins `torch>=2.8` so its bundled cuDNN is compatible with JAX 0.6.2. Upgrade your environment: `pip install -U "torch>=2.8"`.

**Training produces very different results compared to the GUI.**
The CLI's training runner covers the single-run happy path only. Auto-tune hyperparameter search, multi-run sweeps, and embedding-based class balancing remain GUI-only. Either run those features from the Training tab or open an issue if you need them on the CLI.
