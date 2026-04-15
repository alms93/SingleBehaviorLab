# How to Use SingleBehaviorLab

This guide walks through the app using a concrete example: **mice walking and moving in an open cage** (one or more animals, one or more videos). The same steps apply to other behaviors and species once you adapt class names and video setup.

For installation, GPU setup, and troubleshooting, see [README.md](README.md).

---

## Table of Contents

- [Before You Start](#before-you-start)
- **Part 1 — Behavior Sequencing**
  - [1. Labeling](#1-labeling)
  - [2. Training Sequencing Model](#2-training-sequencing-model)
  - [3. Sequencing (Inference)](#3-sequencing-inference)
  - [4. Refine](#4-refine)
  - [5. Downstream Analysis](#5-downstream-analysis)
- **Part 2 — Unbiased Discovery**
  - [6. Segmentation Tracking](#6-segmentation-tracking)
  - [7. Registration](#7-registration)
  - [8. Clustering](#8-clustering)
- [How the Two Parts Fit Together](#how-the-two-parts-fit-together)
- [Tips and Common Pitfalls](#tips-and-common-pitfalls)
- [Quick Reference](#quick-reference)

---

## Before You Start

1. **Launch the app** (from an environment with `singlebehaviorlab` installed):
   - `singlebehaviorlab`, or equivalently `python -m singlebehaviorlab`.
2. **Create or load an experiment** when prompted:
   - **Create New Experiment** — pick a folder and name; the app creates `config.yaml`, `data/`, `models/`, etc.
   - **Load Existing Experiment** — open a previous experiment's `config.yaml`.
3. **Put videos in the experiment** — copy recordings into `data/raw_videos/` (or use the app's file dialogs to open videos where they already live).

The tabs at the top are not strictly linear: you can jump to any tab. This document follows the **behavior sequencing** path first, then the **discovery** path (segmentation, registration, clustering).

---

## Part 1 — Behavior Sequencing

Start here with your **raw videos** — no pre-processed clips needed. The app extracts clips for you as the first step.

### Example scenario: mice in an open cage

- **Goal:** Recognize when the mouse is **walking** vs **standing still** vs **grooming** (add more classes as needed: **rearing**, **digging**, etc.).
- **Footage:** One or more continuous recordings across one or more animals and sessions. You can include multiple videos from multiple animals in the same experiment — the model learns behavior signatures that generalize across individuals.

---

### 1. Labeling

**Tab:** `Labeling`

#### Extract clips from your raw videos

This is always the first step — the app works with short clips, not full videos. Add one or more source video files and set:

- **Target FPS** — **12 fps recommended** for most behavior tasks.
- **Frames per clip** — **8 frames recommended**, giving ~0.67 s per clip at 12 fps. Increase for slow/sustained behaviors.
- **Step frames** — stride between clip start points; controls extraction density.
- **Max clips per video** — cap the number of clips from any single video (0 = unlimited). Useful for very long recordings where thousands of clips would be overwhelming.

Extracted clips are saved into `data/clips/`.

If you have **already run clustering** (Part 2), a **Labeling setup** dialog appears after clustering, offering to populate the clip list from cluster representatives.

#### Define behavior classes

Add class names — for example: `walking`, `standing`, `grooming`. You can add, rename, or remove classes at any time before training.

#### Label each clip

- Select a clip in the list; it plays in the viewer.
- Assign a class with **number keys `1`–`9`** or with the on-screen class buttons.
- Use **clip-level** labeling when the behavior is uniform. Use **per-frame** mode when a transition happens — for example, an 8-frame clip where the mouse walks in frames 1–4 then stops to groom in frames 5–8 can be labeled frame-by-frame as `walking` → `grooming`. These transition clips teach the model what behavior boundaries look like.
- Press **`Ctrl+S`** to save annotations.

#### Optional features in the Labeling tab

- **Clip list filters** — filter by video source, by class, or toggle **Show unlabeled only** / **Next unlabeled** to quickly find clips that still need labels.
- **Fullscreen labeling** — click the fullscreen button for a larger video view. Shortcuts: `A`/`D` (prev/next clip), `R` (random), `Q`/`E` (prev/next frame), `1`–`9` (assign class), `Space` (play/pause).
- **Zoom** — `+`/`-` buttons on the video viewer to inspect details.
- **Multi-label assignment** — click **Multi** to assign more than one behavior to a clip (e.g. a mouse that is simultaneously walking and sniffing). Used with One-vs-Rest (OvR) training mode.
- **Hard-negative round dataset** — advanced feature for building a curated training subset. Select target classes and "near-negative" distractor classes (behaviors that look similar but aren't the target, e.g. `near_negative_resting` for clips that resemble grooming). The app builds a balanced mix for training to sharpen the model on confusing cases.

#### Localization bbox (optional)

VideoPrism processes clips at **288×288 pixels**. For most setups this resolution is sufficient. If the camera is far from the animal and the mouse appears small in frame, subtle postures become harder to distinguish.

**Localization** addresses this: the model learns to find and crop around the animal, then classifies on the zoomed-in crop. To enable: check **Draw localization bbox**, draw a box around the animal, click **Save bbox**. The Localization section in the Training tab activates automatically once clips have bbox labels.

For a standard close-range or medium-range cage, try without localization first.

#### Aim for balance

Dozens of clips per class is a practical starting point. More helps when conditions vary across animals or sessions.

---

### 2. Training Sequencing Model

**Tab:** `Training Sequencing Model`

#### Core setup

1. **Paths** — confirm annotation file, clips directory, and output model path.
2. **Dataset info** — the panel summarizes labeled clips and class counts.
3. **Training profiles** — load a preset (e.g. `LowInputData`, `MoreInputData`, `LocalizationLowData`, `LocalizationMoreData`) or configure manually. Save your own profiles for reproducibility via **Advanced: Profiles**.
4. **Start Training** — click and watch the log. The best checkpoint is saved automatically.

#### Hyperparameters worth knowing

- **Epochs, batch size, learning rate** — defaults from the loaded profile are usually a good start.
- **Temporal decoder** — enables a multi-scale temporal convolutional network (MS-TCN) after spatial attention pooling. Improves frame-level accuracy by using temporal context across the clip. Leave on unless you have very few clips.
- **Validation split** — defaults to 20%. Check "Use all data for training" if your dataset is very small (<50 clips total).
- **Class selection / per-class limits** — train on a subset of classes, or cap how many clips per class are used (useful if one class has far more clips than others).

#### Optional features in the Training tab

- **Data augmentation** — "Use data augmentation" enables flips, color jitter, blur, noise, rotation, speed perturbation, random occlusion overlays, and lighting jitter. Configurable via the **Augmentation options** dialog. Recommended for small datasets.
- **OvR (One-vs-Rest)** — trains an independent binary classifier per class instead of a shared softmax. Useful when behaviors co-occur or when you have many classes with different difficulty levels. Found in the **OvR & Hard Mining** section.
- **Confusion-aware hard mining** — after early epochs, the sampler overweights clips the model gets wrong, focusing training on its weak points. Enabled in OvR & Hard Mining.
- **Fine-tune from pretrained** — load weights from a previously trained model (e.g. from another experiment or an earlier training round) instead of starting from scratch. Check **Fine-tune existing model** and browse to a `.pt` file.
- **Auto-tune** — automated hyperparameter search: the app tries N random configurations on a small number of epochs, then trains from scratch with the best. Requires a validation split.
- **Batch training** — check multiple profiles in **Advanced: Profiles**, then click **Batch Train** to run them sequentially and compare results in a CSV.
- **Training visualization** — click **Visualize training** to open a live monitor with loss curves, accuracy, macro F1, per-class F1, and localization metrics (IoU, center error) if applicable.

#### Localization

Activates only when annotations include saved bounding boxes. The model trains in two stages: first localization (finding the animal), then classification (on crops).

#### Resolution

Set in **`config.yaml`** (e.g. `resolution: 288`), not in the Training tab GUI.

---

### 3. Sequencing (Inference)

**Tab:** `Sequencing`

#### Core steps

1. **Load the trained model** — choose the `.pt` checkpoint.
2. **Select video(s)** — pick one or more recordings. The app processes them in batch.
3. **Match settings** — clip length, step, FPS, and resolution should match training (metadata from the model is used when possible).
4. **Run** — the app produces per-frame behavior predictions and an uncertainty report.

#### Optional features

- **Quick-check sampled inference** — process only a few evenly-spaced chunks of the video instead of the full recording. Set **Chunk duration** and **Number of chunks**. Good for fast sanity checks before committing to a full run.
- **Timeline postprocessing** — **Viterbi smoothing** reduces noisy frame-by-frame flickering (recommended); **merge short segments** cleans up tiny fragments; **per-class thresholds** let you tune sensitivity per behavior.
- **Clip popup** — click any segment on the behavior timeline to preview the clip, see per-class confidence bars, localization bbox overlays, and optionally correct the label or add the clip directly to your training set.
- **Attention heatmaps** — check **Collect attention maps** before running, then **Export attention heatmap** to produce a video showing which parts of the frame the model focuses on.

#### Exports

- **JSON** — full results bundle (auto-saved).
- **CSV / SVG** — behavior timeline segments.
- **Overlay video** — the recording with behavior labels burned onto each frame for visual review.

---

### 4. Refine

**Tab:** `Refine`

After Sequencing finishes, the app sends results to Refine for active-learning review.

#### Three review modes

- **Uncertain review** (default) — clips where the model was least confident, ranked per class.
- **Confident enrichment** — high-confidence clips to bulk-add training volume without much manual effort.
- **Transition mining** — clips near behavior boundaries (e.g. walking → grooming transitions), often the most informative for improving boundary detection.

#### Workflow

1. Select a **scope**: review across all videos or per-video.
2. Inspect each clip: see the model's prediction and per-class score bars.
3. **Accept** if correct, or **reassign** the true label. You can also **mark as hard negative** to flag confusing clips for focused training later.
4. **Save** accepted/corrected clips to your annotation set.
5. **Iterate** — return to Training, retrain, run Sequencing again. Repeat until accuracy is acceptable.

---

### 5. Downstream Analysis

**Tab:** `Downstream Analysis`

Use this after you have inference results to summarize and export.

#### Overview metrics

- **Occurrences** — how many bouts of each behavior.
- **Average bout duration** — typical length of each behavior episode.
- **Total duration / Percent time** — how much of the session each behavior occupies.
- Modes: **General Overview** (all data) or **Group Comparison** (compare experimental conditions, e.g. treated vs control).
- Graph types: bar, box, violin, strip, line.

#### Spatial distribution

Scatter plot of localization centroids — see where in the cage each behavior occurs. Useful for identifying location-specific behaviors (e.g. grooming mostly in one corner).

#### Transition analysis

- **Transition probability matrix** — table showing how likely each behavior is to follow each other.
- **Interactive transition graph** — circular or network layout; optionally filter to **significant transitions only** (Z > 1.96).

#### Statistics

- **Mann-Whitney** (2 groups) or **Kruskal-Wallis** (>2 groups) on the filtered data.

#### Exports

- Graphs: PDF, SVG, PNG, HTML.
- Data tables: CSV.

---

## Part 2 — Unbiased Discovery

Use this path to discover behavioral differences across animals and conditions **without assuming what to look for**. By segmenting and tracking animals across many videos, extracting standardized VideoPrism embeddings, and clustering them, you let the data reveal which behaviors exist and how they differ between individuals or experimental groups — before any manual labeling biases the analysis.

This is especially valuable when working with **many videos across multiple animals** (e.g. treated vs control groups, different genotypes, or longitudinal recordings). Segmentation isolates each animal consistently, registration normalizes appearance and framing, and clustering surfaces the natural structure of the behavioral repertoire. You can then label the discovered clusters rather than deciding upfront which behaviors matter.

**Tab order:** `Segmentation Tracking` → `Registration` → `Clustering` → then **Labeling** (to name the discovered clusters).

---

### 6. Segmentation Tracking

**Tab:** `Segmentation Tracking`

#### Core steps

1. **Load your cage video** — overhead or side view.
2. **Choose a SAM2 model** — **SAM2.1 Base+** is a good default. Larger models (SAM2.1 Large) give more precise masks but are slower; smaller models (Tiny/Small) are faster and fit on smaller GPUs.
3. **Prompt the animal** — scrub to a clear frame, click **positive points** on the mouse (and negative points on background clutter if needed). Use the **Object** selector and **+** button to track multiple animals with separate IDs.
4. **Track** — click **Run tracking** to propagate masks through the video.
5. **Export** — masks are saved as `.h5` files. Enable **Save overlay video** to get an MP4 with colored mask overlays for visual verification.

#### Optional features

- **Tracking resolution** — 256 (fastest) to 1024 (best quality). 512 is a good balance for centroid/bbox extraction without needing full-resolution mask precision.
- **Pause, add prompts, resume** — if the mask drifts mid-tracking, pause, scrub to the problem frame, add corrective positive/negative points, then resume from that frame.
- **Processing range** — limit tracking to a subset of frames (start/end) instead of the full video.
- **Motion-aware tracking** — Kalman-based quality scoring that filters low-quality frames from SAM2's memory, preventing drift from accumulating. Requires the `filterpy` package. Tune the **motion score threshold** and **frames before auto-correct**.
- **OC-SORT drift correction** — optional virtual trajectory correction for sustained drift.
- **Multi-video batch** — load multiple videos, run tracking on all sequentially.

---

### 7. Registration

**Tab:** `Registration`

1. **Video + mask pairs** — load raw videos and their segmentation mask files. Use naming conventions documented in-app so pairs are matched automatically.
2. **Crop and normalize** — set **box size**, **target size**, **background** handling (e.g. black, blur, or original), **normalization** (CLAHE recommended for contrast), **mask feather** (soft edge blending), and whether the ROI is **locked to the first frame** or updated per frame.
3. **Clips from video** — set **target FPS**, **clip length**, and **step** (use the same values you plan to use for labeling, e.g. 12 fps / 8 frames).
4. **Process** — outputs registered clips to `registered_clips/`.
5. **Extract embeddings** — runs VideoPrism on all registered clips, producing a **feature matrix** and **metadata** CSV for clustering.

---

### 8. Clustering

**Tab:** `Clustering`

1. **Load data** — load the embedding matrix and metadata produced by Registration.
2. **Preprocess** (optional) — feature selection, scaling, PCA.
3. **UMAP** — set neighbors, `min_dist`, 2D or 3D.
4. **Cluster** — choose **Leiden** (resolution, k) or **HDBSCAN** (min cluster size, epsilon).
5. **Explore** — click points in the interactive plot to inspect clips. Pick **representative clips** per cluster for labeling — ideal for discovering rare behaviors (e.g. short rears among long walking bouts).
6. **Connect to Labeling** — switch to the Labeling tab and the **Labeling setup** dialog offers to populate the clip list from your clusters.

---

## How the Two Parts Fit Together

```text
Optional discovery:  Segmentation → Registration → Clustering
                              ↓
Core sequencing:       Labeling → Training → Sequencing → Refine
                              ↑______________________________|
                                       (retrain loop)

Analysis:              Downstream Analysis (summaries, exports)
```

For **mice in an open cage**, a practical first project is:

1. Put raw videos in the experiment, open **Labeling**, **extract clips** (12 fps, 8 frames per clip), **define classes**, and **label**.
2. **Train** in **Training Sequencing Model** and run **Sequencing** on new cage videos.
3. **Refine** uncertain clips and retrain.
4. Add **Segmentation → Registration → Clustering** when you want to scale across long recordings or discover rare behaviors automatically.

---

## Tips and Common Pitfalls

- **Start small** — label 30–50 clips per class, train, run inference, then use **Refine** to grow the dataset iteratively. This is faster than labeling hundreds of clips upfront.
- **Confused classes?** — if the model struggles with two similar behaviors (e.g. resting vs grooming), use the **hard-negative round** in Labeling or **hard-pair mining** in Training to force the model to focus on distinguishing them.
- **Multi-animal videos** — when animals look similar, localization becomes more important so the model focuses on one animal at a time rather than ambiguous full-frame input.
- **Unstable training loss** — try loading a preset profile (e.g. `LowInputData`) and reducing the learning rate.
- **Save profiles** — always save your training configuration as a profile so experiments are reproducible.
- **Augmentation** — enable data augmentation for small datasets (<100 clips). It helps the model generalize across lighting, angle, and animal appearance variation.
- **Viterbi smoothing** — after inference, enable Viterbi in the timeline postprocessing. It dramatically reduces noisy frame-by-frame behavior flickering.
- **Quick checks first** — use **Quick-check sampled inference** to spot-check a few chunks of video before committing to a full inference run.

---

## Quick Reference

### Tab names

| Your goal | Tab |
|-----------|-----|
| Extract clips and assign labels | **Labeling** |
| Train the classifier | **Training Sequencing Model** |
| Run model on new videos | **Sequencing** |
| Fix uncertain predictions | **Refine** |
| Plots, stats, exports | **Downstream Analysis** |
| SAM2 masks + track | **Segmentation Tracking** |
| Crop + VideoPrism features | **Registration** |
| UMAP + Leiden / HDBSCAN | **Clustering** |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`–`9` | Assign behavior class |
| `Space` | Play / pause |
| `Q` / `E` | Previous / next frame |
| `A` / `D` | Previous / next clip (fullscreen) |
| `R` | Random clip (fullscreen) |
| `Ctrl+S` | Save annotation |
| `Ctrl+Q` | Quit |

### Experiment folder layout

```text
your_experiment/
├── config.yaml
├── data/
│   ├── raw_videos/
│   ├── clips/
│   └── annotations/annotations.json
├── models/behavior_heads/     # trained .pt checkpoints
├── registered_clips/          # after Registration (if used)
└── results/                   # inference outputs, uncertainty, etc.
```
