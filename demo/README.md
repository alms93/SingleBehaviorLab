# Demo notebooks

Two Jupyter notebooks walking through SingleBehaviorLab's pipelines end-to-end via the Python API. They mirror what the CLI does but run inline so you can follow along and inspect every output as it's produced.

## Contents

| Notebook | What it covers |
|---|---|
| [`01_behavior_sequencing.ipynb`](01_behavior_sequencing.ipynb) | Train a behavior classifier from a labeled experiment directory and run inference on a new video. |
| [`02_segmentation_clustering.ipynb`](02_segmentation_clustering.ipynb) | SAM2 segmentation from point prompts → VideoPrism embedding extraction → UMAP + Leiden clustering, with an inline UMAP plot. |

## What you need to provide

### Segmentation + clustering (notebook 02)

A silent demo video and matching prompts JSON are already bundled under `demo/data/segmentation_clustering/`, so notebook 02 runs out of the box. The default configuration processes the first 2000 frames (a few minutes on a single GPU); flip `FULL_VIDEO = True` in the first code cell to process the whole clip.

Replace the bundled files with your own `Demo_video.mp4` and `sam2_prompts.json` if you want to try a different recording. Point prompts are exported from the GUI Segmentation tab via the **Export prompts** button.

### Behavior sequencing (notebook 01)

Notebook 01 needs a labeled experiment directory, which cannot be pre-bundled because it depends on your class taxonomy. Create one in the app (*File → New Experiment*), label a handful of clips in the Labeling tab, and point the `EXPERIMENT` variable in the notebook at the resulting folder. A minimal sanity-check experiment with 30–50 labeled clips across two or three behavior classes is enough to see the pipeline end-to-end.

## Running the notebooks

Install SingleBehaviorLab and Jupyter into a virtual environment:

```bash
python -m venv sbl_env
source sbl_env/bin/activate
pip install singlebehaviorlab jupyterlab
```

Then launch Jupyter from the repo root:

```bash
jupyter lab demo/
```

Both notebooks use the Python API (`singlebehaviorlab.backend.*`) rather than shelling out to the CLI, so they can be stepped through cell-by-cell and their intermediate outputs inspected interactively. The CLI produces identical results — see **[CLI.md](../CLI.md)** for the equivalent one-shot invocations.
