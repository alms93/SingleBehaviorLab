# Demo notebooks

Two Jupyter notebooks walking through SingleBehaviorLab's pipelines end-to-end via the Python API. They mirror what the CLI does but run inline so you can follow along and inspect every output as it's produced.

## Contents

| Notebook | What it covers |
|---|---|
| [`01_behavior_sequencing.ipynb`](01_behavior_sequencing.ipynb) | Train a behavior classifier from a labeled experiment directory and run inference on a new video. |
| [`02_segmentation_clustering.ipynb`](02_segmentation_clustering.ipynb) | SAM2 segmentation from point prompts → VideoPrism embedding extraction → UMAP + Leiden clustering, with an inline UMAP plot. |

## What you need to provide

Everything lives in the `demo/data/` folder. Drop the following files there before running the notebooks:

| File | Needed for | Description |
|---|---|---|
| `demo.mp4` | both notebooks | The video to process. Any reasonable length works — the segmentation + clustering notebook is tuned for the first ~1000 frames. |
| `sam2_prompts.json` | notebook 02 | Point prompts exported from the GUI Segmentation tab via **Export prompts**. |
| `experiment/` | notebook 01 | A labeled experiment directory (`config.yaml`, `data/annotations/annotations.json`, `data/clips/`). Create it in the GUI — *File → New Experiment* — and label a handful of clips before running the notebook. |

None of the demo data is checked into the repository; the notebooks assume you populate `demo/data/` yourself.

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
