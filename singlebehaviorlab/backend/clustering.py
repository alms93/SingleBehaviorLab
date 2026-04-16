"""Headless UMAP + Leiden/HDBSCAN clustering for the CLI.

Produces a pickle file that the GUI Clustering tab loads via the existing
"Load Analysis State" action. The state schema matches
``clustering_widget._save_analysis_state``.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

__all__ = ["ClusteringParams", "run_clustering", "plot_umap_clusters"]


@dataclass
class ClusteringParams:
    """Knobs for ``run_clustering``. Defaults mirror the GUI sliders."""

    method: str = "leiden"
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    normalization: str = "standard"
    subtract_video_mean: bool = False
    temporal_derivative: bool = False
    leiden_resolution: float = 1.0
    leiden_k: int = 15
    min_cluster_size: int = 10
    min_samples: int = 5
    hdbscan_epsilon: float = 0.0


def _load_matrix_metadata(
    matrix_path: str,
    metadata_path: Optional[str],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if matrix_path.endswith(".npz"):
        npz = np.load(matrix_path, allow_pickle=True)
        matrix = npz["matrix"]
        feature_names = npz["feature_names"]
        snippet_ids = npz["snippet_ids"] if "snippet_ids" in npz else npz.get("span_ids")
        if snippet_ids is None:
            snippet_ids = np.array([f"snippet{i + 1}" for i in range(matrix.shape[1])])
        matrix_df = pd.DataFrame(matrix, index=feature_names, columns=snippet_ids)
    elif matrix_path.endswith(".parquet"):
        matrix_df = pd.read_parquet(matrix_path, engine="pyarrow")
    else:
        matrix_df = pd.read_csv(matrix_path, index_col=0)

    metadata_df: Optional[pd.DataFrame] = None
    if metadata_path:
        if metadata_path.endswith(".npz"):
            meta_npz = np.load(metadata_path, allow_pickle=True)
            metadata_df = pd.DataFrame(meta_npz["metadata"], columns=meta_npz["columns"])
        elif metadata_path.endswith(".parquet"):
            metadata_df = pd.read_parquet(metadata_path, engine="pyarrow")
        else:
            metadata_df = pd.read_csv(metadata_path)
    return matrix_df, metadata_df


def _normalize(X: pd.DataFrame, method: str) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    method = method.lower()
    if method == "standard":
        from sklearn.preprocessing import StandardScaler
        arr = StandardScaler().fit_transform(X)
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        arr = MinMaxScaler().fit_transform(X)
    elif method == "l2":
        from sklearn.preprocessing import Normalizer
        arr = Normalizer(norm="l2").fit_transform(X)
    elif method in ("none", "raw"):
        arr = X.values
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def _run_umap(
    data: pd.DataFrame,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )
    return reducer.fit_transform(data.values)


def _run_leiden(data: pd.DataFrame, leiden_k: int, resolution: float) -> np.ndarray:
    from sklearn.neighbors import kneighbors_graph
    import igraph as ig
    import leidenalg as la

    knn = kneighbors_graph(data.values, n_neighbors=leiden_k, mode="connectivity", include_self=False)
    sources, targets = knn.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))
    graph = ig.Graph(n=data.shape[0], edges=edges, directed=False)
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )
    return np.array(partition.membership)


def _run_hdbscan(data: pd.DataFrame, params: ClusteringParams) -> np.ndarray:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        cluster_selection_epsilon=params.hdbscan_epsilon,
    )
    return clusterer.fit_predict(data.values)


def run_clustering(
    matrix_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    metadata_path: Optional[str | os.PathLike[str]] = None,
    params: Optional[ClusteringParams] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Cluster a feature matrix and write a GUI-loadable analysis pickle.

    Returns the written pickle path.
    """
    params = params or ClusteringParams()

    matrix_path_str = str(Path(matrix_path).expanduser().resolve())
    metadata_path_str: Optional[str] = None
    if metadata_path:
        metadata_path_str = str(Path(metadata_path).expanduser().resolve())

    output_path_obj = Path(output_path).expanduser().resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    _log(f"Loading matrix: {matrix_path_str}")
    matrix_df, metadata_df = _load_matrix_metadata(matrix_path_str, metadata_path_str)
    _log(f"Matrix shape: {matrix_df.shape[0]} features × {matrix_df.shape[1]} samples")

    X = matrix_df.T
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if params.subtract_video_mean and metadata_df is not None:
        group_col = None
        for col in ("group", "video_id"):
            if col in metadata_df.columns:
                group_col = col
                break
        snippet_col = "snippet" if "snippet" in metadata_df.columns else None
        if group_col and snippet_col:
            for grp in metadata_df[group_col].unique():
                grp_snippets = metadata_df.loc[metadata_df[group_col] == grp, snippet_col].values
                mask = X.index.isin(grp_snippets)
                if mask.sum() > 1:
                    X.loc[mask] -= X.loc[mask].mean(axis=0)
            _log("Applied per-video mean subtraction")

    if params.temporal_derivative and metadata_df is not None:
        group_col = None
        for col in ("group", "video_id"):
            if col in metadata_df.columns:
                group_col = col
                break
        snippet_col = "snippet" if "snippet" in metadata_df.columns else None
        if group_col and snippet_col:
            delta_rows = []
            delta_index = []
            for grp in metadata_df[group_col].unique():
                grp_snippets = metadata_df.loc[metadata_df[group_col] == grp, snippet_col].values
                grp_mask = X.index.isin(grp_snippets)
                grp_data = X.loc[grp_mask]
                if len(grp_data) < 2:
                    continue
                vals = grp_data.values
                delta_rows.append(vals[1:] - vals[:-1])
                delta_index.extend(grp_data.index[1:].tolist())
            if delta_rows:
                X = pd.DataFrame(np.concatenate(delta_rows, axis=0), index=delta_index, columns=X.columns)
                _log(f"Applied temporal derivatives ({len(X)} samples)")

    processed = _normalize(X, params.normalization)


    _log(f"Processed shape: {processed.shape} (samples × features)")

    _log(
        f"Running UMAP (n_neighbors={params.n_neighbors}, "
        f"min_dist={params.min_dist}, n_components={params.n_components})"
    )
    embedding = _run_umap(
        processed,
        n_components=params.n_components,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
    )

    if params.method == "leiden":
        _log(f"Running Leiden clustering (k={params.leiden_k}, resolution={params.leiden_resolution})")
        clusters = _run_leiden(processed, params.leiden_k, params.leiden_resolution)
    elif params.method == "hdbscan":
        _log(
            f"Running HDBSCAN (min_cluster_size={params.min_cluster_size}, "
            f"min_samples={params.min_samples}, epsilon={params.hdbscan_epsilon})"
        )
        clusters = _run_hdbscan(processed, params)
    else:
        raise ValueError(f"Unknown clustering method: {params.method}")

    unique_clusters = sorted(set(int(c) for c in clusters))
    _log(f"Clusters found: {len(unique_clusters)} (labels: {unique_clusters})")

    snippet_to_clip_map: dict[str, str] = {}
    if metadata_df is not None and "clip_path" in metadata_df.columns:
        snippet_col = (
            "snippet"
            if "snippet" in metadata_df.columns
            else ("span_id" if "span_id" in metadata_df.columns else None)
        )
        if snippet_col is not None:
            for _, row in metadata_df.iterrows():
                snippet_id = str(row.get(snippet_col, "")).strip()
                clip_path_val = str(row.get("clip_path", "")).strip()
                if snippet_id and clip_path_val and os.path.exists(clip_path_val):
                    snippet_to_clip_map[snippet_id] = clip_path_val
        _log(f"Built snippet→clip map with {len(snippet_to_clip_map)} entries from metadata.")

    state = {
        "matrix_data": matrix_df,
        "metadata": metadata_df,
        "processed_data": processed,
        "embedding": embedding,
        "clusters": clusters,
        "selected_features": list(matrix_df.index),
        "snippet_to_clip_map": snippet_to_clip_map,
        "metadata_file_path": metadata_path_str,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "version": "1.0",
    }

    with open(output_path_obj, "wb") as f:
        pickle.dump(state, f)
    _log(f"Wrote analysis state: {output_path_obj}")
    return str(output_path_obj)


def plot_umap_clusters(
    state: Union[dict[str, Any], str, os.PathLike[str]],
    *,
    show: bool = False,
    save: Optional[str | os.PathLike[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    point_size: float = 6.0,
):
    """Plot the UMAP embedding produced by ``run_clustering``.

    Follows the scanpy ``sc.pl.umap`` convention: pass ``show=True`` to pop up
    an interactive window, pass ``save='foo.pdf'`` to write the figure to
    disk (format inferred from the extension). Both can be combined.

    Args:
        state: Either an analysis-state dict (with ``embedding`` and
            ``clusters`` keys) or the path to a ``.pkl`` file produced by
            ``run_clustering``.
        show: Call ``plt.show()`` to open an interactive window.
        save: Destination path for the rendered figure. PDF, PNG, and SVG
            are all supported.
        title: Optional plot title.
        figsize: Figure size in inches.
        point_size: Scatter marker size.

    Returns:
        The matplotlib ``Figure`` object so the caller can further customise
        or embed it.
    """
    import matplotlib
    if not show and matplotlib.get_backend().lower() not in {"agg", "pdf", "svg", "ps"}:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(state, (str, os.PathLike)):
        with open(state, "rb") as f:
            state = pickle.load(f)
    if not isinstance(state, dict):
        raise TypeError("state must be a dict or a path to a clustering .pkl file.")

    embedding = np.asarray(state.get("embedding"))
    clusters = np.asarray(state.get("clusters"))
    if embedding is None or embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError("Analysis state is missing a valid 2D+ UMAP embedding.")
    if clusters is None or len(clusters) != embedding.shape[0]:
        raise ValueError("Cluster labels are missing or the wrong length.")

    unique_labels = sorted({int(c) for c in clusters})
    non_noise = [c for c in unique_labels if c >= 0]
    cmap = plt.get_cmap("tab20", max(len(non_noise), 1))

    fig, ax = plt.subplots(figsize=figsize)
    noise_mask = clusters < 0
    if np.any(noise_mask):
        ax.scatter(
            embedding[noise_mask, 0],
            embedding[noise_mask, 1],
            s=point_size,
            color="lightgray",
            alpha=0.5,
            label="Noise",
            linewidths=0,
        )
    for i, c in enumerate(non_noise):
        mask = clusters == c
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            color=cmap(i % cmap.N),
            alpha=0.85,
            label=f"Cluster {c}",
            linewidths=0,
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"UMAP + {'Noise + ' if np.any(noise_mask) else ''}{len(non_noise)} clusters")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize="x-small", framealpha=0.85, markerscale=1.5)
    fig.tight_layout()

    if save:
        save_path = Path(save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig
