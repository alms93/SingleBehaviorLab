"""Smoke tests for clustering logic (no display server required)."""

import os
import inspect
import numpy as np
import pandas as pd
import pytest


class TestClusteringImports:
    def test_clustering_widget_importable(self):
        try:
            from singlebehaviorlab.gui.clustering_widget import (
                ClusteringWidget,
                ClusteringWorker,
                LoadDataWorker,
            )
            assert ClusteringWidget is not None
            assert ClusteringWorker is not None
            assert LoadDataWorker is not None
        except ImportError as e:
            if "PyQt6" in str(e):
                pytest.skip("PyQt6 not available in headless env")
            raise

    def test_clustering_widget_init_signature(self):
        try:
            from singlebehaviorlab.gui.clustering_widget import ClusteringWidget
        except ImportError:
            pytest.skip("PyQt6 not available")
        sig = inspect.signature(ClusteringWidget.__init__)
        assert "config" in sig.parameters

    def test_load_data_worker_signature(self):
        try:
            from singlebehaviorlab.gui.clustering_widget import LoadDataWorker
        except ImportError:
            pytest.skip("PyQt6 not available")
        sig = inspect.signature(LoadDataWorker.__init__)
        params = list(sig.parameters.keys())
        assert "matrix_path" in params
        assert "metadata_path" in params


class TestClusteringDependencies:
    """Verify all clustering scientific dependencies are importable."""

    def test_umap_import(self):
        import umap
        assert hasattr(umap, "UMAP")

    def test_leidenalg_import(self):
        import leidenalg as la
        assert hasattr(la, "find_partition")

    def test_igraph_import(self):
        import igraph as ig
        g = ig.Graph(n=5, edges=[(0, 1), (1, 2)])
        assert g.vcount() == 5

    def test_hdbscan_import(self):
        import hdbscan
        assert hasattr(hdbscan, "HDBSCAN")

    def test_plotly_import(self):
        import plotly.express as px
        assert callable(px.scatter)


class TestClusteringLogic:
    """Test clustering algorithms directly without the GUI widget."""

    def test_leiden_clustering_on_synthetic_data(self):
        from sklearn.neighbors import kneighbors_graph
        import leidenalg as la
        import igraph as ig

        rng = np.random.RandomState(42)
        cluster_a = rng.randn(30, 10) + 5
        cluster_b = rng.randn(30, 10) - 5
        data = np.vstack([cluster_a, cluster_b])

        knn = kneighbors_graph(data, n_neighbors=5, mode="connectivity", include_self=False)
        sources, targets = knn.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist()))
        g = ig.Graph(n=data.shape[0], edges=edges, directed=False)
        partition = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=1.0)
        clusters = np.array(partition.membership)

        assert len(clusters) == 60
        assert len(set(clusters)) >= 2

    def test_hdbscan_clustering_on_synthetic_data(self):
        import hdbscan

        rng = np.random.RandomState(42)
        cluster_a = rng.randn(50, 5) + 10
        cluster_b = rng.randn(50, 5) - 10
        data = np.vstack([cluster_a, cluster_b])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
        labels = clusterer.fit_predict(data)

        assert len(labels) == 100
        assert len(set(labels) - {-1}) >= 2  # at least 2 real clusters

    def test_umap_embedding_shape(self):
        import umap

        rng = np.random.RandomState(42)
        data = rng.randn(50, 20)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)

        assert embedding.shape == (50, 2)

    def test_umap_3d_embedding(self):
        import umap

        rng = np.random.RandomState(42)
        data = rng.randn(50, 20)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=3, random_state=42)
        embedding = reducer.fit_transform(data)

        assert embedding.shape == (50, 3)


class TestDataLoading:
    """Test data loading formats without the GUI."""

    def test_load_csv_matrix(self, tmp_path):
        df = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
        path = str(tmp_path / "matrix.csv")
        df.to_csv(path, index=True)
        loaded = pd.read_csv(path, index_col=0)
        assert loaded.shape == (10, 5)

    def test_load_npz_matrix(self, tmp_path):
        matrix = np.random.randn(10, 5).astype(np.float32)
        feature_names = np.array([f"f{i}" for i in range(5)])
        snippet_ids = np.array([f"clip_{i:06d}" for i in range(10)])
        path = str(tmp_path / "matrix.npz")
        np.savez(path, matrix=matrix, feature_names=feature_names, snippet_ids=snippet_ids)

        with np.load(path, allow_pickle=True) as data:
            loaded_matrix = data["matrix"]
            loaded_features = data["feature_names"]
            loaded_ids = data["snippet_ids"]

        assert loaded_matrix.shape == (10, 5)
        assert len(loaded_features) == 5
        assert len(loaded_ids) == 10

    def test_load_npz_metadata(self, tmp_path):
        metadata_values = np.array([["video1.mp4", "0"], ["video1.mp4", "8"]])
        columns = np.array(["source_video", "start_frame"])
        path = str(tmp_path / "metadata.npz")
        np.savez(path, metadata=metadata_values, columns=columns)

        with np.load(path, allow_pickle=True) as data:
            df = pd.DataFrame(data["metadata"], columns=list(data["columns"]))

        assert df.shape == (2, 2)
        assert list(df.columns) == ["source_video", "start_frame"]
