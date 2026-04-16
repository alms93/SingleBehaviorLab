"""SingleBehaviorLab — behavioral video annotation and training tool.

A short Python API re-exports the backend pipeline functions at the package
top level, so the typical usage is::

    import singlebehaviorlab as sbl

    sbl.segment(video, prompts, out_masks)
    sbl.register(video, out_masks, out_matrix)
    sbl.cluster(out_matrix, out_pkl, metadata_path=out_metadata)
    sbl.plot_umap_clusters(out_pkl, show=True, save="umap.pdf")

    sbl.train(experiment_dir, profile="balanced")
    sbl.infer(model, video, out_json, experiment_dir=experiment_dir)

The re-exports use lazy attribute loading (PEP 562 ``__getattr__``) so that
``import singlebehaviorlab`` does not pull in torch, jax, tensorflow, sam2,
or videoprism. Each symbol triggers its underlying backend module only on
first access.
"""

__version__ = "2.1.0"
__author__ = "Almir Aljovic"

# Mapping of public name → (backend module, attribute name).
_PUBLIC_API = {
    "segment": ("singlebehaviorlab.backend.segmentation", "run_sam2_segmentation"),
    "load_prompts_json": ("singlebehaviorlab.backend.segmentation", "load_prompts_json"),
    "save_prompts_json": ("singlebehaviorlab.backend.segmentation", "save_prompts_json"),
    "register": ("singlebehaviorlab.backend.registration", "run_registration"),
    "RegistrationParams": ("singlebehaviorlab.backend.registration", "RegistrationParams"),
    "cluster": ("singlebehaviorlab.backend.clustering", "run_clustering"),
    "ClusteringParams": ("singlebehaviorlab.backend.clustering", "ClusteringParams"),
    "plot_umap_clusters": ("singlebehaviorlab.backend.clustering", "plot_umap_clusters"),
    "infer": ("singlebehaviorlab.backend.inference", "run_inference_on_video"),
    "train": ("singlebehaviorlab.backend.training_runner", "run_training_session"),
    "load_config": ("singlebehaviorlab.config", "load_config"),
}

__all__ = ["__version__", "__author__", *sorted(_PUBLIC_API)]


def __getattr__(name):
    try:
        module_name, attr = _PUBLIC_API[name]
    except KeyError as exc:
        raise AttributeError(f"module 'singlebehaviorlab' has no attribute {name!r}") from exc
    import importlib
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals().keys()) + list(_PUBLIC_API.keys())))
