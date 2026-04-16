"""Command-line interface for SingleBehaviorLab.

Running ``singlebehaviorlab`` with no subcommand launches the graphical
interface. Running it with a subcommand (``train``, ``infer``, ``register``,
``segment``, ``cluster``) runs that pipeline step headlessly from the terminal,
suitable for servers and batch jobs.
"""

import argparse
import logging
import signal
import sys

__all__ = ["main"]

logger = logging.getLogger("singlebehaviorlab")


def _add_common_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Log verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help="Also write log output to this file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (useful for non-TTY server logs).",
    )


def _configure_logging(args: argparse.Namespace) -> None:
    level = getattr(logging, getattr(args, "log_level", "INFO"))
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_file = getattr(args, "log_file", None)
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="singlebehaviorlab",
        description="SingleBehaviorLab — behavior sequencing and unsupervised discovery.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the graphical interface (default when no command is given).",
    )
    _add_common_runtime_flags(gui_parser)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a behavior classifier from an experiment directory.",
    )
    train_parser.add_argument(
        "--experiment", required=True, metavar="DIR",
        help="Experiment directory containing data/, models/, and config.yaml.",
    )
    train_parser.add_argument(
        "--config", metavar="PATH",
        help="Alternate config.yaml to override the one in the experiment directory.",
    )
    train_parser.add_argument(
        "--profile", metavar="NAME",
        help="Training profile name from data/training_profiles.json (e.g. balanced, quick, precise).",
    )
    train_parser.add_argument(
        "--resume", metavar="CHECKPOINT",
        help="Warm-start training from an existing .pt checkpoint (partial load if classes differ).",
    )
    train_parser.add_argument(
        "--epochs", type=int, metavar="N",
        help="Number of training epochs (overrides the profile/config value).",
    )
    train_parser.add_argument(
        "--batch-size", type=int, metavar="N",
        help="Mini-batch size (overrides the profile/config value).",
    )
    train_parser.add_argument(
        "--lr", type=float, metavar="X",
        help="Classification head learning rate (overrides the profile/config value).",
    )
    train_parser.add_argument(
        "--output-name", metavar="NAME",
        help="Basename (without extension) for the saved .pt checkpoint under models/behavior_heads/.",
    )
    _add_common_runtime_flags(train_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run a trained classifier on a video and write a GUI-loadable results JSON.",
    )
    infer_parser.add_argument(
        "--experiment", required=True, metavar="DIR",
        help="Experiment directory (used to locate classes via annotations.json if the model lacks metadata).",
    )
    infer_parser.add_argument(
        "--model", required=True, metavar="PATH",
        help="Trained classifier .pt checkpoint (with sibling .meta.json when available).",
    )
    infer_parser.add_argument(
        "--video", required=True, metavar="PATH",
        help="Input video file to run inference on.",
    )
    infer_parser.add_argument(
        "--out", required=True, metavar="PATH",
        help="Output JSON path. Loadable via the Inference tab's 'Load results' action.",
    )
    infer_parser.add_argument(
        "--target-fps", type=float, metavar="N",
        help="Target FPS for clip extraction (default: value stored in the model's metadata).",
    )
    infer_parser.add_argument(
        "--clip-length", type=int, metavar="N",
        help="Frames per clip (default: value stored in the model's metadata).",
    )
    infer_parser.add_argument(
        "--batch-size", type=int, metavar="N",
        help="Inference mini-batch size (default: auto-scaled to free GPU memory).",
    )
    infer_parser.add_argument(
        "--save-arrays", action="store_true",
        help="Also write a .arrays.npz sidecar with the raw per-frame probability arrays.",
    )
    _add_common_runtime_flags(infer_parser)

    register_parser = subparsers.add_parser(
        "register",
        help="Extract VideoPrism embeddings from a video + mask pair.",
    )
    register_parser.add_argument(
        "--video", required=True, metavar="PATH",
        help="Input video file.",
    )
    register_parser.add_argument(
        "--mask", required=True, metavar="PATH",
        help="HDF5 mask file produced by `singlebehaviorlab segment` (or the GUI Segmentation tab).",
    )
    register_parser.add_argument(
        "--out", required=True, metavar="PATH",
        help="Output matrix .npz. A sibling _metadata.npz is written next to it.",
    )
    register_parser.add_argument(
        "--backbone", metavar="NAME", default="videoprism_public_v1_base",
        help="VideoPrism backbone identifier (default: videoprism_public_v1_base).",
    )
    register_parser.add_argument(
        "--clip-length", type=int, metavar="N",
        help="Number of frames per extracted clip (default: 16).",
    )
    register_parser.add_argument(
        "--step-frames", type=int, metavar="N",
        help="Stride between consecutive clips in frames (default: clip-length / 2).",
    )
    register_parser.add_argument(
        "--target-fps", type=float, metavar="N",
        help="Subsampling FPS for clip extraction (default: 12).",
    )
    register_parser.add_argument(
        "--clahe", dest="clahe", action="store_true", default=None,
        help="Apply CLAHE contrast normalization to extracted clips (default: on).",
    )
    register_parser.add_argument(
        "--no-clahe", dest="clahe", action="store_false", default=None,
        help="Disable CLAHE contrast normalization.",
    )
    register_parser.add_argument(
        "--flip-invariant", action="store_true",
        help="Average original + horizontally flipped embeddings to remove facing-direction bias. 2x extraction time.",
    )
    _add_common_runtime_flags(register_parser)

    segment_parser = subparsers.add_parser(
        "segment",
        help="Run SAM2 tracking on a video using a saved prompts JSON file.",
    )
    segment_parser.add_argument(
        "--video", required=True, metavar="PATH",
        help="Input video file to segment.",
    )
    segment_parser.add_argument(
        "--prompts", required=True, metavar="PATH",
        help="Point/box prompts JSON exported from the GUI Segmentation tab.",
    )
    segment_parser.add_argument(
        "--out", required=True, metavar="PATH",
        help="Output HDF5 mask file. Loadable in the GUI's Registration tab.",
    )
    segment_parser.add_argument(
        "--model", metavar="FILE", default="sam2.1_hiera_large.pt",
        help=(
            "SAM2 checkpoint filename under sam2_checkpoints/. One of: "
            "sam2.1_hiera_tiny.pt, sam2.1_hiera_small.pt, sam2.1_hiera_base_plus.pt, "
            "sam2.1_hiera_large.pt (default: large)."
        ),
    )
    segment_parser.add_argument(
        "--start-frame", type=int, metavar="N",
        help="First frame to track (default: 0).",
    )
    segment_parser.add_argument(
        "--end-frame", type=int, metavar="N",
        help="Last frame (exclusive) to track (default: end of video).",
    )
    _add_common_runtime_flags(segment_parser)

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Run UMAP + Leiden/HDBSCAN on an embedding matrix and write a GUI-loadable analysis file.",
    )
    cluster_parser.add_argument(
        "--matrix", required=True, metavar="PATH",
        help="Feature matrix .npz produced by `register`.",
    )
    cluster_parser.add_argument(
        "--metadata", metavar="PATH",
        help="Companion metadata .npz (enables the snippet→clip map for UMAP click-to-inspect).",
    )
    cluster_parser.add_argument(
        "--out", required=True, metavar="PATH",
        help="Output .pkl loadable via 'Load Analysis State' in the Clustering tab.",
    )
    cluster_parser.add_argument(
        "--method", choices=("leiden", "hdbscan"), default="leiden",
        help="Clustering algorithm to run on the UMAP embedding (default: leiden).",
    )
    cluster_parser.add_argument(
        "--n-components", type=int, default=2, choices=(2, 3),
        help="UMAP output dimensionality (default: 2).",
    )
    cluster_parser.add_argument(
        "--umap-neighbors", type=int, metavar="N",
        help="UMAP n_neighbors and Leiden k (default: 15).",
    )
    cluster_parser.add_argument(
        "--umap-min-dist", type=float, metavar="X",
        help="UMAP min_dist parameter (default: 0.1).",
    )
    cluster_parser.add_argument(
        "--leiden-resolution", type=float, metavar="X",
        help="Leiden resolution; lower values give fewer, larger clusters (default: 1.0).",
    )
    cluster_parser.add_argument(
        "--hdbscan-min-cluster-size", type=int, metavar="N",
        help="HDBSCAN min_cluster_size (default: 10).",
    )
    cluster_parser.add_argument(
        "--plot-save", metavar="PATH",
        help="Render the UMAP scatter and save it. Format inferred from extension (.pdf, .png, .svg).",
    )
    cluster_parser.add_argument(
        "--plot-show", action="store_true",
        help="Open the UMAP scatter in an interactive window (requires a display).",
    )
    _add_common_runtime_flags(cluster_parser)

    return parser


def _not_yet_implemented(command: str) -> int:
    logger.error("The `%s` subcommand will land in a later 2.1.0 step.", command)
    return 2


def _progress_bar(total: int, desc: str, disable: bool):
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, disable=disable, leave=True)


def cmd_train(args: argparse.Namespace) -> int:
    from singlebehaviorlab.backend.training_runner import run_training_session

    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
        overrides["classification_lr"] = args.lr

    bar = {"pbar": None}

    def log_fn(msg: str) -> None:
        logger.info(msg)

    def progress_cb(current: int, total: int) -> None:
        if bar["pbar"] is None:
            bar["pbar"] = _progress_bar(total, "training", disable=args.no_progress)
        pbar = bar["pbar"]
        if pbar is not None:
            pbar.n = current
            pbar.total = total
            pbar.refresh()
            if current >= total:
                pbar.close()

    try:
        result = run_training_session(
            args.experiment,
            config_override=args.config,
            profile=args.profile,
            cli_overrides=overrides,
            output_name=args.output_name,
            log_fn=log_fn,
            progress_callback=progress_cb,
        )
    finally:
        if bar["pbar"] is not None:
            bar["pbar"].close()

    logger.info("Training complete.")
    logger.info("Checkpoint: %s", result.get("output_path", "<unknown>"))
    best_val = result.get("best_val_acc")
    best_f1 = result.get("best_val_f1")
    if best_val is not None:
        logger.info("Best val accuracy: %.4f", float(best_val))
    if best_f1 is not None:
        logger.info("Best val F1: %.4f", float(best_f1))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    from singlebehaviorlab.backend.registration import RegistrationParams, run_registration

    params = RegistrationParams(
        backbone_model=args.backbone,
    )
    if args.clip_length is not None:
        params.clip_length_frames = args.clip_length
    if args.step_frames is not None:
        params.step_frames = args.step_frames
    if args.target_fps is not None:
        params.target_fps = int(args.target_fps)
    if args.clahe is False:
        params.normalization_method = "None"
    if args.flip_invariant:
        params.flip_invariant = True

    bar = {"pbar": None}

    def log_fn(msg: str) -> None:
        logger.info(msg)

    def progress_cb(current: int, total: int) -> None:
        if bar["pbar"] is None:
            bar["pbar"] = _progress_bar(total, "register", disable=args.no_progress)
        pbar = bar["pbar"]
        if pbar is not None:
            pbar.n = current
            pbar.total = total
            pbar.refresh()
            if current >= total:
                pbar.close()

    try:
        result = run_registration(
            args.video,
            args.mask,
            args.out,
            params=params,
            log_fn=log_fn,
            progress_callback=progress_cb,
        )
    finally:
        if bar["pbar"] is not None:
            bar["pbar"].close()

    logger.info("Registration complete.")
    logger.info("Matrix:   %s", result["matrix"])
    logger.info("Metadata: %s", result["metadata"])
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    from singlebehaviorlab.backend.clustering import (
        ClusteringParams,
        plot_umap_clusters,
        run_clustering,
    )

    params = ClusteringParams(
        method=args.method,
        n_components=args.n_components,
    )
    if args.umap_neighbors is not None:
        params.n_neighbors = args.umap_neighbors
        params.leiden_k = args.umap_neighbors
    if args.umap_min_dist is not None:
        params.min_dist = args.umap_min_dist
    if args.leiden_resolution is not None:
        params.leiden_resolution = args.leiden_resolution
    if args.hdbscan_min_cluster_size is not None:
        params.min_cluster_size = args.hdbscan_min_cluster_size

    def log_fn(msg: str) -> None:
        logger.info(msg)

    out = run_clustering(
        args.matrix,
        args.out,
        metadata_path=args.metadata,
        params=params,
        log_fn=log_fn,
    )
    logger.info("Clustering complete.")
    logger.info("Analysis state: %s", out)
    logger.info("Load it via the Clustering tab → 'Load Analysis State'.")

    if args.plot_save or args.plot_show:
        logger.info("Rendering UMAP scatter...")
        plot_umap_clusters(out, show=args.plot_show, save=args.plot_save)
        if args.plot_save:
            logger.info("Saved plot: %s", args.plot_save)
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from singlebehaviorlab.backend.inference import run_inference_on_video

    bar = {"pbar": None}

    def log_fn(msg: str) -> None:
        logger.info(msg)

    def progress_cb(current: int, total: int) -> None:
        if bar["pbar"] is None:
            bar["pbar"] = _progress_bar(total, "infer", disable=args.no_progress)
        pbar = bar["pbar"]
        if pbar is not None:
            pbar.n = current
            pbar.total = total
            pbar.refresh()
            if current >= total:
                pbar.close()

    try:
        out = run_inference_on_video(
            args.model,
            args.video,
            args.out,
            experiment_dir=args.experiment,
            target_fps=args.target_fps,
            clip_length=args.clip_length,
            batch_size=args.batch_size,
            save_arrays=args.save_arrays,
            log_fn=log_fn,
            progress_callback=progress_cb,
        )
    finally:
        if bar["pbar"] is not None:
            bar["pbar"].close()

    logger.info("Inference complete.")
    logger.info("Results: %s", out)
    logger.info("Load via the Inference tab's 'Load results' action to apply smoothing and decoding.")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    from singlebehaviorlab.backend.segmentation import run_sam2_segmentation

    bar = {"pbar": None}

    def log_fn(msg: str) -> None:
        logger.info(msg)

    def progress_cb(current: int, total: int) -> None:
        if bar["pbar"] is None:
            bar["pbar"] = _progress_bar(total, "segment", disable=args.no_progress)
        pbar = bar["pbar"]
        if pbar is not None:
            pbar.n = current
            pbar.total = total
            pbar.refresh()
            if current >= total:
                pbar.close()

    try:
        out = run_sam2_segmentation(
            args.video,
            args.prompts,
            args.out,
            model_name=args.model,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            log_fn=log_fn,
            progress_callback=progress_cb,
        )
    finally:
        if bar["pbar"] is not None:
            bar["pbar"].close()

    logger.info("Segmentation complete.")
    logger.info("Mask file: %s", out)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    command = args.command
    if command == "train":
        return cmd_train(args)
    if command == "infer":
        return cmd_infer(args)
    if command == "register":
        return cmd_register(args)
    if command == "segment":
        return cmd_segment(args)
    if command == "cluster":
        return cmd_cluster(args)
    logger.error("Unknown command: %s", command)
    return 1


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "gui"):
        from singlebehaviorlab.__main__ import run_gui_app
        run_gui_app()
        return

    _configure_logging(args)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

    try:
        exit_code = _run_command(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Command failed with an unhandled exception")
        sys.exit(2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
