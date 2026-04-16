"""Downloadable demo datasets for trying SingleBehaviorLab end-to-end.

Each entry in :data:`DEMOS` maps a short name to a list of ``(filename, url)``
pairs. :func:`load_demo` downloads the listed files into a local cache
directory and returns a dict mapping logical asset names to absolute paths,
so the pipeline functions can be called directly on the returned values::

    import singlebehaviorlab as sbl

    demo = sbl.load_demo("segmentation_clustering")
    sbl.segment(demo["video"], demo["prompts"], "masks.h5")

The asset URLs are pinned to a released tag so existing notebooks keep
working when ``main`` moves forward.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["DEMOS", "DemoAsset", "load_demo"]


_TAG = "v2.1.0"
_RAW_BASE = f"https://raw.githubusercontent.com/alms93/SingleBehaviorLab/{_TAG}"


@dataclass(frozen=True)
class DemoAsset:
    """One file that belongs to a demo dataset."""

    key: str              # logical name, e.g. "video" or "prompts"
    filename: str         # on-disk filename under the cache directory
    url: str              # remote URL to download from


DEMOS: dict[str, list[DemoAsset]] = {
    "segmentation_clustering": [
        DemoAsset(
            key="video",
            filename="Demo_video.mp4",
            url=f"{_RAW_BASE}/demo/data/segmentation_clustering/Demo_video.mp4",
        ),
        DemoAsset(
            key="prompts",
            filename="sam2_prompts.json",
            url=f"{_RAW_BASE}/demo/data/segmentation_clustering/sam2_prompts.json",
        ),
    ],
}


def _default_cache_dir() -> Path:
    override = os.environ.get("SBL_DEMO_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "singlebehaviorlab" / "demos"


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None  # type: ignore[assignment]

    if _tqdm is None:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f)
    else:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length") or 0) or None
            with _tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                leave=False,
            ) as bar:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 64)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
    tmp_path.replace(dest)


def load_demo(
    name: str = "segmentation_clustering",
    *,
    destination: Optional[str | os.PathLike[str]] = None,
    force: bool = False,
) -> dict[str, str]:
    """Download (or reuse a cached copy of) a demo dataset.

    Args:
        name: Registered demo name. Currently ``"segmentation_clustering"``
            is the only entry; more will follow as additional demos land.
        destination: Optional override for the cache directory. Defaults to
            ``$SBL_DEMO_DIR`` if set, otherwise
            ``~/.cache/singlebehaviorlab/demos/``.
        force: Re-download even if the files already exist locally.

    Returns:
        A dict mapping each asset's ``key`` to an absolute path on disk, e.g.
        ``{"video": "/root/.cache/.../Demo_video.mp4", "prompts": "..."}``.
    """
    if name not in DEMOS:
        available = ", ".join(sorted(DEMOS.keys()))
        raise KeyError(f"Unknown demo '{name}'. Available demos: {available}")

    cache_root = Path(destination).expanduser().resolve() if destination else _default_cache_dir()
    demo_dir = cache_root / name
    demo_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for asset in DEMOS[name]:
        target = demo_dir / asset.filename
        if force or not target.exists() or target.stat().st_size == 0:
            _download_with_progress(asset.url, target)
        paths[asset.key] = str(target)
    return paths
