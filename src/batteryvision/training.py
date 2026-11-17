"""Shared training utilities: seeding, cached split loading, threshold selection, run metadata."""
from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

from . import config
from .dataset import CLASSES, load_manifest
from .transforms import build_cache


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_cached_splits(size_wh: tuple[int, int]):
    """Return manifest df and {split: (images, masks, labels, sample_ids)} at size_wh."""
    df = load_manifest(config.manifest_path())
    cache = config.processed_dir() / f"cache_{size_wh[0]}x{size_wh[1]}.npz"
    data = build_cache(df, config.data_root(), size_wh, cache)
    out = {}
    for s in ("train", "val", "test"):
        idx = np.where(df["split"].to_numpy() == s)[0]
        out[s] = (data["images"][idx], data["masks"][idx], df[CLASSES].to_numpy()[idx], df["sample_id"].to_numpy()[idx])
    return df, out


def best_f1_threshold(scores: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Threshold maximising F1 on (validation) scores. Returns (threshold, f1)."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        p = scores >= t
        tp = int((p & (y == 1)).sum())
        fp = int((p & (y == 0)).sum())
        fn = int((~p & (y == 1)).sum())
        f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def manifest_hash() -> str:
    return hashlib.sha256(config.manifest_path().read_bytes()).hexdigest()[:16]


def environment_info() -> dict:
    import cv2
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
