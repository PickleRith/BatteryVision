"""Resizing, caching and aligned image+mask augmentation for training.

Masks are downsampled with a max-pool ("any annotated pixel in the cell"), not
plain nearest-neighbour: nearest sampling can drop 1-2 px wide cracks and pinholes
entirely. Evaluation always happens at the original 640x480 resolution.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import load_image, load_mask
from .models import IMAGENET_MEAN, IMAGENET_STD


def resize_image(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)


def downsample_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Any-pixel (max-pool) downsampling of a bool (H, W, K) mask to size_wh."""
    w, h = size_wh
    H, W, K = mask.shape
    if (W, H) == (w, h):
        return mask.copy()
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        return mask.reshape(h, fy, w, fx, K).any(axis=(1, 3))
    # fallback for non-integer factors
    return cv2.resize(mask.astype(np.uint8), size_wh, interpolation=cv2.INTER_AREA) > 0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def build_cache(df, data_root: Path, size_wh: tuple[int, int], cache_path: Path) -> dict[str, np.ndarray]:
    """Resize all images/masks once and store them in a compressed .npz cache."""
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if list(z["sample_id"]) == list(df["sample_id"]):
            return {k: z[k] for k in z.files}
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        im = resize_image(load_image(data_root / r.image_path), size_wh)
        m = downsample_mask(load_mask(data_root / r.mask_path), size_wh)
        return im, m

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, df.itertuples(index=False)))
    data = {
        "sample_id": df["sample_id"].to_numpy().astype(str),
        "images": np.stack([r[0] for r in res]),
        "masks": np.stack([r[1] for r in res]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **data)
    return data


class CoatingDataset(Dataset):
    """Serves (image tensor, mask tensor (K,H,W), label vector (K,))."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, labels: np.ndarray, augment: bool = False, seed: int = 0):
        self.images, self.masks, self.labels = images, masks, labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img, m = self.images[i], self.masks[i]
        if self.augment:
            # geometric flips are physically plausible for top-down coating imagery
            if self.rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if self.rng.random() < 0.5:
                img, m = img[::-1], m[::-1]
            # mild global brightness/contrast change (lighting variation)
            if self.rng.random() < 0.5:
                a = self.rng.uniform(0.85, 1.15)
                b = self.rng.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
        x = to_tensor(img)
        y = torch.from_numpy(m.astype(np.float32)).permute(2, 0, 1)
        lab = torch.from_numpy(self.labels[i].astype(np.float32))
        return x, y, lab
