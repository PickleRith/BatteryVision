"""Mask overlays. Colors are fixed per class so every view uses the same legend."""
from __future__ import annotations

import cv2
import numpy as np

from .dataset import CLASSES

# One fixed color per class, used by both image overlays and charts.
# Categorical slots 1-4 of a palette validated for colour-vision-deficiency
# separation (adjacent-pair OKLab dE >= 8.4 protan); identity is never colour-alone
# (legends, labels and tables always name the class).
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "surface_crack": (0x2A, 0x78, 0xD6),   # blue
    "delamination": (0xEB, 0x68, 0x34),    # orange
    "pinhole": (0x1B, 0xAF, 0x7A),         # aqua
    "unclassified": (0xED, 0xA1, 0x00),    # yellow
}


def color_hex(c: str) -> str:
    r, g, b = CLASS_COLORS[c]
    return f"#{r:02x}{g:02x}{b:02x}"


def overlay_masks(
    image: np.ndarray,
    mask: np.ndarray,
    classes: list[str] | None = None,
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """Blend a (H, W, 4) bool mask onto an RGB image. Small regions get an outline so
    tiny pinholes stay visible."""
    classes = classes or CLASSES
    out = image.astype(np.float32).copy()
    for c in classes:
        k = CLASSES.index(c)
        m = mask[..., k]
        if not m.any():
            continue
        col = np.array(CLASS_COLORS[c], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * col
    out = out.clip(0, 255).astype(np.uint8)
    if outline:
        for c in classes:
            m = mask[..., CLASSES.index(c)].astype(np.uint8)
            if not m.any():
                continue
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, CLASS_COLORS[c], 1)
    return out


def mask_to_rgb(mask: np.ndarray, classes: list[str] | None = None) -> np.ndarray:
    """Render a (H, W, 4) mask on black, for side-by-side display."""
    h, w = mask.shape[:2]
    return overlay_masks(np.zeros((h, w, 3), np.uint8), mask, classes, alpha=1.0, outline=False)


def error_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Single-channel comparison: green = TP, red = FN (missed), blue = FP (false alarm)."""
    out = image.astype(np.float32).copy()
    for m, col in (((pred & gt), (0, 220, 0)), ((~pred & gt), (255, 0, 0)), ((pred & ~gt), (40, 120, 255))):
        out[m] = (1 - alpha) * out[m] + alpha * np.array(col, np.float32)
    return out.clip(0, 255).astype(np.uint8)
