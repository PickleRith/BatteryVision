"""Classical (non-learned) pinhole candidate detector.

Pinholes in CoatingVision appear as tiny, bright, roughly round specks (median
annotated area ~50 px in a 640x480 patch). The detector:

1. grayscale + light Gaussian denoise
2. white top-hat (image minus its morphological opening) to keep only bright
   structures smaller than the kernel, suppressing illumination gradients
3. threshold the top-hat response at `k_sigma` robust standard deviations above
   its median (robust to per-image brightness/contrast)
4. connected components, filtered by area and circularity

It targets pinholes only. It is not a multi-class defect detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PinholeParams:
    blur_ksize: int = 3
    tophat_ksize: int = 15
    k_sigma: float = 8.0
    min_abs_response: float = 12.0
    min_area: int = 4
    max_area: int = 400
    min_circularity: float = 0.35

    def to_dict(self) -> dict:
        return asdict(self)


def tophat_response(image_rgb: np.ndarray, p: PinholeParams) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if p.blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.tophat_ksize, p.tophat_ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k).astype(np.float32)


def detect_pinholes(image_rgb: np.ndarray, p: PinholeParams | None = None) -> tuple[np.ndarray, list[dict]]:
    """Return (bool mask HxW, list of candidate blobs with geometry)."""
    p = p or PinholeParams()
    th = tophat_response(image_rgb, p)
    med = float(np.median(th))
    mad = float(np.median(np.abs(th - med))) * 1.4826 + 1e-6
    thr = max(med + p.k_sigma * mad, p.min_abs_response)
    binary = (th > thr).astype(np.uint8)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros(binary.shape, bool)
    blobs: list[dict] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (p.min_area <= area <= p.max_area):
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = cv2.arcLength(contours[0], True) if contours else 0.0
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 1.0
        if circ < p.min_circularity:
            continue
        mask |= labels == i
        blobs.append({
            "x": float(cents[i, 0]),
            "y": float(cents[i, 1]),
            "area_px": area,
            "circularity": float(min(circ, 1.0)),
            "peak_response": float(th[labels == i].max()),
        })
    return mask, blobs
