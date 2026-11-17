"""Classification, segmentation and object-level metrics.

Conventions (stated explicitly because they change the numbers):
- Pixel metrics are computed by summing TP/FP/FN over all evaluated images
  ("micro" / dataset-level), so images with empty ground truth still contribute
  their false-positive pixels.
- Per-image Dice is reported only over images where the class is present in the
  ground truth (empty-GT images are excluded from that average); `empty_fp_rate`
  reports how often the model predicts the class on images where it is absent.
- Division by zero returns NaN, never 0 or 1, so undefined values are visible.
"""
from __future__ import annotations

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def binary_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {"precision": p, "recall": r, "f1": f1}


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn = binary_counts(pred, gt)
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn = binary_counts(pred, gt)
    return _safe_div(tp, tp + fp + fn)


class SegAccumulator:
    """Accumulates per-class pixel counts and per-image Dice across a dataset."""

    def __init__(self, classes: list[str]):
        self.classes = classes
        k = len(classes)
        self.tp = np.zeros(k, np.int64)
        self.fp = np.zeros(k, np.int64)
        self.fn = np.zeros(k, np.int64)
        self.img_dice: list[list[float]] = [[] for _ in classes]
        self.empty_gt = np.zeros(k, np.int64)
        self.empty_gt_with_pred = np.zeros(k, np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt: bool arrays (H, W, K)."""
        for k in range(len(self.classes)):
            p, g = pred[..., k], gt[..., k]
            tp, fp, fn = binary_counts(p, g)
            self.tp[k] += tp
            self.fp[k] += fp
            self.fn[k] += fn
            if g.any():
                self.img_dice[k].append(_safe_div(2 * tp, 2 * tp + fp + fn))
            else:
                self.empty_gt[k] += 1
                self.empty_gt_with_pred[k] += int(p.any())

    def results(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, c in enumerate(self.classes):
            tp, fp, fn = int(self.tp[k]), int(self.fp[k]), int(self.fn[k])
            d = prf_from_counts(tp, fp, fn)
            out[c] = {
                "pixel_precision": d["precision"],
                "pixel_recall": d["recall"],
                "dice": d["f1"],  # dataset-level Dice == pixel F1
                "iou": _safe_div(tp, tp + fp + fn),
                "mean_image_dice_gt_present": float(np.mean(self.img_dice[k])) if self.img_dice[k] else float("nan"),
                "images_gt_present": len(self.img_dice[k]),
                "images_gt_empty": int(self.empty_gt[k]),
                "empty_fp_rate": _safe_div(self.empty_gt_with_pred[k], self.empty_gt[k]),
            }
        return out


def match_components(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 3) -> dict[str, int]:
    """Object-level matching for small blobs (e.g. pinholes).

    A GT component counts as detected if any predicted pixel lies within
    `tolerance_px` of it; a predicted component is a true positive if it lies
    within `tolerance_px` of any GT pixel.
    """
    import cv2

    pred = np.asarray(pred, np.uint8)
    gt = np.asarray(gt, np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    pred_d = cv2.dilate(pred, k) > 0
    gt_d = cv2.dilate(gt, k) > 0
    ng, gl = cv2.connectedComponents(gt, connectivity=8)
    npred, pl = cv2.connectedComponents(pred, connectivity=8)
    gt_hit = int(np.count_nonzero(np.unique(gl[pred_d])))  # nonzero labels touched by a prediction
    pred_tp = int(np.count_nonzero(np.unique(pl[gt_d])))
    return {
        "gt_objects": ng - 1,
        "gt_detected": gt_hit,
        "pred_objects": npred - 1,
        "pred_true": pred_tp,
    }
