"""Load trained models and produce results in a stable schema.

Result schema (identical keys regardless of model):

    sample_id, model_name, predicted_labels, class_probabilities,
    predicted_masks (bool HxWx4 at input resolution, segmentation models only),
    predicted_coverage_pct (from predicted masks only), ground_truth_masks,
    review_required, review_reasons

`review_required` is a DEMO REVIEW HEURISTIC (see `review_heuristic`), not a
validated industrial acceptance rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import CLASSES
from .models import SmallUNet, build_classifier
from .transforms import resize_image, to_tensor

# Default band for the demo review heuristic; models override it with a band chosen
# on validation (meta["review_band"], see scripts/calibrate_review.py).
UNCERTAIN_BAND = 0.15


@dataclass
class LoadedModel:
    name: str
    kind: str  # "segmentation" | "classification"
    net: torch.nn.Module
    meta: dict
    input_size_wh: tuple[int, int]
    pixel_thresholds: dict[str, float] = field(default_factory=dict)
    image_thresholds: dict[str, float] = field(default_factory=dict)
    review_band: float = UNCERTAIN_BAND


def load_model(path: str | Path) -> LoadedModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    if "pixel_thresholds" in meta:
        net = SmallUNet(base=ckpt.get("base", 16))
        kind = "segmentation"
        pix, img = meta["pixel_thresholds"], meta["image_thresholds"]
    else:
        net = build_classifier(pretrained=False)
        kind = "classification"
        pix, img = {}, meta["thresholds"]
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return LoadedModel(meta["model_name"], kind, net, meta, tuple(meta["input_size_wh"]), pix, img,
                       float(meta.get("review_band", UNCERTAIN_BAND)))


def review_heuristic(probs: dict[str, float], thresholds: dict[str, float],
                     band: float = UNCERTAIN_BAND) -> list[str]:
    """Flag images where any class score is within `band` of its threshold."""
    reasons = []
    for c in CLASSES:
        if abs(probs[c] - thresholds[c]) < band:
            reasons.append(f"{c} probability {probs[c]:.2f} is close to its threshold {thresholds[c]:.2f}")
    return reasons


@torch.no_grad()
def predict(model: LoadedModel, image_rgb: np.ndarray, sample_id: str = "upload",
            ground_truth: np.ndarray | None = None) -> dict:
    H, W = image_rgb.shape[:2]
    x = to_tensor(resize_image(image_rgb, model.input_size_wh)).unsqueeze(0)
    out = model.net(x)
    masks = coverage = None
    if model.kind == "segmentation":
        prob_map = torch.sigmoid(out)
        prob_map = F.interpolate(prob_map, size=(H, W), mode="bilinear", align_corners=False)[0].numpy()
        probs = {c: float(prob_map[k].max()) for k, c in enumerate(CLASSES)}
        masks = np.stack([prob_map[k] >= model.pixel_thresholds[c] for k, c in enumerate(CLASSES)], axis=-1)
        coverage = {c: 100.0 * float(masks[..., k].mean()) for k, c in enumerate(CLASSES)}
    else:
        p = torch.sigmoid(out)[0].numpy()
        probs = {c: float(p[k]) for k, c in enumerate(CLASSES)}
    labels = {c: probs[c] >= model.image_thresholds[c] for c in CLASSES}
    reasons = review_heuristic(probs, model.image_thresholds, model.review_band)
    return {
        "sample_id": sample_id,
        "model_name": model.name,
        "model_kind": model.kind,
        "predicted_labels": labels,
        "class_probabilities": probs,
        "predicted_masks": masks,
        "predicted_coverage_pct": coverage,
        "ground_truth_masks": ground_truth,
        "review_required": bool(reasons),
        "review_reasons": reasons,
    }
