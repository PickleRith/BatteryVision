"""Re-select a segmentation model's per-class PIXEL thresholds on the validation split
at the original 640x480 resolution (the resolution used for evaluation).

Why: train_segmentation.py picks thresholds at the 320x240 training resolution,
against masks thickened by any-pixel downsampling. Scored at full resolution
against the original thin masks, those thresholds over-segment (e.g. crack pixel
precision ~0.55 on val). This script fixes the mismatch using validation data only.

    python scripts/calibrate_thresholds.py models/unet_v2.pt
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from batteryvision import config
from batteryvision.dataset import CLASSES, load_image, load_manifest, load_mask
from batteryvision.inference import load_model
from batteryvision.transforms import resize_image, to_tensor

GRID = np.round(np.arange(0.10, 0.96, 0.05), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str)
    args = ap.parse_args()
    path = config.resolve(args.model)
    model = load_model(path)
    if model.kind != "segmentation":
        print("Only segmentation models have pixel thresholds.")
        return 1

    root = config.data_root()
    df = load_manifest(config.manifest_path())
    val = df[df.split == "val"]
    # counts[class, threshold] for tp, fp, fn, summed over all val images
    tp = np.zeros((len(CLASSES), len(GRID)), np.int64)
    fp = np.zeros_like(tp)
    fn = np.zeros_like(tp)
    for r in val.itertuples(index=False):
        img = load_image(root / r.image_path)
        gt = load_mask(root / r.mask_path)
        with torch.no_grad():  # same preprocessing/upsampling as inference.predict
            x = to_tensor(resize_image(img, model.input_size_wh)).unsqueeze(0)
            prob = torch.sigmoid(model.net(x))
            prob = F.interpolate(prob, size=img.shape[:2], mode="bilinear", align_corners=False)[0].numpy()
        for k in range(len(CLASSES)):
            g = gt[..., k]
            for j, t in enumerate(GRID):
                p = prob[k] >= t
                tp[k, j] += int((p & g).sum())
                fp[k, j] += int((p & ~g).sum())
                fn[k, j] += int((~p & g).sum())

    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    new = {c: float(GRID[int(np.argmax(dice[k]))]) for k, c in enumerate(CLASSES)}
    old = dict(model.pixel_thresholds)
    for k, c in enumerate(CLASSES):
        j_old = int(np.argmin(np.abs(GRID - old[c])))
        print(f"{c:15s} old thr {old[c]:.2f} (val Dice@640x480 {dice[k, j_old]:.3f}) -> "
              f"new thr {new[c]:.2f} (val Dice {dice[k].max():.3f})")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    meta.setdefault("pixel_thresholds_train_resolution", old)
    meta["pixel_thresholds"] = new
    meta["val_dice_full_res_at_thresholds"] = {c: float(dice[k].max()) for k, c in enumerate(CLASSES)}
    meta["threshold_selection"] = ("validation split only. Pixel: max dataset Dice at 640x480 "
                                   "(scripts/calibrate_thresholds.py); image: max F1 of max-pixel-prob")
    torch.save(ckpt, path)
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
