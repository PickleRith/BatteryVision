"""Set each model's demo review band on the VALIDATION split.

An image is flagged for human review when any class score lies within `band` of
that class's threshold. A fixed band (0.15) flagged 99% of images for the U-Net,
whose scores (max pixel probability) cluster near 1, so the band is chosen per
model as the 15th percentile of each validation image's smallest |score - threshold|,
i.e. ~15% of validation images get flagged.

    python scripts/evaluate.py --split val      # produces val per-image scores
    python scripts/calibrate_review.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from batteryvision import config
from batteryvision.dataset import CLASSES
from batteryvision.inference import load_model

TARGET_FLAG_RATE = 0.15


def main() -> int:
    mdir = config.artifacts_dir() / "metrics"
    for path in sorted(config.models_dir().glob("*.pt")):
        model = load_model(path)
        per = mdir / f"{model.name}_val_per_image.csv"
        if not per.exists():
            print(f"skip {model.name}: run `python scripts/evaluate.py --split val` first")
            continue
        df = pd.read_csv(per)
        thr = model.image_thresholds
        margin = np.min(np.stack([np.abs(df[f"{c}_prob"] - thr[c]) for c in CLASSES], 1), axis=1)
        band = float(np.quantile(margin, TARGET_FLAG_RATE))
        flagged = margin < band
        wrong = np.any(np.stack([df[f"{c}_pred"] != df[f"{c}_true"] for c in CLASSES], 1), axis=1)
        cap = flagged[wrong].mean() if wrong.any() else float("nan")
        print(f"{model.name}: band={band:.4f} val flag rate={flagged.mean():.3f} "
              f"val images with any error={int(wrong.sum())}, of which flagged={cap:.3f}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt["meta"]["review_band"] = band
        ckpt["meta"]["review_band_selection"] = (
            f"validation split: {TARGET_FLAG_RATE:.0%} quantile of min |score - threshold|")
        torch.save(ckpt, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
