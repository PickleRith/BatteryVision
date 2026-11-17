"""Final held-out evaluation on the frozen test split (thresholds come from val).

    python scripts/evaluate.py                       # all models found in models/
    python scripts/evaluate.py --split val           # sanity check on val

Writes, per model:
    artifacts/metrics/<model>_<split>_metrics.json   aggregate metrics
    artifacts/metrics/<model>_<split>_per_image.csv  per-image probabilities, labels, Dice
    artifacts/metrics/<model>_<split>_pr.json        PR-curve points per class
Segmentation is evaluated at the ORIGINAL 640x480 resolution against the
published masks (predictions are bilinearly upsampled before thresholding).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from batteryvision import config
from batteryvision.dataset import CLASSES, load_image, load_manifest, load_mask
from batteryvision.inference import load_model, predict
from batteryvision.metrics import SegAccumulator, dice, match_components, prf_from_counts
from batteryvision.training import manifest_hash, save_json


def image_level_metrics(y: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for k, c in enumerate(CLASSES):
        yt, yp = y[:, k].astype(bool), pred[:, k].astype(bool)
        tp, fp, fn = int((yt & yp).sum()), int((~yt & yp).sum()), int((yt & ~yp).sum())
        d = prf_from_counts(tp, fp, fn)
        d.update({
            "support": int(yt.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~yt & ~yp).sum()),
            "average_precision": float(average_precision_score(yt, scores[:, k])) if yt.any() else float("nan"),
        })
        out[c] = d
    return out


def _errors_flagged(y: np.ndarray, pred: np.ndarray, rows: list[dict]) -> dict:
    wrong = np.any(y.astype(bool) != pred, axis=1)
    flagged = np.array([r["review_required"] for r in rows], bool)
    return {"images_with_any_error": int(wrong.sum()),
            "of_which_flagged": int((wrong & flagged).sum()),
            "share_flagged": float((wrong & flagged).sum() / wrong.sum()) if wrong.any() else float("nan")}


def pr_points(y: np.ndarray, scores: np.ndarray) -> dict:
    out = {}
    for k, c in enumerate(CLASSES):
        if not y[:, k].any():
            out[c] = None
            continue
        p, r, t = precision_recall_curve(y[:, k], scores[:, k])
        out[c] = {"precision": p.round(4).tolist(), "recall": r.round(4).tolist(), "thresholds": t.round(4).tolist()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--models", nargs="*", default=None, help="model .pt files (default: all in models/)")
    args = ap.parse_args()

    root = config.data_root()
    df = load_manifest(config.manifest_path())
    d = df[df.split == args.split].reset_index(drop=True)
    paths = [config.resolve(p) for p in args.models] if args.models else sorted(
        p for p in config.models_dir().glob("*.pt") if not p.stem.startswith("smoke"))
    if not paths:
        print("No model checkpoints found in", config.models_dir())
        return 1

    mdir = config.artifacts_dir() / "metrics"
    for path in paths:
        model = load_model(path)
        print(f"== {model.name} ({model.kind}) on {args.split}: {len(d)} images")
        y = d[CLASSES].to_numpy()
        scores = np.zeros((len(d), len(CLASSES)), np.float32)
        pred = np.zeros_like(scores, dtype=bool)
        seg = SegAccumulator(CLASSES) if model.kind == "segmentation" else None
        obj = np.zeros(4, np.int64)
        rows = []
        for i, r in enumerate(d.itertuples(index=False)):
            img = load_image(root / r.image_path)
            res = predict(model, img, r.sample_id)
            scores[i] = [res["class_probabilities"][c] for c in CLASSES]
            pred[i] = [res["predicted_labels"][c] for c in CLASSES]
            row = {"sample_id": r.sample_id, "review_required": res["review_required"]}
            for k, c in enumerate(CLASSES):
                row[f"{c}_true"] = int(y[i, k])
                row[f"{c}_prob"] = round(float(scores[i, k]), 4)
                row[f"{c}_pred"] = int(pred[i, k])
            if seg is not None:
                gt = load_mask(root / r.mask_path)
                pm = res["predicted_masks"]
                seg.update(pm, gt)
                ph = CLASSES.index("pinhole")
                mc = match_components(pm[..., ph], gt[..., ph])
                obj += [mc["gt_objects"], mc["gt_detected"], mc["pred_objects"], mc["pred_true"]]
                for k, c in enumerate(CLASSES):
                    row[f"{c}_dice"] = dice(pm[..., k], gt[..., k]) if gt[..., k].any() else np.nan
                    row[f"{c}_pred_coverage_pct"] = round(res["predicted_coverage_pct"][c], 4)
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"   {i + 1}/{len(d)}")

        metrics = {
            "model_name": model.name,
            "model_kind": model.kind,
            "split": args.split,
            "n_images": len(d),
            "manifest_sha256_16": manifest_hash(),
            "thresholds_from": "validation split",
            "image_level": image_level_metrics(y, scores, pred),
            "image_level_score": "sigmoid probability" if model.kind == "classification"
            else "max pixel probability of the class channel",
            "review_rate": float(np.mean([r["review_required"] for r in rows])),
            "review_band": model.review_band,
            "errors_flagged_for_review": _errors_flagged(y, pred, rows),
        }
        if seg is not None:
            metrics["pixel_level"] = seg.results()
            metrics["pixel_level_notes"] = (
                "Evaluated at 640x480. dice/iou/precision/recall sum pixels over all images "
                "(empty-GT images included). mean_image_dice_gt_present averages only images "
                "where the class is annotated. empty_fp_rate = share of images without the "
                "class where any pixel is predicted.")
            g, gd, po, pt = (int(v) for v in obj)
            metrics["pinhole_objects"] = {"gt_objects": g, "gt_detected": gd, "pred_objects": po, "pred_true": pt,
                                          "object_recall": gd / g if g else float("nan"),
                                          "object_precision": pt / po if po else float("nan"),
                                          "matching": "3 px tolerance, same as the OpenCV baseline"}
        mac = [metrics["image_level"][c]["f1"] for c in CLASSES if metrics["image_level"][c]["support"]]
        metrics["image_level_macro_f1"] = float(np.nanmean(mac))

        stem = f"{model.name}_{args.split}"
        save_json(metrics, mdir / f"{stem}_metrics.json")
        pd.DataFrame(rows).to_csv(mdir / f"{stem}_per_image.csv", index=False)
        save_json(pr_points(y, scores), mdir / f"{stem}_pr.json")
        print(json.dumps({c: {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}
                          for c, m in metrics["image_level"].items()}, indent=1))
        if seg is not None:
            print(json.dumps({c: {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}
                              for c, m in metrics["pixel_level"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
