"""Classical OpenCV pinhole baseline: tune on train+val, report once on test.

    python scripts/baseline_opencv.py            # small grid search + test report
    python scripts/baseline_opencv.py --no-tune  # use default parameters

Outputs artifacts/metrics/baseline_pinhole.json and
artifacts/metrics/baseline_pinhole_per_image.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from batteryvision import config
from batteryvision.classical_cv import PinholeParams, detect_pinholes
from batteryvision.dataset import MASK_CHANNEL, load_image, load_manifest, load_mask
from batteryvision.metrics import binary_counts, match_components, prf_from_counts

CH = MASK_CHANNEL["pinhole"]


def load_split(df: pd.DataFrame, root) -> list[tuple[str, np.ndarray, np.ndarray]]:
    def one(r):
        return r.sample_id, load_image(root / r.image_path), load_mask(root / r.mask_path)[..., CH]

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, df.itertuples(index=False)))


def evaluate(data, p: PinholeParams, keep_rows: bool = False):
    obj = np.zeros(4, np.int64)
    px = np.zeros(3, np.int64)
    img_tp = img_fp = img_fn = img_tn = 0
    rows = []
    for sid, im, gt in data:
        pred, blobs = detect_pinholes(im, p)
        m = match_components(pred, gt)
        obj += [m["gt_objects"], m["gt_detected"], m["pred_objects"], m["pred_true"]]
        px += binary_counts(pred, gt)
        has_gt, has_pred = bool(gt.any()), bool(pred.any())
        img_tp += has_gt and has_pred
        img_fp += (not has_gt) and has_pred
        img_fn += has_gt and not has_pred
        img_tn += (not has_gt) and not has_pred
        if keep_rows:
            rows.append({"sample_id": sid, "gt_pinhole": int(has_gt), "pred_pinhole": int(has_pred),
                         "n_candidates": len(blobs), **m})
    gt_obj, gt_det, pr_obj, pr_true = (int(v) for v in obj)
    res = {
        "object_recall": gt_det / gt_obj if gt_obj else float("nan"),
        "object_precision": pr_true / pr_obj if pr_obj else float("nan"),
        "gt_objects": gt_obj,
        "pred_objects": pr_obj,
        "false_candidates_per_image": (pr_obj - pr_true) / max(len(data), 1),
        "pixel": prf_from_counts(*[int(v) for v in px]),
        "image_level": {"tp": img_tp, "fp": img_fp, "fn": img_fn, "tn": img_tn,
                        **prf_from_counts(img_tp, img_fp, img_fn)},
        "n_images": len(data),
    }
    r, pr = res["object_recall"], res["object_precision"]
    res["object_f1"] = 2 * r * pr / (r + pr) if (r + pr) > 0 else 0.0
    return res, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()

    root = config.data_root()
    df = load_manifest(config.manifest_path())
    tune_df = df[df.split.isin(["train", "val"])]
    # Tune on ALL train+val images so pinhole prevalence matches deployment/test.
    # (An earlier version tuned on a 50/50 balanced subset, which overstated precision.)
    tune = load_split(tune_df, root)
    print(f"tuning on {len(tune)} train+val images ({int(tune_df.pinhole.sum())} with pinholes)")

    best = PinholeParams()
    if not args.no_tune:
        # Earlier grids peaked at their upper edge, so the threshold range extends well beyond it.
        grid = {
            "tophat_ksize": [11, 15],
            "k_sigma": [20.0, 30.0],
            "min_abs_response": [30.0, 40.0, 55.0, 70.0, 90.0],
            "min_circularity": [0.5, 0.7],
        }
        best_f1 = -1.0
        for vals in itertools.product(*grid.values()):
            p = PinholeParams(**dict(zip(grid.keys(), vals)))
            res, _ = evaluate(tune, p)
            if res["object_f1"] > best_f1:
                best_f1, best = res["object_f1"], p
                print(f"  {vals} object F1={res['object_f1']:.3f} R={res['object_recall']:.3f} P={res['object_precision']:.3f}")
        print("best params:", best.to_dict())

    tune_res, _ = evaluate(tune, best)
    test_data = load_split(df[df.split == "test"], root)
    test_res, rows = evaluate(test_data, best, keep_rows=True)
    out = {
        "method": "white top-hat + robust threshold + area/circularity filter",
        "target_class": "pinhole",
        "params": best.to_dict(),
        "tuned_on": "all train+val images (natural pinhole prevalence); object-level F1",
        "tuning_set": tune_res,
        "test": test_res,
        "matching": "component matched if within 3 px (dilation tolerance)",
    }
    mdir = config.artifacts_dir() / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "baseline_pinhole.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(mdir / "baseline_pinhole_per_image.csv", index=False)
    print(json.dumps(test_res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
