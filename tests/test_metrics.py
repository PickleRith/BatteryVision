import math

import numpy as np

from batteryvision.metrics import SegAccumulator, dice, iou, match_components, prf_from_counts


def test_dice_iou_basic():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2, :2] = True  # 4 px
    b[:2, :] = True   # 8 px, overlap 4
    assert dice(a, b) == 2 * 4 / (4 + 8)
    assert iou(a, b) == 4 / 8


def test_empty_masks_are_undefined_not_perfect():
    z = np.zeros((3, 3), bool)
    assert math.isnan(dice(z, z))
    assert math.isnan(iou(z, z))
    assert math.isnan(prf_from_counts(0, 0, 0)["precision"])


def test_seg_accumulator_excludes_empty_gt_from_image_dice_but_counts_false_positives():
    acc = SegAccumulator(["a"])
    gt_present = np.zeros((4, 4, 1), bool)
    gt_present[0, 0, 0] = True
    acc.update(gt_present.copy(), gt_present)            # perfect on a positive image
    empty = np.zeros((4, 4, 1), bool)
    fp = empty.copy()
    fp[3, 3, 0] = True
    acc.update(fp, empty)                                # false alarm on an empty image
    r = acc.results()["a"]
    assert r["mean_image_dice_gt_present"] == 1.0
    assert r["images_gt_present"] == 1 and r["images_gt_empty"] == 1
    assert r["empty_fp_rate"] == 1.0
    assert r["pixel_precision"] == 0.5                   # FP pixel counted in dataset-level metric


def test_match_components_with_tolerance():
    gt = np.zeros((20, 20), bool)
    gt[5, 5] = True
    gt[15, 15] = True
    pred = np.zeros_like(gt)
    pred[6, 7] = True    # within 3 px of first GT pinhole
    pred[0, 19] = True   # spurious
    m = match_components(pred, gt, tolerance_px=3)
    assert m == {"gt_objects": 2, "gt_detected": 1, "pred_objects": 2, "pred_true": 1}
