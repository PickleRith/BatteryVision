import numpy as np
import torch

from batteryvision.dataset import CLASSES
from batteryvision.inference import LoadedModel, predict, review_heuristic
from batteryvision.models import SmallUNet, build_classifier


def _seg_model():
    torch.manual_seed(0)
    net = SmallUNet(base=4).eval()
    t = {c: 0.5 for c in CLASSES}
    return LoadedModel("test_unet", "segmentation", net, {}, (64, 48), t, t)


def test_segmentation_result_schema_and_shapes():
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    r = predict(_seg_model(), img, "x")
    assert set(r) >= {"sample_id", "model_name", "predicted_labels", "class_probabilities",
                      "predicted_masks", "ground_truth_masks", "review_required"}
    assert r["predicted_masks"].shape == (480, 640, 4)
    assert set(r["predicted_coverage_pct"]) == set(CLASSES)
    assert all(0 <= v <= 1 for v in r["class_probabilities"].values())


def test_classifier_never_returns_masks_or_coverage():
    net = build_classifier(pretrained=False).eval()
    t = {c: 0.5 for c in CLASSES}
    m = LoadedModel("test_cls", "classification", net, {}, (64, 48), {}, t)
    r = predict(m, np.zeros((480, 640, 3), np.uint8))
    assert r["predicted_masks"] is None
    assert r["predicted_coverage_pct"] is None


def test_review_heuristic():
    t = {c: 0.5 for c in CLASSES}
    assert review_heuristic({c: 0.05 for c in CLASSES}, t) == []
    p = {c: 0.05 for c in CLASSES} | {"pinhole": 0.55}
    assert len(review_heuristic(p, t)) == 1
