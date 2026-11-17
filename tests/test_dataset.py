"""Pairing, mask loading and channel-mapping tests.

Synthetic fixtures run everywhere; tests marked `needs_data` also check the real
downloaded dataset and are skipped if it is absent.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from batteryvision import config
from batteryvision.dataset import (
    CLASSES,
    CSV_COLUMNS,
    MASK_CHANNEL,
    load_mask,
    pair_samples,
    parse_original_name,
    read_labels,
)
from batteryvision.splits import make_group_ids, stratified_group_split
from batteryvision.transforms import downsample_mask


def _make_root(tmp_path, n=3, drop_mask_for=None):
    (tmp_path / "segmentation/images").mkdir(parents=True)
    (tmp_path / "segmentation/masks").mkdir(parents=True)
    (tmp_path / "classification").mkdir(parents=True)
    rows = []
    for i in range(1, n + 1):
        Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(tmp_path / f"segmentation/images/image_{i}.jpg")
        if i != drop_mask_for:
            m = np.zeros((480, 640, 4), np.uint8)
            m[10:20, 10:20, 2] = 255  # pinhole channel only
            Image.fromarray(m, "RGBA").save(tmp_path / f"segmentation/masks/image_{i}.png")
        rows.append({"original_file_name": f"R1-600um_frame_{i}_patch_0.png", "file_name": f"image_{i}.jpg",
                     "Surface_Crack": 0, "Delamination": 0, "Pinhole": 1, "unclassified": 0})
    pd.DataFrame(rows).to_csv(tmp_path / "classification/labels.csv", index=False)
    return tmp_path


def test_pairing_ok(tmp_path):
    root = _make_root(tmp_path)
    df, problems = pair_samples(root, read_labels(root))
    assert problems == []
    assert len(df) == 3
    assert df.iloc[0]["mask_path"] == "segmentation/masks/image_1.png"


def test_pairing_reports_missing_mask(tmp_path):
    root = _make_root(tmp_path, drop_mask_for=2)
    df, problems = pair_samples(root, read_labels(root))
    assert len(df) == 2
    assert problems == ["missing mask: segmentation/masks/image_2.png"]


def test_missing_labels_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_labels(tmp_path)


def test_mask_channel_order_is_pillow_rgba(tmp_path):
    root = _make_root(tmp_path, n=1)
    m = load_mask(root / "segmentation/masks/image_1.png")
    assert m.shape == (480, 640, 4) and m.dtype == bool
    assert m[..., MASK_CHANNEL["pinhole"]].any()
    assert not m[..., MASK_CHANNEL["surface_crack"]].any()
    assert list(MASK_CHANNEL) == CLASSES == list(CSV_COLUMNS)


def test_non_rgba_mask_rejected(tmp_path):
    p = tmp_path / "bad.png"
    Image.fromarray(np.zeros((4, 4), np.uint8), "L").save(p)
    with pytest.raises(ValueError):
        load_mask(p)


def test_corrupt_image_file_raises(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"not an image")
    from batteryvision.dataset import load_image

    with pytest.raises(Exception):
        load_image(p)


def test_parse_original_name():
    s = parse_original_name("R1-1100um-top-to-bottom-center-1_frame_220_patch_7.png")
    assert (s.run_id, s.frame, s.patch) == ("R1-1100um-top-to-bottom-center-1", 220, 7)
    assert parse_original_name("garbage.png") is None


def test_downsample_mask_keeps_single_pixel_defects():
    m = np.zeros((480, 640, 4), bool)
    m[101, 201, 2] = True  # one-pixel pinhole at an odd coordinate
    small = downsample_mask(m, (320, 240))
    assert small.shape == (240, 320, 4)
    assert small[..., 2].sum() == 1


def test_group_split_keeps_groups_together():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"run_id": rng.choice(["a", "b", "c"], n), "frame": rng.integers(0, 600, n)})
    for c in CLASSES:
        df[c] = rng.random(n) < 0.2
    df[CLASSES] = df[CLASSES].astype(int)
    df["group_id"] = make_group_ids(df, 50)
    df["split"] = stratified_group_split(df, "group_id", {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
    assert set(df.split) == {"train", "val", "test"}


needs_data = pytest.mark.skipif(not (config.data_root() / "segmentation").exists(), reason="dataset not downloaded")


@needs_data
def test_real_dataset_channel_mapping_matches_labels_on_sample():
    root = config.data_root()
    df, problems = pair_samples(root, read_labels(root))
    assert not problems
    for r in df.sample(60, random_state=0).itertuples(index=False):
        m = load_mask(root / r.mask_path)
        for c, k in MASK_CHANNEL.items():
            assert bool(m[..., k].any()) == bool(getattr(r, c)), (r.sample_id, c)


@needs_data
def test_manifest_splits_disjoint():
    p = config.manifest_path()
    if not p.exists():
        pytest.skip("manifest not built")
    df = pd.read_csv(p)
    assert df.sample_id.is_unique
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
