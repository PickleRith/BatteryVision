"""Dataset layout, label/mask loading and manifest construction for CoatingVision.

Layout observed in the Figshare archive (CoatingVision.zip, v1):

    classification/labels.csv        original_file_name,file_name,Surface_Crack,Delamination,Pinhole,unclassified
    classification/images/image_N.jpg
    segmentation/images/image_N.jpg  (byte-identical to classification/images)
    segmentation/masks/image_N.png   RGBA, 640x480, values {0, 255}
    detection/images, detection/labels (YOLO txt; not used here)

Mask channel mapping was verified for every image: when the PNG is read with
Pillow in RGBA order, channel k is non-empty exactly when the k-th label column is 1
(see scripts/inspect_dataset.py). OpenCV's imread returns BGRA, which would swap
crack and pinhole, so always load masks through `load_mask`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES: list[str] = ["surface_crack", "delamination", "pinhole", "unclassified"]
CLASS_LABELS: dict[str, str] = {
    "surface_crack": "Surface crack",
    "delamination": "Delamination",
    "pinhole": "Pinhole",
    "unclassified": "Unclassified",
}
# Column names in classification/labels.csv
CSV_COLUMNS: dict[str, str] = {
    "surface_crack": "Surface_Crack",
    "delamination": "Delamination",
    "pinhole": "Pinhole",
    "unclassified": "unclassified",
}
# Index into the mask array returned by `load_mask` (Pillow RGBA order).
MASK_CHANNEL: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

LABELS_CSV = Path("classification") / "labels.csv"
IMAGE_DIR = Path("segmentation") / "images"
MASK_DIR = Path("segmentation") / "masks"

_ORIG_RE = re.compile(r"^(?P<run>.+)_frame_(?P<frame>\d+)_patch_(?P<patch>\d+)\.\w+$")


@dataclass(frozen=True)
class SourceInfo:
    run_id: str
    frame: int
    patch: int


def parse_original_name(name: str) -> SourceInfo | None:
    """Parse e.g. 'R7-700um-middle_frame_489_patch_1.png'. Returns None if unparseable."""
    m = _ORIG_RE.match(str(name))
    if not m:
        return None
    return SourceInfo(m["run"], int(m["frame"]), int(m["patch"]))


def mask_name_for(image_name: str) -> str:
    return Path(image_name).stem + ".png"


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as uint8 RGB (H, W, 3)."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    """Load a 4-channel annotation mask as bool (H, W, 4), channels ordered as CLASSES."""
    with Image.open(path) as im:
        if im.mode != "RGBA":
            raise ValueError(f"Expected RGBA mask, got {im.mode}: {path}")
        arr = np.array(im)
    if arr.ndim != 3 or arr.shape[2] != len(CLASSES):
        raise ValueError(f"Unexpected mask shape {arr.shape}: {path}")
    return arr > 127


def read_labels(data_root: Path) -> pd.DataFrame:
    path = Path(data_root) / LABELS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    df = pd.read_csv(path)
    missing = {"original_file_name", "file_name", *CSV_COLUMNS.values()} - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing columns: {sorted(missing)}")
    return df


def pair_samples(data_root: Path, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pair each labelled image with its mask. Returns (paired rows, problems)."""
    data_root = Path(data_root)
    records, problems = [], []
    for row in labels.itertuples(index=False):
        img = IMAGE_DIR / row.file_name
        msk = MASK_DIR / mask_name_for(row.file_name)
        if not (data_root / img).exists():
            problems.append(f"missing image: {img.as_posix()}")
            continue
        if not (data_root / msk).exists():
            problems.append(f"missing mask: {msk.as_posix()}")
            continue
        src = parse_original_name(row.original_file_name)
        if src is None:
            problems.append(f"unparseable original_file_name: {row.original_file_name}")
        rec = {
            "sample_id": Path(row.file_name).stem,
            "image_path": img.as_posix(),
            "mask_path": msk.as_posix(),
            "original_file_name": row.original_file_name,
            "run_id": src.run_id if src else "",
            "frame": src.frame if src else -1,
            "patch": src.patch if src else -1,
        }
        for c, col in CSV_COLUMNS.items():
            rec[c] = int(getattr(row, col))
        records.append(rec)
    return pd.DataFrame.from_records(records), problems


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    """Per-class pixel coverage (% of image pixels) and connected-component counts."""
    import cv2

    out: dict[str, float] = {}
    total = mask.shape[0] * mask.shape[1]
    for c, k in MASK_CHANNEL.items():
        ch = mask[..., k].astype(np.uint8)
        out[f"{c}_coverage_pct"] = 100.0 * float(ch.sum()) / total
        n, _ = cv2.connectedComponents(ch, connectivity=8)
        out[f"{c}_components"] = int(n - 1)
    return out


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in CLASSES:
        df[c] = df[c].astype(int)
    return df
