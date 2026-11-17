"""Inspect the downloaded CoatingVision archive and run data sanity checks.

Reports actual folder structure, image/mask modes and sizes, label columns and
counts, and verifies that each mask channel agrees with its label column for every
image. Exits non-zero if any check fails.

    python scripts/inspect_dataset.py [--data-root PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from batteryvision import config
from batteryvision.dataset import (
    CLASSES,
    CSV_COLUMNS,
    IMAGE_DIR,
    MASK_CHANNEL,
    load_mask,
    pair_samples,
    read_labels,
)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args()
    root = args.data_root or config.data_root()
    if not root.exists():
        print(f"Data root not found: {root}\nDownload CoatingVision.zip from "
              "https://doi.org/10.6084/m9.figshare.29260121.v1 and extract it there.")
        return 2

    print(f"Data root: {root}\n\n== Folder structure ==")
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        files = [f for f in d.iterdir() if f.is_file()]
        exts = Counter(f.suffix.lower() for f in files)
        print(f"  {d.relative_to(root).as_posix():32s} {len(files):5d} files {dict(exts)}")

    labels = read_labels(root)
    print(f"\n== labels.csv ==\n  columns: {list(labels.columns)}\n  rows: {len(labels)}")
    for c, col in CSV_COLUMNS.items():
        print(f"  {col:15s} positives={int(labels[col].sum()):5d}  values={sorted(labels[col].unique())}")
    no_label = int((labels[list(CSV_COLUMNS.values())].sum(axis=1) == 0).sum())
    print(f"  rows with no defect label: {no_label}")

    paired, problems = pair_samples(root, labels)
    print(f"\n== Pairing ==\n  paired: {len(paired)}  problems: {len(problems)}")
    for p in problems[:20]:
        print("   ", p)

    failures: list[str] = []
    img_modes, mask_modes, values = Counter(), Counter(), set()
    agree = np.zeros((len(CLASSES), 2, 2), dtype=int)  # [class, label, mask_nonempty]
    cls_dir = root / "classification" / "images"
    identical = 0
    for r in paired.itertuples(index=False):
        ip, mp = root / r.image_path, root / r.mask_path
        with Image.open(ip) as im:
            img_modes[(im.mode, im.size)] += 1
            isize = im.size
        with Image.open(mp) as m:
            mask_modes[(m.mode, m.size)] += 1
            values |= set(np.unique(np.array(m)).tolist())
            msize = m.size
        if isize != msize:
            failures.append(f"size mismatch {r.sample_id}: image {isize} mask {msize}")
        mask = load_mask(mp)
        for c, k in MASK_CHANNEL.items():
            agree[k, getattr(r, c), int(mask[..., k].any())] += 1
        cp = cls_dir / Path(r.image_path).name
        if cp.exists() and md5(cp) == md5(ip):
            identical += 1

    print(f"\n== Images ({IMAGE_DIR.as_posix()}) ==  {dict(img_modes)}")
    print(f"   identical to classification/images copy: {identical}/{len(paired)}")
    print(f"== Masks ==  {dict(mask_modes)}  unique values: {sorted(values)}")
    if not values <= {0, 255}:
        failures.append(f"non-binary mask values: {sorted(values)}")

    print("\n== Channel <-> label agreement (Pillow RGBA order) ==")
    print(f"  {'class':15s} {'ch':>3s} {'lab1&mask':>10s} {'lab1&empty':>11s} {'lab0&mask':>10s} {'lab0&empty':>11s}")
    for c, k in MASK_CHANNEL.items():
        a = agree[k]
        print(f"  {c:15s} {k:3d} {a[1,1]:10d} {a[1,0]:11d} {a[0,1]:10d} {a[0,0]:11d}")
        if a[1, 0] or a[0, 1]:
            failures.append(f"channel {k} disagrees with label '{c}' on {a[1,0] + a[0,1]} images")

    print("\n== Acquisition structure (from original_file_name) ==")
    print(f"  runs: {paired['run_id'].nunique()}  unique frames: "
          f"{paired.groupby(['run_id', 'frame']).ngroups}")
    for run, n in paired["run_id"].value_counts().items():
        print(f"   {run:40s} {n}")
    dup = paired["original_file_name"].duplicated().sum()
    if dup:
        failures.append(f"{dup} duplicate original_file_name entries")

    print("\n== Result ==")
    if failures or problems:
        for f in failures:
            print("  FAIL:", f)
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
