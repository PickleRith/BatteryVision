"""Copy a small set of TEST-split examples into data/samples/ so the app works without
the full download (e.g. on Streamlit Community Cloud).

CoatingVision is published under CC BY 4.0 (Figshare record 29260121), which permits
redistribution with attribution; the attribution is written next to the samples.

    python scripts/export_demo_assets.py [--per-class 4]
"""
from __future__ import annotations

import argparse
import shutil
import sys

import pandas as pd

from batteryvision import config
from batteryvision.dataset import CLASSES, load_manifest

ATTRIBUTION = """Sample images and masks in this folder are from:

  Sampath et al., "CoatingVision: A Defect Dataset for Coating Manufacturing",
  Scientific Data (2026). https://doi.org/10.1038/s41597-025-06419-1
  Dataset: https://doi.org/10.6084/m9.figshare.29260121.v1

Licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Files are copied unmodified; only a subset of the test split is included.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()
    root = config.data_root()
    df = load_manifest(config.manifest_path())
    test = df[df.split == "test"]
    picks = []
    for c in CLASSES:
        d = test[(test[c] == 1) & ~test.sample_id.isin(picks)]
        picks += d.sample(min(args.per_class, len(d)), random_state=42).sample_id.tolist()
    none = test[test[CLASSES].sum(axis=1) == 0]
    picks += none.sample(min(2, len(none)), random_state=42).sample_id.tolist()

    out = config.resolve("data/samples")
    sel = df[df.sample_id.isin(picks)].copy()
    for r in sel.itertuples(index=False):
        for rel in (r.image_path, r.mask_path):
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dst)
    sel.to_csv(out / "samples.csv", index=False)
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    print(f"exported {len(sel)} samples to {out}")
    print(sel[CLASSES].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
