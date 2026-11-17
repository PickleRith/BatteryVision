"""Build data/processed/manifest.csv with labels, mask statistics, groups and splits.

    python scripts/prepare_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd

from batteryvision import config
from batteryvision.dataset import CLASSES, load_mask, mask_stats, pair_samples, read_labels
from batteryvision.splits import make_group_ids, split_summary, stratified_group_split


def main() -> int:
    s = config.load_settings()
    root = config.data_root()
    labels = read_labels(root)
    df, problems = pair_samples(root, labels)
    if problems:
        print("Pairing problems:\n  " + "\n  ".join(problems[:20]))
        return 1

    stats = [mask_stats(load_mask(root / p)) for p in df["mask_path"]]
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    df["n_labels"] = df[CLASSES].sum(axis=1)

    df["group_id"] = make_group_ids(df, s["split"]["frame_block_size"])
    fr = {k: s["split"][k] for k in ("train", "val", "test")}
    df["split"] = stratified_group_split(df, "group_id", fr, seed=s["seed"])

    # sanity: no group straddles splits
    assert (df.groupby("group_id")["split"].nunique() == 1).all()

    df = df.sort_values("sample_id", key=lambda x: x.str.extract(r"(\d+)")[0].astype(int))
    out = config.manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]

    summ = split_summary(df)
    print(summ.to_string(index=False))
    meta = {
        "manifest_sha256_16": digest,
        "n_images": len(df),
        "seed": s["seed"],
        "frame_block_size": s["split"]["frame_block_size"],
        "split_summary": summ.to_dict(orient="records"),
    }
    (out.parent / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    # A copy of split assignments is versioned so results are reproducible.
    split_out = config.artifacts_dir() / "metrics" / "split_assignments.csv"
    split_out.parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "group_id", "split"]].to_csv(split_out, index=False)
    print(f"\nWrote {out} (sha256[:16]={digest}) and {split_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
