"""Leakage-aware train/val/test split.

Images are 640x480 patches cut from video frames of 8 coating runs. Patches of the
same frame, and frames close in time, look nearly identical, so splitting by image
would leak. We group by (run, frame // block_size) and assign whole groups to
splits with a greedy multi-label stratifier (groups holding the rarest classes
are placed first). Residual risk: frames just either side of a block boundary
can still be similar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import CLASSES

SPLITS = ("train", "val", "test")


def make_group_ids(df: pd.DataFrame, block_size: int) -> pd.Series:
    return df["run_id"].astype(str) + "|b" + (df["frame"] // block_size).astype(str)


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    fractions: dict[str, float],
    seed: int = 42,
) -> pd.Series:
    """Assign each group to one split so per-class counts approach the target fractions."""
    rng = np.random.default_rng(seed)
    y = df[CLASSES].to_numpy()
    groups = df[group_col].to_numpy()
    uniq = list(pd.unique(groups))
    g_counts = {g: y[groups == g].sum(0) for g in uniq}
    g_sizes = {g: int((groups == g).sum()) for g in uniq}

    totals = y.sum(0).astype(float)
    n_total = float(len(df))
    weights = 1.0 / np.maximum(totals, 1.0)  # rare classes dominate the choice

    targets = {s: totals * fractions[s] for s in SPLITS}
    size_targets = {s: n_total * fractions[s] for s in SPLITS}
    assigned = {s: np.zeros(len(CLASSES)) for s in SPLITS}
    assigned_n = {s: 0.0 for s in SPLITS}

    rng.shuffle(uniq)

    def key(g):
        c = g_counts[g]
        rarest = min((totals[k] for k in range(len(CLASSES)) if c[k] > 0), default=np.inf)
        return (rarest, -g_sizes[g])

    uniq.sort(key=key)  # stable sort keeps the seeded shuffle as tie-breaker
    out: dict = {}
    for g in uniq:
        c = g_counts[g]
        best, best_score = None, -np.inf
        for s in SPLITS:
            # fraction of each split's own target still unfilled (negative = overshoot)
            rel = (targets[s] - assigned[s]) / np.maximum(targets[s], 1e-9)
            present = c > 0
            label_deficit = float((rel * weights * present).sum() / max((weights * present).sum(), 1e-12)) if present.any() else 0.0
            size_deficit = (size_targets[s] - assigned_n[s]) / size_targets[s]
            score = (label_deficit if present.any() else size_deficit) + 0.25 * size_deficit
            if score > best_score:
                best, best_score = s, score
        out[g] = best
        assigned[best] += c
        assigned_n[best] += g_sizes[g]
    return pd.Series([out[g] for g in groups], index=df.index, name="split")


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SPLITS:
        d = df[df["split"] == s]
        r = {"split": s, "images": len(d), "groups": int(d["group_id"].nunique())}
        r.update({c: int(d[c].sum()) for c in CLASSES})
        rows.append(r)
    return pd.DataFrame(rows)
