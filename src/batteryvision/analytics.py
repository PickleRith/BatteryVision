"""Aggregate statistics computed from the manifest (never hardcoded)."""
from __future__ import annotations

import pandas as pd

from .dataset import CLASSES


def prevalence(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = [{"class": c, "images": int(df[c].sum()), "pct_of_images": 100.0 * df[c].sum() / max(n, 1)} for c in CLASSES]
    return pd.DataFrame(rows)


def cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Number of images labelled with both class i and class j (diagonal = class count)."""
    y = df[CLASSES].to_numpy()
    m = y.T @ y
    return pd.DataFrame(m, index=CLASSES, columns=CLASSES)


def coverage_long(df: pd.DataFrame) -> pd.DataFrame:
    """Per-image mask coverage for images where the class is present (long format)."""
    parts = []
    for c in CLASSES:
        d = df[df[c] == 1]
        parts.append(pd.DataFrame({
            "class": c,
            "coverage_pct": d[f"{c}_coverage_pct"].to_numpy(),
            "components": d[f"{c}_components"].to_numpy(),
            "split": d["split"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def label_combinations(df: pd.DataFrame) -> pd.DataFrame:
    combo = df[CLASSES].apply(lambda r: " + ".join(c for c in CLASSES if r[c]) or "(no annotated defect)", axis=1)
    return combo.value_counts().rename_axis("labels").reset_index(name="images")


def run_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("run_id")
    out = g[CLASSES].sum()
    out.insert(0, "images", g.size())
    return out.reset_index()


def describe_coverage(df: pd.DataFrame) -> pd.DataFrame:
    cl = coverage_long(df)
    return cl.groupby("class")["coverage_pct"].describe(percentiles=[0.5, 0.9]).reindex(CLASSES).reset_index()


