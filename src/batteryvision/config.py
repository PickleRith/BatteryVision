"""Settings loading. All relative paths resolve against the repository root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@lru_cache(maxsize=1)
def load_settings() -> dict:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    env = os.environ.get("BATTERYVISION_DATA_ROOT")
    return resolve(env) if env else resolve(load_settings()["data_root"])


def processed_dir() -> Path:
    return resolve(load_settings()["processed_dir"])


def models_dir() -> Path:
    return resolve(load_settings()["models_dir"])


def artifacts_dir() -> Path:
    return resolve(load_settings()["artifacts_dir"])


def manifest_path() -> Path:
    return processed_dir() / "manifest.csv"
