from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .augment import augment_sequence
from .features import LANDMARK_DIM, normalize_sequence, resample_sequence


def load_manifest(data_dir: Path) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/generate_synthetic_sequences.py or collect_sequences.py")
    return json.loads(path.read_text(encoding="utf-8"))


def list_samples(data_dir: Path, split: str = "train") -> list[dict]:
    manifest = load_manifest(data_dir)
    return [s for s in manifest["samples"] if s["split"] == split]


def load_sample(data_dir: Path, sample: dict, seq_len: int) -> tuple[np.ndarray, int]:
    arr = np.load(Path(data_dir) / sample["path"])["seq"].astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != LANDMARK_DIM:
        raise ValueError(f"Bad sequence shape {arr.shape} in {sample['path']}")
    arr = resample_sequence(arr, seq_len)
    arr = normalize_sequence(arr)
    return arr, int(sample["label_id"])


def load_split(
    data_dir: Path,
    split: str,
    seq_len: int,
    *,
    augment: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    samples = list_samples(data_dir, split)
    if not samples:
        raise ValueError(f"No samples for split={split} in {data_dir}")
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for s in samples:
        x, y = load_sample(data_dir, s, seq_len)
        if augment and split == "train":
            x = normalize_sequence(augment_sequence(x, rng=rng))
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def label_names(data_dir: Path) -> list[str]:
    return list(load_manifest(data_dir)["labels"])
