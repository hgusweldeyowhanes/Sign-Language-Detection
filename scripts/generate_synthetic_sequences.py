#!/usr/bin/env python
"""Generate synthetic landmark sequences for CSLR smoke-training / CI.

Each gloss gets a characteristic motion pattern so models can overfit quickly
for pipeline validation. Replace with real MediaPipe captures for research.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CSLR_LABELS  # noqa: E402
from app.cslr.features import LANDMARK_DIM  # noqa: E402

DEFAULT_LABELS = list(CSLR_LABELS)


def synth_sequence(label_id: int, length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
    seq = np.zeros((length, LANDMARK_DIM), dtype=np.float32)
    # base resting hand
    for i in range(21):
        seq[:, i * 3] = 0.1 * np.sin(t * (0.5 + 0.1 * i))
        seq[:, i * 3 + 1] = 0.1 * np.cos(t * (0.4 + 0.05 * i))
        seq[:, i * 3 + 2] = 0.02 * np.sin(t)
    # class-specific trajectory on fingertip channels
    phase = 0.7 * label_id
    amp = 0.15 + 0.02 * label_id
    freq = 1.0 + 0.35 * label_id
    tip_idxs = [4, 8, 12, 16, 20]
    for tip in tip_idxs:
        seq[:, tip * 3] += amp * np.sin(freq * t + phase)
        seq[:, tip * 3 + 1] += amp * np.cos(freq * t + phase * 0.5)
    seq += rng.normal(0, 0.01, size=seq.shape).astype(np.float32)
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sequences")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    samples = []
    for lid, name in enumerate(DEFAULT_LABELS):
        for i in range(args.per_class):
            split = "train" if i < int(args.per_class * 0.7) else (
                "val" if i < int(args.per_class * 0.85) else "test"
            )
            seq = synth_sequence(lid, args.seq_len, rng)
            rel = f"{split}/{name}_{i:03d}.npz"
            path = args.out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, seq=seq)
            samples.append(
                {
                    "path": rel.replace("\\", "/"),
                    "label": name,
                    "label_id": lid,
                    "split": split,
                    "frames": args.seq_len,
                }
            )

    manifest = {
        "labels": DEFAULT_LABELS,
        "feature_dim": LANDMARK_DIM,
        "seq_len_default": args.seq_len,
        "synthetic": True,
        "samples": samples,
        "note": "Synthetic motion patterns for pipeline tests. Collect real MediaPipe sequences for research claims.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} sequences → {args.out}")


if __name__ == "__main__":
    main()
