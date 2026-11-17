#!/usr/bin/env python
"""Evaluate CSLR models: accuracy, F1, latency, model size (+ optional robustness)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cslr.augment import augment_sequence  # noqa: E402
from app.cslr.dataset import label_names, load_split  # noqa: E402
from app.cslr.features import normalize_sequence  # noqa: E402
from app.cslr.metrics import classification_metrics, measure_latency_ms, model_efficiency  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sequences")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--split", default="test")
    parser.add_argument("--robustness", action="store_true", help="Also eval speed/noise-augmented copy")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    labels = label_names(args.data)
    model = tf.keras.models.load_model(args.model)
    x, y = load_split(args.data, args.split, args.seq_len)
    y_pred = model.predict(x, verbose=0).argmax(axis=1)

    result = {
        "split": args.split,
        "model": str(args.model),
        **classification_metrics(y, y_pred, labels),
        **model_efficiency(model),
        **measure_latency_ms(model, x[: min(16, len(x))]),
    }

    if args.robustness:
        rng = np.random.default_rng(1)
        x_rob = np.stack([normalize_sequence(augment_sequence(s, rng=rng)) for s in x])
        y_rob = model.predict(x_rob, verbose=0).argmax(axis=1)
        result["robustness"] = classification_metrics(y, y_rob, labels)

    print(json.dumps({k: result[k] for k in result if k != "report"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
