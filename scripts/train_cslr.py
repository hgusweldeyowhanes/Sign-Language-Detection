#!/usr/bin/env python
"""Train a lightweight continuous SLR model (cnn1d | lstm | transformer)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cslr.dataset import label_names, load_split  # noqa: E402
from app.cslr.features import LANDMARK_DIM  # noqa: E402
from app.cslr.metrics import classification_metrics, measure_latency_ms, model_efficiency  # noqa: E402
from app.cslr.models import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sequences")
    parser.add_argument("--arch", choices=["cnn1d", "lstm", "transformer"], default="lstm")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "cslr")
    args = parser.parse_args()

    labels = label_names(args.data)
    x_train, y_train = load_split(args.data, "train", args.seq_len, augment=True, seed=0)
    x_val, y_val = load_split(args.data, "val", args.seq_len, augment=False)

    model = build_model(args.arch, args.seq_len, LANDMARK_DIM, len(labels))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"{args.arch}_best.keras"
    model.save(model_path)

    # quick val metrics
    import numpy as np

    probs = model.predict(x_val, verbose=0)
    y_pred = probs.argmax(axis=1)
    metrics = classification_metrics(y_val, y_pred, labels)
    metrics.update(model_efficiency(model))
    metrics.update(measure_latency_ms(model, x_val[: min(16, len(x_val))]))
    metrics["architecture"] = args.arch
    metrics["seq_len"] = args.seq_len
    metrics["labels"] = labels
    metrics["history"] = {k: [float(x) for x in v] for k, v in history.history.items()}

    (args.out / f"{args.arch}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    meta = {"architecture": args.arch, "seq_len": args.seq_len, "labels": labels, "model": str(model_path.name)}
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("accuracy", "f1_macro", "params", "size_mb_fp32", "per_sample_ms_mean")}, indent=2))
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
