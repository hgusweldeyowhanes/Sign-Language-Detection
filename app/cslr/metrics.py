from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .models import count_params, estimate_model_mb


def classification_metrics(y_true, y_pred, label_names: Optional[list[str]] = None) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": report,
    }


def measure_latency_ms(model, x_batch: np.ndarray, warmup: int = 5, runs: int = 30) -> dict:
    # warmup
    for _ in range(warmup):
        model.predict(x_batch, verbose=0)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(x_batch, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    per_sample = arr / max(1, x_batch.shape[0])
    return {
        "batch_latency_ms_mean": float(arr.mean()),
        "batch_latency_ms_p95": float(np.percentile(arr, 95)),
        "per_sample_ms_mean": float(per_sample.mean()),
    }


def model_efficiency(model) -> dict:
    return {
        "params": count_params(model),
        "size_mb_fp32": round(estimate_model_mb(model), 4),
    }
