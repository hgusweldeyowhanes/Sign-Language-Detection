"""Continuous Sign Language Recognition (CSLR) — spatiotemporal landmark models."""

from .features import SequenceBuffer, landmarks_to_vector, normalize_sequence

__all__ = [
    "SequenceBuffer",
    "landmarks_to_vector",
    "normalize_sequence",
    "ContinuousRecognizer",
    "build_model",
]


def __getattr__(name: str):
    if name == "ContinuousRecognizer":
        from .continuous import ContinuousRecognizer

        return ContinuousRecognizer
    if name in {"build_model", "MODEL_REGISTRY", "count_params", "estimate_model_mb"}:
        from . import models

        return getattr(models, name)
    raise AttributeError(name)
