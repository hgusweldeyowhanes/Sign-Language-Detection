"""Load the Keras sign classifier with Teachable-Machine compatibility."""
from __future__ import annotations

from pathlib import Path

from keras.layers import DepthwiseConv2D
from keras.models import load_model

from .config import MODEL_PATH


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


def load_classifier(model_path: Path | str | None = None):
    path = Path(model_path or MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Place best_model.keras in the project root "
            "(or pass --model path)."
        )
    return load_model(path, custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D})
