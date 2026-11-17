from __future__ import annotations

from typing import Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_REGISTRY: dict[str, Callable[..., keras.Model]] = {}


def register(name: str):
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn

    return deco


@register("cnn1d")
def build_cnn1d(seq_len: int, feat_dim: int, n_classes: int, width: int = 64) -> keras.Model:
    """Lightweight temporal CNN — strong baseline for low-resource devices."""
    inp = keras.Input(shape=(seq_len, feat_dim), name="landmarks")
    x = layers.Conv1D(width, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(width * 2, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(width, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cslr_cnn1d")


@register("lstm")
def build_lstm(seq_len: int, feat_dim: int, n_classes: int, width: int = 64) -> keras.Model:
    """Spatiotemporal CNN front-end + bidirectional LSTM."""
    inp = keras.Input(shape=(seq_len, feat_dim), name="landmarks")
    x = layers.Conv1D(width, 3, padding="same", activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(width, return_sequences=False))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(width, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cslr_lstm")


@register("transformer")
def build_transformer(
    seq_len: int,
    feat_dim: int,
    n_classes: int,
    width: int = 64,
    heads: int = 4,
    blocks: int = 2,
) -> keras.Model:
    """Compact Transformer encoder over landmark frames."""
    inp = keras.Input(shape=(seq_len, feat_dim), name="landmarks")
    x = layers.Dense(width)(inp)
    # learnable-ish positional encoding via Conv1D depthwise mix
    pos = layers.Conv1D(width, 1, padding="same")(x)
    x = x + pos
    for _ in range(blocks):
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=width // heads)(x, x)
        x = layers.LayerNormalization()(x + attn)
        ff = layers.Dense(width * 2, activation="relu")(x)
        ff = layers.Dense(width)(ff)
        x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cslr_transformer")


def build_model(
    architecture: str,
    seq_len: int,
    feat_dim: int,
    n_classes: int,
    **kwargs,
) -> keras.Model:
    key = architecture.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](seq_len, feat_dim, n_classes, **kwargs)


def count_params(model: keras.Model) -> int:
    return int(model.count_params())


def estimate_model_mb(model: keras.Model) -> float:
    """Rough float32 weight size in megabytes."""
    return count_params(model) * 4 / (1024 * 1024)
