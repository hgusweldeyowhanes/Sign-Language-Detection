#!/usr/bin/env python
"""Export a trained CSLR Keras model to TFLite (optionally quantized) for edge devices."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quantize", action="store_true", help="Dynamic-range quantization")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out = args.out or args.model.with_suffix(".tflite")
    out.write_bytes(tflite_model)
    print(f"Wrote {out} ({len(tflite_model)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
