# Real-Time Sign Language Recognition Using Deep Learning

**Research focus:** *A Lightweight Spatiotemporal Deep Learning Framework for Continuous Sign Language Recognition.*

Fields: Computer Vision · Deep Learning · Accessibility  

This repo extends the earlier isolated Amharic/ASL-style webcam demo into a **continuous** pipeline: **MediaPipe landmarks → CNN / LSTM / Transformer**, with robustness-oriented augmentation, edge export (TFLite), and an evaluation harness (accuracy, F1, latency, model size).

> Isolated phrase CNN + rule-based fingerspelling remain available. Continuous CSLR is the research path.

---

## Problem

Recognizing individual static signs is easier than understanding **continuous** sign language in real environments—variable lighting, backgrounds, and signing speeds. This project targets continuous gloss streams on **low-resource devices**.

## What’s included

| Mode | Entry | Notes |
|---|---|---|
| Continuous CSLR (new) | `python run_continuous.py` or `python run_advanced.py --mode continuous` | Sliding-window landmark sequences |
| Phrase CNN | `python run_advanced.py` | Isolated RGB crops → Keras CNN |
| Fingerspell A–Z | `python run_advanced.py --mode fingerspell` | MediaPipe geometry (no CNN) |

**Research contribution:** a lightweight spatiotemporal stack for continuous recognition under resource constraints, built on MediaPipe landmarks rather than heavy RGB video backbones.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 1) synthetic sequences (pipeline smoke test)
python scripts/generate_synthetic_sequences.py

# 2) train a lightweight model
python scripts/train_cslr.py --arch lstm --epochs 15

# 3) evaluate (accuracy, F1, latency, size)
python scripts/eval_cslr.py --model artifacts/cslr/lstm_best.keras --robustness

# 4) realtime continuous recognition
python run_continuous.py
```

Compare architectures:

```bash
python scripts/train_cslr.py --arch cnn1d
python scripts/train_cslr.py --arch transformer
python scripts/export_lightweight.py --model artifacts/cslr/cnn1d_best.keras --quantize
```

Collect **real** MediaPipe sequences (replace synthetic data for paper-quality claims):

```bash
python scripts/collect_sequences.py
```

## Continuous architecture

```
Webcam
  → MediaPipe Hands (21 landmarks × xyz)
  → wrist-centered / scale-normalized features (63-D)
  → sliding window (T=48) + speed/noise augmentation (train)
  → cnn1d | BiLSTM | compact Transformer
  → gloss debounce → continuous sentence buffer
```

Optional: export TFLite for phones / edge CPUs.

## Evaluation protocol

`scripts/eval_cslr.py` reports:

- **Accuracy**, **macro/weighted F1**
- **Inference latency** (batch + per-sample ms)
- **Model size** (parameter count, FP32 MB)
- **`--robustness`**: speed jitter + landmark noise (proxy for signing speed / sensor noise)

## Project layout

```
app/
  cslr/                 # continuous SLR framework
    features.py         # landmark vectors, buffers, resample
    models.py           # cnn1d / lstm / transformer
    continuous.py       # streaming recognizer
    augment.py          # speed / noise / dropout
    dataset.py / metrics.py
  landmarks.py          # MediaPipe tracker (shared)
  ui_advanced.py        # Conversation Studio (phrases / fingerspell)
scripts/
  generate_synthetic_sequences.py
  collect_sequences.py
  train_cslr.py
  eval_cslr.py
  export_lightweight.py
run_continuous.py
run_advanced.py
```

## Technologies

Python · OpenCV · MediaPipe · TensorFlow/Keras · LSTM · Transformers · scikit-learn · TFLite

## Legacy phrase / fingerspell

Phrase mode still expects `best_model.keras` for the 4-class RGB CNN (`GOODBYE`, `HELLO`, `I LOVE YOU`, `THANK YOU`). Fingerspell works without weights.

```bash
python run_advanced.py
python run_advanced.py --mode fingerspell
python -m pytest
```

## Extending to Amharic / EthSL

1. Edit `CSLR_LABELS` in `app/config.py` with your gloss inventory.  
2. Record clips via `scripts/collect_sequences.py`.  
3. Retrain and evaluate; report latency/size on the target device.

## Future work

- CTC / attention decoding for unsegmented multi-gloss sentences  
- Pose + two-hand fusion  
- On-device TFLite benchmark table (phone / Pi)  
- Larger EthSL / Amharic continuous corpus  

Apache-2.0 · see [LICENSE](LICENSE)
