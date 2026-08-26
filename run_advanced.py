#!/usr/bin/env python
"""Run the advanced Conversation Studio UI.

Usage:
  python run_advanced.py
  python run_advanced.py --mode fingerspell
  python run_advanced.py --model path/to/best_model.keras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Sign Language Conversation Studio")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to best_model.keras (default: ./best_model.keras)",
    )
    parser.add_argument(
        "--mode",
        choices=["phrases", "fingerspell"],
        default="phrases",
        help="Start in phrase-sign mode or A–Z fingerspelling",
    )
    args = parser.parse_args()
    from app.ui_advanced import main as run_ui

    run_ui(model_path=args.model, mode=args.mode)


if __name__ == "__main__":
    main()
