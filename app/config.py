"""Sign Language Detection — shared config."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "best_model.keras"

LABELS = ["GOODBYE", "HELLO", "I LOVE YOU", "THANK YOU"]

IMG_SIZE = 224
OFFSET = 20
CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAMES = 5
COOLDOWN_SECONDS = 1.6

# Conversation / practice phrases (composed from labels)
PRACTICE_TARGETS = [
    "HELLO",
    "THANK YOU",
    "I LOVE YOU",
    "GOODBYE",
    "HELLO THANK YOU",
    "HELLO I LOVE YOU",
    "THANK YOU GOODBYE",
]
