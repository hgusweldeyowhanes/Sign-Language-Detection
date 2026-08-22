"""Advanced Conversation Mode — stable gestures → sentence buffer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path

from .config import COOLDOWN_SECONDS, STABLE_FRAMES


@dataclass
class TranscriptEntry:
    phrase: str
    confidence: float
    at: str


@dataclass
class PhraseEngine:
    """Debounce repeated frames, build a spoken sentence, keep a transcript."""

    stable_frames: int = STABLE_FRAMES
    cooldown_seconds: float = COOLDOWN_SECONDS
    _stable_count: int = 0
    _last_candidate: str | None = None
    _last_committed: str | None = None
    _last_commit_at: float = 0.0
    tokens: list[str] = field(default_factory=list)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    last_event: str = ""

    def update(self, label: str | None, confidence: float = 0.0) -> str | None:
        """
        Feed one frame's label. Returns a newly committed phrase, or None.
        """
        if not label:
            self._stable_count = 0
            self._last_candidate = None
            return None

        if label == self._last_candidate:
            self._stable_count += 1
        else:
            self._last_candidate = label
            self._stable_count = 1

        if self._stable_count < self.stable_frames:
            return None

        now = time.time()
        if (
            label == self._last_committed
            and (now - self._last_commit_at) < self.cooldown_seconds
        ):
            return None

        self._commit(label, confidence)
        return label

    def _commit(self, label: str, confidence: float) -> None:
        self.tokens.append(label)
        entry = TranscriptEntry(
            phrase=label,
            confidence=round(confidence, 3),
            at=datetime.now().strftime("%H:%M:%S"),
        )
        self.transcript.append(entry)
        self._last_committed = label
        self._last_commit_at = time.time()
        self._stable_count = 0
        self.last_event = f"Added “{label}” ({confidence:.0%})"

    def sentence(self) -> str:
        return " ".join(self.tokens).strip()

    def undo(self) -> str | None:
        if not self.tokens:
            return None
        removed = self.tokens.pop()
        if self.transcript:
            self.transcript.pop()
        self.last_event = f"Removed “{removed}”"
        return removed

    def clear(self) -> None:
        self.tokens.clear()
        self.transcript.clear()
        self._stable_count = 0
        self._last_candidate = None
        self._last_committed = None
        self.last_event = "Conversation cleared"

    def end_sentence(self) -> str:
        text = self.sentence()
        if text and not text.endswith((".", "!", "?")):
            text = f"{text}."
        self.last_event = "Sentence ready"
        return text

    def export_txt(self, path: Path) -> Path:
        lines = [
            f"[{e.at}] {e.phrase}  (conf {e.confidence:.0%})" for e in self.transcript
        ]
        body = "Sentence: " + self.sentence() + "\n\n" + "\n".join(lines)
        path.write_text(body, encoding="utf-8")
        self.last_event = f"Exported {path.name}"
        return path

    def export_json(self, path: Path) -> Path:
        payload = {
            "sentence": self.sentence(),
            "entries": [e.__dict__ for e in self.transcript],
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.last_event = f"Exported {path.name}"
        return path
