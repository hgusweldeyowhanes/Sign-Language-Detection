"""Fingerspelling buffer: stable letters → current word → sentence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Callable

from .config import (
    LETTER_COOLDOWN_SECONDS,
    LETTER_MIN_CONFIDENCE,
    LETTER_STABLE_FRAMES,
    WORD_IDLE_SECONDS,
)
from .phrase_engine import TranscriptEntry


@dataclass(frozen=True)
class SpellEvent:
    kind: str  # "letter" | "word"
    value: str


@dataclass
class FingerspellEngine:
    """Debounce letters, assemble a word, then append it to the sentence."""

    stable_frames: int = LETTER_STABLE_FRAMES
    idle_seconds: float = WORD_IDLE_SECONDS
    cooldown_seconds: float = LETTER_COOLDOWN_SECONDS
    min_confidence: float = LETTER_MIN_CONFIDENCE
    _stable_count: int = 0
    _last_candidate: str | None = None
    _last_committed: str | None = None
    _last_commit_at: float = 0.0
    _last_seen_at: float = 0.0
    _released: bool = True
    current_word: str = ""
    words: list[str] = field(default_factory=list)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    last_event: str = ""
    last_letter: str | None = None
    last_confidence: float = 0.0
    _clock: Callable[[], float] = field(default=time.time, repr=False)

    def _now(self) -> float:
        return self._clock()

    def update(self, letter: str | None, confidence: float = 0.0) -> SpellEvent | None:
        """
        Feed one frame. Returns a letter or auto-committed word event.
        """
        now = self._now()
        if not letter:
            self._stable_count = 0
            self._last_candidate = None
            self._released = True
            if (
                self.current_word
                and self._last_seen_at
                and (now - self._last_seen_at) >= self.idle_seconds
            ):
                word = self.commit_word()
                if word:
                    return SpellEvent("word", word)
            return None

        self._last_seen_at = now
        self.last_letter = letter
        self.last_confidence = confidence

        if letter == self._last_candidate:
            self._stable_count += 1
        else:
            self._last_candidate = letter
            self._stable_count = 1
            if letter != self._last_committed:
                self._released = True

        if self._stable_count < self.stable_frames:
            return None
        if confidence < self.min_confidence:
            return None
        if not self._released:
            return None
        if (
            letter == self._last_committed
            and (now - self._last_commit_at) < self.cooldown_seconds
        ):
            return None

        self._commit_letter(letter, confidence)
        return SpellEvent("letter", letter)

    def _commit_letter(self, letter: str, confidence: float) -> None:
        self.current_word += letter
        self._last_committed = letter
        self._last_commit_at = self._now()
        self._stable_count = 0
        self._released = False
        self.last_event = f"Letter “{letter}” → {self.current_word}"

    def commit_word(self) -> str | None:
        word = self.current_word.strip()
        if not word:
            return None
        self.words.append(word)
        self.transcript.append(
            TranscriptEntry(
                phrase=word,
                confidence=round(self.last_confidence, 3),
                at=datetime.now().strftime("%H:%M:%S"),
            )
        )
        self.current_word = ""
        self._last_committed = None
        self._released = True
        self.last_event = f"Word “{word}”"
        return word

    def backspace(self) -> str | None:
        if self.current_word:
            removed = self.current_word[-1]
            self.current_word = self.current_word[:-1]
            self._released = True
            self.last_event = f"Deleted letter “{removed}”"
            return removed
        if self.words:
            removed = self.words.pop()
            if self.transcript:
                self.transcript.pop()
            self.last_event = f"Removed word “{removed}”"
            return removed
        return None

    def sentence(self) -> str:
        parts = list(self.words)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts).strip()

    def undo(self) -> str | None:
        return self.backspace()

    def clear(self) -> None:
        self.current_word = ""
        self.words.clear()
        self.transcript.clear()
        self._stable_count = 0
        self._last_candidate = None
        self._last_committed = None
        self._released = True
        self.last_event = "Conversation cleared"

    def end_sentence(self) -> str:
        self.commit_word()
        text = " ".join(self.words).strip()
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
            "current_word": self.current_word,
            "words": list(self.words),
            "entries": [e.__dict__ for e in self.transcript],
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.last_event = f"Exported {path.name}"
        return path
