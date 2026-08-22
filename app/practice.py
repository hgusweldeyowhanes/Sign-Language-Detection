"""Guided practice mode — hit a target phrase sequence."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import LABELS, PRACTICE_TARGETS


@dataclass
class PracticeSession:
    targets: list[str] = field(default_factory=lambda: list(PRACTICE_TARGETS))
    current: str = ""
    progress: list[str] = field(default_factory=list)
    score: int = 0
    attempts: int = 0
    message: str = "Press Next Target to start practicing."

    def next_target(self) -> str:
        self.current = random.choice(self.targets)
        self.progress.clear()
        self.message = f"Sign: {self.current}"
        return self.current

    def on_phrase(self, phrase: str) -> bool:
        """Return True if the practice target was completed."""
        if not self.current:
            self.message = "Choose a target first."
            return False
        self.attempts += 1
        expected = self._next_expected_label()
        if phrase != expected:
            self.message = f"Expected “{expected}”, got “{phrase}”."
            return False
        self.progress.append(phrase)
        built = " ".join(self.progress)
        if built == self.current:
            self.score += 1
            self.message = f"Perfect! +1  (score {self.score})"
            return True
        self.message = f"Good — next: {self._next_expected_label()}"
        return False

    def _next_expected_label(self) -> str | None:
        remaining = self.current
        for done in self.progress:
            if remaining.startswith(done):
                remaining = remaining[len(done) :].strip()
            else:
                return None
        for lab in sorted(LABELS, key=len, reverse=True):
            if remaining == lab or remaining.startswith(lab + " "):
                return lab
        return remaining or None
