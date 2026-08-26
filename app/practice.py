"""Guided practice mode — hit a target phrase sequence."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import LABELS, PRACTICE_TARGETS, SPELL_TARGETS


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


@dataclass
class SpellPractice:
    """Guided A–Z practice — spell a target word letter by letter."""

    targets: list[str] = field(default_factory=lambda: list(SPELL_TARGETS))
    current: str = ""
    index: int = 0
    score: int = 0
    attempts: int = 0
    message: str = "Press Next Target, then fingerspell the word."

    def next_target(self) -> str:
        self.current = random.choice(self.targets)
        self.index = 0
        self.message = f"Spell: {self.current}  (next: {self.current[0]})"
        return self.current

    def expected_letter(self) -> str | None:
        if not self.current or self.index >= len(self.current):
            return None
        return self.current[self.index]

    def on_letter(self, letter: str) -> bool:
        if not self.current:
            self.message = "Choose a target first."
            return False
        self.attempts += 1
        expected = self.expected_letter()
        if letter != expected:
            self.message = f"Expected “{expected}”, got “{letter}”."
            return False
        self.index += 1
        if self.index >= len(self.current):
            self.score += 1
            self.message = f"Perfect! {self.current}  (score {self.score})"
            return True
        typed = self.current[: self.index]
        self.message = f"{typed}_  next: {self.expected_letter()}"
        return False
