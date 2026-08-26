from app.fingerspell import FingerspellEngine
from app.practice import SpellPractice


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def add(self, seconds):
        self.t += seconds


def _hold(engine, letter, frames=6, confidence=0.9):
    event = None
    for _ in range(frames):
        event = engine.update(letter, confidence)
    return event


def test_stable_letter_appends_to_word():
    clock = Clock()
    engine = FingerspellEngine(_clock=clock)
    event = _hold(engine, "H")
    assert event is not None
    assert event.kind == "letter"
    assert event.value == "H"
    assert engine.current_word == "H"


def test_same_letter_needs_release_for_double():
    clock = Clock()
    engine = FingerspellEngine(_clock=clock)
    _hold(engine, "L")
    extra = engine.update("L", 0.9)
    assert extra is None
    assert engine.current_word == "L"

    engine.update(None)
    clock.add(0.5)
    event = _hold(engine, "L")
    assert event is not None
    assert engine.current_word == "LL"


def test_idle_pause_commits_word():
    clock = Clock()
    engine = FingerspellEngine(_clock=clock, idle_seconds=1.0)
    _hold(engine, "H")
    engine.update(None)
    clock.add(1.05)
    event = engine.update(None)
    assert event is not None
    assert event.kind == "word"
    assert event.value == "H"
    assert engine.words == ["H"]
    assert engine.current_word == ""


def test_sentence_joins_words():
    clock = Clock()
    engine = FingerspellEngine(_clock=clock)
    _hold(engine, "H")
    engine.commit_word()
    _hold(engine, "I")
    assert engine.sentence() == "H I"


def test_spell_practice_letter_by_letter():
    session = SpellPractice(targets=["HI"])
    session.targets = ["HI"]
    session.next_target()
    assert session.current == "HI"
    assert session.on_letter("H") is False
    assert session.on_letter("X") is False
    assert session.on_letter("I") is True
    assert session.score == 1
