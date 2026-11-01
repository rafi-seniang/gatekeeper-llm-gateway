import numpy as np

from app.engines.identifier_tagger import IdentifierSpan, decode_spans

TAGS = ["O", "B-NAME", "I-NAME", "B-PHONE", "I-PHONE"]


def one_hot(*tags: int) -> np.ndarray:
    return np.eye(len(TAGS), dtype=np.float32)[list(tags)]


def test_tokens_become_spans() -> None:
    text = "Halo Budi Santoso, 0812"
    offsets = [(0, 4), (5, 9), (10, 17), (19, 23)]
    spans = decode_spans(text, offsets, one_hot(0, 1, 2, 3), TAGS, threshold=0.5)
    assert spans == [IdentifierSpan("NAME", 5, 17), IdentifierSpan("PHONE", 19, 23)]


def test_a_span_is_trimmed_of_punctuation() -> None:
    assert decode_spans("(Budi)", [(0, 6)], one_hot(1), TAGS, 0.5) == [IdentifierSpan("NAME", 1, 5)]


def test_a_weak_span_is_dropped() -> None:
    probabilities = np.array([[0.7, 0.3, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0, 0.0]], dtype=np.float32)
    probabilities[1] = [0.2, 0.45, 0.0, 0.35, 0.0]
    assert decode_spans("x Budi", [(0, 1), (2, 6)], probabilities, TAGS, 0.9) == []


def test_a_name_on_the_next_line_is_a_new_span() -> None:
    spans = decode_spans("Budi\nSari", [(0, 4), (5, 9)], one_hot(1, 2), TAGS, 0.5)
    assert spans == [IdentifierSpan("NAME", 0, 4), IdentifierSpan("NAME", 5, 9)]
