"""Finds personal data with a local token tagger, so nothing leaves the machine before it is masked.

The weights are IndoPII Base (Apache 2.0), an ONNX tagger for 21 kinds of personal data in UU PDP. The repository's own
Python is not used: only the weights, the tokenizer and the label list are read, and spans are decoded here.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

EDGE_CHARACTERS = " \t\r\n,.;:()[]{}\"'"


@dataclass(frozen=True)
class IdentifierSpan:
    label: str
    start: int
    end: int


class IdentifierFinder(Protocol):
    async def find(self, text: str) -> list[IdentifierSpan]:
        """Spans left to right, none overlapping."""
        ...


def decode_spans(
    text: str, offsets: list[tuple[int, int]], probabilities: np.ndarray, tags: list[str], threshold: float
) -> list[IdentifierSpan]:
    """BIO tags to spans. Each span's score is the mean of its tokens' B and I probability."""
    spans: list[IdentifierSpan] = []
    label, start, end = "", 0, 0
    scores: list[float] = []

    def close() -> None:
        trimmed_start, trimmed_end = start, end
        while trimmed_start < trimmed_end and text[trimmed_start] in EDGE_CHARACTERS:
            trimmed_start += 1
        while trimmed_end > trimmed_start and text[trimmed_end - 1] in EDGE_CHARACTERS:
            trimmed_end -= 1
        if label and trimmed_end > trimmed_start and sum(scores) / len(scores) >= threshold:
            spans.append(IdentifierSpan(label, trimmed_start, trimmed_end))

    best_tags = probabilities.argmax(-1).tolist()
    for position, tag_index in enumerate(best_tags):  # each token depends on the span the last one left open
        tag = tags[tag_index]
        if tag == "O":
            close()
            label, scores = "", []
            continue
        prefix, tag_label = tag[0], tag[2:]
        pair_index = tag_index + 1 if prefix == "B" else tag_index - 1
        score = float(probabilities[position, tag_index] + probabilities[position, pair_index])
        token_start, token_end = offsets[position]
        same_span = label == tag_label and "\n" not in text[end:token_start]
        if same_span and (prefix == "I" or token_start == end):
            end = token_end
            scores.append(score)
        else:
            close()
            label, start, end, scores = tag_label, token_start, token_end, [score]
    close()
    return spans


class OnnxIdentifierTagger:
    """Loaded once at startup. Long texts are read in overlapping windows, weighted towards each window's middle."""

    def __init__(self, repository: str, revision: str, threshold: float, threads: int) -> None:
        folder = Path(
            snapshot_download(
                repository, revision=revision, allow_patterns=["model.onnx", "tokenizer.json", "indopii.json"]
            )
        )
        meta = json.loads((folder / "indopii.json").read_text(encoding="utf-8"))
        self._tags: list[str] = meta["labels"]
        self._window = int(meta["max_len"]) - 2
        self._stride = int(meta["stride"])
        self._cls_id, self._sep_id = int(meta["cls_id"]), int(meta["sep_id"])
        self._threshold = threshold
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = onnxruntime.InferenceSession(
            str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

    def _token_probabilities(self, ids: list[int]) -> np.ndarray:
        totals = np.zeros((len(ids), len(self._tags)), dtype=np.float32)
        weights = np.zeros(len(ids), dtype=np.float32)
        start = 0
        while True:  # one pass per window, the last one ends at the text's end
            end = min(len(ids), start + self._window)
            window = np.array([[self._cls_id, *ids[start:end], self._sep_id]], dtype=np.int64)
            logits = self._session.run(None, {"input_ids": window, "attention_mask": np.ones_like(window)})[0][0][1:-1]
            logits = logits - logits.max(-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(-1, keepdims=True)
            positions = np.arange(end - start, dtype=np.float32)
            edge_weight = np.minimum(positions + 1, end - start - positions) + 1.0
            totals[start:end] += probabilities * edge_weight[:, None]
            weights[start:end] += edge_weight
            if end == len(ids):
                return totals / weights[:, None]
            start = end - self._stride

    def find_now(self, text: str) -> list[IdentifierSpan]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        if not encoding.ids:
            return []
        probabilities = self._token_probabilities(encoding.ids)
        return decode_spans(text, encoding.offsets, probabilities, self._tags, self._threshold)

    async def find(self, text: str) -> list[IdentifierSpan]:
        return await asyncio.to_thread(self.find_now, text)
