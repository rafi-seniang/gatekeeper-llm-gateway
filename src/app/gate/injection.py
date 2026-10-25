"""Prompt-injection guards. Each returns the probability that a text is an injection."""

import asyncio
from typing import Protocol

from app.gate.decisions import DecisionClient


class InjectionGuard(Protocol):
    name: str

    async def probability(self, text: str) -> float: ...


class ClassifierGuard:
    """A fine-tuned text classifier (DeBERTa), loaded once at startup, run off the event loop."""

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import pipeline  # heavy import, only where the classifier is used

        self.name = model_id
        self._classify = pipeline("text-classification", model=model_id, device=device, truncation=True, max_length=512)
        labels = list((self._classify.model.config.id2label or {}).values())
        injection_labels = [label for label in labels if label.upper() == "INJECTION"]
        if not injection_labels:
            raise RuntimeError(f"{model_id} has no INJECTION label (it has {labels}); choose another classifier")
        self._injection_label = injection_labels[0]

    async def probability(self, text: str) -> float:
        scores = await asyncio.to_thread(self._classify, text, top_k=None)
        return next(float(score["score"]) for score in scores if score["label"] == self._injection_label)


class DecisionGuard:
    """The same question asked of a decision model through the Jev wire format."""

    def __init__(self, decisions: DecisionClient) -> None:
        self.name = f"decision:{decisions.model}"
        self._decisions = decisions

    async def probability(self, text: str) -> float:
        return await self._decisions.injection_probability(text)
