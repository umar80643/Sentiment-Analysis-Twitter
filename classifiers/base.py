"""Shared interface every classifier wrapper implements.

train.py, evaluation/metrics.py, and (from Phase 2) app.py all talk to
NaiveBayes, LogisticRegression, and LinearSVC through this one interface, so
none of them need per-model branching logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sentiment import FeatureSettings

Tweet = tuple[str, str, str, list[str]]


@dataclass(frozen=True)
class Prediction:
    """Result of classifying one piece of text.

    confidence / class_probabilities are None when a model can't produce a
    calibrated probability (this shouldn't happen for any model built through
    classifiers.registry, but callers should not assume it's always present).
    """

    label: str
    confidence: float | None
    class_probabilities: dict[str, float] | None = None


@runtime_checkable
class SentimentClassifier(Protocol):
    """Every wrapper trains on raw (text, label, subject, query) tuples and
    predicts on raw text; callers never touch tokenisation or vectorisation
    directly.
    """

    name: str
    feature_mode: str | None

    def fit(self, tweets: list[Tweet], settings: FeatureSettings) -> None: ...

    def predict(self, text: str) -> Prediction: ...

    @property
    def raw_model(self) -> object:
        """The underlying trained model object (an NLTK classifier, or a
        fitted sklearn estimator), for saving legacy-compatible artifacts.
        """
        ...
