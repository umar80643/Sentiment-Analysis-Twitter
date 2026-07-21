"""Single place that knows how to construct each supported classifier, so
train.py never hard-codes per-model branching logic.
"""
from __future__ import annotations

from collections.abc import Sequence

from classifiers.base import SentimentClassifier
from classifiers.nltk_models import NLTKClassifierModel
from classifiers.sklearn_models import SklearnSentimentModel
from config import FeatureMode, ModelName


def build_classifier(model: ModelName, feature_mode: FeatureMode = "tfidf") -> SentimentClassifier:
    """Return a fresh, untrained classifier. feature_mode is ignored for
    NaiveBayesClassifier, which always trains on raw NLTK feature dicts.
    """
    if model == "NaiveBayesClassifier":
        return NLTKClassifierModel(name=model)
    if model in ("LogisticRegression", "LinearSVC"):
        return SklearnSentimentModel(name=model, feature_mode=feature_mode)
    raise ValueError(f"Unsupported model: {model}")


def model_runs(
    models: Sequence[ModelName], feature_modes: Sequence[FeatureMode]
) -> list[tuple[ModelName, FeatureMode | None]]:
    """Expand requested models x feature modes into the list of runs to
    perform. NaiveBayesClassifier ignores feature_mode, so it appears at most
    once per comparison regardless of how many feature modes were requested;
    LogisticRegression and LinearSVC appear once per requested feature mode.
    """
    runs: list[tuple[ModelName, FeatureMode | None]] = []
    for model in models:
        if model == "NaiveBayesClassifier":
            if ("NaiveBayesClassifier", None) not in runs:
                runs.append(("NaiveBayesClassifier", None))
        else:
            for mode in feature_modes:
                runs.append((model, mode))
    return runs
