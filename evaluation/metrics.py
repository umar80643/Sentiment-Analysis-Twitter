"""Cross-validated metric computation shared by every classifier type.

This reuses sentiment.k_fold_cross_validation (unchanged) so the fold-split
logic that NaiveBayes has always used is exactly what LogisticRegression and
LinearSVC are evaluated with too.

cross_validate() returns two objects:
- RunMetrics: small scalar summary, cheap to keep forever in models/metrics.json
- RunDiagnostics: confusion matrix + per-class ROC/PR curves, sized to the
  dataset. This is only embedded in that run's own model artifact (.pkl),
  which the Streamlit dashboard reads to plot curves for whichever model the
  user selects, without bloating the long-lived metrics history.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from classifiers.base import SentimentClassifier, Tweet
from sentiment import FeatureSettings, k_fold_cross_validation


@dataclass
class RunMetrics:
    classifier: str
    feature_mode: str | None
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    train_seconds: float
    predict_seconds_per_1000: float
    tweet_count: int
    fold_accuracies: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunDiagnostics:
    labels: list[str]
    confusion_matrix: list[list[int]]
    # Per label (one-vs-rest): {"fpr": [...], "tpr": [...], "auc": float}
    roc_curves: dict[str, dict[str, object]] = field(default_factory=dict)
    pr_curves: dict[str, dict[str, object]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def cross_validate(
    build_model: Callable[[], SentimentClassifier],
    tweets: Sequence[Tweet],
    settings: FeatureSettings,
    folds: int,
    seed: int,
    classifier_name: str,
    feature_mode: str | None = None,
) -> tuple[RunMetrics, RunDiagnostics]:
    """Run k-fold cross-validation, aggregating predictions across all folds
    before computing metrics (rather than averaging per-fold metrics), which
    is the more standard and less noisy approach for imbalanced label sets.
    build_model() must return a fresh, untrained classifier each time it's
    called, since each fold needs its own model instance.
    """
    all_true: list[str] = []
    all_predicted: list[str] = []
    all_probabilities: list[dict[str, float] | None] = []
    fold_accuracies: list[float] = []
    total_train_seconds = 0.0
    total_predict_seconds = 0.0
    total_predicted = 0

    for train_items, test_items in k_fold_cross_validation(list(tweets), folds, randomise=True, seed=seed):
        model = build_model()

        start = time.perf_counter()
        model.fit(train_items, settings)
        total_train_seconds += time.perf_counter() - start

        fold_true = [label for _text, label, _subject, _query in test_items]
        start = time.perf_counter()
        fold_predictions = [model.predict(text) for text, _label, _subject, _query in test_items]
        total_predict_seconds += time.perf_counter() - start
        total_predicted += len(test_items)

        fold_predicted = [prediction.label for prediction in fold_predictions]
        all_true.extend(fold_true)
        all_predicted.extend(fold_predicted)
        all_probabilities.extend(prediction.class_probabilities for prediction in fold_predictions)
        fold_accuracies.append(accuracy_score(fold_true, fold_predicted))

    predict_seconds_per_1000 = (total_predict_seconds / total_predicted) * 1000 if total_predicted else 0.0
    metrics = RunMetrics(
        classifier=classifier_name,
        feature_mode=feature_mode,
        accuracy=accuracy_score(all_true, all_predicted),
        precision_macro=precision_score(all_true, all_predicted, average="macro", zero_division=0),
        recall_macro=recall_score(all_true, all_predicted, average="macro", zero_division=0),
        f1_macro=f1_score(all_true, all_predicted, average="macro", zero_division=0),
        train_seconds=total_train_seconds,
        predict_seconds_per_1000=predict_seconds_per_1000,
        tweet_count=len(tweets),
        fold_accuracies=fold_accuracies,
    )
    diagnostics = _compute_diagnostics(all_true, all_predicted, all_probabilities)
    return metrics, diagnostics


def _compute_diagnostics(
    all_true: list[str], all_predicted: list[str], all_probabilities: list[dict[str, float] | None]
) -> RunDiagnostics:
    labels = sorted(set(all_true) | set(all_predicted))
    matrix = confusion_matrix(all_true, all_predicted, labels=labels).tolist()

    roc_curves: dict[str, dict[str, object]] = {}
    pr_curves: dict[str, dict[str, object]] = {}
    if all_probabilities and all(probability is not None for probability in all_probabilities):
        # One-vs-rest binarisation done manually (rather than via
        # sklearn.preprocessing.label_binarize) so 2-class and 3-class label
        # sets are both handled the same way -- label_binarize collapses to a
        # single column for exactly 2 classes, which would break the
        # per-label loop below.
        true_matrix = np.array([[1 if truth == label else 0 for label in labels] for truth in all_true])
        for index, label in enumerate(labels):
            scores = [probability.get(label, 0.0) for probability in all_probabilities]
            fpr, tpr, _ = roc_curve(true_matrix[:, index], scores)
            precision_values, recall_values, _ = precision_recall_curve(true_matrix[:, index], scores)
            roc_curves[label] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
            pr_curves[label] = {
                "precision": precision_values.tolist(),
                "recall": recall_values.tolist(),
                "auc": float(auc(recall_values, precision_values)),
            }

    return RunDiagnostics(labels=labels, confusion_matrix=matrix, roc_curves=roc_curves, pr_curves=pr_curves)
