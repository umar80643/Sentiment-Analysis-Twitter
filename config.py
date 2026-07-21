"""Central paths, default experiment settings, and model hyperparameters.

Every path and per-classifier hyperparameter used by train.py, the classifiers/
package, and (from Phase 2) app.py is defined here, so nothing is hard-coded
at the call site.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
# Original 1.1.0 artifact path/schema. train.py keeps writing a NaiveBayes
# model here unchanged so predict.py and app.py never need to know the new
# multi-model machinery exists.
DEFAULT_MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"

ModelName = Literal["NaiveBayesClassifier", "LogisticRegression", "LinearSVC"]
FeatureMode = Literal["bow", "tfidf"]

ALL_MODELS: tuple[ModelName, ...] = ("NaiveBayesClassifier", "LogisticRegression", "LinearSVC")
ALL_FEATURE_MODES: tuple[FeatureMode, ...] = ("bow", "tfidf")


@dataclass(frozen=True)
class Settings:
    folds: int = 10
    random_seed: int = 42
    ngram: int = 1
    use_negation: bool = False


@dataclass(frozen=True)
class LogisticRegressionParams:
    C: float = 1.0
    max_iter: int = 1000
    # Sentiment140/Sanders labels are imbalanced (few "neu" tweets); balancing
    # class weight keeps the minority class from being ignored.
    class_weight: str | None = "balanced"


@dataclass(frozen=True)
class LinearSVCParams:
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str | None = "balanced"
    # LinearSVC has no predict_proba; calibrating it fits a sigmoid on top of
    # the decision function so the UI can show a real confidence percentage
    # instead of an unnormalised margin.
    calibrate: bool = True


@dataclass(frozen=True)
class HyperParams:
    logistic_regression: LogisticRegressionParams = field(default_factory=LogisticRegressionParams)
    linear_svc: LinearSVCParams = field(default_factory=LinearSVCParams)


DEFAULT_HYPERPARAMS = HyperParams()
