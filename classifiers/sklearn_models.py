"""Scikit-learn linear-model wrappers (Logistic Regression, Linear SVM).

Both train on the project's existing engineered features (see
features/vectorization.py) rather than raw text, so every classifier in the
comparison sees identical inputs -- the only thing that differs between runs
is the learning algorithm and, optionally, the BOW/TF-IDF weighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from classifiers.base import Prediction, Tweet
from config import DEFAULT_HYPERPARAMS, FeatureMode
from features.vectorization import build_vectorizer, featurize_text, featurize_tweets
from sentiment import FeatureSettings


def _build_estimator(name: str):
    params = DEFAULT_HYPERPARAMS
    if name == "LogisticRegression":
        settings = params.logistic_regression
        return LogisticRegression(C=settings.C, max_iter=settings.max_iter, class_weight=settings.class_weight)
    if name == "LinearSVC":
        settings = params.linear_svc
        base = LinearSVC(C=settings.C, max_iter=settings.max_iter, class_weight=settings.class_weight)
        if not settings.calibrate:
            return base
        # LinearSVC has no predict_proba; CalibratedClassifierCV fits a
        # sigmoid on top of its decision function so predictions carry a real,
        # normalised confidence percentage instead of an unbounded margin.
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    raise ValueError(f"Unsupported classifier: {name}")


@dataclass
class SklearnSentimentModel:
    name: str  # "LogisticRegression" or "LinearSVC"
    feature_mode: FeatureMode = "tfidf"
    _vectorizer: object = field(default=None, repr=False)
    _estimator: object = field(default=None, repr=False)
    _settings: FeatureSettings = field(default_factory=FeatureSettings, repr=False)

    def fit(self, tweets: list[Tweet], settings: FeatureSettings) -> None:
        self._settings = settings
        features, labels = featurize_tweets(tweets, settings)
        self._vectorizer = build_vectorizer(self.feature_mode)
        design_matrix = self._vectorizer.fit_transform(features)
        self._estimator = _build_estimator(self.name)
        self._estimator.fit(design_matrix, labels)

    def predict(self, text: str) -> Prediction:
        features = featurize_text(text, self._settings)
        design_matrix = self._vectorizer.transform([features])
        label = self._estimator.predict(design_matrix)[0]

        confidence, distribution = None, None
        if hasattr(self._estimator, "predict_proba"):
            probabilities = self._estimator.predict_proba(design_matrix)[0]
            distribution = {
                class_label: float(probability)
                for class_label, probability in zip(self._estimator.classes_, probabilities)
            }
            confidence = distribution.get(label)
        return Prediction(label=label, confidence=confidence, class_probabilities=distribution)

    @property
    def raw_model(self) -> object:
        return {"vectorizer": self._vectorizer, "estimator": self._estimator}

    def explain(self, text: str, top_n: int = 8) -> list[tuple[str, float]]:
        """Signed per-feature contribution to the predicted label, taken
        directly from the linear model's coefficients (coefficient x feature
        value present in this tweet). Works for LogisticRegression and for
        LinearSVC, whether or not it's wrapped in CalibratedClassifierCV.
        """
        base_estimator = self._underlying_linear_estimator()
        if base_estimator is None or not hasattr(base_estimator, "coef_"):
            return []

        features = featurize_text(text, self._settings)
        dict_vectorizer: Any = self._vectorizer.named_steps["dict"]
        feature_index = {name: index for index, name in enumerate(dict_vectorizer.get_feature_names_out())}

        classes = list(base_estimator.classes_)
        predicted_label = self.predict(text).label
        if predicted_label not in classes:
            return []
        # Binary linear models store a single coefficient row for the
        # positive class; multiclass models store one row per class.
        row = 0 if base_estimator.coef_.shape[0] == 1 else classes.index(predicted_label)
        coefficients = base_estimator.coef_[row]

        contributions = [
            (name, float(coefficients[feature_index[name]] * value))
            for name, value in features.items()
            if name in feature_index
        ]
        contributions.sort(key=lambda item: item[1], reverse=True)
        if len(contributions) <= 2 * top_n:
            return contributions
        return contributions[:top_n] + contributions[-top_n:]

    def _underlying_linear_estimator(self):
        estimator = self._estimator
        if hasattr(estimator, "calibrated_classifiers_"):
            calibrated = estimator.calibrated_classifiers_[0]
            return getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
        return estimator
