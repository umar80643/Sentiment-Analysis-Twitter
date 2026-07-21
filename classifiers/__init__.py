"""Model-agnostic classifier wrappers: NaiveBayes (NLTK, unchanged) plus new
LogisticRegression / LinearSVC (scikit-learn) behind one shared interface.
"""
from classifiers.base import Prediction, SentimentClassifier
from classifiers.registry import build_classifier, model_runs

__all__ = ["Prediction", "SentimentClassifier", "build_classifier", "model_runs"]
