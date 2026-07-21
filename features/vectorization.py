"""Numeric vectorization of the project's existing dict-based NLTK features.

Rather than re-implementing bag-of-words / TF-IDF from raw text -- which would
duplicate the tokenisation, stemming, n-gram, and negation logic already in
sentiment.py -- this module wraps the *existing* extract_features() output
with scikit-learn's DictVectorizer. That is what lets LogisticRegression and
LinearSVC train on exactly the same engineered features NaiveBayesClassifier
uses, so the accuracy/F1 comparison across models is a fair one (same inputs,
different learning algorithm) instead of comparing different feature
engineering pipelines.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from config import FeatureMode
from sentiment import FeatureSettings, extract_features, tokenise

Tweet = tuple[str, str, str, list[str]]


def build_vectorizer(mode: FeatureMode) -> Pipeline:
    """Bag-of-words keeps raw feature counts. TF-IDF reweights those same
    features by how distinctive they are across the training set, which
    typically helps linear models by down-weighting ubiquitous tokens
    relative to rarer, more sentiment-bearing ones.
    """
    steps: list[tuple[str, Any]] = [("dict", DictVectorizer(sparse=True))]
    if mode == "tfidf":
        steps.append(("tfidf", TfidfTransformer()))
    return Pipeline(steps)


def featurize_tweets(
    tweets: Sequence[Tweet], settings: FeatureSettings
) -> tuple[list[dict[str, float]], list[str]]:
    """Tokenise and extract_features() for every tweet, reusing sentiment.py
    exactly -- no tokenisation or feature-engineering logic is duplicated here.
    """
    features = [
        extract_features(tokenise(text, subject, query), settings)
        for text, _label, subject, query in tweets
    ]
    labels = [label for _text, label, _subject, _query in tweets]
    return features, labels


def featurize_text(text: str, settings: FeatureSettings) -> dict[str, float]:
    return extract_features(tokenise(text), settings)
