"""Wraps the original, untouched NLTK classifier pipeline in sentiment.py so
it satisfies the same SentimentClassifier interface as the new scikit-learn
models. No tokenisation, feature-extraction, or training logic is duplicated
or modified here -- this only adapts the existing functions to a common shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from classifiers.base import Prediction, Tweet
from sentiment import ClassifierName, FeatureSettings, Method, extract_features, tokenise, train_final_model


@dataclass
class NLTKClassifierModel:
    """Covers NaiveBayesClassifier (used in the new model comparison) as well
    as the original MaxentClassifier / DecisionTreeClassifier options, which
    remain fully importable via sentiment.py even though they're not part of
    the new --models comparison flag.
    """

    name: ClassifierName = "NaiveBayesClassifier"
    method: Method = "1step"
    feature_mode: str | None = None  # NLTK models always use raw feature dicts, never a vectorizer
    _model: object = field(default=None, repr=False)
    _settings: FeatureSettings = field(default_factory=FeatureSettings, repr=False)

    def fit(self, tweets: list[Tweet], settings: FeatureSettings) -> None:
        self._settings = settings
        self._model = train_final_model(tweets, self.name, self.method, settings.__dict__)

    def predict(self, text: str) -> Prediction:
        features = extract_features(tokenise(text), self._settings)
        model = self._model
        if isinstance(model, tuple):
            object_model, sentiment_model = model
            label = object_model.classify(features)
            classifier_for_probs = sentiment_model if label == "obj" else object_model
            label = sentiment_model.classify(features) if label == "obj" else label
        else:
            classifier_for_probs = model
            label = model.classify(features)

        confidence, distribution = None, None
        if hasattr(classifier_for_probs, "prob_classify"):
            # NaiveBayesClassifier (and Maxent) expose per-label probabilities;
            # DecisionTreeClassifier does not, so confidence stays None for it.
            probabilities = classifier_for_probs.prob_classify(features)
            distribution = {sample: probabilities.prob(sample) for sample in probabilities.samples()}
            confidence = distribution.get(label)
        return Prediction(label=label, confidence=confidence, class_probabilities=distribution)

    @property
    def raw_model(self) -> object:
        return self._model

    def explain(self, text: str, top_n: int = 8) -> list[tuple[str, float]]:
        """Signed per-feature contribution to the predicted label: positive
        values support the prediction, negative values argue against it.
        Only meaningful for NaiveBayesClassifier / MaxentClassifier, which
        expose per-(label, feature) log-probabilities; returns [] otherwise
        (e.g. for a 2-step model or DecisionTreeClassifier).
        """
        model = self._model
        classifier = model[1] if isinstance(model, tuple) else model
        if not hasattr(classifier, "_feature_probdist"):
            return []

        features = extract_features(tokenise(text), self._settings)
        predicted_label = self.predict(text).label
        other_labels = [label for label in classifier.labels() if label != predicted_label]

        contributions: list[tuple[str, float]] = []
        for feature_name, feature_value in features.items():
            if not feature_value:
                continue
            predicted_dist = classifier._feature_probdist.get((predicted_label, feature_name))
            if predicted_dist is None:
                continue
            other_logprobs = [
                classifier._feature_probdist[(other, feature_name)].logprob(feature_value)
                for other in other_labels
                if (other, feature_name) in classifier._feature_probdist
            ]
            if not other_logprobs:
                continue
            predicted_logprob = predicted_dist.logprob(feature_value)
            contributions.append((feature_name, predicted_logprob - sum(other_logprobs) / len(other_logprobs)))

        contributions.sort(key=lambda item: item[1], reverse=True)
        if len(contributions) <= 2 * top_n:
            return contributions
        return contributions[:top_n] + contributions[-top_n:]
