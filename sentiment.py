"""Original NLTK sentiment pipeline, made safe and runnable on Python 3.12."""
from __future__ import annotations
import logging
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import nltk
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier

import preprocessing
from sanderstwitter02 import Tweet

LOGGER = logging.getLogger(__name__)
FOLDS = 10
ClassifierName = Literal["NaiveBayesClassifier", "MaxentClassifier", "DecisionTreeClassifier"]
Method = Literal["1step", "2step"]
NEGATION_RE = re.compile(r"^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$|n't", re.I)

@dataclass(frozen=True)
class FeatureSettings:
    ngram: int = 1
    negtn: bool = False
    def __post_init__(self) -> None:
        if self.ngram not in {1, 2, 3}: raise ValueError("ngram must be 1, 2, or 3")

def k_fold_cross_validation(items: Sequence[Any], folds: int, randomise: bool = False, seed: int | None = None) -> Iterable[tuple[list[Any], list[Any]]]:
    if folds < 2 or folds > len(items): raise ValueError("folds must be between 2 and the number of records")
    values = list(items)
    if randomise: random.Random(seed).shuffle(values)
    for fold in range(folds): yield ([value for index, value in enumerate(values) if index % folds != fold], [value for index, value in enumerate(values) if index % folds == fold])

def tokenise(text: str, subject: str = "", query: Sequence[str] = ()) -> list[str]:
    normalised = preprocessing.processAll(text, subject, query)
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word if word.startswith("__") else word.lower()) for word in normalised.split() if len(word) >= 3]

def extract_features(words: Sequence[str], settings: FeatureSettings) -> dict[str, float]:
    features: dict[str, float] = {f"has({word})": 1.0 for word in words}
    if settings.ngram >= 2: features.update({f"has({','.join(pair)})": 1.0 for pair in nltk.bigrams(words)})
    if settings.ngram >= 3: features.update({f"has({','.join(triple)})": 1.0 for triple in nltk.trigrams(words)})
    if settings.negtn:
        negations = [bool(NEGATION_RE.search(word)) for word in words]
        left, right = _negation_distances(negations), list(reversed(_negation_distances(list(reversed(negations)))))
        features.update({f"neg_l({word})": distance for word, distance in zip(words, left)})
        features.update({f"neg_r({word})": distance for word, distance in zip(words, right)})
    return features

def _negation_distances(flags: Sequence[bool]) -> list[float]:
    values: list[float] = []; previous = 0.0
    for is_negation in flags:
        previous = 1.0 if is_negation else max(0.0, previous - 0.1); values.append(previous)
    return values

def getTrainingAndTestData(tweets: Sequence[Tweet], K: int, k: int, method: Method, feature_set: dict[str, Any]) -> tuple[Any, ...]:
    settings = FeatureSettings(ngram=int(feature_set.get("ngram", 1)), negtn=bool(feature_set.get("negtn", False)))
    all_tweets = [(tokenise(text, subject, query), sentiment) for text, sentiment, subject, query in tweets]
    train = [item for index, item in enumerate(all_tweets) if index % K != k]; test = [item for index, item in enumerate(all_tweets) if index % K == k]
    convert = lambda records: [(extract_features(words, settings), label) for words, label in records]
    if method == "1step": return convert(train), convert(test)
    objective = lambda label: "obj" if label in {"neg", "pos"} else label
    train_obj, test_obj = [(words, objective(label)) for words, label in train], [(words, objective(label)) for words, label in test]
    train_sentiment, test_sentiment = [(words, label) for words, label in train if label in {"neg", "pos"}], [(words, label) for words, label in test if label in {"neg", "pos"}]
    return convert(train_obj), convert(train_sentiment), convert(test_obj), convert(test_sentiment), [label for _, label in test]

def _trainer(name: ClassifierName):
    if name == "NaiveBayesClassifier": return NaiveBayesClassifier.train
    if name == "MaxentClassifier": return lambda data: MaxentClassifier.train(data, algorithm="GIS", max_iter=10, trace=0)
    if name == "DecisionTreeClassifier": return lambda data: DecisionTreeClassifier.train(data, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10, binary=False)
    raise ValueError(f"Unsupported classifier: {name}")

def train_final_model(
    tweets: Sequence[Tweet],
    classifier: ClassifierName = "NaiveBayesClassifier",
    method: Method = "1step",
    feature_set: dict[str, Any] | None = None,
) -> Any:
    """Fit a deployable model on every supplied labelled tweet."""
    settings = FeatureSettings(**(feature_set or {}))
    labelled = [
        (extract_features(tokenise(text, subject, query), settings), label)
        for text, label, subject, query in tweets
    ]
    trainer = _trainer(classifier)
    if method == "1step":
        return trainer(labelled)

    objectivity = [(features, "obj" if label in {"neg", "pos"} else label) for features, label in labelled]
    polarity = [(features, label) for features, label in labelled if label in {"neg", "pos"}]
    return trainer(objectivity), trainer(polarity)

def trainAndClassify(tweets: Sequence[Tweet], classifier: ClassifierName = "NaiveBayesClassifier", method: Method = "1step", feature_set: dict[str, Any] | None = None, fileprefix: str = "", folds: int = FOLDS) -> Any:
    """Cross-validate, then train and return a deployable full-data model."""
    if len(tweets) < folds: raise ValueError(f"Need at least {folds} tweets for {folds}-fold validation")
    settings = feature_set or {"ngram": 1, "negtn": False}; train = _trainer(classifier); accuracies: list[float] = []; final_model: Any = None
    for fold in range(folds):
        parts = getTrainingAndTestData(tweets, folds, fold, method, settings)
        if method == "1step":
            training, testing = parts; final_model = train(training); accuracy = nltk.classify.accuracy(final_model, testing)
        else:
            train_obj, train_sent, test_obj, test_sent, truth = parts; object_model, sentiment_model = train(train_obj), train(train_sent)
            predicted = [sentiment_model.classify(features) if object_model.classify(features) == "obj" else object_model.classify(features) for features, _ in test_obj]
            final_model, accuracy = (object_model, sentiment_model), sum(a == b for a, b in zip(truth, predicted)) / len(truth)
        accuracies.append(accuracy); LOGGER.info("fold %d/%d accuracy: %.4f", fold + 1, folds, accuracy)
    LOGGER.info("average accuracy: %.4f", sum(accuracies) / len(accuracies))
    LOGGER.info("training final model on %d tweets", len(tweets))
    return train_final_model(tweets, classifier, method, settings)

def predict(text: str, model: Any, settings: FeatureSettings = FeatureSettings()) -> str:
    features = extract_features(tokenise(text), settings)
    if isinstance(model, tuple):
        object_model, sentiment_model = model; classification = object_model.classify(features)
        return sentiment_model.classify(features) if classification == "obj" else classification
    return model.classify(features)
