from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifiers.nltk_models import NLTKClassifierModel
from classifiers.registry import build_classifier, model_runs
from classifiers.sklearn_models import SklearnSentimentModel
from evaluation.metrics import cross_validate
from sentiment import FeatureSettings

TWEETS = [
    ("I love this", "pos", "", []),
    ("great product", "pos", "", []),
    ("amazing experience", "pos", "", []),
    ("bad service", "neg", "", []),
    ("I hate this", "neg", "", []),
    ("terrible outcome", "neg", "", []),
    ("fine okay", "neu", "", []),
    ("ordinary day", "neu", "", []),
    ("nothing special", "neu", "", []),
]


def test_model_runs_expands_sklearn_per_feature_mode_and_dedupes_naive_bayes():
    runs = model_runs(["NaiveBayesClassifier", "LogisticRegression"], ["bow", "tfidf"])
    assert runs == [
        ("NaiveBayesClassifier", None),
        ("LogisticRegression", "bow"),
        ("LogisticRegression", "tfidf"),
    ]


def test_nltk_classifier_model_trains_and_predicts():
    model = NLTKClassifierModel(name="NaiveBayesClassifier")
    model.fit(TWEETS, FeatureSettings())
    prediction = model.predict("I love it")
    assert prediction.label in {"pos", "neg", "neu"}
    assert prediction.confidence is not None and 0.0 <= prediction.confidence <= 1.0


def test_sklearn_model_trains_and_predicts_with_calibrated_confidence():
    for name in ("LogisticRegression", "LinearSVC"):
        model = SklearnSentimentModel(name=name, feature_mode="tfidf")
        model.fit(TWEETS, FeatureSettings())
        prediction = model.predict("I hate it")
        assert prediction.label in {"pos", "neg", "neu"}
        assert prediction.confidence is not None and 0.0 <= prediction.confidence <= 1.0
        assert prediction.class_probabilities is not None
        assert abs(sum(prediction.class_probabilities.values()) - 1.0) < 1e-6


def test_build_classifier_factory_covers_all_supported_models():
    for name in ("NaiveBayesClassifier", "LogisticRegression", "LinearSVC"):
        classifier = build_classifier(name, "bow")
        assert classifier.name == name


def test_cross_validate_produces_metrics_and_diagnostics_in_valid_ranges():
    metrics, diagnostics = cross_validate(
        build_model=lambda: build_classifier("LogisticRegression", "tfidf"),
        tweets=TWEETS,
        settings=FeatureSettings(),
        folds=3,
        seed=0,
        classifier_name="LogisticRegression",
        feature_mode="tfidf",
    )
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert len(metrics.fold_accuracies) == 3
    assert metrics.tweet_count == len(TWEETS)

    assert set(diagnostics.labels) <= {"pos", "neg", "neu"}
    assert len(diagnostics.confusion_matrix) == len(diagnostics.labels)
    # LogisticRegression always yields calibrated probabilities, so ROC/PR
    # curves should be populated for every label.
    assert set(diagnostics.roc_curves) == set(diagnostics.labels)
