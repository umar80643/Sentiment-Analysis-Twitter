"""Train, cross-validate, and save one or more classical Twitter sentiment
models, writing every run's metrics to models/metrics.json.

NaiveBayesClassifier keeps training on the original NLTK dict-feature
pipeline in sentiment.py, completely unchanged. LogisticRegression and
LinearSVC are new: they train on the *same* engineered features (via
features/vectorization.py), so the accuracy/F1 comparison across models is
apples-to-apples rather than comparing different feature engineering.

MaxentClassifier and DecisionTreeClassifier -- the other two classifiers
sentiment.py has always supported -- are untouched and still importable from
sentiment.py directly; they're just not part of this comparison CLI, since
the brief asked specifically for Naive Bayes vs. Logistic Regression vs.
Linear SVM.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import config
from classifiers.registry import build_classifier, model_runs
from evaluation.metrics import cross_validate
from sanderstwitter02 import getTweetsRawData
from sentiment import FeatureSettings
from stanfordcorpus import getNormalisedTweets
from utils.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)


def _load_metrics_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("%s was unreadable; starting a fresh metrics log", path)
        return []


def _save_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)
    LOGGER.info("saved %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sanders", type=Path)
    parser.add_argument("--sentiment140", type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=[*config.ALL_MODELS, "all"], default=["NaiveBayesClassifier"],
        help="One or more classifiers to train and compare, or 'all'.",
    )
    parser.add_argument(
        "--features", nargs="+", choices=[*config.ALL_FEATURE_MODES, "all"], default=["tfidf"],
        help="Feature weighting for LogisticRegression/LinearSVC ('bow' or 'tfidf'); ignored by NaiveBayesClassifier.",
    )
    parser.add_argument("--ngram", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--negation", action="store_true")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--metrics-out", type=Path, default=config.METRICS_PATH)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a model/feature combination if its artifact file already exists in --model-dir.",
    )
    args = parser.parse_args()

    configure_logging()

    tweets = ([] if args.sanders is None else getTweetsRawData(args.sanders))
    tweets += [] if args.sentiment140 is None else getNormalisedTweets(args.sentiment140)
    if not tweets:
        parser.error("supply --sanders and/or --sentiment140")
    random.Random(args.seed).shuffle(tweets)

    models = list(config.ALL_MODELS) if "all" in args.models else args.models
    feature_modes = list(config.ALL_FEATURE_MODES) if "all" in args.features else args.features
    settings = FeatureSettings(args.ngram, args.negation)

    LOGGER.info("loaded %d labelled tweets", len(tweets))
    metrics_log = _load_metrics_log(args.metrics_out)
    run_summaries: list[tuple[str, str, float]] = []

    for classifier_name, feature_mode in model_runs(models, feature_modes):
        run_label = classifier_name + (f" ({feature_mode})" if feature_mode else "")
        suffix = feature_mode or "nltk"
        artifact_path = args.model_dir / f"{classifier_name}_{suffix}.pkl"
        if args.skip_existing and artifact_path.exists():
            LOGGER.info("skipping %s: %s already exists", run_label, artifact_path)
            continue

        LOGGER.info("=== %s: %d-fold cross-validation ===", run_label, args.folds)

        metrics, diagnostics = cross_validate(
            build_model=lambda: build_classifier(classifier_name, feature_mode or "tfidf"),
            tweets=tweets,
            settings=settings,
            folds=args.folds,
            seed=args.seed,
            classifier_name=classifier_name,
            feature_mode=feature_mode,
        )
        LOGGER.info(
            "%s -> accuracy %.4f | precision %.4f | recall %.4f | f1 %.4f | train %.2fs | predict %.2fms/1k",
            run_label, metrics.accuracy, metrics.precision_macro, metrics.recall_macro,
            metrics.f1_macro, metrics.train_seconds, metrics.predict_seconds_per_1000,
        )

        LOGGER.info("training final %s on all %d tweets", run_label, len(tweets))
        final_model = build_classifier(classifier_name, feature_mode or "tfidf")
        final_model.fit(tweets, settings)

        trained_at = datetime.now(timezone.utc).isoformat()
        artifact = {
            "model": final_model,
            "settings": settings,
            "classifier": classifier_name,
            "feature_mode": feature_mode,
            "trained_at": trained_at,
            "tweet_count": len(tweets),
            "metrics": metrics.as_dict(),
            "diagnostics": diagnostics.as_dict(),
        }
        _save_artifact(artifact_path, artifact)

        if classifier_name == "NaiveBayesClassifier" and feature_mode is None:
            # Preserve the original 1.1.0 artifact schema/path exactly, so
            # predict.py and app.py keep working without any changes.
            legacy_artifact = {
                "model": final_model.raw_model,
                "settings": settings,
                "classifier": classifier_name,
                "method": "1step",
                "trained_at": trained_at,
                "tweet_count": len(tweets),
            }
            _save_artifact(config.DEFAULT_MODEL_PATH, legacy_artifact)

        metrics_log.append({
            "run_at": trained_at,
            "dataset_size": len(tweets),
            **metrics.as_dict(),
        })
        run_summaries.append((run_label, artifact_path.name, metrics.accuracy))

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")
    LOGGER.info("metrics log updated: %s (%d runs total)", args.metrics_out, len(metrics_log))

    LOGGER.info("=== summary (best accuracy first) ===")
    for label, filename, accuracy in sorted(run_summaries, key=lambda item: item[2], reverse=True):
        LOGGER.info("%-40s accuracy %.4f  ->  %s", label, accuracy, filename)
    LOGGER.info("Dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
