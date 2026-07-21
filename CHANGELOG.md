# Changelog

## 2.0.0 — Multi-model comparison + interactive dashboard

- Added `classifiers/` with a shared `SentimentClassifier` interface
  (`fit`, `predict`, `explain`, `raw_model`) wrapping the original NLTK
  pipeline plus two new scikit-learn models, Logistic Regression and a
  calibrated Linear SVM.
- Added `features/` with configurable Bag-of-Words / TF-IDF vectorization of
  the project's existing engineered features (no feature-engineering logic
  duplicated).
- Added `evaluation/` for cross-validated accuracy/precision/recall/F1,
  confusion matrices, and per-class ROC/PR curves, computed identically
  across every classifier.
- Added `utils/logging_setup.py`, replacing ad hoc prints with leveled
  logging to console and `logs/training.log`.
- Extended `config.py` with model/feature options and per-classifier
  hyperparameters.
- Rewrote `train.py` to train and cross-validate any combination of models
  and feature modes in one run, save one artifact per combination, append
  every run to `models/metrics.json`, and skip already-trained combinations
  with `--skip-existing`. Continues to also write the original
  `models/sentiment_model.pkl` schema unchanged so `predict.py` needed no
  changes.
- Rewrote `app.py`: model/feature selector, calibrated confidence scores,
  per-class probability chart, word-level prediction explanations, a
  model-comparison dashboard (table + bar charts + confusion matrix + ROC
  curves), and batch prediction from pasted text or an uploaded CSV/TXT file
  with CSV download of the results.
- Added `tests/test_classifiers.py` covering the new wrappers, the
  model/feature-mode expansion logic, and cross-validation.

## 1.1.0 — Portfolio application

- Added full-data model training and safe serialized model artifacts.
- Added `app.py`, a Streamlit dashboard for single-tweet and batch sentiment analysis.
- Updated command-line prediction to use the saved artifact and its original feature settings.
- Added training metadata to the saved artifact: classifier, method, data size, timestamp, and feature configuration.

## 1.0.0 — Python 3.12 modernization

- Modernized every Python module from Python 2 syntax and binary CSV handling to UTF-8, `pathlib`, context managers, annotations, and package-relative imports.
- Repaired preprocessing's unsafe emoticon regex and empty-query regex while preserving normalisation order and feature tokens.
- Repaired Sanders and Sentiment140 loaders: newline handling, encoding, malformed-row errors, label mapping, and reservoir sampling.
- Replaced obsolete `mdp` PCA with training-only `sklearn.decomposition.PCA`.
- Replaced hard-coded Twitter credentials and obsolete `python-twitter` calls with Tweepy and environment credentials.
- Rewrote the Python 2 experiment runner as a typed NLTK pipeline; removed its unavailable SVM option instead of retaining a broken interface.
- Fixed `xrange`, exception syntax, output redirection, integer division, imports, and fold validation.
- Added configuration, command-line training/prediction, dependencies, tests, ignored runtime directories, and this README.

### Modernized original files

`preprocessing/__init__.py`, `sanderstwitter02/__init__.py`, `sanderstwitter02/install.py`, `stanfordcorpus/__init__.py`, `sandersfeatures/tweet_features.py`, `sandersfeatures/tweet_pca.py`, `sandersfeatures/__init__.py`, `sentiment.py`, and `stats.py`.
