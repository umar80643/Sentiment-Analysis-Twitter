# Tweet Sentiment Lab

A Python 3.12+ Twitter/X sentiment analysis project, evolved from a classical
NLTK research pipeline into a small production-style ML system: multiple
comparable classifiers, configurable feature extraction, cross-validated
evaluation with saved metrics, and an interactive Streamlit dashboard for
single-post and batch prediction.

No transformer or LLM has been substituted — every classifier here is a
classical, fully-interpretable model (Naive Bayes, Logistic Regression,
Linear SVM), trained on hand-engineered n-gram / negation features.

## Overview

- **Task:** classify a tweet or short social-media post as positive,
  negative, or neutral.
- **Data:** the Sanders Twitter Corpus and/or the Sentiment140 (1.6M tweets)
  dataset.
- **Pipeline:** normalise → Porter-stem → n-gram / negation features →
  classify → cross-validate → save a reusable, versioned model artifact.
- **Models:** Naive Bayes (NLTK), Logistic Regression, and Linear SVM
  (scikit-learn), each trainable with Bag-of-Words or TF-IDF features.
- **Interfaces:** a CLI for training/prediction (`train.py`, `predict.py`)
  and a Streamlit dashboard (`app.py`) for exploring predictions and
  comparing models.

## Architecture

```text
Sanders / Sentiment140 CSV
        │
        ▼
preprocessing/            URL, handle, hashtag, emoticon, punctuation handling
        │
        ▼
sentiment.tokenise()       Porter stemming, min-length filtering
        │
        ▼
sentiment.extract_features()   unigram/bigram/trigram + negation-distance features
        │
        ├──────────────► NaiveBayesClassifier (NLTK)        — trains directly on feature dicts
        │
        ▼
features/vectorization.py      DictVectorizer (+ optional TfidfTransformer)
        │
        ├──────────────► LogisticRegression (scikit-learn)
        └──────────────► LinearSVC + CalibratedClassifierCV (scikit-learn)
        │
        ▼
evaluation/metrics.py     k-fold cross-validation → accuracy/precision/recall/F1,
                           confusion matrix, ROC/PR curves, timing
        │
        ▼
train.py                  saves one artifact per model/feature combo + models/metrics.json
        │
        ▼
predict.py / app.py       CLI prediction  /  Streamlit dashboard
```

Every classifier shares one interface (`classifiers.base.SentimentClassifier`:
`fit`, `predict`, `explain`, `raw_model`), so `train.py`, `evaluation/`, and
`app.py` never branch on which model they're holding — new classifiers plug
into the same pipeline.

**Why a shared DictVectorizer instead of raw-text vectorizers:** Logistic
Regression and Linear SVM train on the exact same engineered n-gram/negation
features Naive Bayes uses (see `features/vectorization.py`), rather than a
separate raw-text `CountVectorizer`/`TfidfVectorizer`. That keeps the model
comparison about the *learning algorithm*, not about which feature engineering
happened to be used — and avoids re-implementing tokenisation twice.

## Features

- **Three classifiers, one interface:** Naive Bayes, Logistic Regression,
  Linear SVM (calibrated for real confidence scores), trained and compared
  through `classifiers/registry.py`.
- **Configurable features:** Bag-of-Words or TF-IDF, plus n-gram order
  (1–3) and optional negation features, all via CLI flags or `config.py`.
- **Cross-validated evaluation:** accuracy, macro precision/recall/F1,
  confusion matrix, and per-class ROC/PR curves for every model, computed the
  same way regardless of classifier type (`evaluation/metrics.py`).
- **Every run recorded:** `models/metrics.json` accumulates classifier,
  feature mode, dataset size, timing, and scores for every training run —
  never overwritten, only appended to.
- **Interactive dashboard (`app.py`):** model + feature selector, confidence
  percentage, per-class probability chart, word-level explanation of each
  prediction, a full model-comparison table with bar charts, confusion
  matrix / ROC curves for the selected model, and batch prediction from a
  pasted list or an uploaded CSV/TXT file with CSV download of the results.
- **Backward compatible:** `predict.py` and the original `sentiment_model.pkl`
  artifact schema are untouched — training always keeps writing that file
  alongside the new per-model artifacts.
- **Logging, not prints:** every entry point logs to console and
  `logs/training.log` via `utils/logging_setup.py`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Usage

Place datasets under `data/` (a 5,000-tweet Sentiment140 sample and a
500-tweet tiny sample are included for a quick run), then train:

```bash
# Naive Bayes only (also writes the legacy models/sentiment_model.pkl)
python train.py --sentiment140 data/training.sample.norm.csv --folds 5

# Compare every model against every feature mode
python train.py --sentiment140 data/training.sample.norm.csv --models all --features all --folds 5

# A specific combination, with bigrams and negation features
python train.py --sentiment140 data/training.sample.norm.csv \
    --models LogisticRegression --features tfidf --ngram 2 --negation

# Skip a combination if it's already been trained
python train.py --sentiment140 data/training.sample.norm.csv --models all --features all --skip-existing
```

Then launch the dashboard:

```bash
streamlit run app.py
```

Or predict from the command line (uses the original NaiveBayes artifact):

```bash
python predict.py "I absolutely love this"
```

Convert the original six-column Sentiment140 source to the UTF-8 normalised
format once:

```bash
python -c "from stanfordcorpus import getNormalisedCSV; getNormalisedCSV('data/training.1600000.processed.noemoticon.csv', 'data/training.norm.csv')"
```

## Datasets

- **Sanders:** five-column `topic,sentiment,tweet_id,date,text` full-corpus
  CSV. Labels map to `pos`, `neg`, and `neu`.
- **Sentiment140:** its six-column CSV is decoded as Latin-1 then emitted as
  UTF-8 normalised CSV. Polarity `0/1`, `2`, and `3/4` map to `neg`, `neu`,
  and `pos`.

## Results

5-fold cross-validation on the included 5,000-tweet Sentiment140 sample
(`data/training.sample.norm.csv`), unigram features, no negation:

| Model | Features | Accuracy | Precision | Recall | F1 | Train time | Predict time |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Bag-of-Words | **0.733** | 0.733 | 0.733 | 0.733 | 5.3 s | 1.0 ms / 1k |
| Logistic Regression | TF-IDF | 0.732 | 0.732 | 0.732 | 0.732 | 5.0 s | 1.5 ms / 1k |
| Linear SVM | TF-IDF | 0.727 | 0.727 | 0.727 | 0.727 | 4.8 s | 5.5 ms / 1k |
| Linear SVM | Bag-of-Words | 0.723 | 0.723 | 0.723 | 0.723 | 5.3 s | 5.1 ms / 1k |
| Naive Bayes | (NLTK dict features) | 0.715 | 0.715 | 0.715 | 0.715 | 5.7 s | 0.3 ms / 1k |

Reproduce with:

```bash
python train.py --sentiment140 data/training.sample.norm.csv --models all --features all --folds 5
```

Numbers move with dataset size, seed, n-gram order, and negation features —
run the command above to regenerate this table, or open the dashboard's
**Model comparison** tab to see it interactively. On the full 1.6M-tweet
Sentiment140 corpus, expect all models to improve, with Logistic Regression
and Linear SVM typically pulling further ahead of Naive Bayes as more
training data becomes available for their smoother decision boundaries.

## Layout

```text
data/                 local datasets
preprocessing/         URL, handle, hashtag, emoticon and punctuation handling
sanderstwitter02/       Sanders loader and optional Tweepy retriever
stanfordcorpus/         Sentiment140 conversion and loader
sandersfeatures/        legacy hand-built features and sklearn PCA
sentiment.py            tokenisation, feature extraction, NLTK classifiers, prediction
config.py               paths, feature/model options, hyperparameters
features/               Bag-of-Words / TF-IDF vectorization of the shared features
classifiers/            shared classifier interface + NaiveBayes/LogReg/LinearSVC wrappers
evaluation/             cross-validated metrics, confusion matrix, ROC/PR curves
utils/                  logging setup
train.py, predict.py    CLI entry points
app.py                  Streamlit dashboard
models/, logs/          ignored runtime outputs (metrics.json is the exception, see below)
tests/                  loader, normalisation, feature, classifier, and training tests
```

## Future improvements

- Add a held-out (non-cross-validated) test split for a final, single
  unbiased accuracy number alongside the CV estimate.
- Hyperparameter search (grid/random search over `C`, n-gram order) with
  results folded into `models/metrics.json`.
- A lightweight transformer baseline (e.g. DistilBERT) for comparison against
  the classical models — kept out of this project so far to keep the
  pipeline fully interpretable and CPU-trainable.
- Package the trained artifacts behind a small FastAPI service for
  programmatic (non-Streamlit) inference.
