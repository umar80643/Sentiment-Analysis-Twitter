"""Interactive portfolio dashboard for the trained classical sentiment models.

Reads whichever model artifacts train.py has produced in models/ (any
combination of NaiveBayesClassifier / LogisticRegression / LinearSVC, and for
the latter two, bag-of-words or TF-IDF) and lets the user:

- pick a model + feature mode and classify a single post, with a calibrated
  confidence percentage and a word-level explanation of the prediction
- compare every trained model/feature combination on accuracy, precision,
  recall, F1, training time, and prediction time, plus that model's own
  confusion matrix and ROC curves
- upload a CSV or TXT file of posts for batch prediction and download the
  results
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config

LABELS = {"pos": "Positive", "neg": "Negative", "neu": "Neutral"}
LABEL_COLOR = {"pos": "#2ecc71", "neg": "#e74c3c", "neu": "#95a5a6"}


def discover_artifacts(model_dir: Path) -> dict[tuple[str, str | None], Path]:
    """Map (classifier, feature_mode) -> artifact path for every model
    train.py has saved, by parsing the "{classifier}_{suffix}.pkl" filenames
    it writes. The legacy models/sentiment_model.pkl is intentionally not
    included here; it exists solely so predict.py keeps working unchanged.
    """
    artifacts: dict[tuple[str, str | None], Path] = {}
    if not model_dir.exists():
        return artifacts
    for path in sorted(model_dir.glob("*.pkl")):
        for model_name in config.ALL_MODELS:
            prefix = f"{model_name}_"
            if path.stem.startswith(prefix):
                suffix = path.stem[len(prefix):]
                feature_mode = None if suffix == "nltk" else suffix
                artifacts[(model_name, feature_mode)] = path
                break
    return artifacts


@st.cache_resource
def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_label(classifier: str, feature_mode: str | None) -> str:
    return classifier + (f" ({feature_mode})" if feature_mode else "")


st.set_page_config(page_title="Tweet Sentiment Lab", page_icon="💬", layout="wide")
st.title("Tweet Sentiment Lab")
st.caption("Classical NLP sentiment analysis · NLTK & scikit-learn · Naive Bayes / Logistic Regression / Linear SVM")

artifacts = discover_artifacts(config.MODEL_DIR)
if not artifacts:
    st.warning("No trained models found. Train at least one model first.")
    st.code("python train.py --sentiment140 data/training.sample.norm.csv --models all --features all --folds 3")
    st.stop()

# ---------------------------------------------------------------------------
# Model + feature selection
# ---------------------------------------------------------------------------
selector_col, card_col = st.columns([2, 1])
with selector_col:
    available_models = sorted({model for model, _ in artifacts})
    selected_model = st.selectbox("Model", available_models)
    available_features = sorted(
        feature_mode for model, feature_mode in artifacts if model == selected_model and feature_mode is not None
    )
    selected_feature = st.selectbox("Feature extraction", available_features) if available_features else None

artifact_path = artifacts[(selected_model, selected_feature)]
artifact = load_artifact(artifact_path)
model = artifact["model"]

with card_col:
    st.subheader("Model card")
    st.metric("Training tweets", f"{artifact['tweet_count']:,}")
    if artifact.get("metrics"):
        st.metric("Cross-validated accuracy", f"{artifact['metrics']['accuracy'] * 100:.2f}%")
    st.write(f"**Feature mode:** {selected_feature or 'NLTK dict features'}")
    st.write(f"**N-grams:** {artifact['settings'].ngram}")
    st.write(f"**Negation features:** {'On' if artifact['settings'].negtn else 'Off'}")

st.divider()

# ---------------------------------------------------------------------------
# Single-post prediction
# ---------------------------------------------------------------------------
st.header("Analyse a post")
text = st.text_area("Paste a tweet or short social-media post", placeholder="I absolutely love the new album!", height=120)
if st.button("Analyse sentiment", type="primary", disabled=not text.strip()):
    prediction = model.predict(text)
    label = LABELS.get(prediction.label, prediction.label)

    result_col, explain_col = st.columns([1, 1])
    with result_col:
        st.subheader(f"Prediction: {label}")
        if prediction.confidence is not None:
            st.metric("Confidence", f"{prediction.confidence * 100:.2f}%")
        if prediction.class_probabilities:
            probabilities = pd.Series(
                {LABELS.get(k, k): v for k, v in prediction.class_probabilities.items()}
            ).sort_values(ascending=False)
            st.bar_chart(probabilities)

    with explain_col:
        st.write("**What drove this prediction**")
        contributions = model.explain(text) if hasattr(model, "explain") else []
        if not contributions:
            st.caption("Word-level explanations aren't available for this model/text combination.")
        else:
            supporting = sorted((c for c in contributions if c[1] > 0), key=lambda item: -item[1])[:5]
            opposing = sorted((c for c in contributions if c[1] <= 0), key=lambda item: item[1])[:5]
            if supporting:
                st.write("Supports this prediction:")
                for name, weight in supporting:
                    st.write(f":green[**{name}**]  (+{weight:.3f})")
            if opposing:
                st.write("Argues against it:")
                for name, weight in opposing:
                    st.write(f":red[**{name}**]  ({weight:.3f})")

st.divider()

# ---------------------------------------------------------------------------
# Model comparison dashboard
# ---------------------------------------------------------------------------
st.header("Model comparison")
if config.METRICS_PATH.exists():
    runs = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    latest_run_per_combo = {(run["classifier"], run["feature_mode"]): run for run in runs}
    comparison = pd.DataFrame(latest_run_per_combo.values())
    comparison["label"] = comparison.apply(lambda row: run_label(row["classifier"], row["feature_mode"]), axis=1)
    comparison = comparison.sort_values("accuracy", ascending=False).set_index("label")

    st.dataframe(
        comparison[["accuracy", "precision_macro", "recall_macro", "f1_macro", "train_seconds", "predict_seconds_per_1000"]]
        .rename(columns={
            "accuracy": "Accuracy", "precision_macro": "Precision", "recall_macro": "Recall",
            "f1_macro": "F1", "train_seconds": "Train time (s)", "predict_seconds_per_1000": "Predict time (ms/1k)",
        }),
        use_container_width=True,
    )

    chart_col, time_col = st.columns(2)
    with chart_col:
        fig, ax = plt.subplots()
        comparison[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot.bar(ax=ax)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.legend(["Accuracy", "Precision", "Recall", "F1"], loc="lower right")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
    with time_col:
        fig, ax = plt.subplots()
        comparison[["train_seconds", "predict_seconds_per_1000"]].plot.bar(ax=ax, secondary_y="predict_seconds_per_1000")
        ax.set_ylabel("Training time (s)")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    st.subheader(f"Diagnostics: {run_label(selected_model, selected_feature)}")
    diagnostics = artifact.get("diagnostics")
    if diagnostics:
        matrix_col, roc_col = st.columns(2)
        with matrix_col:
            st.caption("Confusion matrix")
            labels = diagnostics["labels"]
            matrix = np.array(diagnostics["confusion_matrix"])
            fig, ax = plt.subplots()
            ax.imshow(matrix, cmap="Blues")
            ax.set_xticks(range(len(labels))); ax.set_xticklabels([LABELS.get(l, l) for l in labels])
            ax.set_yticks(range(len(labels))); ax.set_yticklabels([LABELS.get(l, l) for l in labels])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            st.pyplot(fig)
        with roc_col:
            st.caption("ROC curve (one-vs-rest)")
            if diagnostics.get("roc_curves"):
                fig, ax = plt.subplots()
                for label, curve in diagnostics["roc_curves"].items():
                    ax.plot(curve["fpr"], curve["tpr"], label=f"{LABELS.get(label, label)} (AUC {curve['auc']:.2f})")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.caption("Not available for this model.")
    else:
        st.caption("This artifact predates the diagnostics feature — retrain it to see a confusion matrix and ROC curve.")
else:
    st.info("No metrics yet. Run train.py to populate the comparison dashboard.")

st.divider()

# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
st.header("Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV or TXT file of posts", type=["csv", "txt"])
manual_batch = st.text_area("...or paste one post per line", placeholder="Amazing performance!\nThis is disappointing.\nIt is fine.")

batch_texts: list[str] = []
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".csv"):
        uploaded_df = pd.read_csv(uploaded_file)
        text_column = st.selectbox("Which column contains the post text?", uploaded_df.columns)
        batch_texts = uploaded_df[text_column].astype(str).tolist()
    else:
        batch_texts = [line for line in uploaded_file.read().decode("utf-8", errors="replace").splitlines() if line.strip()]
elif manual_batch.strip():
    batch_texts = [line for line in manual_batch.splitlines() if line.strip()]

if batch_texts and st.button("Run batch prediction", disabled=not batch_texts):
    predictions = [model.predict(post) for post in batch_texts]
    results = pd.DataFrame({
        "post": batch_texts,
        "sentiment": [LABELS.get(p.label, p.label) for p in predictions],
        "confidence": [p.confidence for p in predictions],
    })
    st.dataframe(results, use_container_width=True)
    st.bar_chart(results["sentiment"].value_counts())
    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
