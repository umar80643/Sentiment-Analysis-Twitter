"""Predict sentiment with a saved project model."""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
from sentiment import predict
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("text"); parser.add_argument("--model", type=Path, default=Path("models/sentiment_model.pkl")); args = parser.parse_args()
    with args.model.open("rb") as handle: artifact = pickle.load(handle)
    print(predict(args.text, artifact["model"], artifact["settings"]))
if __name__ == "__main__": main()
