"""Lightweight descriptive statistics for loaded tweets."""
from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
import preprocessing
from sanderstwitter02 import Tweet
def class_counts(tweets: Sequence[Tweet]) -> Counter[str]: return Counter(tweet[1] for tweet in tweets)
def feature_averages(tweets: Sequence[Tweet]) -> dict[str, float]:
    if not tweets: return {name: 0.0 for name in ("handles", "hashtags", "urls", "emoticons", "words", "characters")}
    values = {"handles": [preprocessing.countHandles(t[0]) for t in tweets], "hashtags": [preprocessing.countHashtags(t[0]) for t in tweets], "urls": [preprocessing.countUrls(t[0]) for t in tweets], "emoticons": [preprocessing.countEmoticons(t[0]) for t in tweets], "words": [len(t[0].split()) for t in tweets], "characters": [len(t[0]) for t in tweets]}
    return {name: sum(items) / len(items) for name, items in values.items()}
