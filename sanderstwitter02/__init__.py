"""Sanders Twitter Corpus loader."""
from __future__ import annotations
import csv
from pathlib import Path
Tweet = tuple[str, str, str, list[str]]
QUERY_TERMS = {"apple": ["@apple"], "microsoft": ["#microsoft"], "google": ["#google"], "twitter": ["#twitter"]}
_LABELS = {"positive": "pos", "negative": "neg", "irrelevant": "neu", "neutral": "neu"}

def getTweetsRawData(fileName: str | Path) -> list[Tweet]:
    """Load Sanders' five-column full corpus CSV as (text, label, topic, query)."""
    path = Path(fileName)
    tweets: list[Tweet] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, escapechar="\\"), 1):
            if len(row) < 5: raise ValueError(f"{path}:{row_number}: expected at least 5 columns")
            topic, label = row[0].lower(), row[1].lower()
            if topic not in QUERY_TERMS: raise ValueError(f"{path}:{row_number}: unknown topic {topic!r}")
            tweets.append((row[4], _LABELS.get(label, label), row[0], QUERY_TERMS[topic]))
    return tweets

get_tweets_raw_data = getTweetsRawData
