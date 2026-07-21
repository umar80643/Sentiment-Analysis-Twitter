"""Optional Sanders raw-tweet retriever using credentials supplied through env vars.

Twitter's historical statuses are frequently unavailable; the existing full corpus
CSV is therefore the recommended input for training.
"""
from __future__ import annotations
import csv, json, logging, os, time
from pathlib import Path
import tweepy
LOGGER = logging.getLogger(__name__)
def read_total_list(filename: str | Path) -> list[list[str]]:
    with Path(filename).open("r", encoding="utf-8", newline="") as handle: return list(csv.reader(handle))
def twitter_client() -> tweepy.API:
    keys = [os.getenv(key) for key in ("TWITTER_CONSUMER_KEY", "TWITTER_CONSUMER_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET")]
    if not all(keys): raise RuntimeError("Set TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN and TWITTER_ACCESS_TOKEN_SECRET")
    auth = tweepy.OAuth1UserHandler(*keys); return tweepy.API(auth)
def parse_tweet_json(filename: str | Path) -> list[str]:
    with Path(filename).open("r", encoding="utf-8") as handle: payload = json.load(handle)
    if "error" in payload or "errors" in payload: raise RuntimeError("downloaded tweet contains an API error")
    return [payload["created_at"], payload.get("full_text", payload["text"])]
def download_tweets(fetch_list: list[list[str]], raw_dir: str | Path, pause_seconds: float = 5.0) -> None:
    directory = Path(raw_dir); directory.mkdir(parents=True, exist_ok=True); api = twitter_client()
    for item in fetch_list:
        destination = directory / f"{item[2]}.json"
        try:
            status = api.get_status(item[2], tweet_mode="extended"); destination.write_text(json.dumps(status._json, ensure_ascii=False), encoding="utf-8")
        except tweepy.TweepyException as error: LOGGER.warning("Cannot download tweet %s: %s", item[2], error)
        time.sleep(max(0.0, pause_seconds))
