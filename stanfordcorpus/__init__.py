"""Sentiment140 (historically called Stanford Twitter Corpus) loaders."""
from __future__ import annotations
import csv, random, re
from pathlib import Path

FULLDATA = "training.1600000.processed.noemoticon.csv"; TESTDATA = "testdata.manual.2009.06.14.csv"
POLARITY, TWID, DATE, SUBJ, USER, TEXT = range(6)
Tweet = tuple[str, str, str, list[str]]
QUERY_RE = re.compile(r'\w+|".*?"')
def get_class(polarity: str) -> str: return "neg" if polarity in {"0", "1"} else "pos" if polarity in {"3", "4"} else "neu" if polarity == "2" else "err"
def get_query(subject: str) -> list[str]: return [] if subject == "NO_QUERY" else QUERY_RE.findall(subject)
def getNormalisedCSV(in_file: str | Path, out_file: str | Path) -> None:
    with Path(in_file).open("r", encoding="latin-1", newline="") as source, Path(out_file).open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, quoting=csv.QUOTE_ALL)
        for row in csv.reader(source):
            if len(row) < 6: continue
            queries = get_query(row[SUBJ]); writer.writerow([row[TEXT], get_class(row[POLARITY]), row[SUBJ], len(queries), *queries])
def getNormalisedTweets(in_file: str | Path) -> list[Tweet]:
    tweets: list[Tweet] = []
    with Path(in_file).open("r", encoding="utf-8", newline="") as source:
        for number, row in enumerate(csv.reader(source), 1):
            if len(row) < 4: raise ValueError(f"{in_file}:{number}: invalid normalised record")
            count = int(row[3]); tweets.append((row[0], row[1], row[2], row[4:4 + count]))
    return tweets
def randomSampleCSV(in_file: str | Path, out_file: str | Path, K: int = 100, seed: int | None = None) -> None:
    rng = random.Random(seed); sample: list[str] = []
    with Path(in_file).open("r", encoding="latin-1") as source:
        for index, line in enumerate(source):
            if index < K: sample.append(line)
            else:
                replacement = rng.randrange(index + 1)
                if replacement < K: sample[replacement] = line
    Path(out_file).write_text("".join(sample), encoding="utf-8")

get_normalised_tweets = getNormalisedTweets
