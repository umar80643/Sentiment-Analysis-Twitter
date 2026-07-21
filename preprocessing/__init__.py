"""Deterministic tweet normalisation used by the original research pipeline."""
from __future__ import annotations

import re
from collections.abc import Sequence

HASH_RE = re.compile(r"#(\w+)")
HANDLE_RE = re.compile(r"@(\w+)")
URL_RE = re.compile(r"(?:http|https|ftp)://[a-zA-Z0-9./]+")
WORD_BOUNDARY_RE = re.compile(r"\W+")
REPEATING_RE = re.compile(r"(.)\1{1,}", re.IGNORECASE)

EMOTICONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("__EMOT_SMILEY", (":-)", ":)", "(:", "(-:")),
    ("__EMOT_LAUGH", (":-D", ":D", "X-D", "XD", "xD")),
    ("__EMOT_LOVE", ("<3", r":\*")),
    ("__EMOT_WINK", (";-)", ";)", ";-D", ";D", "(;", "(-;")),
    ("__EMOT_FROWN", (":-(", ":(", "(:", "(-:")),
    ("__EMOT_CRY", (":,(", r":\'(", ':"(', ":((")),
)
PUNCTUATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("__PUNC_EXCL", ("!", "¡")), ("__PUNC_QUES", ("?", "¿")),
    ("__PUNC_ELLP", ("...", "…")),
)

def _emoticon_pattern(values: Sequence[str]) -> re.Pattern[str]:
    # Escaping is intentional: the original hand-built regex misinterpreted '*'.
    return re.compile("(?:" + "|".join(re.escape(value) for value in values) + ")")

EMOTICON_PATTERNS = tuple((replacement, _emoticon_pattern(values)) for replacement, values in EMOTICONS)

def hash_repl(match: re.Match[str]) -> str: return f"__HASH_{match.group(1).upper()}"
def hndl_repl(match: re.Match[str]) -> str: return "__HNDL"

def _punctuation_replacement(match: re.Match[str]) -> str:
    text = match.group(0)
    tokens = [name for name, values in PUNCTUATIONS if any(value in text for value in values)]
    return f" {' '.join(tokens)} " if tokens else " "

def processHashtags(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    return HASH_RE.sub(hash_repl, text)
def processHandles(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    return HANDLE_RE.sub(hndl_repl, text)
def processUrls(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    return URL_RE.sub(" __URL ", text)
def processEmoticons(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    for replacement, pattern in EMOTICON_PATTERNS: text = pattern.sub(f" {replacement} ", text)
    return text
def processPunctuations(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    return WORD_BOUNDARY_RE.sub(_punctuation_replacement, text)
def processRepeatings(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    return REPEATING_RE.sub(lambda match: match.group(1) * 2, text)
def processQueryTerm(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    return re.sub("|".join(re.escape(item) for item in query), "__QUER", text, flags=re.IGNORECASE) if query else text

def processAll(text: str, subject: str = "", query: Sequence[str] = ()) -> str:
    """Apply the legacy normalisation order, preserving its feature tokens."""
    text = processQueryTerm(text, subject, query)
    text = processHashtags(text, subject, query)
    text = processHandles(text, subject, query)
    text = processUrls(text, subject, query)
    text = processEmoticons(text, subject, query)
    text = processPunctuations(text, subject, query)
    return processRepeatings(text, subject, query)

def countHandles(text: str) -> int: return len(HANDLE_RE.findall(text))
def countHashtags(text: str) -> int: return len(HASH_RE.findall(text))
def countUrls(text: str) -> int: return len(URL_RE.findall(text))
def countEmoticons(text: str) -> int: return sum(len(pattern.findall(text)) for _, pattern in EMOTICON_PATTERNS)

# PEP 8 aliases for new callers; legacy public names remain above.
process_all = processAll
