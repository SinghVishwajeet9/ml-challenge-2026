"""Native-script to English word dictionary, learned without labels.

Names of S2/S3 records are often written in an Indian script ("ভিশন সলিউশনস প্রাইভেট
লিমিটেড" for "Vision Solutions Private Limited"). Letter-level transliteration loses
too much for some scripts (Bengali, Odia, Malayalam), so words are mapped instead.

Source: confident first-pass matches (final probability >= MIN_P) between an English S1
name and a native-script S2/S3 name, over train subsets and test. No labels are used.
When both names have the same number of words, words are aligned by position. A native
word is kept if it was seen at least MIN_COUNT times and one English word accounts for at
least MIN_SHARE of its alignments.

Output: configs/native_dict.json  {native word: english word}
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from anyascii import anyascii

from common import Paths, log, timed

NATIVE_RUN = re.compile("[ऀ-෿‌‍]+")
WORD = re.compile("[ऀ-෿‌‍]+|[A-Za-z0-9]+")
JOINERS = re.compile("[‌‍]")
MIN_P = 0.95
MIN_COUNT = 2
MIN_SHARE = 0.5
DICT_FILE = Path(__file__).resolve().parents[1] / "configs" / "native_dict.json"
SOURCES = {"tune": "train", "val": "train", "val2": "train", "rest": "train", "test": "test"}


def _clean(word: str) -> str:
    return JOINERS.sub("", word)


def _native_words(name: str) -> list[str]:
    return [w for w in (_clean(t) for t in WORD.findall(name or "")) if w]


def _english_words(name: str) -> list[str]:
    return re.findall("[a-z0-9]+", anyascii(name or "").lower())


def build(paths: Paths) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    n_pairs = 0
    for subset, split in SOURCES.items():
        f = paths.work / "first_pass" / f"{subset}.parquet"
        if not f.exists():
            continue
        raw = pl.read_parquet(paths.records(split), columns=["rid", "name"])
        pairs = (pl.read_parquet(f).filter(pl.col("p") >= MIN_P).select("s1_rid", "other_rid")
                 .join(raw.rename({"rid": "other_rid", "name": "n2"}), on="other_rid")
                 .filter(pl.col("n2").str.contains(NATIVE_RUN.pattern))
                 .join(raw.rename({"rid": "s1_rid", "name": "n1"}), on="s1_rid"))
        for n1, n2 in pairs.select("n1", "n2").iter_rows():
            a, b = _english_words(n1), _native_words(n2)
            if len(a) != len(b) or not a:
                continue
            for x, y in zip(b, a):
                if NATIVE_RUN.fullmatch(x):
                    counts[x][y] += 1
        n_pairs += pairs.height
        log.info("native dictionary: %s %d confident native pairs", subset, pairs.height)
    out = {}
    for x, c in counts.items():
        y, k = c.most_common(1)[0]
        if k >= MIN_COUNT and k / sum(c.values()) >= MIN_SHARE:
            out[x] = y
    log.info("native dictionary: %d words from %d pairs", len(out), n_pairs)
    return out


def run(paths: Paths) -> None:
    with timed("build native dictionary"):
        d = build(paths)
    DICT_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=0, sort_keys=True),
                         encoding="utf-8")


_DICT: dict[str, str] | None = None


def translate(name: str) -> str:
    """Replace every native-script word that is in the dictionary by its English word."""
    global _DICT
    if _DICT is None:
        _DICT = (json.loads(DICT_FILE.read_text(encoding="utf-8")) if DICT_FILE.exists()
                 else {})
    return NATIVE_RUN.sub(lambda m: _DICT.get(_clean(m.group(0)), m.group(0)), name)
