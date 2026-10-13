"""Stage `block`: candidate generation (S1 x S2/S3), always within one country.

S1 records are searched within their own state/region (see `block`), so the cost grows
linearly with the data. Two steps, run per chunk of S1 records:

1. Retrieve a wide pool from several sources (union):
     name    IDF-weighted inverted index on name skeleton tokens + token pairs
     concat  exact key: joined-up name == other's joined-up name or website label
     addr    IDF-weighted inverted index on address tokens/numbers + pairs
     tfidf   character-trigram TF-IDF cosine on name skeletons (letter-level variants)
2. Re-rank the pool with one combined score and keep the best `N_KEEP` per S1:
     name_sim (trigram cosine) + addr_sim (word TF-IDF cosine) + house match + concat hit.
   A record that is decent on both name and address beats one that is strong on
   only one, which is what crowded blocks (many similar names in one city) need.

The kept set is exactly what the matching model scores, i.e. candidate_pairs.tsv.
Output: work/block/<subset>.parquet with pool features (reused as model features).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import Paths, log, switch, timed

# Pool sizes per source.
K_NAME = 50
K_ADDR = 50
K_TFIDF = 30
# The country-wide pass over S2/S3 records without a state (mostly no address) gets a
# small quota: those records rank on name alone and would otherwise crowd out
# address-only matches.
K_NOSTATE = 10
# Country-wide exact-name key: only rare names (shared by at most this many S2/S3
# records), flagged separately and given no re-rank bonus, so it adds cross-state
# candidates without crowding out address matches.
COUNTRY_CONCAT_MAX = 5
# S1 states whose true matches often sit in another state (Telangana -> Andhra Pradesh,
# split in 2014; DC -> WA, "Washington") also search that state. Learned from training.
MERGE_MIN_SHARE = 0.02
MERGE_MIN_PAIRS = 200
# If more than this share of a country's S2/S3 records have no detected state (e.g.
# France: city-only addresses and no training data to learn city -> region), the
# country is searched as a single region instead of per state.
MAX_NOSTATE_SHARE = 0.10
TFIDF_MIN_SIM = 0.2
# Final candidates per S1 after re-ranking.
N_KEEP = 40
# Re-rank weights.
W_NAME, W_ADDR, W_HOUSE, W_CONCAT = 1.0, 1.0, 0.3, 0.5

# No-address records rank on name alone, so for common names ("Lille Club") hundreds of
# them tie at the top score and push out same-name records whose address also matches.
# They get their own small quota, ranked by name similarity.
NOADDR_KEEP = int(os.environ.get("BER_NOADDR_KEEP", "5"))
# Joint name + address index: ties between same-name records are broken by the address.
# Its top hits are kept on top of the N_KEEP list.
K_JOINT = 5
JOINT_MAX_DF = 100   # joint keys must be selective; also bounds query memory
JOINT_CHUNK = 5_000  # queries per joint-index lookup
# Second pass from confident first-pass matches (seeds): records near a seed record
# (joint index) or sharing a rare exact address / name with it.
K_SIB_NEIGHBOURS = 3
SIB_RARE_MAX = 5
SIB_MAX_PER_S1 = 10
SEED_MIN_P = float(os.environ.get("BER_SEED_MIN_P", "0.9"))

MAX_DF = int(os.environ.get("BER_MAX_DF", "400"))  # skip index keys shared by more S2/S3 records than this (fixed cap)
# Normalized columns blocking uses (the rest is left on disk to bound memory).
BLOCK_COLUMNS = ["rid", "entity_id", "source", "country", "addr_state", "name_skel",
                 "name_concat", "name_domain", "name_core", "name_translit", "addr_core",
                 "addr_numbers", "addr_house", "addr_missing"]
PAIR_TOKENS = 6       # pair keys from each record's 6 rarest tokens
MIN_TOKEN_LEN = int(os.environ.get("BER_MIN_TOKEN_LEN", "2"))
S1_CHUNK = 30_000


# --------------------------------------------------------------------------- tokens

def _tokens(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.select("rid", "country", pl.col(col).fill_null("").str.split(" ").alias("tok"))
            .explode("tok")
            .filter(pl.col("tok").str.len_chars() >= MIN_TOKEN_LEN)
            .unique())


def _addr_tokens(df: pl.DataFrame) -> pl.DataFrame:
    words = _tokens(df, "addr_core").filter(~pl.col("tok").str.contains(r"^\d+$"))
    nums = (df.select("rid", "country", pl.col("addr_numbers").alias("tok")).explode("tok")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "0"))
            .with_columns(("#" + pl.col("tok")).alias("tok")))
    return pl.concat([words, nums]).unique()


def _joint_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Name skeleton tokens and address tokens in one key space (prefixed)."""
    name = _tokens(df, "name_skel").with_columns(("n:" + pl.col("tok")).alias("tok"))
    addr = _addr_tokens(df).with_columns(("a:" + pl.col("tok")).alias("tok"))
    return pl.concat([name, addr]).unique()


def _with_pairs(tok: pl.DataFrame, freq: pl.DataFrame) -> pl.DataFrame:
    """Single tokens plus sorted pairs of the record's rarest tokens. A pair of common
    words ("consulting" + "sk") is often rare enough to index."""
    top = (tok.join(freq, on="tok", how="left")
           .with_columns(pl.col("df").fill_null(0))
           .sort(["rid", "df", "tok"])
           .group_by("rid", maintain_order=True).head(PAIR_TOKENS)
           .select("rid", "country", "tok"))
    pairs = (top.join(top.select("rid", pl.col("tok").alias("tok_b")), on="rid")
             .filter(pl.col("tok") < pl.col("tok_b"))
             .select("rid", "country", (pl.col("tok") + "|" + pl.col("tok_b")).alias("tok")))
    return pl.concat([tok, pairs])


def _addr_text(df: pl.DataFrame) -> pl.Series:
    words = pl.col("addr_core").fill_null("")
    nums = pl.col("addr_numbers").list.eval("#" + pl.element()).list.join(" ").fill_null("")
    return df.select(pl.concat_str([words, nums], separator=" ").str.strip_chars()).to_series()


# --------------------------------------------------------------------------- index

class InvertedIndex:
    """IDF-weighted index over one subset of S2/S3 records (name or address keys).
    Keys shared by more than MAX_DF records are skipped; the cap is fixed, so the work
    per S1 does not grow with the size of the data."""

    def __init__(self, other_tok: pl.DataFrame, max_df: int = MAX_DF):
        self.single_df = other_tok.group_by("tok").len("df")
        keys = _with_pairs(other_tok, self.single_df)
        n = max(other_tok["rid"].n_unique(), 1)
        self.dfreq = (keys.group_by("tok").len("df").filter(pl.col("df") <= max_df)
                      .select("tok", (pl.lit(float(n)) / pl.col("df")).log().alias("w")))
        self.postings = keys.select("rid", "tok").join(self.dfreq.select("tok"), on="tok")

    def query(self, s1_tok: pl.DataFrame, k: int, label: str) -> pl.DataFrame:
        q = _with_pairs(s1_tok, self.single_df).select("rid", "tok").join(self.dfreq, on="tok")
        pairs = (q.join(self.postings, on="tok", suffix="_o")
                 .group_by("rid", "rid_o").agg(pl.col("w").sum().alias("score")))
        top = (pairs.sort(["rid", "score"], descending=[False, True])
               .group_by("rid", maintain_order=True).head(k))
        return top.select(pl.col("rid").alias("s1_rid"), pl.col("rid_o").alias("other_rid"),
                          pl.col("score").alias(f"{label}_score"))


class CountrySpace:
    """TF-IDF spaces of one country's S2/S3 records (name trigrams, address words).
    Fitted country-wide so similarities mean the same thing in every state."""

    def __init__(self, other: pl.DataFrame):
        self.rids = other["rid"].to_numpy()
        self.row_of = dict(zip(self.rids.tolist(), range(len(self.rids))))
        self.name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                        dtype=np.float32, sublinear_tf=True)
        self.name_m = self.name_vec.fit_transform(other["name_skel"].fill_null("").to_list())
        self.addr_vec = TfidfVectorizer(analyzer=str.split, min_df=1, dtype=np.float32,
                                        sublinear_tf=True)
        self.addr_m = self.addr_vec.fit_transform(_addr_text(other).to_list())
        self.house = other["addr_house"].to_list()
        self.addr_missing = other["addr_missing"].to_numpy()


def _row_cos(a: sparse.csr_matrix, ia: np.ndarray, b: sparse.csr_matrix,
             ib: np.ndarray, step: int = 500_000) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), step):
        out[s:s + step] = np.asarray(
            a[ia[s:s + step]].multiply(b[ib[s:s + step]]).sum(axis=1)).ravel()
    return out


# --------------------------------------------------------------------------- blocking

SRC = ["src_name", "src_addr", "src_concat", "src_tfidf", "src_xconcat", "src_joint", "src_sib"]
POOL_SCHEMA = {"s1_rid": pl.Int32, "other_rid": pl.Int32, **{c: pl.Boolean for c in SRC},
               "name_score": pl.Float64, "addr_score": pl.Float64}


def _concat_index(other: pl.DataFrame, max_shared: int = 50) -> pl.DataFrame:
    keys = pl.concat([
        other.select("rid", pl.col("name_concat").alias("key")),
        other.select("rid", pl.col("name_domain").alias("key")),
    ]).filter(pl.col("key").is_not_null() & (pl.col("key").str.len_chars() >= 6)).unique()
    common = keys.group_by("key").len().filter(pl.col("len") > max_shared)
    return keys.join(common.select("key"), on="key", how="anti")


def _as_pool(df: pl.DataFrame, src: str, score: str | None = None) -> pl.DataFrame:
    cols = [pl.col("s1_rid").cast(pl.Int32), pl.col("other_rid").cast(pl.Int32)]
    cols += [pl.lit(c == src).alias(c) for c in SRC]
    for s in ("name_score", "addr_score"):
        cols.append((pl.col(s) if score == s else pl.lit(None)).cast(pl.Float64).alias(s))
    return df.select(cols)


def _merge_pools(parts: list[pl.DataFrame]) -> pl.DataFrame:
    parts = [p for p in parts if p.height]
    if not parts:
        return pl.DataFrame(schema=POOL_SCHEMA)
    return pl.concat(parts).group_by("s1_rid", "other_rid").agg(
        *[pl.col(c).any() for c in SRC], pl.col("name_score").max(), pl.col("addr_score").max())


class Retriever:
    """Candidate retrieval over a subset (e.g. one state) of a country's S2/S3 records.
    with_index=False uses TF-IDF top-K only (cheap enough country-wide for a few S1)."""

    def __init__(self, space: CountrySpace, rows: np.ndarray, other_sub: pl.DataFrame | None,
                 with_index: bool, k: int | None = None):
        self.space, self.rows, self.with_index = space, rows, with_index
        self.k_name, self.k_addr, self.k_tfidf = (k, k, k) if k else (K_NAME, K_ADDR, K_TFIDF)
        if len(rows) == 0:
            return
        self.name_mt = space.name_m[rows].T.tocsr()
        self.joint_ix = None
        if with_index:
            self.name_ix = InvertedIndex(_tokens(other_sub, "name_skel"))
            self.addr_ix = InvertedIndex(_addr_tokens(other_sub))
            self.concat_ix = _concat_index(other_sub)
            if switch("joint_index") or switch("second_pass"):
                self.joint_ix = InvertedIndex(_joint_tokens(other_sub), JOINT_MAX_DF)
        else:
            self.addr_mt = space.addr_m[rows].T.tocsr()

    def _topn(self, s1: pl.DataFrame, va: sparse.csr_matrix, mt: sparse.csr_matrix,
              k: int) -> pl.DataFrame:
        m = sp_matmul_topn(va, mt, top_n=k, threshold=TFIDF_MIN_SIM,
                           n_threads=os.cpu_count()).tocoo()
        return pl.DataFrame({"s1_rid": s1["rid"].gather(m.row),
                             "other_rid": pl.Series(self.space.rids[self.rows[m.col]],
                                                    dtype=pl.Int32)})

    def query(self, s1: pl.DataFrame, va_name: sparse.csr_matrix,
              va_addr: sparse.csr_matrix) -> pl.DataFrame:
        if len(self.rows) == 0:
            return pl.DataFrame(schema=POOL_SCHEMA)
        parts = [_as_pool(self._topn(s1, va_name, self.name_mt, self.k_tfidf), "src_tfidf")]
        if self.with_index:
            parts.append(_as_pool(self.name_ix.query(_tokens(s1, "name_skel"), self.k_name,
                                                     "name"), "src_name", "name_score"))
            parts.append(_as_pool(self.addr_ix.query(_addr_tokens(s1), self.k_addr, "addr"),
                                  "src_addr", "addr_score"))
            concat = (s1.select("rid", pl.col("name_concat").alias("key"))
                      .join(self.concat_ix, on="key", suffix="_o")
                      .select(pl.col("rid").alias("s1_rid"), pl.col("rid_o").alias("other_rid"))
                      .unique())
            parts.append(_as_pool(concat, "src_concat"))
            if self.joint_ix is not None and switch("joint_index"):
                jt = _joint_tokens(s1)
                for i in range(0, s1.height, JOINT_CHUNK):
                    ids = s1["rid"].slice(i, JOINT_CHUNK)
                    q = jt.join(pl.DataFrame({"rid": ids}), on="rid")
                    parts.append(_as_pool(self.joint_ix.query(q, K_JOINT, "joint"), "src_joint"))
        else:
            parts.append(_as_pool(self._topn(s1, va_addr, self.addr_mt, self.k_addr),
                                  "src_addr"))
        return _merge_pools(parts)


def _rerank(s1: pl.DataFrame, pool: pl.DataFrame, space: CountrySpace,
            va_name: sparse.csr_matrix, va_addr: sparse.csr_matrix) -> pl.DataFrame:
    """Combined name + address score over the pooled pairs; keep the best N_KEEP per S1.
    With the no-address quota, no-address records fill at most NOADDR_KEEP of the slots
    (best by name similarity); joint-index hits are kept on top of the list."""
    if pool.height == 0:
        return pool
    pool = _score(s1, pool, space, va_name, va_addr)
    ranked = (pool.sort(["s1_rid", "block_score"], descending=[False, True])
              .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("block_rank")))
    if switch("noaddr_quota"):
        na = (ranked.filter(pl.col("_na"))
              .sort(["s1_rid", "name_sim", "block_score"], descending=[False, True, True])
              .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("_r"))
              .filter(pl.col("_r") < NOADDR_KEEP))
        n_na = na.group_by("s1_rid").len("_n_na")
        wa = (ranked.filter(~pl.col("_na"))
              .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("_r"))
              .join(n_na, on="s1_rid", how="left")
              .filter(pl.col("_r") < N_KEEP - pl.col("_n_na").fill_null(0)))
        kept = (pl.concat([na.drop("_n_na", strict=False), wa.drop("_n_na")], how="diagonal")
                .drop("_na", "_r")
                .sort(["s1_rid", "block_score"], descending=[False, True])
                .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("block_rank")))
    else:
        kept = ranked.filter(pl.col("block_rank") < N_KEEP).drop("_na")
    if switch("joint_index"):
        extra = (ranked.filter(pl.col("src_joint"))
                 .join(kept.select("s1_rid", "other_rid"), on=["s1_rid", "other_rid"], how="anti")
                 .with_columns(pl.lit(N_KEEP).cast(kept["block_rank"].dtype).alias("block_rank")))
        kept = pl.concat([kept, extra.select(kept.columns)])
    return kept


def _score(s1: pl.DataFrame, pool: pl.DataFrame, space: CountrySpace,
           va_name: sparse.csr_matrix, va_addr: sparse.csr_matrix) -> pl.DataFrame:
    """Pool pairs with name_sim, addr_sim, house_eq, addr_sim_rank and block_score."""
    s1_row = dict(zip(s1["rid"].to_list(), range(s1.height)))
    ia = np.fromiter((s1_row[r] for r in pool["s1_rid"].to_list()), np.int64, pool.height)
    ib = np.fromiter((space.row_of[r] for r in pool["other_rid"].to_list()), np.int64,
                     pool.height)
    house_s1 = s1["addr_house"].to_list()
    house_eq = np.fromiter(
        (bool(set(house_s1[i]) & set(space.house[j])) if house_s1[i] and space.house[j]
         else False for i, j in zip(ia.tolist(), ib.tolist())), bool, pool.height)
    pool = pool.with_columns(
        pl.Series("name_sim", _row_cos(va_name, ia, space.name_m, ib)),
        pl.Series("addr_sim", _row_cos(va_addr, ia, space.addr_m, ib)),
        pl.Series("house_eq", house_eq),
    )
    # A candidate without an address can't earn address similarity; let its name
    # similarity stand in so it is not pushed out by partial address matches.
    addr_missing = pl.Series(space.addr_missing[ib])
    pool = pool.with_columns(
        pl.when(addr_missing).then(pl.col("name_sim")).otherwise(pl.col("addr_sim"))
        .alias("addr_sim_rank"), addr_missing.alias("_na"))
    pool = pool.with_columns(
        (W_NAME * pl.col("name_sim") + W_ADDR * pl.col("addr_sim_rank")
         + W_HOUSE * pl.col("house_eq").cast(pl.Float32)
         + W_CONCAT * pl.col("src_concat").cast(pl.Float32)).alias("block_score"))
    return pool


def _country_concat(s1: pl.DataFrame, concat_ix: pl.DataFrame) -> pl.DataFrame:
    """Exact joined-name / website key across the whole country: identical names are
    candidates even when the two records sit in different states."""
    pairs = (s1.select("rid", pl.col("name_concat").alias("key"))
             .join(concat_ix, on="key", suffix="_o")
             .select(pl.col("rid").alias("s1_rid"), pl.col("rid_o").alias("other_rid"))
             .unique())
    return _as_pool(pairs, "src_xconcat")


def _run_group(s1: pl.DataFrame, retrievers: list[Retriever], space: CountrySpace,
               out: list[pl.DataFrame], country_concat: pl.DataFrame | None = None) -> None:
    for i in range(0, s1.height, S1_CHUNK):
        chunk = s1.slice(i, S1_CHUNK)
        va_name = space.name_vec.transform(chunk["name_skel"].fill_null("").to_list())
        va_addr = space.addr_vec.transform(_addr_text(chunk).to_list())
        parts = [r.query(chunk, va_name, va_addr) for r in retrievers]
        if country_concat is not None:
            parts.append(_country_concat(chunk, country_concat))
        pool = _merge_pools(parts)
        out.append(_rerank(chunk, pool, space, va_name, va_addr))


def fit_state_neighbours(paths: Paths, all_train: bool = False) -> pl.DataFrame:
    """(country, state, neighbour): S1 records of `state` also search S2/S3 records of
    `neighbour` when >= MERGE_MIN_SHARE and >= MERGE_MIN_PAIRS of the state's true pairs
    sit there (Telangana -> Andhra Pradesh, DC -> WA). Not transitive, so one link never
    pulls in a third state. Learned from train minus the val slice, or all of train for
    test (`all_train`); cached."""
    f = paths.norm / ("state_neighbours_all.parquet" if all_train else "state_neighbours.parquet")
    if f.exists():
        return pl.read_parquet(f)
    norm = pl.read_parquet(paths.norm / "train.parquet",
                           columns=["rid", "entity_id", "country", "addr_state"])
    if not all_train:
        val = pl.read_parquet(paths.slices / "val_records.parquet", columns=["rid"])
        norm = norm.join(val, on="rid", how="anti")
    gt = (pl.read_parquet(paths.gt_pairs())
          .join(norm.select(pl.col("entity_id").alias("s1_id"), "country",
                            pl.col("addr_state").alias("state")), on="s1_id")
          .join(norm.select(pl.col("entity_id").alias("match_id"),
                            pl.col("addr_state").alias("neighbour")), on="match_id")
          .drop_nulls(["state", "neighbour"]))
    nb = (gt.group_by("country", "state", "neighbour").len()
          .with_columns((pl.col("len") / pl.col("len").sum().over("country", "state"))
                        .alias("share"))
          .filter((pl.col("state") != pl.col("neighbour")) & (pl.col("len") >= MERGE_MIN_PAIRS)
                  & (pl.col("share") >= MERGE_MIN_SHARE))
          .select("country", "state", "neighbour"))
    # Symmetric: if WA records often sit in DC, DC records also search WA.
    nb = pl.concat([nb, nb.select("country", pl.col("neighbour").alias("state"),
                                  pl.col("state").alias("neighbour"))]).unique()
    nb.write_parquet(f)
    log.info("state neighbours: %s", nb.rows())
    return nb


def _seed_neighbours(seeds: pl.DataFrame, other_sub: pl.DataFrame, r: "Retriever",
                     chunk: int = JOINT_CHUNK) -> pl.DataFrame:
    """(s1_rid, other_rid): records near each seed record in the joint index."""
    q = other_sub.join(seeds.select(pl.col("other_rid").alias("rid")).unique(), on="rid")
    parts = []
    for i in range(0, q.height, chunk):
        nb = r.joint_ix.query(_joint_tokens(q.slice(i, chunk)), K_SIB_NEIGHBOURS + 1, "nb")
        parts.append(nb.filter(pl.col("s1_rid") != pl.col("other_rid"))
                     .select(pl.col("s1_rid").alias("seed"), "other_rid"))
    if not parts:
        return pl.DataFrame(schema={"s1_rid": pl.Int32, "other_rid": pl.Int32})
    nb = pl.concat(parts)
    return (seeds.select("s1_rid", pl.col("other_rid").alias("seed")).join(nb, on="seed")
            .select(pl.col("s1_rid").cast(pl.Int32), pl.col("other_rid").cast(pl.Int32)))


def _rare_key_siblings(seeds: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    """(s1_rid, other_rid): records sharing a rare exact address or name with a seed record."""
    keys = pl.concat([
        other.filter(~pl.col("addr_missing") & (pl.col("addr_core").str.len_chars() >= 8))
        .select("rid", ("a:" + pl.col("addr_core")).alias("key")),
        other.filter(pl.col("name_skel").str.len_chars() >= 4)
        .select("rid", ("n:" + pl.col("name_skel")).alias("key"))])
    rare = keys.join(keys.group_by("key").len("df").filter(pl.col("df") <= SIB_RARE_MAX)
                     .select("key"), on="key")
    return (seeds.join(rare.rename({"rid": "other_rid"}), on="other_rid")
            .select("s1_rid", "key", pl.col("other_rid").alias("seed"))
            .join(rare, on="key").filter(pl.col("rid") != pl.col("seed"))
            .select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32).alias("other_rid")))


def _second_pass(s1: pl.DataFrame, sib: pl.DataFrame, kept: pl.DataFrame,
                 space: CountrySpace, columns: list[str]) -> pl.DataFrame:
    """Score the sibling pairs that are not candidates yet; best SIB_MAX_PER_S1 per S1."""
    sib = sib.unique().join(kept.select("s1_rid", "other_rid"), on=["s1_rid", "other_rid"],
                            how="anti")
    if sib.height == 0:
        return pl.DataFrame(schema=kept.schema)
    sub = s1.join(sib.select(pl.col("s1_rid").alias("rid")).unique(), on="rid")
    out = []
    for i in range(0, sub.height, S1_CHUNK):
        chunk = sub.slice(i, S1_CHUNK)
        va_name = space.name_vec.transform(chunk["name_skel"].fill_null("").to_list())
        va_addr = space.addr_vec.transform(_addr_text(chunk).to_list())
        pool = _as_pool(sib.join(chunk.select(pl.col("rid").alias("s1_rid")), on="s1_rid"),
                        "src_sib")
        out.append(_score(chunk, pool, space, va_name, va_addr))
    res = (pl.concat(out).sort(["s1_rid", "block_score"], descending=[False, True])
           .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("_r"))
           .filter(pl.col("_r") < SIB_MAX_PER_S1).drop("_r", "_na")
           .with_columns(pl.lit(N_KEEP).alias("block_rank")))
    return res.select([pl.col(c).cast(kept.schema[c]) for c in columns])


def block(norm: pl.DataFrame, neighbours: pl.DataFrame | None = None,
          seeds: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per country: S1 records are searched within their own state (true matches share
    the state 99.9% of the time), plus a country-wide pass over S2/S3 records without a
    detected state (e.g. no address). S1 records without a state are searched
    country-wide with TF-IDF only. Cost grows linearly with the data.
    seeds (s1_rid, other_rid): confident first-pass matches for the second pass."""
    out: list[pl.DataFrame] = []
    use_seeds = seeds is not None and seeds.height > 0 and switch("second_pass")
    nbrs: dict[tuple[str, str], list[str]] = {}
    for c, st, nb in (neighbours.rows() if neighbours is not None else []):
        nbrs.setdefault((c, st), []).append(nb)
    for country in sorted(norm["country"].unique()):
        s1 = norm.filter((pl.col("source") == 1) & (pl.col("country") == country))
        other = norm.filter((pl.col("source") != 1) & (pl.col("country") == country))
        if s1.height == 0 or other.height == 0:
            continue
        if other["addr_state"].is_null().mean() > MAX_NOSTATE_SHARE:
            log.info("%s: %.0f%% of S2/S3 without state -> single region", country,
                     100 * other["addr_state"].is_null().mean())
            s1 = s1.with_columns(pl.lit("ALL").alias("addr_state"))
            other = other.with_columns(pl.lit("ALL").alias("addr_state"))
        with timed(f"{country}: TF-IDF spaces ({other.height} S2/S3)"):
            space = CountrySpace(other)
        country_concat = _concat_index(other, max_shared=COUNTRY_CONCAT_MAX)
        o_state = other["addr_state"].fill_null("").to_numpy()
        nostate_rows = np.flatnonzero(o_state == "")
        r_nostate = Retriever(space, nostate_rows, other[nostate_rows], with_index=True,
                              k=K_NOSTATE)
        states = sorted(s1["addr_state"].drop_nulls().unique().to_list())
        s1_nostate = s1.filter(pl.col("addr_state").is_null())
        log.info("%s: %d states | S2/S3 without state %d | S1 without state %d",
                 country, len(states), len(nostate_rows), s1_nostate.height)
        c_out: list[pl.DataFrame] = []
        sib: list[pl.DataFrame] = []
        c_seeds = (seeds.join(s1.select(pl.col("rid").alias("s1_rid")), on="s1_rid")
                   .join(other.select(pl.col("rid").alias("other_rid")), on="other_rid")
                   if use_seeds else None)
        # Finished states are spilled to disk so memory stays flat across big countries.
        import tempfile
        spill = Path(tempfile.mkdtemp(prefix="block_"))
        spill_files: list[Path] = []

        def _spill() -> None:
            done = [o for o in c_out if o.height]
            if done:
                f = spill / f"part_{len(spill_files):04d}.parquet"
                pl.concat(done).write_parquet(f)
                spill_files.append(f)
            c_out.clear()

        with timed(f"{country}: block {s1.height} S1"):
            for st in states:
                rows = np.flatnonzero(np.isin(o_state, [st, *nbrs.get((country, st), [])]))
                r_state = Retriever(space, rows, other[rows], with_index=True)
                s1_st = s1.filter(pl.col("addr_state") == st)
                _run_group(s1_st, [r_state, r_nostate], space, c_out, country_concat)
                _spill()
                if use_seeds and len(rows):
                    st_seeds = c_seeds.join(s1_st.select(pl.col("rid").alias("s1_rid")),
                                            on="s1_rid")
                    sib.append(_seed_neighbours(st_seeds, other[rows], r_state))
                r_state = None
                if len(rows) > 100_000:
                    import psutil
                    log.info("  %s: %d S2/S3 | process memory %.1f GB | free %.1f GB", st,
                             len(rows), psutil.Process().memory_info().rss / 1e9,
                             psutil.virtual_memory().available / 1e9)
            if s1_nostate.height:
                r_country = Retriever(space, np.arange(other.height), None, with_index=False)
                _run_group(s1_nostate, [r_country, r_nostate], space, c_out, country_concat)
                _spill()
        c_out = [pl.read_parquet(f) for f in spill_files]
        shutil.rmtree(spill, ignore_errors=True)
        if use_seeds and c_out:
            with timed(f"{country}: second pass from {c_seeds.height} seeds"):
                kept = pl.concat(c_out)
                sib.append(_rare_key_siblings(c_seeds, other))
                extra = _second_pass(s1, pl.concat(sib), kept, space, kept.columns)
                log.info("%s: second pass adds %d pairs (%.2f per S1)", country, extra.height,
                         extra.height / max(s1.height, 1))
                c_out.append(extra)
        out.extend(c_out)
    return pl.concat([o for o in out if o.height])


# --------------------------------------------------------------------------- report

def report(cands: pl.DataFrame, norm: pl.DataFrame, gt_pairs: pl.DataFrame) -> None:
    """Pair recall per source and of the kept set, for hard groups; oracle F0.5."""
    import evaluate

    ids = norm.select("rid", "entity_id")
    gt = (gt_pairs.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
          .join(ids.rename({"rid": "other_rid", "entity_id": "match_id"}), on="match_id"))
    o = norm.select(pl.col("rid").alias("other_rid"), "name_translit", "addr_missing",
                    pl.col("name_core").alias("core_o"))
    s = norm.select(pl.col("rid").alias("s1_rid"), pl.col("name_core").alias("core_s"))
    gt = gt.join(o, on="other_rid").join(s, on="s1_rid").with_columns(
        (pl.col("core_s").str.split(" ").list.set_intersection(pl.col("core_o").str.split(" "))
         .list.len() == 0).alias("name_jacc0"))
    hit = gt.join(cands, on=["s1_rid", "other_rid"], how="left")

    def rec(df: pl.DataFrame, cond: pl.Expr) -> str:
        return f"{100 * df.select(cond.fill_null(False).mean()).item():.1f}%"

    views = {"all": hit, "nonlatin": hit.filter("name_translit"),
             "addr_missing": hit.filter("addr_missing"),
             "name_jacc0": hit.filter(pl.col("name_jacc0") & ~pl.col("name_translit"))}
    for v, df in views.items():
        log.info("recall %-12s n=%7d | kept via: name %s concat %s addr %s tfidf %s | KEPT %s",
                 v, df.height, rec(df, pl.col("src_name")), rec(df, pl.col("src_concat")),
                 rec(df, pl.col("src_addr")), rec(df, pl.col("src_tfidf")),
                 rec(df, pl.col("block_rank").is_not_null()))

    n_s1 = norm.filter(pl.col("source") == 1).height
    per = cands.group_by("s1_rid").len()
    log.info("kept: %d pairs, %.1f per S1 (p50 %d, max %d)", cands.height,
             cands.height / n_s1, per["len"].median(), per["len"].max())
    for n in (10, 20, 30, N_KEEP):
        log.info("  recall if keeping top %2d: %s", n, rec(hit, pl.col("block_rank") < n))
    s1_ids = norm.filter(pl.col("source") == 1)["entity_id"]
    cand_ids = (cands.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_rid")
                .join(ids.rename({"rid": "other_rid", "entity_id": "match_id"}),
                      on="other_rid")
                .select("s1_id", "match_id"))
    rc = evaluate.recall_ceiling(cand_ids, gt_pairs, s1_ids)
    log.info("pair recall %.2f%% | oracle F0.5 %.4f | cands/S1 %.1f",
             100 * rc["pair_recall"], rc["oracle_f05"], rc["cands_per_s1"])


GT_FOR_SUBSET = {"dev": "dev_gt_pairs", "tune": "tune_gt_pairs", "val": "val_gt_pairs",
                 "val2": "val2_gt_pairs"}


def read_candidates(paths: Paths, subset: str) -> pl.DataFrame:
    return pl.read_parquet(paths.work / "block" / subset / "*.parquet")


def run(paths: Paths, subset: str, country: str | None = None) -> None:
    """country: block only that country (its own process on test, to bound memory);
    other countries' files are kept."""
    scan = pl.scan_parquet(paths.norm / f"{subset}.parquet").select(BLOCK_COLUMNS)
    norm = (scan.filter(pl.col("country") == country) if country else scan).collect()
    neighbours = fit_state_neighbours(paths, all_train=(subset == "test"))
    out = paths.work / "block" / subset
    out.mkdir(parents=True, exist_ok=True)
    for f in ([out / f"{country}.parquet"] if country else list(out.glob("*.parquet"))):
        if f.exists():
            f.unlink()
    # Second-pass seeds: confident matches of the first-pass pipeline (no labels).
    seeds = None
    seed_file = paths.work / "first_pass" / f"{subset}.parquet"
    if switch("second_pass") and seed_file.exists():
        rid_type = norm.schema["rid"]
        seeds = (pl.read_parquet(seed_file).filter(pl.col("p") >= SEED_MIN_P)
                 .select(pl.col("s1_rid").cast(rid_type), pl.col("other_rid").cast(rid_type)))
        log.info("second pass: %d seeds (p >= %.2f)", seeds.height, SEED_MIN_P)
    # One country at a time, written straight to disk, to bound memory on test.
    with timed(f"block {subset}"):
        for country in sorted(norm["country"].unique()):
            block(norm.filter(pl.col("country") == country), neighbours, seeds).write_parquet(
                out / f"{country}.parquet")
    if subset not in GT_FOR_SUBSET and subset != "train":
        return
    cands = read_candidates(paths, subset)
    if subset in GT_FOR_SUBSET:
        report(cands, norm, pl.read_parquet(paths.slices / f"{GT_FOR_SUBSET[subset]}.parquet"))
    elif subset == "train":
        report(cands, norm, pl.read_parquet(paths.gt_pairs()))
