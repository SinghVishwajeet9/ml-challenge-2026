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

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import Paths, log, timed

# Pool sizes per source.
K_NAME = 50
K_ADDR = 50
K_TFIDF = 30
# The country-wide pass over S2/S3 records without a state (mostly no address) gets a
# small quota: those records rank on name alone and would otherwise crowd out
# address-only matches.
K_NOSTATE = 10
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

MAX_DF = 400         # skip index keys shared by more S2/S3 records than this (fixed cap)
PAIR_TOKENS = 6       # pair keys from each record's 6 rarest tokens
MIN_TOKEN_LEN = 2
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

    def __init__(self, other_tok: pl.DataFrame):
        self.single_df = other_tok.group_by("tok").len("df")
        keys = _with_pairs(other_tok, self.single_df)
        n = max(other_tok["rid"].n_unique(), 1)
        self.dfreq = (keys.group_by("tok").len("df").filter(pl.col("df") <= MAX_DF)
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

SRC = ["src_name", "src_addr", "src_concat", "src_tfidf"]
POOL_SCHEMA = {"s1_rid": pl.Int32, "other_rid": pl.Int32, **{c: pl.Boolean for c in SRC},
               "name_score": pl.Float64, "addr_score": pl.Float64}


def _concat_index(other: pl.DataFrame) -> pl.DataFrame:
    keys = pl.concat([
        other.select("rid", pl.col("name_concat").alias("key")),
        other.select("rid", pl.col("name_domain").alias("key")),
    ]).filter(pl.col("key").is_not_null() & (pl.col("key").str.len_chars() >= 6)).unique()
    common = keys.group_by("key").len().filter(pl.col("len") > 50)
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
        if with_index:
            self.name_ix = InvertedIndex(_tokens(other_sub, "name_skel"))
            self.addr_ix = InvertedIndex(_addr_tokens(other_sub))
            self.concat_ix = _concat_index(other_sub)
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
        else:
            parts.append(_as_pool(self._topn(s1, va_addr, self.addr_mt, self.k_addr),
                                  "src_addr"))
        return _merge_pools(parts)


def _rerank(s1: pl.DataFrame, pool: pl.DataFrame, space: CountrySpace,
            va_name: sparse.csr_matrix, va_addr: sparse.csr_matrix) -> pl.DataFrame:
    """Combined name + address score over the pooled pairs; keep the best N_KEEP per S1."""
    if pool.height == 0:
        return pool
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
        .alias("addr_sim_rank"))
    pool = pool.with_columns(
        (W_NAME * pl.col("name_sim") + W_ADDR * pl.col("addr_sim_rank")
         + W_HOUSE * pl.col("house_eq").cast(pl.Float32)
         + W_CONCAT * pl.col("src_concat").cast(pl.Float32)).alias("block_score"))
    return (pool.sort(["s1_rid", "block_score"], descending=[False, True])
            .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("block_rank"))
            .filter(pl.col("block_rank") < N_KEEP))


def _run_group(s1: pl.DataFrame, retrievers: list[Retriever], space: CountrySpace,
               out: list[pl.DataFrame]) -> None:
    for i in range(0, s1.height, S1_CHUNK):
        chunk = s1.slice(i, S1_CHUNK)
        va_name = space.name_vec.transform(chunk["name_skel"].fill_null("").to_list())
        va_addr = space.addr_vec.transform(_addr_text(chunk).to_list())
        pool = _merge_pools([r.query(chunk, va_name, va_addr) for r in retrievers])
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
    nb.write_parquet(f)
    log.info("state neighbours: %s", nb.rows())
    return nb


def block(norm: pl.DataFrame, neighbours: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per country: S1 records are searched within their own state (true matches share
    the state 99.9% of the time), plus a country-wide pass over S2/S3 records without a
    detected state (e.g. no address). S1 records without a state are searched
    country-wide with TF-IDF only. Cost grows linearly with the data."""
    out: list[pl.DataFrame] = []
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
        o_state = other["addr_state"].fill_null("").to_numpy()
        nostate_rows = np.flatnonzero(o_state == "")
        r_nostate = Retriever(space, nostate_rows, other[nostate_rows], with_index=True,
                              k=K_NOSTATE)
        states = sorted(s1["addr_state"].drop_nulls().unique().to_list())
        s1_nostate = s1.filter(pl.col("addr_state").is_null())
        log.info("%s: %d states | S2/S3 without state %d | S1 without state %d",
                 country, len(states), len(nostate_rows), s1_nostate.height)
        with timed(f"{country}: block {s1.height} S1"):
            for st in states:
                rows = np.flatnonzero(np.isin(o_state, [st, *nbrs.get((country, st), [])]))
                r_state = Retriever(space, rows, other[rows], with_index=True)
                _run_group(s1.filter(pl.col("addr_state") == st), [r_state, r_nostate],
                           space, out)
            if s1_nostate.height:
                r_country = Retriever(space, np.arange(other.height), None, with_index=False)
                _run_group(s1_nostate, [r_country, r_nostate], space, out)
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


GT_FOR_SUBSET = {"dev": "dev_gt_pairs", "tune": "tune_gt_pairs", "val": "val_gt_pairs"}


def read_candidates(paths: Paths, subset: str) -> pl.DataFrame:
    return pl.read_parquet(paths.work / "block" / subset / "*.parquet")


def run(paths: Paths, subset: str) -> None:
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet")
    neighbours = fit_state_neighbours(paths, all_train=(subset == "test"))
    out = paths.work / "block" / subset
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.parquet"):
        f.unlink()
    # One country at a time, written straight to disk, to bound memory on test.
    with timed(f"block {subset}"):
        for country in sorted(norm["country"].unique()):
            block(norm.filter(pl.col("country") == country), neighbours).write_parquet(
                out / f"{country}.parquet")
    if subset not in GT_FOR_SUBSET and subset != "train":
        return
    cands = read_candidates(paths, subset)
    if subset in GT_FOR_SUBSET:
        report(cands, norm, pl.read_parquet(paths.slices / f"{GT_FOR_SUBSET[subset]}.parquet"))
    elif subset == "train":
        report(cands, norm, pl.read_parquet(paths.gt_pairs()))
