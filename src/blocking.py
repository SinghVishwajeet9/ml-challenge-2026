"""Stage `block`: candidate generation (S1 x S2/S3), always within one country.

Two steps, run per chunk of S1 records so memory stays bounded on the full data:

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
TFIDF_MIN_SIM = 0.2
# Final candidates per S1 after re-ranking.
N_KEEP = 40
# Re-rank weights.
W_NAME, W_ADDR, W_HOUSE, W_CONCAT = 1.0, 1.0, 0.3, 0.5

MAX_DF_FRAC = 0.002   # skip index keys present in > 0.2% of the country's S2/S3 records
MIN_MAX_DF = 200
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
    top = (tok.join(freq, on=["country", "tok"], how="left")
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
    """IDF-weighted index over S2/S3 keys of one kind (name or address)."""

    def __init__(self, other_tok: pl.DataFrame, n_other: dict[str, int]):
        single_df = other_tok.group_by("country", "tok").len("df")
        keys = _with_pairs(other_tok, single_df)
        dfreq = keys.group_by("country", "tok").len("df")
        cap = pl.col("country").replace_strict(
            {c: max(MIN_MAX_DF, int(MAX_DF_FRAC * n)) for c, n in n_other.items()},
            return_dtype=pl.Int64)
        n_expr = pl.col("country").replace_strict(
            {c: float(n) for c, n in n_other.items()}, return_dtype=pl.Float64)
        self.single_df = single_df
        self.dfreq = (dfreq.filter(pl.col("df") <= cap)
                      .with_columns((n_expr / pl.col("df")).log().alias("w"))
                      .select("country", "tok", "w"))
        self.postings = keys.join(self.dfreq.select("country", "tok"), on=["country", "tok"])

    def query(self, s1_tok: pl.DataFrame, k: int, label: str) -> pl.DataFrame:
        q = _with_pairs(s1_tok, self.single_df).join(self.dfreq, on=["country", "tok"])
        pairs = (q.join(self.postings, on=["country", "tok"], suffix="_o")
                 .group_by("rid", "rid_o").agg(pl.col("w").sum().alias("score")))
        top = (pairs.sort(["rid", "score"], descending=[False, True])
               .group_by("rid", maintain_order=True).head(k))
        return top.select(pl.col("rid").alias("s1_rid"), pl.col("rid_o").alias("other_rid"),
                          pl.col("score").alias(f"{label}_score"))


class CountrySpace:
    """TF-IDF spaces of one country's S2/S3 records (name trigrams, address words)."""

    def __init__(self, other: pl.DataFrame):
        self.rids = other["rid"].to_numpy()
        self.row_of = dict(zip(self.rids.tolist(), range(len(self.rids))))
        self.name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                        dtype=np.float32, sublinear_tf=True)
        self.name_m = self.name_vec.fit_transform(other["name_skel"].fill_null("").to_list())
        self.name_mt = self.name_m.T.tocsr()
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

def _concat_index(other: pl.DataFrame) -> pl.DataFrame:
    keys = pl.concat([
        other.select("rid", "country", pl.col("name_concat").alias("key")),
        other.select("rid", "country", pl.col("name_domain").alias("key")),
    ]).filter(pl.col("key").is_not_null() & (pl.col("key").str.len_chars() >= 6)).unique()
    common = keys.group_by("country", "key").len().filter(pl.col("len") > 50)
    return keys.join(common.select("country", "key"), on=["country", "key"], how="anti")


def _block_country(s1: pl.DataFrame, space: CountrySpace, name_ix: InvertedIndex,
                   addr_ix: InvertedIndex, concat_ix: pl.DataFrame) -> pl.DataFrame:
    name = name_ix.query(_tokens(s1, "name_skel"), K_NAME, "name")
    addr = addr_ix.query(_addr_tokens(s1), K_ADDR, "addr")
    concat = (s1.select("rid", "country", pl.col("name_concat").alias("key"))
              .join(concat_ix, on=["country", "key"], suffix="_o")
              .select(pl.col("rid").alias("s1_rid"), pl.col("rid_o").alias("other_rid"))
              .unique())

    va_name = space.name_vec.transform(s1["name_skel"].fill_null("").to_list())
    m = sp_matmul_topn(va_name, space.name_mt, top_n=K_TFIDF, threshold=TFIDF_MIN_SIM,
                       n_threads=os.cpu_count()).tocoo()
    tfidf = pl.DataFrame({"s1_rid": s1["rid"].gather(m.row),
                          "other_rid": pl.Series(space.rids[m.col], dtype=pl.Int32)})

    sources = {"name": name, "addr": addr, "concat": concat, "tfidf": tfidf}
    pool = pl.concat([d.select("s1_rid", "other_rid") for d in sources.values()]).unique()
    for label, d in sources.items():
        pool = pool.join(d.select("s1_rid", "other_rid", pl.lit(True).alias(f"src_{label}")),
                         on=["s1_rid", "other_rid"], how="left")
    pool = (pool.join(name.select("s1_rid", "other_rid", "name_score"),
                      on=["s1_rid", "other_rid"], how="left")
            .join(addr.select("s1_rid", "other_rid", "addr_score"),
                  on=["s1_rid", "other_rid"], how="left")
            .with_columns(pl.col("^src_.*$").fill_null(False)))

    # Re-rank features for every pooled pair.
    s1_row = dict(zip(s1["rid"].to_list(), range(s1.height)))
    ia = np.fromiter((s1_row[r] for r in pool["s1_rid"].to_list()), np.int64, pool.height)
    ib = np.fromiter((space.row_of[r] for r in pool["other_rid"].to_list()), np.int64,
                     pool.height)
    va_addr = space.addr_vec.transform(_addr_text(s1).to_list())
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
    addr_missing = pl.Series(np.asarray(space.addr_missing)[ib])
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


def block(norm: pl.DataFrame) -> pl.DataFrame:
    out = []
    for country in sorted(norm["country"].unique()):
        s1 = norm.filter((pl.col("source") == 1) & (pl.col("country") == country))
        other = norm.filter((pl.col("source") != 1) & (pl.col("country") == country))
        if s1.height == 0 or other.height == 0:
            continue
        n_other = {country: other.height}
        with timed(f"{country}: build indexes ({other.height} S2/S3)"):
            space = CountrySpace(other)
            name_ix = InvertedIndex(_tokens(other, "name_skel"), n_other)
            addr_ix = InvertedIndex(_addr_tokens(other), n_other)
            concat_ix = _concat_index(other)
        with timed(f"{country}: block {s1.height} S1"):
            for i in range(0, s1.height, S1_CHUNK):
                out.append(_block_country(s1.slice(i, S1_CHUNK), space, name_ix, addr_ix,
                                          concat_ix))
    return pl.concat(out)


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


def run(paths: Paths, subset: str) -> None:
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet")
    with timed(f"block {subset}"):
        cands = block(norm)
    out = paths.work / "block"
    out.mkdir(exist_ok=True)
    cands.write_parquet(out / f"{subset}.parquet")
    if subset in GT_FOR_SUBSET:
        report(cands, norm, pl.read_parquet(paths.slices / f"{GT_FOR_SUBSET[subset]}.parquet"))
    elif subset == "train":
        report(cands, norm, pl.read_parquet(paths.gt_pairs()))
