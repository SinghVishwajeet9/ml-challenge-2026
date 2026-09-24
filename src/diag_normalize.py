"""Diagnostics for normalization on the sample: raw vs normalized similarity, for true
pairs (overall and per hard group) vs random / look-alike non-matches.

python src/diag_normalize.py [--work-dir work]
Good normalization: true-pair similarity goes up, non-match similarity stays flat.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

REPO = Path(__file__).resolve().parents[1]


def _simple_tokens(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ")
            .str.strip_chars().str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.unique())


def _split(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")).list.unique())


def _jacc(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(0.0)


def build(work: Path) -> pl.DataFrame:
    pairs = pl.read_parquet(work / "sample" / "pairs.parquet")
    raw = pl.read_parquet(work / "sample" / "records.parquet").select("rid", "name", "address")
    norm = pl.read_parquet(work / "norm" / "sample.parquet")
    side = raw.join(norm, on="rid")
    d = (pairs.join(side.rename(lambda c: c if c == "rid" else c + "_1").rename({"rid": "s1_rid"}),
                    on="s1_rid")
         .join(side.rename(lambda c: c if c == "rid" else c + "_2").rename({"rid": "other_rid"}),
               on="other_rid"))
    d = d.with_columns(
        _simple_tokens("name_1").alias("nr1"), _simple_tokens("name_2").alias("nr2"),
        _split("name_core_1").alias("nc1"), _split("name_core_2").alias("nc2"),
        _split("name_skel_1").alias("ns1"), _split("name_skel_2").alias("ns2"),
        _simple_tokens("address_1").alias("ar1"), _simple_tokens("address_2").alias("ar2"),
        _split("addr_core_1").alias("ac1"), _split("addr_core_2").alias("ac2"),
    )
    concat_hit = (
        ((pl.col("name_concat_1") == pl.col("name_concat_2")) & (pl.col("name_concat_1") != ""))
        | (pl.col("name_concat_1") == pl.col("name_domain_2"))
        | (pl.col("name_concat_2") == pl.col("name_domain_1"))
    ).fill_null(False)
    initials_hit = (
        ((pl.col("name_core_2") == pl.col("name_initials_1")) & (pl.col("name_initials_1").str.len_chars() >= 2))
        | ((pl.col("name_core_1") == pl.col("name_initials_2")) & (pl.col("name_initials_2").str.len_chars() >= 2))
    ).fill_null(False)
    d = d.with_columns(
        _jacc("nr1", "nr2").alias("name_raw_jacc"),
        _jacc("nc1", "nc2").alias("name_core_jacc"),
        _jacc("ns1", "ns2").alias("name_skel_jacc"),
        concat_hit.alias("name_concat_hit"),
        initials_hit.alias("name_initials_hit"),
        _jacc("ar1", "ar2").alias("addr_raw_jacc"),
        _jacc("ac1", "ac2").alias("addr_core_jacc"),
        (pl.col("addr_state_1") == pl.col("addr_state_2")).alias("state_eq"),
        pl.col("addr_house_1").list.set_intersection("addr_house_2").list.len().gt(0).alias("house_eq"),
        pl.col("addr_numbers_1").list.set_intersection("addr_numbers_2").list.len().gt(0).alias("nums_eq"),
    )
    d = d.with_columns(
        pl.max_horizontal("name_core_jacc", "name_skel_jacc",
                          pl.col("name_concat_hit").cast(pl.Float64),
                          pl.col("name_initials_hit").cast(pl.Float64)).alias("name_best"),
    )
    d = d.with_columns(
        pl.Series("name_raw_tset", process.cpdist(
            d["name_1"].str.to_lowercase().to_list(), d["name_2"].str.to_lowercase().to_list(),
            scorer=fuzz.token_set_ratio, workers=-1) / 100),
        pl.Series("name_norm_tset", process.cpdist(
            d["name_norm_1"].fill_null("").to_list(), d["name_norm_2"].fill_null("").to_list(),
            scorer=fuzz.token_set_ratio, workers=-1) / 100),
    )
    return d


SIM = ["name_raw_jacc", "name_core_jacc", "name_skel_jacc", "name_best",
       "name_raw_tset", "name_norm_tset", "addr_raw_jacc", "addr_core_jacc"]
FLAGS = ["name_concat_hit", "name_initials_hit", "state_eq", "house_eq", "nums_eq"]


def report(d: pl.DataFrame) -> None:
    views = {
        "true (base)": d.filter(pl.col("label") & (pl.col("origin") == "base")),
        "true nonlatin": d.filter(pl.col("label") & pl.col("nonlatin")),
        "true name_jacc0": d.filter(pl.col("label") & pl.col("name_jacc0") & ~pl.col("nonlatin")),
        "true addr_missing": d.filter(pl.col("label") & pl.col("addr_missing")),
        "neg_random": d.filter(pl.col("kind") == "neg_random"),
        "neg_lookalike": d.filter(pl.col("kind") == "neg_lookalike"),
    }
    rows = []
    for name, v in views.items():
        is_true = name.startswith("true")
        q = 0.10 if is_true else 0.90  # true: weak tail; negatives: strong tail
        row = {"view": name, "n": v.height}
        for c in SIM:
            x = v[c].drop_nulls().to_numpy()
            row[c] = f"{np.median(x):.2f}/{np.quantile(x, q):.2f}" if len(x) else "-"
        for c in FLAGS:
            row[c] = f"{v[c].fill_null(False).mean():.2f}"
        rows.append(row)
    out = pl.DataFrame(rows)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=20, tbl_hide_dataframe_shape=True):
        print("similarity cells: median/p10 for true pairs, median/p90 for negatives")
        print(out.select("view", "n", *SIM))
        print("flag rates")
        print(out.select("view", *FLAGS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(REPO / "work"))
    report(build(Path(ap.parse_args().work_dir)))
