"""Per-S1 expected-F0.5 decision (replaces the global threshold).

The score is a macro average of per-S1 F0.5, so the best set for one S1 depends on its other
candidates: an extra uncertain match lowers the expected F0.5 of an S1 that already has several
confident matches, and raises it for an S1 whose only candidate is uncertain. With calibrated
stacker probabilities, the expected F0.5 of every prefix of an S1's candidates (sorted by p) is
estimated by Monte Carlo (independent Bernoulli draws), including the empty set, and the best
prefix is kept. One S1 per S2/S3 record is kept as before (only a record's best S1 is eligible).

F0.5 of a prefix of size k: 1.25 * TP / (0.25 * N + k), N = number of true matches of the S1;
the empty set scores 1 when N = 0.
Measured (stacker with F1 column): val +0.00000, val2 +0.00008, delhi +0.00016 over the tuned
global threshold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

NMAX = 16          # candidates per S1 considered (by p)
PMIN = 0.005       # candidates below this are ignored


def decide(sc: pl.DataFrame, samples: int = 400, lo: float = 0.02, hi: float = 0.98,
           base_t: float = 0.5, seed: int = 0) -> pl.DataFrame:
    """sc: (s1_rid, other_rid, p) calibrated scores. Returns the chosen (s1_rid, other_rid)."""
    rng = np.random.default_rng(seed)
    d = (sc.filter(pl.col("p") >= PMIN)
         .with_columns((pl.col("p") == pl.col("p").max().over("other_rid")).alias("elig")))
    unc = d.group_by("s1_rid").agg(((pl.col("p") > lo) & (pl.col("p") < hi)).any().alias("u"))
    d = d.join(unc, on="s1_rid")
    # S1s whose candidates are all near 0 or 1: the decision is obvious
    easy = d.filter(~pl.col("u") & pl.col("elig") & (pl.col("p") >= base_t)).select("s1_rid", "other_rid")
    h = (d.filter(pl.col("u")).sort(["s1_rid", "elig", "p"], descending=[False, True, True])
         .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("j")).filter(pl.col("j") < NMAX))
    s1s = h["s1_rid"].unique(maintain_order=True)
    pos = {int(r): i for i, r in enumerate(s1s.to_list())}
    n = len(pos)
    P = np.zeros((n, NMAX), np.float32)
    EL = np.zeros((n, NMAX), bool)
    OR = np.full((n, NMAX), -1, np.int64)
    si = np.array([pos[int(r)] for r in h["s1_rid"].to_list()], dtype=np.int64)
    jj = h["j"].to_numpy()
    P[si, jj] = h["p"].to_numpy()
    EL[si, jj] = h["elig"].to_numpy()
    OR[si, jj] = h["other_rid"].to_numpy()
    n_elig = EL.sum(1)
    best_k = np.zeros(n, np.int64)
    ks = np.arange(1, NMAX + 1, dtype=np.float32)
    for a in range(0, n, 3000):
        b = min(n, a + 3000)
        X = rng.random((b - a, samples, NMAX), dtype=np.float32) < P[a:b, None, :]
        N = X.sum(-1).astype(np.float32)
        TP = np.cumsum(X, -1, dtype=np.float32)      # eligible candidates come first
        F = 1.25 * TP / (0.25 * N[..., None] + ks)
        EF = np.concatenate([(N == 0).mean(1, keepdims=True), F.mean(1)], 1)
        EF[np.arange(NMAX + 1)[None, :] > n_elig[a:b, None]] = -1
        best_k[a:b] = EF.argmax(1)
    take = np.arange(NMAX)[None, :] < best_k[:, None]
    chosen = pl.DataFrame({"s1_rid": np.repeat(s1s.to_numpy(), NMAX)[take.ravel()].astype(np.int32),
                           "other_rid": OR[take].astype(np.int32)})
    return pl.concat([easy.cast({"s1_rid": pl.Int32, "other_rid": pl.Int32}), chosen])
