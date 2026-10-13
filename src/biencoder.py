"""Fine-tuned multilingual-e5 bi-encoder: dense similarity feature and candidate recovery.

Each record is embedded once from its raw text ("query: <name> | <address>"), so similarity
for any (S1, record) pair is a dot product and nearest-neighbour search finds true S1s that
the TF-IDF blocking missed (typo-heavy names, trade names, domains, native scripts).

Training: contrastive (InfoNCE, symmetric, in-batch negatives) on labelled train pairs of S1s
outside the holdouts (val, val2, delhi, dev), plus confident French test pairs (pseudo-labels,
no labels used) so the model sees French vocabulary. Batches are grouped by (country, state)
so in-batch negatives are same-state businesses. One pair per S1 per batch.

  python src/biencoder.py train    --work-dir work --batch 128 --hours 1.0
  python src/biencoder.py embed    --work-dir work --subset val2      (val, val2, delhi, test)
  python src/biencoder.py retrieve --work-dir work --subset val2 --k 10
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HOLDOUTS = ("val", "val2", "delhi", "dev")
MAX_LEN = 64
DIM_DT = np.float16


def _text(name: pl.Expr, address: pl.Expr) -> pl.Expr:
    return "query: " + name.fill_null("") + " | " + address.fill_null("")


def _pool(out, mask):
    h = out.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
    return F.normalize(h.sum(1) / mask.sum(1, keepdim=True), dim=-1)


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- training data
def _hard_negatives(blocks: list[Path], recs: pl.DataFrame, true_pairs: pl.DataFrame) -> pl.DataFrame:
    """For each record (other_rid): the wrong S1 with the highest blocking score among its
    blocking candidates. These are the look-alikes (same-name twins, same-address siblings)."""
    b = pl.concat([pl.scan_parquet(f).select("s1_rid", "other_rid", "block_score")
                   .join(recs.lazy(), on="other_rid", how="semi").collect() for f in blocks])
    b = b.join(true_pairs, on=["s1_rid", "other_rid"], how="anti")
    return (b.sort("block_score", descending=True).unique("other_rid", keep="first")
            .select("other_rid", pl.col("s1_rid").alias("hn_rid")))


def training_pairs(work: Path, n_pairs: int, n_france: int, seed: int = 0) -> pl.DataFrame:
    raw = pl.read_parquet(work / "raw" / "train_records.parquet",
                          columns=["rid", "entity_id", "name", "address", "country"])
    ids = raw.select("rid", "entity_id")
    gt = pl.read_parquet(work / "raw" / "train_gt_pairs.parquet")
    held = pl.concat([pl.read_parquet(work / "slices" / f"{s}_records.parquet",
                                      columns=["entity_id", "source"])
                      .filter(pl.col("source") == 1).select("entity_id") for s in HOLDOUTS])
    gt = gt.join(held.rename({"entity_id": "s1_id"}), on="s1_id", how="anti")
    gt = (gt.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
          .join(ids.rename({"rid": "other_rid", "entity_id": "match_id"}), on="match_id")
          .select("s1_rid", "other_rid"))
    # hard negatives exist for records of S1s that went through blocking (tune, rest)
    blocks = sorted((work / "block" / "tune").glob("*.parquet")) + sorted(
        (work / "block" / "rest").glob("*.parquet"))
    hn = _hard_negatives(blocks, gt.select("other_rid"), gt)
    gt = gt.join(hn, on="other_rid")
    # one random positive per S1, then cap
    gt = (gt.with_columns(pl.int_range(pl.len()).shuffle(seed).alias("_r"))
          .sort("_r").unique("s1_rid", keep="first").drop("_r"))
    gt = gt.sample(n=min(n_pairs, gt.height), seed=seed)
    st = pl.read_parquet(work / "norm" / "train.parquet", columns=["rid", "addr_state"])
    txt = raw.select("rid", "country", _text(pl.col("name"), pl.col("address")).alias("t"))
    d = (gt.join(txt.rename({"rid": "s1_rid", "t": "t_s1"}), on="s1_rid")
         .join(txt.select(pl.col("rid").alias("other_rid"), pl.col("t").alias("t_rec")),
               on="other_rid")
         .join(txt.select(pl.col("rid").alias("hn_rid"), pl.col("t").alias("t_hn")), on="hn_rid")
         .join(st.rename({"rid": "s1_rid", "addr_state": "state"}), on="s1_rid", how="left")
         .select(pl.col("s1_rid").cast(pl.String).alias("s1_id"), "country", "state",
                 "t_s1", "t_rec", "t_hn"))
    del raw, txt
    if n_france:
        d = pl.concat([d, france_pseudo(work, n_france, seed)])
    return d


def france_pseudo(work: Path, n: int, seed: int) -> pl.DataFrame:
    """Confident French test pairs (final stacker p >= 0.99, record's best S1), one per S1,
    with the record's best other blocking candidate as hard negative (a record has one S1)."""
    norm = pl.read_parquet(work / "norm" / "test.parquet", columns=["rid", "country", "addr_state"])
    fr = norm.filter((pl.col("country") == "France"))
    sc = (pl.scan_parquet(work / "trap" / "test_refined_f1.parquet")
          .filter(pl.col("p") >= 0.99).select("s1_rid", "other_rid", "p").collect())
    sc = sc.filter(pl.col("p") == pl.col("p").max().over("other_rid"))
    sc = sc.join(fr.select(pl.col("rid").alias("s1_rid"), pl.col("addr_state").alias("state")),
                 on="s1_rid")
    sc = (sc.with_columns(pl.int_range(pl.len()).shuffle(seed).alias("_r")).sort("_r")
          .unique("s1_rid", keep="first").drop("_r"))
    hn = _hard_negatives([work / "block" / "test" / "France.parquet"], sc.select("other_rid"),
                         sc.select("s1_rid", "other_rid"))
    sc = sc.join(hn, on="other_rid")
    sc = sc.sample(n=min(n, sc.height), seed=seed)
    raw = pl.read_parquet(work / "raw" / "test_records.parquet", columns=["rid", "name", "address"])
    txt = raw.select("rid", _text(pl.col("name"), pl.col("address")).alias("t"))
    return (sc.join(txt.rename({"rid": "s1_rid", "t": "t_s1"}), on="s1_rid")
            .join(txt.rename({"rid": "other_rid", "t": "t_rec"}), on="other_rid")
            .join(txt.rename({"rid": "hn_rid", "t": "t_hn"}), on="hn_rid")
            .select(pl.col("s1_rid").cast(pl.String).alias("s1_id"), pl.lit("France").alias("country"),
                    "state", "t_s1", "t_rec", "t_hn"))


def state_batches(d: pl.DataFrame, bs: int, seed: int) -> list[np.ndarray]:
    """Row indices grouped into batches of one (country, state); batch order shuffled."""
    rng = np.random.default_rng(seed)
    g = (d.with_row_index("i").with_columns(pl.col("state").fill_null("?"))
         .with_columns(pl.lit(rng.random(d.height)).alias("_r")).sort(["country", "state", "_r"]))
    idx = g["i"].to_numpy()
    batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    batches = [b for b in batches if len(b) >= bs // 4]
    rng.shuffle(batches)
    return batches


def train(a) -> None:
    work = Path(a.work_dir)
    out = work / "bienc"
    out.mkdir(parents=True, exist_ok=True)
    _log("building training pairs")
    d = training_pairs(work, a.pairs, a.france)
    _log(f"pairs: {d.height} ({d.group_by('country').len().rows()})")
    batches = state_batches(d, a.batch, 0)
    ts1, trec, thn = d["t_s1"].to_list(), d["t_rec"].to_list(), d["t_hn"].to_list()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.model)
    net = AutoModel.from_pretrained(a.model).to(dev)
    net.gradient_checkpointing_enable() if a.checkpointing else None
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(batches)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / total))
    scale = 1.0 / a.temp
    t0 = time.time()
    net.train()

    def enc(texts):
        t = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        t = {k: v.to(dev) for k, v in t.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _pool(net(**t), t["attention_mask"]).float()

    for step, b in enumerate(batches):
        q = enc([trec[i] for i in b])
        p = enc([ts1[i] for i in b] + [thn[i] for i in b])     # positives, then hard negatives
        s = q @ p.T * scale
        lab = torch.arange(len(b), device=dev)
        # record -> S1 (positives + all hard negatives), S1 -> record (positives only)
        loss = (F.cross_entropy(s, lab) + F.cross_entropy(s[:, :len(b)].T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            el = time.time() - t0
            _log(f"step {step}/{total} loss {loss.item():.4f} {el / (step + 1):.3f}s/step")
            if step == 200:
                # shrink the schedule so it ends inside the time budget
                fit = int(a.hours * 3600 / (el / (step + 1)))
                if fit < total:
                    total = fit
                    _log(f"budget: stopping after {total} steps")
        if step + 1 >= total:
            break
    net.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    _log(f"saved {out / 'model'} after {step + 1} steps, {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------------- embedding
def _records(work: Path, subset: str) -> pl.DataFrame:
    if subset == "test":
        return pl.read_parquet(work / "raw" / "test_records.parquet",
                               columns=["rid", "source", "name", "address", "country"])
    return pl.read_parquet(work / "slices" / f"{subset}_records.parquet",
                           columns=["rid", "source", "name", "address", "country"])


@torch.no_grad()
def embed(a) -> None:
    work = Path(a.work_dir)
    out = work / "bienc"
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(out / "model")
    net = AutoModel.from_pretrained(out / "model", torch_dtype=torch.float16).to(dev).eval()
    r = _records(work, a.subset).sort("rid")
    texts = r.select(_text(pl.col("name"), pl.col("address")).alias("t"))["t"].to_list()
    rid = r["rid"].to_numpy()
    n = len(texts)
    emb = np.lib.format.open_memmap(out / f"{a.subset}_emb.npy", mode="w+", dtype=DIM_DT,
                                    shape=(n, net.config.hidden_size))
    order = np.argsort([len(t) for t in texts], kind="stable")
    t0 = time.time()
    for i in range(0, n, a.batch):
        idx = order[i:i + a.batch]
        t = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True,
                return_tensors="pt")
        t = {k: v.to(dev) for k, v in t.items()}
        emb[idx] = _pool(net(**t), t["attention_mask"]).cpu().numpy().astype(DIM_DT)
        if (i // a.batch) % 500 == 0:
            _log(f"{a.subset}: {i}/{n} ({(time.time() - t0) / 60:.1f} min)")
    emb.flush()
    np.save(out / f"{a.subset}_rid.npy", rid)
    pl.DataFrame({"rid": rid, "source": r["source"].to_numpy(), "country": r["country"]}).write_parquet(
        out / f"{a.subset}_meta.parquet")
    _log(f"{a.subset}: embedded {n} records in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------------- retrieval
@torch.no_grad()
def retrieve(a) -> None:
    """For every S2/S3 record: its top-k S1 of the same country by cosine."""
    work = Path(a.work_dir)
    out = work / "bienc"
    dev = torch.device("cuda")
    emb = np.load(out / f"{a.subset}_emb.npy", mmap_mode="r")
    meta = pl.read_parquet(out / f"{a.subset}_meta.parquet").with_row_index("i")
    parts = []
    for country in meta["country"].unique().to_list():
        m = meta.filter(pl.col("country") == country)
        s1 = m.filter(pl.col("source") == 1)
        ot = m.filter(pl.col("source") != 1)
        S = torch.from_numpy(np.ascontiguousarray(emb[s1["i"].to_numpy()])).to(dev)
        s1_rid = torch.from_numpy(s1["rid"].to_numpy()).to(dev)
        oi, orid = ot["i"].to_numpy(), ot["rid"].to_numpy()
        k = min(a.k, S.shape[0])
        for j in range(0, len(oi), a.chunk):
            Q = torch.from_numpy(np.ascontiguousarray(emb[oi[j:j + a.chunk]])).to(dev)
            v, ix = (Q @ S.T).topk(k, dim=1)          # fp16: a float32 copy would not fit in memory
            parts.append(pl.DataFrame({
                "other_rid": np.repeat(orid[j:j + a.chunk], k).astype(np.int32),
                "s1_rid": s1_rid[ix].flatten().cpu().numpy().astype(np.int32),
                "cos": v.flatten().cpu().numpy().astype(np.float32),
                "rank": np.tile(np.arange(k, dtype=np.int8), len(Q))}))
        _log(f"{a.subset} {country}: {len(oi)} records vs {S.shape[0]} S1")
        del S
    pl.concat(parts).write_parquet(out / f"{a.subset}_knn.parquet")
    _log(f"wrote {out / f'{a.subset}_knn.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "embed", "retrieve"])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--subset", default="val2")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--pairs", type=int, default=1_600_000)
    ap.add_argument("--france", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    if a.cmd == "embed" and a.batch == 256:
        a.batch = 1024
    {"train": train, "embed": embed, "retrieve": retrieve}[a.cmd](a)


if __name__ == "__main__":
    main()
