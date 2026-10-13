"""Hybrid cross-encoder for uncertain candidate pairs.

A multilingual transformer reads both records (raw name and address, native scripts
included) as one text pair; the LightGBM feature vector is fed in alongside it, and a
small head combines the two. It is only applied to pairs LightGBM is unsure about, and
its probability is blended with LightGBM's there.

Each trained variant lives in work/xenc/<tag>/ (weights, settings, val scores, blend).

Stages (run_pipeline.py):
  xenc-oof    2-fold LightGBM on the training features -> out-of-fold probabilities, so
              "uncertain" training pairs are chosen the same way as at prediction time
  xenc-train  fine-tune on uncertain training pairs; score the val band
  xenc-blend  choose blend weight and threshold on val -> <tag>/blend.json
  xenc-score  score the uncertain test pairs saved by `predict`
  xenc-apply  blend test scores and decide -> work/pred/test_matches.parquet
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from common import Paths, log, switch, timed

BAND = (0.01, 0.99)        # LightGBM probability range treated as uncertain
SEED = 13
# Training features keep only this share of non-matching pairs; they are re-weighted.
NEG_FRAC = 0.15


@dataclass
class XConfig:
    model: str = "intfloat/multilingual-e5-small"
    epochs: int = 2
    max_len: int = 128
    batch: int = 64
    lr_encoder: float = 3e-5
    lr_head: float = 1e-3
    lora: bool = False           # train low-rank adapters instead of all encoder weights
    lora_r: int = 16
    train_frac: float = 1.0      # sample of the training band (faster experiments)
    addr_placeholder: bool = False   # "address unavailable" instead of an empty address
    rest_frac: float = 1.0       # random share of rest's uncertain pairs to train on
    use_rest: bool = False       # add uncertain pairs from the 'rest' subset (train minus
                                 # tune and val), scored out-of-sample by the current LightGBM


def _dir(paths: Paths, tag: str):
    d = paths.work / "xenc" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- data

def oof_predictions(paths: Paths, subset: str = "tune", neg_frac: float = NEG_FRAC) -> None:
    """Out-of-fold LightGBM probabilities for the training features (2 folds by S1)."""
    import model as M
    feats = pl.read_parquet(paths.work / "features" / f"{subset}.parquet")
    policy = json.loads((paths.work / "models" / "policy.json").read_text())
    cols = policy["features"]
    fold = (feats["s1_rid"].hash(SEED) % 2).to_numpy()
    y = feats["label"].to_numpy()
    X = M._matrix(feats, cols)
    p = np.zeros(len(y), dtype=np.float32)
    for k in (0, 1):
        tr = fold != k
        w = np.where(y[tr], 1.0, 1.0 / neg_frac)
        with timed(f"oof fold {k}"):
            b = lgb.train(M.PARAMS, lgb.Dataset(X[tr], y[tr], weight=w),
                          policy.get("best_iteration") or 800)
            p[~tr] = b.predict(X[~tr])
    out = feats.select("s1_rid", "other_rid", "label").with_columns(pl.Series("p", p))
    out.write_parquet(paths.work / "models" / f"{subset}_oof.parquet")
    band = out.filter((pl.col("p") >= BAND[0]) & (pl.col("p") <= BAND[1]))
    log.info("oof written; band pairs %d (%.1f%%), positives in band %.1f%%", band.height,
             100 * band.height / out.height, 100 * band["label"].mean())


def _raw_records(paths: Paths, subset: str) -> pl.DataFrame:
    if subset in ("test", "rest"):
        split = "test" if subset == "test" else "train"
        raw = pl.read_parquet(paths.records(split), columns=["rid", "name", "address"])
    else:
        # val2 records live in the slices folder like dev / tune / val.
        raw = pl.read_parquet(paths.slices / f"{subset}_records.parquet",
                              columns=["rid", "name", "address"])
    if switch("xenc_translated"):
        # Native-script names shown to the transformer in English words (native_dict.py).
        import native_dict
        from normalize import INDIC
        uniq = raw.filter(pl.col("name").fill_null("").str.contains(INDIC))["name"].unique()
        if uniq.len():
            new = pl.Series([native_dict.translate(s) for s in uniq.to_list()], dtype=pl.String)
            raw = raw.with_columns(pl.col("name").replace(uniq, new))
    return raw


def _text(name: pl.Expr, address: pl.Expr, placeholder: bool = False) -> pl.Expr:
    if placeholder:
        address = (pl.when(address.is_null() | (address.str.strip_chars() == ""))
                   .then(pl.lit("address unavailable")).otherwise(address))
    return name.fill_null("") + " | " + address.fill_null("")


def _with_text(paths: Paths, subset: str, pairs: pl.DataFrame,
               placeholder: bool = False) -> pl.DataFrame:
    raw = _raw_records(paths, subset)
    txt = _text(pl.col("name"), pl.col("address"), placeholder)
    return (pairs.join(raw.select(pl.col("rid").alias("s1_rid"), txt.alias("text_a")),
                       on="s1_rid")
            .join(raw.select(pl.col("rid").alias("other_rid"), txt.alias("text_b")),
                  on="other_rid"))


def band_pairs(paths: Paths, subset: str, scored: pl.DataFrame, features_file,
               placeholder: bool = False) -> pl.DataFrame:
    """Uncertain pairs with both texts and the numeric feature columns. The feature file
    is scanned lazily and filtered to the band, so memory stays small."""
    band = scored.filter((pl.col("p") >= BAND[0]) & (pl.col("p") <= BAND[1]))
    feats = (pl.scan_parquet(features_file)
             .join(band.lazy(), on=["s1_rid", "other_rid"], how="inner").collect())
    return _with_text(paths, subset, feats, placeholder)


def _rest_band(paths: Paths, cols: list[str], placeholder: bool = False) -> pl.DataFrame:
    """Uncertain pairs of the 'rest' subset (saved by `predict --subset rest`), labelled
    from the ground truth. All candidates were scored, so no re-weighting is needed."""
    norm = pl.read_parquet(paths.norm / "rest.parquet", columns=["rid", "entity_id"])
    gt = (pl.read_parquet(paths.gt_pairs())
          .join(norm.select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("s1_id")),
                on="s1_id")
          .join(norm.select(pl.col("rid").alias("other_rid"),
                            pl.col("entity_id").alias("match_id")), on="match_id")
          .select("s1_rid", "other_rid", pl.lit(True).alias("label")))
    band = (pl.read_parquet(paths.work / "pred" / "rest_band_feats.parquet")
            .join(gt, on=["s1_rid", "other_rid"], how="left")
            .with_columns(pl.col("label").fill_null(False), pl.lit(1.0).alias("w")))
    return _with_text(paths, "rest", band, placeholder)


# --------------------------------------------------------------------------- model

def _build(cfg: XConfig, n_num: int):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            enc = AutoModel.from_pretrained(cfg.model)
            if cfg.lora:
                from peft import LoraConfig, get_peft_model
                enc = get_peft_model(enc, LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r,
                                                     lora_dropout=0.05,
                                                     target_modules=["query", "value"]))
            self.encoder = enc
            h = enc.config.hidden_size
            self.text_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 128), nn.ReLU(),
                                           nn.Dropout(0.1))
            self.num_head = nn.Sequential(nn.Linear(n_num, 128), nn.LayerNorm(128), nn.ReLU(),
                                          nn.Dropout(0.1))
            self.out = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, input_ids, attention_mask, num):
            hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean pooling
            return self.out(torch.cat([self.text_head(pooled), self.num_head(num)], -1)).squeeze(-1)

    return Hybrid()


class NumScaler:
    """Median-imputed, standardised, clipped numeric features."""

    def fit(self, X: np.ndarray) -> "NumScaler":
        self.med = np.nanmedian(X, axis=0)
        Xf = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xf = np.where(np.isnan(X), self.med, X)
        return np.clip((Xf - self.mu) / self.sd, -8, 8).astype(np.float32)

    def to_json(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("med", "mu", "sd")}

    @classmethod
    def from_json(cls, d: dict) -> "NumScaler":
        s = cls()
        for k, v in d.items():
            setattr(s, k, np.asarray(v, dtype=np.float64))
        return s


def _batches(tok, cfg: XConfig, a, b, num: np.ndarray, idx: np.ndarray, device):
    import torch
    for s in range(0, len(idx), cfg.batch):
        i = idx[s:s + cfg.batch]
        enc = tok([a[j] for j in i], [b[j] for j in i], truncation=True, max_length=cfg.max_len,
                  padding=True, return_tensors="pt")
        yield (enc["input_ids"].to(device), enc["attention_mask"].to(device),
               torch.from_numpy(num[i]).to(device), i)


def _predict(model, tok, cfg: XConfig, a, b, num, device) -> np.ndarray:
    import torch
    model.eval()
    out = np.zeros(len(a), dtype=np.float32)
    big = XConfig(**{**asdict(cfg), "batch": cfg.batch * 4})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for ids, mask, x, i in _batches(tok, big, a, b, num, np.arange(len(a)), device):
            out[i] = torch.sigmoid(model(ids, mask, x).float()).cpu().numpy()
    return out


def _load(paths: Paths, tag: str):
    import torch
    from transformers import AutoTokenizer
    d = _dir(paths, tag)
    meta = json.loads((d / "meta.json").read_text())
    cfg = XConfig(**meta["config"])
    net = _build(cfg, len(meta["features"]))
    net.load_state_dict(torch.load(d / "hybrid.pt", map_location="cuda"))
    return net.to("cuda"), AutoTokenizer.from_pretrained(cfg.model), cfg, meta


# --------------------------------------------------------------------------- stages

def train(paths: Paths, tag: str, cfg: XConfig) -> None:
    import torch
    from transformers import AutoTokenizer
    from sklearn.metrics import log_loss, roc_auc_score

    import model as M
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    out_dir = _dir(paths, tag)
    cols = json.loads((paths.work / "models" / "policy.json").read_text())["features"]

    tr = band_pairs(paths, "tune",
                    pl.read_parquet(paths.work / "models" / "tune_oof.parquet").drop("label"),
                    paths.work / "features" / "tune.parquet", cfg.addr_placeholder)
    # Tune features keep only NEG_FRAC of non-matches, so those rows are up-weighted.
    tr = tr.with_columns(pl.when(pl.col("label")).then(1.0).otherwise(1.0 / NEG_FRAC).alias("w"))
    if cfg.use_rest:
        rest = _rest_band(paths, cols, cfg.addr_placeholder)
        if cfg.rest_frac < 1:
            rest = rest.sample(fraction=cfg.rest_frac, seed=SEED)
        keep = ["s1_rid", "other_rid", "p", "label", "w", "text_a", "text_b", *cols]
        tr = pl.concat([tr.select(keep), rest.select(keep)], how="vertical_relaxed")
        log.info("[%s] + rest band pairs %d", tag, rest.height)
    if cfg.train_frac < 1:
        tr = tr.sample(fraction=cfg.train_frac, seed=SEED)
    va = band_pairs(paths, "val", pl.read_parquet(paths.work / "models" / "val_scored.parquet"),
                    paths.work / "features" / "val.parquet", cfg.addr_placeholder)
    log.info("[%s] %s | train band pairs %d (pos %.1f%%) | val band pairs %d", tag, cfg,
             tr.height, 100 * tr["label"].mean(), va.height)

    scaler = NumScaler().fit(M._matrix(tr, cols).astype(np.float64))
    xtr, xva = scaler.transform(M._matrix(tr, cols)), scaler.transform(M._matrix(va, cols))
    ytr = tr["label"].to_numpy().astype(np.float32)
    wtr = tr["w"].to_numpy().astype(np.float32)
    wtr /= wtr.mean()
    a_tr, b_tr = tr["text_a"].to_list(), tr["text_b"].to_list()
    a_va, b_va = va["text_a"].to_list(), va["text_b"].to_list()

    tok = AutoTokenizer.from_pretrained(cfg.model)
    net = _build(cfg, len(cols)).to(device)
    enc_params = [p for p in net.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    log.info("[%s] trainable parameters: encoder %.1fM, head %.2fM", tag,
             sum(p.numel() for p in enc_params) / 1e6, sum(p.numel() for p in head_params) / 1e6)
    lr_enc = cfg.lr_encoder * (10 if cfg.lora else 1)   # adapters take a larger rate
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_enc},
                             {"params": head_params, "lr": cfg.lr_head}], weight_decay=0.01)
    steps = cfg.epochs * math.ceil(len(ytr) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_enc, cfg.lr_head],
                                                total_steps=steps, pct_start=0.06)
    amp = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(SEED)
    yva = va["label"].to_numpy()
    for ep in range(cfg.epochs):
        net.train()
        t0, seen = time.time(), 0
        for ids, mask, x, i in _batches(tok, cfg, a_tr, b_tr, xtr, rng.permutation(len(ytr)),
                                        device):
            with torch.autocast("cuda", dtype=torch.float16):
                per = lossf(net(ids, mask, x).float(), torch.from_numpy(ytr[i]).to(device))
                loss = (per * torch.from_numpy(wtr[i]).to(device)).mean()
            opt.zero_grad(set_to_none=True)
            amp.scale(loss).backward()
            amp.step(opt)
            amp.update()
            sched.step()
            seen += len(i)
        pv = _predict(net, tok, cfg, a_va, b_va, xva, device)
        log.info("[%s] epoch %d (%.0f pairs/s): val band AUC %.4f logloss %.4f | LightGBM AUC "
                 "%.4f logloss %.4f", tag, ep, seen / (time.time() - t0),
                 roc_auc_score(yva, pv), log_loss(yva, np.clip(pv, 1e-6, 1 - 1e-6)),
                 roc_auc_score(yva, va["p"].to_numpy()),
                 log_loss(yva, np.clip(va["p"].to_numpy(), 1e-6, 1 - 1e-6)))

    torch.save(net.state_dict(), out_dir / "hybrid.pt")
    (out_dir / "meta.json").write_text(json.dumps({"features": cols, "scaler": scaler.to_json(),
                                                   "config": asdict(cfg), "band": BAND}))
    va.select("s1_rid", "other_rid").with_columns(pl.Series("px", pv)).write_parquet(
        out_dir / "val_scored.parquet")
    log.info("[%s] saved to %s", tag, out_dir)


def _blend(scored: pl.DataFrame, px: pl.DataFrame, w: float) -> pl.DataFrame:
    return (scored.join(px, on=["s1_rid", "other_rid"], how="left")
            .with_columns(pl.when(pl.col("px").is_not_null())
                          .then((1 - w) * pl.col("p") + w * pl.col("px"))
                          .otherwise(pl.col("p")).alias("p"))
            .select("s1_rid", "other_rid", "p"))


def blend(paths: Paths, tag: str) -> None:
    """Choose blend weight and threshold on val; save them with the model."""
    import model as M
    gt, s1 = M._gt_rids(paths, "val")
    sc = pl.read_parquet(paths.work / "models" / "val_scored.parquet")
    px = pl.read_parquet(_dir(paths, tag) / "val_scored.parquet")
    results = []
    for w in (0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
        s = _blend(sc, px, w)
        f, t = max((M._score_decision(M.decide_threshold(s, t), gt, s1), t)
                   for t in np.arange(0.5, 0.95, 0.02))
        results.append((f, w, float(t)))
        log.info("[%s]   blend weight %.2f: val F0.5 %.4f @t=%.2f", tag, w, f, t)
    f, w, t = max(results)
    (_dir(paths, tag) / "blend.json").write_text(json.dumps({"w": w, "t": t, "val_f05": f}))
    log.info("[%s] best: weight %.2f, threshold %.2f, val F0.5 %.4f", tag, w, t, f)


def score(paths: Paths, subset: str, tag: str) -> None:
    """Transformer probabilities for the uncertain pairs saved by `predict`."""
    net, tok, cfg, meta = _load(paths, tag)
    band = pl.read_parquet(paths.work / "pred" / f"{subset}_band_feats.parquet")
    band = _with_text(paths, subset, band, cfg.addr_placeholder)
    x = NumScaler.from_json(meta["scaler"]).transform(
        band.select(pl.col(c).cast(pl.Float32) for c in meta["features"]).to_numpy())
    with timed(f"[{tag}] score {band.height} uncertain {subset} pairs"):
        px = _predict(net, tok, cfg, band["text_a"].to_list(), band["text_b"].to_list(), x, "cuda")
    band.select("s1_rid", "other_rid").with_columns(pl.Series("px", px)).write_parquet(
        _dir(paths, tag) / f"{subset}_scored.parquet")


def apply(paths: Paths, subset: str, tag: str) -> None:
    """Blend LightGBM and transformer scores with the val-chosen weight and threshold."""
    import model as M
    params = json.loads((_dir(paths, tag) / "blend.json").read_text())
    sc = pl.read_parquet(paths.work / "pred" / f"{subset}_scored.parquet")
    px = pl.read_parquet(_dir(paths, tag) / f"{subset}_scored.parquet")
    matches = M.decide_threshold(_blend(sc, px, params["w"]), params["t"])
    matches.write_parquet(paths.work / "pred" / f"{subset}_matches.parquet")
    log.info("[%s] %s: %d matches (blend weight %.2f, threshold %.2f)", tag, subset,
             matches.height, params["w"], params["t"])
