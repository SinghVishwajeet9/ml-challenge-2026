"""Score one shard of a score file written by rival_data.py with a trained checkpoint.

  python score_rival.py --tag f1 --mode select --set test --shard 0 --nshards 4
writes results/<tag>_<set>_<shard>.parquet (s1_rid, other_rid, px)
"""
import argparse
import os

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--mode", choices=["select", "pair"], required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--max-len", type=int, default=0)
ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()
MAX_LEN = args.max_len or (256 if args.mode == "select" else 160)
dev = torch.device("cuda")

df = pl.read_parquet(f"score_{args.set}.parquet").with_row_index("i")
df = df.filter(pl.col("i") % args.nshards == args.shard)
a = df["text_rec"].to_list()
b = (df["text_s1"] if args.mode == "pair" else "candidate: " + df["text_s1"] + " </s> rival: " + df["text_rival"]).to_list()
tok = AutoTokenizer.from_pretrained(f"ckpt_{args.tag}")
net = AutoModelForSequenceClassification.from_pretrained(f"ckpt_{args.tag}").to(dev).eval()
order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
out = []
with torch.no_grad():
    for i in range(0, len(a), args.batch):
        idx = order[i:i + args.batch]
        t = tok([a[j] for j in idx], [b[j] for j in idx], truncation="longest_first",
                max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            out.append(net(**t).logits.float().squeeze(-1).cpu())
p = torch.sigmoid(torch.cat(out)).numpy()
px = np.empty_like(p)
px[order] = p
os.makedirs("results", exist_ok=True)
df.select("s1_rid", "other_rid").with_columns(pl.Series("px", px)).write_parquet(
    f"results/{args.tag}_{args.set}_{args.shard}.parquet")
print(f"{args.tag} {args.set} shard {args.shard}: {len(px)} pairs", flush=True)
