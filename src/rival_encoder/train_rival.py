"""Fine-tune a cross-encoder on train.parquet (written by rival_data.py); saves the best checkpoint by val AUC.

  python train_rival.py --model BAAI/bge-reranker-v2-m3 --mode select --tag f1 --hours 2.2
  python train_rival.py --model intfloat/multilingual-e5-large --mode pair --tag f2 --hours 2.2

mode select: first = record, second = "candidate: <S1> </s> rival: <strongest other S1>"
mode pair:   first = record, second = <S1>
"""
import argparse
import math
import os
import time

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", choices=["select", "pair"], required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--hours", type=float, default=2.2)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-len", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=3000)
ap.add_argument("--freeze-emb", action="store_true",
                help="freeze the word embeddings (large vocabularies: halves optimizer memory)")
args = ap.parse_args()
MAX_LEN = args.max_len or (256 if args.mode == "select" else 160)
torch.manual_seed(13)
dev = torch.device("cuda")
# GPUs without fast bf16 (e.g. T4): fp16 with loss scaling
DT = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=DT == torch.float16)


def second(df):
    if args.mode == "pair":
        return df["text_s1"].to_list()
    return ("candidate: " + df["text_s1"] + " </s> rival: " + df["text_rival"]).to_list()


tr = pl.read_parquet("train.parquet")
va = pl.read_parquet("score_val.parquet").sample(n=min(60_000, pl.read_parquet("score_val.parquet").height), seed=1)
a_tr, b_tr = tr["text_rec"].to_list(), second(tr)
y = tr["label"].to_numpy().astype(np.float32)
w = tr["w"].to_numpy().astype(np.float32)
w /= w.mean()
a_va, b_va, y_va = va["text_rec"].to_list(), second(va), va["label"].to_numpy()

tok = AutoTokenizer.from_pretrained(args.model)
net = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1,
                                                         ignore_mismatched_sizes=True).to(dev)
if args.freeze_emb:
    net.get_input_embeddings().weight.requires_grad_(False)
opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=args.lr,
                        weight_decay=0.01)
epoch_steps = math.ceil(len(y) / args.batch)
total = [epoch_steps]          # shrunk after timing so the schedule ends inside the budget
warm = 500
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 1 - s / total[0]))
lossf = torch.nn.BCEWithLogitsLoss(reduction="none")


def enc(a, b):
    t = tok(a, b, truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in t.items()}


@torch.no_grad()
def predict(a, b, bs=256):
    net.eval()
    out = []
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    for i in range(0, len(a), bs):
        idx = order[i:i + bs]
        with torch.autocast("cuda", dtype=DT):
            out.append(net(**enc([a[j] for j in idx], [b[j] for j in idx])).logits.float().squeeze(-1).cpu())
    p = torch.sigmoid(torch.cat(out)).numpy()
    res = np.empty_like(p)
    res[order] = p
    net.train()
    return res


os.makedirs(f"ckpt_{args.tag}", exist_ok=True)
t0, best, step = time.time(), -1.0, 0
budget = args.hours * 3600
perm = np.random.default_rng(13).permutation(len(y))
net.train()
for i in range(0, len(perm), args.batch):
    idx = perm[i:i + args.batch]
    with torch.autocast("cuda", dtype=DT):
        logit = net(**enc([a_tr[j] for j in idx], [b_tr[j] for j in idx])).logits.float().squeeze(-1)
    loss = (lossf(logit, torch.from_numpy(y[idx]).to(dev)) * torch.from_numpy(w[idx]).to(dev)).mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    step += 1
    if step == 300:   # fit the schedule to the time budget (eval time included roughly)
        per = (time.time() - t0) / step
        total[0] = min(epoch_steps, int(0.9 * budget / per))
        print(f"[{args.tag}] {per:.3f}s/step -> {total[0]} of {epoch_steps} steps", flush=True)
    out_of_time = time.time() - t0 > budget
    if step % args.eval_every == 0 or step >= total[0] or out_of_time:
        auc = roc_auc_score(y_va, predict(a_va, b_va))
        print(f"[{args.tag}] step {step} loss {loss.item():.4f} val AUC {auc:.5f} "
              f"(LightGBM {roc_auc_score(y_va, va['p_lgb'].to_numpy()):.5f}) {(time.time() - t0) / 60:.0f} min", flush=True)
        if auc > best:
            best = auc
            net.save_pretrained(f"ckpt_{args.tag}")
            tok.save_pretrained(f"ckpt_{args.tag}")
    if step >= total[0] or out_of_time:
        break
print(f"[{args.tag}] done after {step} steps ({(time.time() - t0) / step:.3f}s/step), best val AUC {best:.5f}", flush=True)
