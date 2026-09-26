"""Weight-space patching on OLMo 2 1B: is the reversible part of the update low-rank?

THE QUESTION. At OLMo's readout state the crash is carried by fact-specific displacements
that reverse together (decomp_offline: removing everything the facts' changes share makes
accuracy worse, 0.21 -> 0.02). The mechanism's statement is about WEIGHTS: the shared write
is one direction of the weight update, withdrawn as one thing; through a deep nonlinear
network one weight direction displaces every fact differently. So the place to look for
"one thing" is the update itself.

WHAT IS DONE, at every eval step, with training paused:
    for every 2-D parameter matrix W:  dW = W(t) - W(0)
    PATCH r:   W_patched = W(t) - (top-r singular directions of dW)      r in RANKS
    KEEP r:    W_patched = W(0) + (top-r singular directions of dW)      the low-rank part alone
    CONTROL r: W_patched = W(t) - (random rank-r component of dW with the same Frobenius
               mass as the top-r part)
then score A (non-copy and copy) and B with the patched weights, restore W(t), continue.
Also logged per matrix at the trough: the share of ||dW||_F^2 in its top 1 / 4 / 16
directions (how low-rank the update is at all).

Reading. If the rank-1-per-matrix patch restores A at the trough while B keeps most of its
accuracy, suppression is low-rank in weight space and projecting it out is the mitigation.
If only the control-matched patch helps, or neither, the reversible part is not low-rank
per matrix either.

Same injection as llm/inject_decomp.py (lr, seed, data, schedule). Writes
llm/out/patchw_lr<lr>_seed<seed>.json after every eval.

  .venv-llm/bin/python -m llm.patch_weights --lr 1e-5 --steps 1000 --seed 1

Also scored under every weight setting: the mean next-token loss on a fixed batch of
pre-cutoff prose (llm_real's generic pool), so "does the patched model still model ordinary
text" is answered in the same run.
"""
import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.evalsets import build_a, build_b
from llm.inject import DATASET_ID, MODEL_ID, DocSampler, eval_schedule, score, summarise

RANKS = (1, 4, 16)
SKIP = ("embed_tokens", "lm_head")      # patched matrices are the transformer blocks only
GENERIC_ROWS = 64                       # 64 x 512 tokens of pre-cutoff prose, fixed once


def matrices(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.dim() == 2 and not any(s in n for s in SKIP)]


@torch.no_grad()
def lowrank_parts(dW, r, gen):
    """(top-r part of dW, random rank-r part with the same Frobenius norm)."""
    q = min(r + 8, min(dW.shape))
    U, S, V = torch.svd_lowrank(dW, q=q, niter=4)
    top = (U[:, :r] * S[:r]) @ V[:, :r].T
    a = torch.randn(dW.shape[0], r, generator=gen, device=dW.device, dtype=dW.dtype)
    b = torch.randn(dW.shape[1], r, generator=gen, device=dW.device, dtype=dW.dtype)
    rnd = a @ b.T
    rnd = rnd * (top.norm() / rnd.norm().clamp_min(1e-12))
    return top, rnd, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--tokens_per_step", type=int, default=32768)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--facts", default="llm/out/b_facts.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--generic", default="llm_real/out/generic_tokens.npy",
                    help="pre-tokenized pre-cutoff documents (llm_real fetch-generic); '' disables")
    a = ap.parse_args()
    out_path = a.out or f"llm/out/patchw_lr{a.lr:g}_seed{a.seed}.json"

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    ds = load_dataset(DATASET_ID, split="train")
    a_items, a_pools, a_info = build_a(tok, a.gate, ds)
    b_items, b_pools, facts = build_b(tok, a.facts)
    mats = matrices(model)
    W0 = {n: p.detach().clone() for n, p in mats}
    print(f"device={device} lr={a.lr:g} steps={a.steps} seed={a.seed} | {len(mats)} matrices patched, "
          f"{sum(p.numel() for _, p in mats) / 1e6:.0f}M params | A {a_info['n_A']} B {len(b_items)}")

    # generic text: does the patched model still model ordinary prose? A fixed batch of packed
    # rows from the pre-cutoff pool, mean next-token loss under every weight setting.
    generic_rows = None
    if a.generic and os.path.exists(a.generic):
        import numpy as np
        from llm_real.tokens import TokenPool
        pool = TokenPool(list(np.load(a.generic, allow_pickle=True)), a.seed + 3)
        sep = tok("\n\n", add_special_tokens=False).input_ids
        generic_rows = torch.tensor([pool.row(a.seq_len, sep) for _ in range(GENERIC_ROWS)]).to(device)
        print(f"generic text: {GENERIC_ROWS} rows x {a.seq_len} tokens from {len(pool.docs)} documents")
    else:
        print(f"generic text: not scored ({a.generic!r} not found)")

    @torch.no_grad()
    def generic_loss():
        if generic_rows is None:
            return float("nan")
        model.eval(); tot = 0.0
        for i in range(0, GENERIC_ROWS, 8):
            b = generic_rows[i:i + 8]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                tot += float(model(input_ids=b, labels=b).loss.item()) * len(b)
        return tot / GENERIC_ROWS

    sampler = DocSampler(facts, tok, a.seq_len, a.seed)
    micro = a.batch_size * a.seq_len; accum = max(a.tokens_per_step // micro, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    schedule = eval_schedule(a.steps); curve = []
    gen = torch.Generator(device=device); gen.manual_seed(a.seed + 7)

    def score_all(tag, rec):
        aa, ar = score(model, tok, a_items, a_pools, device, a.eval_batch)
        ba, br = score(model, tok, b_items, b_pools, device, a.eval_batch)
        s = summarise(aa, ar, a_items, "A", "copy"); sb = summarise(ba, br, b_items, "B", "stratum")
        rec[f"{tag}/A_noncopy"] = s["A/noncopy/acc"]; rec[f"{tag}/A_noncopy_rank"] = s.get("A/noncopy/rank")
        rec[f"{tag}/A_copy"] = s["A/copy/acc"]; rec[f"{tag}/B"] = sb["B/ALL/acc"]
        rec[f"{tag}/generic_loss"] = generic_loss()

    @torch.no_grad()
    def evaluate(step):
        rec = {"step": step}
        score_all("actual", rec)
        Wt = {n: p.detach().clone() for n, p in mats}
        spec = {}
        for r in RANKS:
            # patch: matrix by matrix, in place (nothing stored but the top part's norm)
            norms = {}
            for n, p in mats:
                dW = Wt[n] - W0[n]
                top, _, S = lowrank_parts(dW, r, gen)
                norms[n] = float(top.norm())
                p.copy_(Wt[n] - top)
                if r == RANKS[-1]:
                    f2 = float((dW ** 2).sum()); s2 = (S ** 2)
                    spec[n] = {"top1": float(s2[0] / f2) if f2 > 0 else 0.0,
                               "top4": float(s2[:4].sum() / f2) if f2 > 0 else 0.0,
                               "top16": float(s2[:16].sum() / f2) if f2 > 0 else 0.0}
                del dW, top
            score_all(f"patch_r{r}", rec)
            # KEEP: the low-rank part alone -- W(0) + top-r of dW, nothing else (the
            # complement of the patch: the toy's "common part alone")
            for n, p in mats:
                dW = Wt[n] - W0[n]
                top, _, _ = lowrank_parts(dW, r, gen)
                p.copy_(W0[n] + top)
                del dW, top
            score_all(f"keep_r{r}", rec)
            # control: a random rank-r part of the same Frobenius norm, matrix by matrix
            for n, p in mats:
                a_ = torch.randn(p.shape[0], r, generator=gen, device=p.device, dtype=p.dtype)
                b_ = torch.randn(p.shape[1], r, generator=gen, device=p.device, dtype=p.dtype)
                rnd = a_ @ b_.T
                rnd.mul_(norms[n] / float(rnd.norm()) if float(rnd.norm()) > 0 else 0.0)
                p.copy_(Wt[n] - rnd)
                del rnd
            score_all(f"control_r{r}", rec)
            for n, p in mats:
                p.copy_(Wt[n])
        del Wt
        rec["spectrum"] = spec
        curve.append(rec)
        print(f"  [step {step:>4}] A noncopy actual={rec['actual/A_noncopy']:.3f} | patch r=1/4/16: "
              + "/".join(f"{rec[f'patch_r{r}/A_noncopy']:.3f}" for r in RANKS)
              + " | control r=1/4/16: " + "/".join(f"{rec[f'control_r{r}/A_noncopy']:.3f}" for r in RANKS)
              + f" | B actual={rec['actual/B']:.3f} patch r=1/4/16: "
              + "/".join(f"{rec[f'patch_r{r}/B']:.3f}" for r in RANKS)
              + f" | keep r=1/4/16: A " + "/".join(f"{rec[f'keep_r{r}/A_noncopy']:.3f}" for r in RANKS)
              + " B " + "/".join(f"{rec[f'keep_r{r}/B']:.3f}" for r in RANKS)
              + f" | generic loss actual={rec['actual/generic_loss']:.3f} patch r=4 {rec['patch_r4/generic_loss']:.3f} "
              + f"control {rec['control_r4/generic_loss']:.3f}"
              + f" | mean top1 share {sum(v['top1'] for v in spec.values()) / len(spec):.2f}", flush=True)
        model.train()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump({"lr": a.lr, "steps": a.steps, "seed": a.seed, "ranks": RANKS, "curve": curve},
                  open(out_path, "w"), indent=1)

    evaluate(0)
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(step / max(a.warmup, 1), 1.0)
        for _ in range(accum):
            batch = sampler.batch(a.batch_size).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step in schedule:
            evaluate(step)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
