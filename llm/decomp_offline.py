"""Offline decomposition of OLMo's old-fact displacement at every grain, from the state dumps
of llm/inject_decomp.py (decomp_*_states.npz).

THE QUESTION. Seed 0 showed the reversible part of OLMo's crash is not one vector over all
old facts; seed 1 (v2) showed one vector PER RELATION carries most of it and a reversible
residual remains. At what grain is the change "common"? Groupings tested, coarse to fine:

    all          one mean over the stratum
    relation     one mean per CounterFact relation            (>= MIN_GROUP facts)
    frame        one mean per (relation, prompt frame)         (>= MIN_GROUP facts)
    kmeans-K     K cluster means of the change vectors, K in KMEANS_K (fit on the change
                 at the trough, then applied at every step: the same partition throughout)

For each grouping and step: accuracy of (x0 + group mean of dx) -- "common alone" -- and of
(x0 + dx - group mean) -- "residual alone" -- read through the PRETRAINED unembedding U(0),
so the head's own drift is out of the picture (the online run read through U(t)). Also the
patch curve at the trough: subtract the group mean estimated from k random facts per group.

Runs on a login node (CPU, no GPU needed; loads the pretrained model once for U(0) and the
tokenizer/dataset for the items and pools):

  .venv-llm/bin/python -m llm.decomp_offline --states llm/out/decomp_lr1e-05_seed1_states.npz
"""
import argparse
import json
import math
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.evalsets import build_a
from llm.inject import DATASET_ID, MODEL_ID

MIN_GROUP = 15
KMEANS_K = (2, 4, 8, 16, 32)
PATCH_K = (1, 4, 16)
PATCH_REPS = 3
SUBSPACE_R = (1, 4, 16, 64, 200)     # rank of the shared subspace removed (fit on half the facts)


def frame_key(it):
    return it["relation_id"] + "|" + re.sub(re.escape(it["subject"]), "{}", it["prompt"], count=1)


def kmeans(X, K, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                C[k] = X[lab == k].mean(0)
    return lab


def read(U, x, items, pools):
    # torch matmul: multi-threaded BLAS; numpy's default build on the cluster is not
    logits = (torch.from_numpy(np.ascontiguousarray(x)) @ U_T).numpy()
    acc, rank = [], []
    top = logits.argmax(1)
    for j, it in enumerate(items):
        acc.append(1.0 if int(top[j]) == it["first_token"] else 0.0)
        if not it["rank_eligible"]:
            rank.append(float("nan")); continue
        ids = pools[it["relation_id"]]
        mine = logits[j, it["first_token"]]; cand = logits[j, ids]
        rank.append(1.0 + int((cand > mine).sum()) + 0.5 * max(int((cand == mine).sum()) - 1, 0))
    rs = [r for r in rank if not math.isnan(r)]
    return float(np.mean(acc)), (float(np.mean(rs)) if rs else float("nan"))


def group_means(dx, labels, min_group=1):
    out = np.zeros_like(dx); covered = np.zeros(len(dx), bool)
    for g in set(labels):
        gi = np.where(labels == g)[0]
        if len(gi) >= min_group:
            out[gi] = dx[gi].mean(0); covered[gi] = True
    return out, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--stratum", default="noncopy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or a.states.replace("_states.npz", "_offline.json")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    z = np.load(a.states, allow_pickle=True)
    steps = z["steps"]; X = z["x"].astype(np.float32); x0 = z["x0"].astype(np.float32)
    print(f"states: {len(steps)} evals (to step {steps[-1]}), {X.shape[1]} facts, d={X.shape[2]}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    U0 = model.lm_head.weight.detach().numpy().astype(np.float32); del model
    global U_T
    U_T = torch.from_numpy(np.ascontiguousarray(U0.T))
    print("pretrained unembedding loaded", flush=True)
    items, pools, _ = build_a(tok, a.gate, load_dataset(DATASET_ID, split="train"))
    print("items and pools built", flush=True)
    assert len(items) == X.shape[1], (len(items), X.shape)
    idx = np.array([k for k, it in enumerate(items) if it["copy"] == a.stratum])
    sub = [items[k] for k in idx]; x0s = x0[idx]
    rel = np.array([it["relation_id"] for it in sub]); frm = np.array([frame_key(it) for it in sub])
    print(f"{a.stratum}: {len(sub)} facts, {len(set(rel))} relations, {len(set(frm))} frames "
          f"({sum(1 for g in set(frm) if (frm == g).sum() >= MIN_GROUP)} with >= {MIN_GROUP} facts)")

    # the trough, for the k-means partition and the patch curve: min of the actual curve
    actual = [read(U0, X[t][idx], sub, pools)[0] for t in range(len(steps))]
    t_tr = int(np.argmin(actual))
    print("actual curve through U(0): " + " ".join(f"{int(s)}:{v:.2f}" for s, v in zip(steps, actual)), flush=True)
    dx_tr = X[t_tr][idx] - x0s
    groupings = {"all": np.zeros(len(sub), int), "relation": rel, "frame": frm}
    for K in KMEANS_K:
        groupings[f"kmeans-{K}"] = kmeans(dx_tr, K)
    print(f"trough at step {steps[t_tr]} (actual {actual[t_tr]:.3f} through U(0))")

    # fixed draws of k facts per group, reused at every step: the patch as a time course
    prng = np.random.default_rng(1)
    draws = {}
    for name in ("all", "relation", "frame"):
        lab = groupings[name]; draws[name] = {}
        for k in PATCH_K:
            reps = []
            for rep in range(PATCH_REPS):
                est_idx = {}
                for g in set(lab):
                    gi = np.where(lab == g)[0]
                    if len(gi) >= max(MIN_GROUP if name != "all" else 1, k + 1):
                        est_idx[g] = prng.choice(gi, k, replace=False)
                reps.append(est_idx)
            draws[name][k] = reps

    def patch_curve(dx, name, k):
        lab = groupings[name]; accs = []
        for est_idx in draws[name][k]:
            est = np.zeros_like(dx); used = np.zeros(len(sub), bool)
            for g, pick in est_idx.items():
                gi = np.where(lab == g)[0]; est[gi] = dx[pick].mean(0); used[pick] = True
            tgt = (est != 0).any(1) & ~used
            if tgt.any():
                accs.append(read(U0, (x0s + dx - est)[tgt], [it for it, c in zip(sub, tgt) if c], pools)[0])
        return float(np.mean(accs)) if accs else float("nan")

    # SUBSPACE patch: the top-r principal directions of the change, fitted on one half of
    # the facts (at every step, on that step's change), removed from each fact of the other
    # half -- each fact loses ITS OWN component along the shared directions, not a mean.
    # Two halves, swapped; the fitting half never reads its own directions.
    hrng = np.random.default_rng(2)
    halves = []
    for rep in range(2):
        perm = hrng.permutation(len(sub)); halves.append((perm[: len(sub) // 2], perm[len(sub) // 2:]))

    def subspace_patch(dx, r):
        accs = []
        for fit, apply in halves:
            for A_, B_ in ((fit, apply), (apply, fit)):
                D = dx[A_] - dx[A_].mean(0)
                _, _, Vt = np.linalg.svd(D, full_matrices=False)
                P = Vt[:r]                                          # (r, d)
                comp = (dx[B_] @ P.T) @ P                            # each fact's own component
                mean_part = dx[A_].mean(0)                           # plus the fitted mean
                x = x0s[B_] + dx[B_] - comp - mean_part
                accs.append(read(U0, x, [sub[i] for i in B_], pools)[0])
        return float(np.mean(accs))

    curve = []
    for t, s in enumerate(steps):
        dx = X[t][idx] - x0s
        rec = {"step": int(s), "actual": actual[t], "actual_rank": read(U0, X[t][idx], sub, pools)[1]}
        for name in ("all", "relation", "frame"):
            for k in PATCH_K:
                rec[f"patch/{name}/k{k}"] = patch_curve(dx, name, k)
        for r in SUBSPACE_R:
            rec[f"subspace/r{r}"] = subspace_patch(dx, r) if t > 0 else actual[t]
        for name, lab in groupings.items():
            mg = MIN_GROUP if name in ("relation", "frame") else 1
            dmean, cov = group_means(dx, lab, mg)
            for part, x in (("common", x0s + dmean), ("residual", x0s + dx - dmean)):
                acc, rk = read(U0, x[cov], [it for it, c in zip(sub, cov) if c], pools)
                rec[f"{name}/{part}/acc"] = acc; rec[f"{name}/{part}/rank"] = rk
            rec[f"{name}/covered"] = int(cov.sum())
            rec[f"{name}/delta_rms"] = float(np.sqrt((dmean[cov] ** 2).sum(1).mean())) if cov.any() else float("nan")
            rec[f"{name}/eps_rms"] = float(np.sqrt(((dx - dmean)[cov] ** 2).sum(1).mean())) if cov.any() else float("nan")
        curve.append(rec)
        print(f"  [step {s:>4}] actual {rec['actual']:.3f} | common alone: " +
              " ".join(f"{n}={rec[f'{n}/common/acc']:.2f}" for n in groupings) +
              " | residual alone: " + " ".join(f"{n}={rec[f'{n}/residual/acc']:.2f}" for n in groupings) +
              " | patch k=4: " + " ".join(f"{n}={rec[f'patch/{n}/k4']:.2f}" for n in ("all", "relation", "frame")) +
              " | subspace r: " + " ".join(f"{r}={rec[f'subspace/r{r}']:.2f}" for r in SUBSPACE_R), flush=True)

    # patch at the trough: subtract the group mean estimated from k random facts per group
    rng = np.random.default_rng(0); patch = {}
    for name in ("all", "relation", "frame"):
        lab = groupings[name]; patch[name] = {}
        for k in PATCH_K:
            accs = []
            for rep in range(PATCH_REPS):
                est = np.zeros_like(dx_tr); used = np.zeros(len(sub), bool)
                for g in set(lab):
                    gi = np.where(lab == g)[0]
                    if len(gi) < max(MIN_GROUP if name != "all" else 1, k + 1):
                        continue
                    pick = rng.choice(gi, k, replace=False); est[gi] = dx_tr[pick].mean(0); used[pick] = True
                    used_g = np.zeros(len(sub), bool); used_g[gi] = True
                tgt = (est != 0).any(1) & ~used
                if tgt.any():
                    accs.append(read(U0, (x0s + dx_tr - est)[tgt], [it for it, c in zip(sub, tgt) if c], pools)[0])
            patch[name][k] = float(np.mean(accs)) if accs else float("nan")
        print(f"patch at the trough, grouping {name}: " + " ".join(f"k={k}: {patch[name][k]:.3f}" for k in PATCH_K))

    json.dump({"states": a.states, "stratum": a.stratum, "trough_step": int(steps[t_tr]),
               "curve": curve, "patch": patch,
               "groups": {n: int(len(set(l.tolist()))) for n, l in groupings.items()}}, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
