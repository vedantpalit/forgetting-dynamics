"""Replay schedules in the attention-only Transformer of Figure 1.

A wrapper around src/experiments/mlpfree_dose_injection.py, which is unchanged: same pretrained checkpoint, optimizer,
learning rate, data and B' stream. A fixed random subset R of n_rep old individuals of A is replayed; R is removed from
A's evaluation set, so `dataA` in the logs is the never-replayed old facts in every arm, including "none".

Replayed rows are ADDED to the full B' batch, and on those steps the loss keeps the B' rows' token count as its
denominator, so at equal parameters the gradient from the new facts is the same in every arm and replay adds to it (as
in the toy, where the loss is a sum over a fixed count). Steps without replay are the unmodified step. The optimizer and gradient clip
still act on the sum, so the new facts' speed can change: report the step at which B is learned in every arm. Under Adam
rows are not update size, and from about step 200 even a few rows outweigh the new facts' gradient, so budgets compare
rows, not doses.

--source C draws the rows from fresh people of the never-trained population instead of R: their answers are in half X
like A's, but they are not old facts, and nearly every answer of A also occurs in R. This separates rehearsing facts
from pushing the old answers' region back up.

Eval sets: dataA = never-replayed old facts; dataC = the replayed facts R (the held-out population, unused here, is
swapped out; R lies in half X like the default C, so own-region metrics come for free). An arm whose R did not hold up
says nothing about replay.

    every      the budget spread evenly over all steps
    int<k>     a burst every k steps, offset by k/2
    early<w>   all of it before step w
    late<w>    all of it after step w
    win<a>_<b> all of it in steps a..b (same rate as early<b-a>, a different window: rate and timing come apart)
    none       no replay, scored on the same A \\ R

Run from the repository root:
  python -m src.experiments.mlpfree_replay_injection --arm every --budget 9600 --total 1200 --inject_seed 0 \
      --lr_mult 1 --inject_batch 256
"""
import argparse
import copy
import importlib
import os
import sys

import numpy as np
import jax.numpy as jnp

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--arm", required=True)
ap.add_argument("--budget", type=int, default=9600)
ap.add_argument("--n_rep", type=int, default=250)
ap.add_argument("--total", type=int, default=1200)
ap.add_argument("--source", choices=["A", "C"], default="A")
known, rest = ap.parse_known_args()


def schedule(arm, budget, steps):
    """Replayed rows at each step; sums exactly to the budget."""
    w = np.zeros(steps)
    if arm == "none":
        return w.astype(int)
    if arm == "every":
        w[:] = 1.0
    elif arm.startswith("int"):
        k = int(arm[3:]); w[k // 2::k] = 1.0
    elif arm.startswith("early"):
        w[:int(arm[5:])] = 1.0
    elif arm.startswith("late"):
        w[int(arm[4:]):] = 1.0
    elif arm.startswith("win"):
        lo, hi = arm[3:].split("_"); w[int(lo):int(hi)] = 1.0
    else:
        raise SystemExit(arm)
    c = np.floor(np.cumsum(w / w.sum()) * budget + 1e-9).astype(int)
    n = np.diff(np.r_[0, c]); n[np.flatnonzero(w)[-1]] += budget - n.sum()
    assert n.sum() == budget and (n >= 0).all()
    return n


base = importlib.import_module("src.experiments.mlpfree_dose_injection")
from src.data.biography import BiographyDataset
from src.data.metrics import compute_cross_entropy
sched = schedule(known.arm, known.budget, known.total)
sys.argv = [sys.argv[0]] + rest + ["--inject.eval_interval", "5", "--inject.total_steps", str(known.total),
                                   "--inject.checkpoint_steps", f"0,20,40,60,80,100,140,200,400,800,{known.total}"]
_build, _key = base.build, base.experiment_key
STATE = {"t": 0, "rows": 0, "batch": None, "stream": None, "tag": None}
TAG = f"{known.arm}-{known.budget}-{known.n_rep}-{known.source}"


def build_replay(cfg, max_eval):
    out = list(_build(cfg, max_eval))
    data_a, data_b = out[4], out[6]
    STATE["batch"] = cfg.inject.batch_size
    seed = cfg.seed + cfg.inject.seed
    R = np.random.default_rng(seed + 77).choice(data_a.person_ids, known.n_rep, replace=False)

    keep = ~np.isin(data_a._eval_ids, R)
    assert len(data_a._eval_ids) == len(data_a.person_ids) and (~keep).sum() == known.n_rep, "A's eval set is capped"
    pool = data_a if known.source == "A" else out[7]
    ids = R if known.source == "A" else np.random.default_rng(seed + 78).choice(pool.person_ids, known.n_rep, replace=False)
    stream = copy.copy(pool)                             # a second stream, over the replayed individuals only
    stream.person_ids = ids
    stream._np_rng = np.random.default_rng(seed + 1005)
    stream._cache, stream._cache_idx = None, 0
    stream.exposure_counts = np.zeros_like(pool.exposure_counts)
    rep = copy.copy(data_a); rep.name = "dataC"          # evaluation of R, whatever the rows are
    for d, m in ((rep, ~keep), (data_a, keep)):          # dataA: never replayed; dataC: replayed
        for f in ("_eval_ids", "eval_inputs", "eval_targets", "eval_mask", "_eval_langs"):
            setattr(d, f, getattr(data_a, f)[m])
    out[7] = rep

    orig, width = data_b.get_batch, int(sched.max())

    def get_batch(rng, batch_size):
        b = orig(rng, batch_size)
        r = int(sched[STATE["t"]]) if STATE["t"] < len(sched) else 0
        STATE["t"] += 1
        if not r:
            return b
        assert batch_size == STATE["batch"], (batch_size, STATE["batch"])
        x = stream.get_batch(rng, r); STATE["rows"] += r
        pad = {k: jnp.zeros((width - r,) + b[k].shape[1:], b[k].dtype) for k in b}   # mask 0: no loss; one compiled shape
        return {k: jnp.concatenate([b[k], x[k], pad[k]]) for k in b}

    data_b.get_batch = get_batch
    STATE["stream"], STATE["tag"] = stream, f"{TAG}-s{cfg.inject.seed}"
    print(f"replay: arm={known.arm} budget={int(sched.sum())} max rows/step={width} steps with replay={int((sched > 0).sum())} "
          f"source={known.source} | R {len(rep._eval_ids)} (dataC), never replayed {len(data_a._eval_ids)} (dataA)", flush=True)
    return tuple(out)


def loss_fn(logits, targets, mask):
    """On a replay step, summed loss over the B' rows' token count; otherwise the original loss."""
    m = (mask != 0).astype(jnp.float32)
    ce = compute_cross_entropy(logits, targets, m)
    if logits.shape[0] == STATE["batch"]:
        return ce
    return ce * jnp.sum(m) / jnp.sum(m[:STATE["batch"]])


def key_replay(*args, **kw):
    return _key(*args, **kw, replay=TAG)


base.build, base.experiment_key = build_replay, key_replay
BiographyDataset.loss_fn = staticmethod(loss_fn)
print(f"replay_injection: argv={sys.argv[1:]}", flush=True)
base.main()
st = STATE["stream"]; c = st.exposure_counts[st.person_ids]
print(f"replay done: {STATE['t']} steps, {STATE['rows']} rows; per replayed person min {c.min()} mean {c.mean():.1f} max {c.max()}", flush=True)
out_dir = "replay_exposures"; os.makedirs(out_dir, exist_ok=True)
np.savez(os.path.join(out_dir, f"{STATE['tag']}.npz"), ids=st.person_ids, counts=c, sched=sched)
assert STATE["t"] == known.total and STATE["rows"] == sched.sum()
