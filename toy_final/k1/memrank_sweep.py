"""Memories per parameter: the effective rank of the write, against individuals vs. values.

PLAN.md section 9.3 asked for this and deferred it: effective rank of W(t) - W(0), read
against B's accuracy, predicted to sit near 1 through the crash and rise toward n_B as B
individuates. An earlier, informal run of a different (K-block) toy found the rank rising
to ~10-16 and SATURATING there rather than climbing to n_B = 50 -- which is exactly what the
vocabulary size predicts, not what the population size predicts, since |Y| = 16 there. This
sweep tests that directly by varying n_B and |V| SEPARATELY, on the same grids as sw_nb.json
and sw_vocab.json, so the two are comparable.

WHY THE SHARED ROW HAS TO BE REMOVED FIRST. dW splits exactly (THEORY.md section 2) into one
rank-1 term along mu (the coherent, individual-blind write) and an individual-only remainder
dW_perp = (I - mu mu^T) dW. Reporting the rank of dW itself would count the shared direction
as "using up capacity" the same way an individuating write does, which conflates the two
channels the rest of this project keeps separate. dW_perp is the part any capacity argument
about INDIVIDUALS should be about.

THE CANDIDATE CEILINGS, made explicit before fitting anything:
  min(n_B, d)               -- one direction per individual, capped by the ambient dimension
  min(n_distinct_B, d)      -- one direction per DISTINCT VALUE actually present in B, capped
                               by d. n_distinct_B <= min(n_B, |Y|) because values are drawn
                               WITH REPLACEMENT (k1.build), so n_B individuals typically cover
                               fewer than n_B distinct values once n_B approaches |Y|.
Effective rank (participation ratio of the singular values, R_PR = (sum s^2)^2 / sum s^4) is
used throughout: it is 1 for a rank-1 matrix and rises smoothly as more directions carry
comparable weight, unlike the discrete matrix rank, which is not meaningful under floating
point noise.
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp

import k1
import sweeps


def eff_rank(M):
    """Participation ratio of M's singular values: 1 for rank-1, rises toward min(M.shape)."""
    s = np.linalg.svd(np.asarray(M), compute_uv=False)
    s2 = s.astype(np.float64) ** 2
    tot = s2.sum()
    return float(tot ** 2 / (s2 ** 2).sum()) if tot > 1e-30 else 0.0


def run_cell(axis, value, seed, match_gate=200, steps=5000, d=128, vocab=32, na=50, nb=50,
            beta=0.5, gain=1.0, u_scale=0.01):
    cfg = dict(d=d, v=vocab, na=na, nd=na, nb=nb)
    if axis == "nb":
        cfg["nb"] = int(value)
    elif axis == "vocab":
        cfg["v"] = int(value)
    else:
        raise ValueError(axis)
    k1.configure(**cfg)

    p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, beta, 1, gain)
    base_ilr = lr / k1.INJECT_RATIO
    ilr, gate, ok = sweeps.match_gate(p0, A, Dd, B, 1, gain, u_scale, base_ilr, seed, match_gate)

    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    n_distinct = int(np.unique(vB).size)
    W0 = np.asarray(p0["W"])

    import numpy.random as npr
    rng = npr.default_rng(seed + 5)
    p = p0
    gi = sorted(set(k1.grid(steps)))
    rows = []
    for t in range(steps + 1):
        if t in gi:
            W = np.asarray(p["W"]); dW = W - W0
            dW_perp = dW - np.outer(mu, mu @ dW)
            rows.append(dict(
                step=t,
                A=k1.accuracy(p, kA, vA, 1, gain), B=k1.accuracy(p, kB, vB, 1, gain),
                rank_full=eff_rank(dW), rank_perp=eff_rank(dW_perp),
                dW_norm=float(np.linalg.norm(dW)), dW_perp_norm=float(np.linalg.norm(dW_perp)),
            ))
        i = rng.integers(0, k1.NB, 32)
        p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, 1, gain)

    return dict(axis=axis, value=value, seed=seed, d=d, vocab=vocab, na=na, nb=nb, beta=beta,
                n_distinct_B=n_distinct, inject_lr=ilr, matched=bool(ok), B_gate=gate,
                pretrain_lr=lr, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=("nb", "vocab"))
    ap.add_argument("--values", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--match_gate", type=int, default=200)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {}
    for value in [float(x) for x in a.values.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            r = run_cell(a.axis, value, seed, a.match_gate, a.steps)
            out[f"{a.axis}{value}-s{seed}"] = r
            last = r["rows"][-1]
            pk = max(r["rows"], key=lambda x: x["rank_perp"])
            print(f"{a.axis}={value:g} seed={seed} | n_distinct_B={r['n_distinct_B']} "
                  f"Bgate={r['B_gate']} | rank_perp end={last['rank_perp']:.2f} "
                  f"peak={pk['rank_perp']:.2f}@{pk['step']} | rank_full end={last['rank_full']:.2f} "
                  f"| A end={last['A']:.2f}", flush=True)
        json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
