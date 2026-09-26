"""Hyperparameter sweeps over the K=1 toy, with B's clock matched across columns.

WHY GATE MATCHING. Every axis here (d, n_B, load, vocabulary) changes how fast B is learned
at a fixed nominal learning rate. Since the crash is caused by B's acquisition, a column that
learns B faster gets a different dose, not just a different geometry, and the crash depths
are then not comparable. So for every configuration we bisect the injection learning rate
until B reaches 0.99 at a fixed target step. The dose sweep is the deliberate exception: it
varies the rate on purpose, to test whether the trough collapses on B's accuracy axis.

AXES
  d       key/residual dimension          (the fix to the un-matched sweep)
  nb      number of injected individuals  (tests the n_B^2 vs n_B amplification)
  load    n_A = n_D, at fixed n_B         (capacity; the scaling-law axis)
  vocab   |V|
  beta    shared key fraction
  dose    injection-rate multiplier       (NOT gate matched, by design)

Each cell: pretrain once (learning-rate probe), bisect, then one full injection run.
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp

import k1


def gate_of(p0, A, Dd, B, norm, gain, u_scale, ilr, seed, cap=4000, every=5):
    """Steps until B first reaches the gate, or `cap` if it never does."""
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    p = p0
    rng = np.random.default_rng(seed + 5)
    for t in range(cap + 1):
        if t % every == 0 and k1.accuracy(p, kB, vB, norm, gain) >= k1.GATE:
            return t
        i = rng.integers(0, k1.NB, 32)
        p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm, gain)
    return cap


def match_gate(p0, A, Dd, B, norm, gain, u_scale, base_ilr, seed, target, iters=11, tol=0.15):
    """Bisect the injection rate in log space so B's gate lands on `target`.

    The gate is monotone decreasing in the rate, so this is a plain bisection on log2 of a
    multiplier. Returns (rate, achieved gate, matched?)."""
    lo, hi = -6.0, 6.0                      # log2 multiplier bracket
    best = (base_ilr, gate_of(p0, A, Dd, B, norm, gain, u_scale, base_ilr, seed), False)
    if abs(best[1] - target) <= tol * target:
        return best[0], best[1], True
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ilr = base_ilr * 2.0 ** mid
        g = gate_of(p0, A, Dd, B, norm, gain, u_scale, ilr, seed)
        if abs(g - target) < abs(best[1] - target):
            best = (ilr, g, abs(g - target) <= tol * target)
        if best[2]:
            break
        if g > target:                      # too slow -> raise the rate
            lo = mid
        else:
            hi = mid
    return best


AXES = {"d": "d", "nb": "n_B", "load": "n_A = n_D", "vocab": "|V|",
        "beta": "beta", "dose": "injection-rate multiplier"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, choices=sorted(AXES))
    ap.add_argument("--values", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--na", type=int, default=50)
    ap.add_argument("--nb", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--u_scale", type=float, default=0.01)
    ap.add_argument("--norms", default="1,0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--match_gate", type=int, default=200, help="0 disables matching")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vals = [float(x) for x in a.values.split(",")]
    out = {}
    for val in vals:
        cfg = dict(d=a.d, v=a.vocab, na=a.na, nd=a.na, nb=a.nb)
        beta, dose = a.beta, 1.0
        if a.sweep == "d":
            cfg["d"] = int(val)
        elif a.sweep == "nb":
            cfg["nb"] = int(val)
        elif a.sweep == "load":
            cfg["na"] = cfg["nd"] = int(val)
        elif a.sweep == "vocab":
            cfg["v"] = int(val)
        elif a.sweep == "beta":
            beta = val
        elif a.sweep == "dose":
            dose = val
        k1.configure(**cfg)

        for norm in [int(x) for x in a.norms.split(",")]:
            for seed in [int(x) for x in a.seeds.split(",")]:
                try:
                    p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, beta, norm, a.gain)
                except SystemExit as e:
                    print(f"{a.sweep}={val} norm={norm} seed={seed}  PRETRAIN FAILED: {e}",
                          flush=True)
                    continue
                base = lr / k1.INJECT_RATIO
                if a.match_gate:
                    ilr, g, ok = match_gate(p0, A, Dd, B, norm, a.gain, a.u_scale, base,
                                            seed, a.match_gate)
                else:
                    ilr, g, ok = base * dose, -1, True
                rows, _ = k1.inject(p0, lr, mu, A, Dd, B, norm, a.gain, a.u_scale,
                                    a.steps, seed, ilr=ilr)
                r = dict(sweep=a.sweep, value=val, norm=norm, seed=seed, beta=beta,
                         d=k1.D, v=k1.V, na=k1.NA, nd=k1.ND, nb=k1.NB, gain=a.gain,
                         u_scale=a.u_scale, pretrain_lr=lr, pretrain_gate=pg,
                         inject_lr=ilr, matched=bool(ok), probe_gate=g, dose=dose, rows=rows)
                q = k1.summarise(r)
                out[f"{a.sweep}{val}-n{norm}-s{seed}"] = r
                flag = "" if ok else "  [GATE NOT MATCHED]"
                print(f"{a.sweep}={val:g} norm={norm} seed={seed} | ilr {ilr:.2e} "
                      f"Bgate {q['B_gate']} | A trough {q['trough']:.2f}@{q['trough_step']} "
                      f"-> after {q['after']:.2f} (rec {q['recovery']:+.2f}) end {q['end']:.2f} "
                      f"| s.w {q['sw_peak']:.3f}->{q['sw_end']:.3f} "
                      f"({-100 * q['sw_drop']:+.0f}%) | eps {q['eps_end']:.2f} "
                      f"| a {q['a_peak']:.2f} | Aown {q['A_own_end']:.2f}{flag}", flush=True)
        json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out}  ({len(out)} runs)")


if __name__ == "__main__":
    main()
