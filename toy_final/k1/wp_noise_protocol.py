"""Can the toy's between-seed spread be reduced by protocol, without touching the model?

wp_noise_sources.py showed the spread of the trough (sd 0.18 across model seeds at n_A = 50)
is not minibatch noise (sd 0.012) and is larger than 50 independent threshold crossings could
give (max binomial sd 0.07), so it is a shared per-instance factor. Two protocol candidates:

  (1) PRETRAINING STOPS AT THE GATE. k1.pretrain returns at the first 100-step check where A
      and D are both >= 0.99, i.e. after 100 to 200 steps, with margins wherever they were at
      that moment. Here: pretrain a FIXED number of steps past the gate instead.
  (2) INJECTION AT A FIXED RATE. B's clock then varies 2.4x across seeds (gate 150 to 359),
      so the trough time varies 40 to 200 and averaged curves smear. Here: gate-match.

Arms, 8 seeds each: baseline (gate-stop pretrain, fixed rate) / fixed-length pretrain only /
gate-matched injection only / both. Reports the between-seed sd of the trough, the trough
step, and the end level, per arm.

    python wp_noise_protocol.py --seeds 0-7 --pre_steps 2000 --out wp_noise_protocol.json
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp

import k1
import sweeps


def pretrain_fixed(seed, beta, norm, gain, lr, n_steps):
    """Same as k1.pretrain but runs exactly n_steps at a given lr; asserts the gate is met."""
    mu, (kA, vA), (kD, vD), (kB, vB) = k1.build(1000 + seed, beta)
    kp = np.concatenate([kA, kD]); vp = np.concatenate([vA, vD])
    rng = np.random.default_rng(seed)
    p = k1.init(seed)
    for t in range(n_steps):
        i = rng.integers(0, len(kp), 32)
        p = k1.step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0, norm, gain)
    ok = min(k1.accuracy(p, kA, vA, norm, gain), k1.accuracy(p, kD, vD, norm, gain)) >= k1.GATE
    return p, ok, mu, (kA, vA), (kD, vD), (kB, vB)


def summ(rows):
    A = np.array([r["A"] for r in rows]); st = np.array([r["step"] for r in rows])
    B = np.array([r["B"] for r in rows])
    tr = int(A.argmin()); bg = int(st[np.argmax(B >= 0.99)]) if (B >= 0.99).any() else -1
    return dict(trough=float(A[tr]), trough_step=int(st[tr]), Bgate=bg, end=float(A[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--pre_steps", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--match", type=int, default=200)
    ap.add_argument("--out", default="wp_noise_protocol.json")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    k1.configure(d=128, v=32, na=50, nd=50, nb=50)
    beta, gain, u = 0.5, 1.0, 0.01

    out = {"gate-stop / fixed-rate": [], "gate-stop / matched": [],
           "fixed-length / fixed-rate": [], "fixed-length / matched": []}
    for seed in range(lo, hi + 1):
        # gate-stop pretraining (k1.pretrain), as everywhere else
        p0, lr, t_pre, mu, A, Dd, B = k1.pretrain(seed, beta, 1, gain)
        base = lr / k1.INJECT_RATIO
        rows, _ = k1.inject(p0, lr, mu, A, Dd, B, 1, gain, u, a.steps, seed)
        out["gate-stop / fixed-rate"].append(dict(seed=seed, t_pre=int(t_pre), **summ(rows)))
        ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, 1, gain, u, base, seed, a.match)
        rows, _ = k1.inject(p0, lr, mu, A, Dd, B, 1, gain, u, a.steps, seed, ilr=ilr)
        out["gate-stop / matched"].append(dict(seed=seed, matched=bool(ok), **summ(rows)))

        # fixed-length pretraining at the same lr
        p1, okp, mu, A, Dd, B = pretrain_fixed(seed, beta, 1, gain, lr, a.pre_steps)
        rows, _ = k1.inject(p1, lr, mu, A, Dd, B, 1, gain, u, a.steps, seed)
        out["fixed-length / fixed-rate"].append(dict(seed=seed, pre_ok=bool(okp), **summ(rows)))
        ilr, g, ok = sweeps.match_gate(p1, A, Dd, B, 1, gain, u, base, seed, a.match)
        rows, _ = k1.inject(p1, lr, mu, A, Dd, B, 1, gain, u, a.steps, seed, ilr=ilr)
        out["fixed-length / matched"].append(dict(seed=seed, matched=bool(ok), **summ(rows)))
        print(f"seed {seed}: " + " | ".join(f"{k.split(' / ')[0][:5]}/{k.split(' / ')[1][:5]} "
              f"{out[k][-1]['trough']:.2f}@{out[k][-1]['trough_step']}" for k in out), flush=True)
    json.dump(out, open(a.out, "w"))

    print(f"\n{'arm':>28} {'trough':>14} {'trough step':>14} {'B gate':>14} {'end':>14}")
    for k, v in out.items():
        x = {q: np.array([r[q] for r in v], float) for q in ("trough", "trough_step", "Bgate", "end")}
        print(f"{k:>28} " + " ".join(f"{x[q].mean():>7.3f}±{x[q].std(ddof=1):<6.3f}" for q in x))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
