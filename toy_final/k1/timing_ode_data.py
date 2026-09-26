"""TASK 3 data: the dose sweep re-run with B's MEAN MARGIN logged, which sw_dose.json does
not carry. Pretraining is identical across doses (same seed/beta/norm), so it is done once."""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json

import numpy as np
import jax
import jax.numpy as jnp

import k1

DOSES = [0.25, 0.5, 1.0, 2.0, 4.0]
STEPS = 6000


def dense(steps):
    return sorted(set(range(0, 1001, 2)) | set(range(1000, 3001, 10)) |
                  set(range(3000, steps + 1, 50)))


def probe(p, kB, vB, kA, vA, norm):
    z, h = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(kB), norm, 1.0)]
    pr = np.asarray(jax.nn.softmax(jnp.asarray(z), -1))
    M = z[np.arange(len(vB)), vB] - (pr * z).sum(-1)
    pY = pr[:, k1.Y].sum(-1)          # mass B puts on its own half
    return (float(M.mean()), float(np.linalg.norm(h, axis=-1).mean()),
            float((z.argmax(-1) == vB).mean()), float(pY.mean()))


def main():
    out = {}
    for seed in (0, 1):
        p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, 0.5, 1, 1.0)
        (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
        W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
        w = U0[:, k1.Y].mean(1) - U0[:, k1.X].mean(1); w /= np.linalg.norm(w)
        for dose in DOSES:
            ilr = lr / k1.INJECT_RATIO * dose
            p = p0
            rng = np.random.default_rng(seed + 5)
            gi = set(dense(STEPS)); rows = []
            for t in range(STEPS + 1):
                if t in gi:
                    M, hn, bacc, pY = probe(p, kB, vB, kA, vA, 1)
                    s = mu @ (np.asarray(p["W"]) - W0)
                    rows.append(dict(step=t, x=float(s @ w), m=M, hn=hn, B=bacc, pY=pY,
                                     A=k1.accuracy(p, kA, vA, 1, 1.0),
                                     s_norm=float(np.linalg.norm(s))))
                i = rng.integers(0, k1.NB, 32)
                p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 0.01, 1, 1.0)
            out[f"dose{dose}-s{seed}"] = dict(dose=dose, seed=seed, ilr=ilr, rows=rows)
            xs = np.array([r["x"] for r in rows])
            print(f"seed {seed} dose {dose}: x peak {xs.max():.3f} @ "
                  f"{rows[int(xs.argmax())]['step']}  m0 {rows[0]['m']:.2f}  "
                  f"x_end {xs[-1]:.3f}", flush=True)
            json.dump(out, open("timing_ode_data.json", "w"))


main()
