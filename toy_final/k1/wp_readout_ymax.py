"""The residual on H1: at n_D = 0 with a FREE store, A recovers while the readout bias keeps
growing.  What recovers, if not the bias?

A fails when its correct X logit is beaten by the BEST Y logit, not by the MEAN one, so the
crash is set by

    z_a[y_a]  -  max_{v in Y} z_a[v]   =   [z_a[y_a] - mean_Y z_a]  -  [max_Y z_a - mean_Y z_a]
                                            ------- the bias ------     --- the Y spread ---

This logs both halves at every checkpoint, for one free-store cell and one throttled-store
cell, so the recovery can be attributed to one of them.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp

import k1
import ballast_k1 as bk
import wp_readout_lib as wl

k1.configure(d=128, v=32, na=50, nb=50)
BETA, GAIN, NORM = 0.5, 1, 1.0


def probe(nd, us, ws, seed=0, steps=5000):
    bk.configure(nd=nd)
    p0, lr, pg, mu, A, Dd, B = bk.pretrain(seed, BETA, NORM, 1.0, nd)
    ilr, g, ok = wl.match_gate2(p0, B, NORM, 1.0, us, ws, lr / k1.INJECT_RATIO, seed, 200)
    (kA, vA) = A; (kB, vB) = B
    p = p0
    rng = np.random.default_rng(seed + 5)
    gi = set(k1.grid(steps)); out = []
    i_ = np.arange(len(vA))
    for t in range(steps + 1):
        if t in gi:
            z = np.asarray(k1.fwd(p, jnp.asarray(kA), NORM, 1.0)[0])
            zy = z[:, k1.Y]
            out.append((t,
                        float((z.argmax(1) == vA).mean()),
                        float((z[i_, vA] - zy.mean(1)).mean()),      # correct minus MEAN of Y
                        float((zy.max(1) - zy.mean(1)).mean()),      # the Y spread
                        float((z[i_, vA] - zy.max(1)).mean()),       # the margin that decides
                        float(np.asarray(k1.accuracy(p, kB, vB, NORM, 1.0)))))
        i = rng.integers(0, k1.NB, 32)
        p = wl.step2(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, us, ws, NORM, 1.0)
    return np.array(out)


for nd, us, ws in ((0, 0.3, 1.0), (0, 0.3, 0.01), (50, 1.0, 1.0), (50, 1.0, 0.01)):
    r = probe(nd, us, ws)
    tr = int(r[:, 1].argmin())
    print(f"nD={nd} u={us} w={ws}")
    print(f"  {'step':>6} {'A':>5} {'B':>5} {'corr-meanY':>11} {'Yspread':>9} {'margin':>8}")
    for j in (0, tr, len(r) // 2, len(r) - 1):
        tag = " <- trough" if j == tr else ""
        print(f"  {r[j,0]:>6.0f} {r[j,1]:>5.2f} {r[j,5]:>5.2f} {r[j,2]:>11.2f} "
              f"{r[j,3]:>9.2f} {r[j,4]:>8.2f}{tag}")
    print(f"  trough -> end:  correct-minus-meanY {r[-1,2]-r[tr,2]:+.2f}, "
          f"Y spread {r[-1,3]-r[tr,3]:+.2f}, deciding margin {r[-1,4]-r[tr,4]:+.2f}, "
          f"A {r[-1,1]-r[tr,1]:+.2f}")
