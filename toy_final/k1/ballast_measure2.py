"""Task 2, corrected: the margin split against the OTHER half.

`ballast_measure.py` split the gap to the best wrong class over the whole vocabulary. At
100% pretrained accuracy the best wrong class is almost always in A's OWN half, and the
shared row is a per-class bias common to every individual, so the argmax over competitors
selects whichever class the bias favours: the shared route then scores NEGATIVE on average
by construction. That is a selection artefact, not a fact about the shared row.

The margin the injection actually eats is the CROSS-HALF one -- eq (6) of THEORY.md: a fails
when the Y-ward bias gap exceeds a's margin over the best class in the other half. So split

    m_cross(a) = z[a, y_a] - max_{v in Y} z[a, v]

into mu_row / mu_res / individual, exactly as before. Same-half margin reported alongside.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import numpy as np
import jax.numpy as jnp

import k1
import ballast_k1 as bk

BETA, GAIN = 0.5, 1.0
NDS = [0, 5, 12, 25, 50]
SEEDS = [0, 1, 2]


def split_vs(zparts, vA, pool):
    """Gap to the best class in `pool`, decomposed over the three routes."""
    z = sum(zparts.values())
    m = np.full(k1.V, -np.inf); m[pool] = 0.0
    zz = z + m[None, :]
    i = np.arange(len(vA))
    zz[i, vA] = -np.inf                       # exclude the correct class if it is in the pool
    j = zz.argmax(1)
    return {k: v[i, vA] - v[i, j] for k, v in zparts.items()}


k1.configure(d=128, v=32, na=50, nb=50)
out = []
for nd in NDS:
    for norm in (1, 0):
        for seed in SEEDS:
            bk.configure(nd=nd)
            p, lr, gate, mu, (kA, vA), (kD, vD), (kB, vB) = bk.pretrain(seed, BETA, norm, GAIN, nd)
            W = np.asarray(p["W"]); U = np.asarray(p["U"])
            c = np.sqrt(k1.D) / np.linalg.norm(kA, axis=1)
            g = (kA - np.sqrt(BETA) * mu[None]) / np.sqrt(1.0 - BETA)
            scale = np.ones(len(kA))
            if norm:
                h = kA + GAIN * (c[:, None] * kA) @ W
                scale = np.sqrt(k1.D) / np.linalg.norm(h, axis=1)
            parts = dict(
                mu_row=scale[:, None] * np.outer(np.sqrt(BETA) * GAIN * c, (mu @ W) @ U),
                mu_res=scale[:, None] * np.tile(np.sqrt(BETA) * (mu @ U), (len(kA), 1)),
                ind=scale[:, None] * (np.sqrt(1 - BETA) * ((g + GAIN * c[:, None] * (g @ W)) @ U)),
            )
            cr = split_vs(parts, vA, k1.Y)     # cross-half: the margin the Y-ward bias eats
            ow = split_vs(parts, vA, k1.X)     # own-half
            tot_c = cr["mu_row"] + cr["mu_res"] + cr["ind"]
            tot_o = ow["mu_row"] + ow["mu_res"] + ow["ind"]
            row = dict(nd=nd, norm=norm, seed=seed,
                       mc=float(tot_c.mean()), mc_murow=float(cr["mu_row"].mean()),
                       mc_mures=float(cr["mu_res"].mean()), mc_ind=float(cr["ind"].mean()),
                       f_murow=float(cr["mu_row"].mean() / tot_c.mean()),
                       f_shared=float((cr["mu_row"] + cr["mu_res"]).mean() / tot_c.mean()),
                       mc_min=float(tot_c.min()), mc_p10=float(np.percentile(tot_c, 10)),
                       mo=float(tot_o.mean()), mo_murow=float(ow["mu_row"].mean()),
                       mo_ind=float(ow["ind"].mean()),
                       mo_min=float(tot_o.min()))
            out.append(row)
            print(f"nd={nd:2d} norm={norm} s{seed} | m_cross {row['mc']:6.2f} = murow "
                  f"{row['mc_murow']:+6.2f} + mures {row['mc_mures']:+5.2f} + ind "
                  f"{row['mc_ind']:+6.2f} | f_murow {row['f_murow']:+.3f} f_shared "
                  f"{row['f_shared']:+.3f} | mc_min {row['mc_min']:+5.2f} p10 {row['mc_p10']:+5.2f}"
                  f" | m_own {row['mo']:5.2f} (murow {row['mo_murow']:+5.2f} ind "
                  f"{row['mo_ind']:+5.2f}, min {row['mo_min']:+.2f})", flush=True)
json.dump(out, open("ballast_measure2.json", "w"))
print("wrote ballast_measure2.json")
