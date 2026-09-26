"""Task 2: what pretraining writes into the shared direction, as a function of ballast size.

For each (n_D, seed, arm) we pretrain to the 0.99 gate and measure, BEFORE any injection:
  s0          = mu @ W_pre                 the shared row of the store
  s0_on_X     = <s0, u_bar_X - u_bar_Y>    positive = the shared row is pro-X
  s0_cos      = the same over ||s0||       (the requested normalised version)
  mu_on_X     = <mu, u_bar_X - u_bar_Y>    the OTHER shared route: mu straight down the residual
  A's margin gap (correct minus best wrong) split into mu_row / mu_res / individual
  A's accuracy with the mu row of W deleted, W -> (I - mu mu^T) W
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

k1.configure(d=128, v=32, na=50, nb=50)
out = []
for nd in NDS:
    for norm in (1, 0):
        for seed in SEEDS:
            bk.configure(nd=nd)
            p, lr, gate, mu, (kA, vA), (kD, vD), (kB, vB) = bk.pretrain(seed, BETA, norm, GAIN, nd)
            W = np.asarray(p["W"]); U = np.asarray(p["U"])
            s0 = mu @ W
            wdir = U[:, k1.X].mean(1) - U[:, k1.Y].mean(1)      # pro-X direction in logit space
            wu = wdir / np.linalg.norm(wdir)

            sp = bk.margin_split(p, kA, vA, mu, BETA, GAIN, norm)
            # exactness check of the split against the real forward pass
            z_true = np.asarray(k1.fwd(p, jnp.asarray(kA), norm, GAIN)[0])
            m = z_true.copy(); m[np.arange(len(vA)), vA] = -np.inf
            g_true = z_true[np.arange(len(vA)), vA] - m.max(1)
            split_err = float(np.abs(g_true - sp["total"]).max())

            pz = bk.zero_mu_row(p, mu)
            row = dict(
                nd=nd, norm=norm, seed=seed, pretrain_lr=lr, pretrain_steps=gate,
                s0_norm=float(np.linalg.norm(s0)),
                s0_on_X=float(s0 @ wdir),
                s0_cos=float(s0 @ wu / np.linalg.norm(s0)),
                mu_on_X=float(mu @ wdir),
                wdir_norm=float(np.linalg.norm(wdir)),
                gap_mean=float(sp["total"].mean()),
                gap_mu_row=float(sp["mu_row"].mean()),
                gap_mu_res=float(sp["mu_res"].mean()),
                gap_ind=float(sp["ind"].mean()),
                frac_mu_row=float(sp["mu_row"].mean() / sp["total"].mean()),
                frac_shared=float((sp["mu_row"] + sp["mu_res"]).mean() / sp["total"].mean()),
                frac_mu_row_perind=float(np.median(sp["mu_row"] / sp["total"])),
                A_acc=float(sp["correct"].mean()),
                A_acc_no_mu_row=float(bk.acc(pz, kA, vA, norm, GAIN)),
                A_acc_no_mu_row_ownhalf=float(bk.acc(pz, kA, vA, norm, GAIN, restrict=k1.X)),
                split_err=split_err,
            )
            out.append(row)
            print(f"nd={nd:2d} norm={norm} s{seed} lr={lr} gate@{gate:5d} | "
                  f"||s0|| {row['s0_norm']:.3f} s0.X {row['s0_on_X']:+.3f} "
                  f"cos {row['s0_cos']:+.3f} | mu.X {row['mu_on_X']:+.3f} | "
                  f"gap {row['gap_mean']:.3f} = murow {row['gap_mu_row']:+.3f} "
                  f"+ mures {row['gap_mu_res']:+.3f} + ind {row['gap_ind']:+.3f} | "
                  f"f_murow {row['frac_mu_row']:+.3f} f_shared {row['frac_shared']:+.3f} | "
                  f"A {row['A_acc']:.2f} -> nomurow {row['A_acc_no_mu_row']:.2f} "
                  f"(own {row['A_acc_no_mu_row_ownhalf']:.2f}) | err {split_err:.2e}", flush=True)
json.dump(out, open("ballast_measure.json", "w"))
print("wrote ballast_measure.json")
