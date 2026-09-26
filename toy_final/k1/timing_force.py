"""TASK 2c: measure the FORCE, not just the time.

(D1) predicts dM/dt = eta g^2 d R with R = (1-b)(1-1/V)/n_B + b/V in the linear arm, and
the same times d/||h||^2 in the normalized arm. Both are directly measurable: run the
injection, log B's mean margin per step, and fit its early slope. This turns the single
fitted constant M* into a measured quantity and exposes the ||h||^2 factor.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp

import k1


def margins(p, kB, vB, norm, gain):
    z, h = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(kB), norm, gain)]
    pr = np.asarray(jax.nn.softmax(jnp.asarray(z), -1))
    M = z[np.arange(len(vB)), vB] - (pr * z).sum(-1)
    return float(M.mean()), float(np.linalg.norm(h, axis=-1).mean()), \
        float((z.argmax(-1) == vB).mean())


def probe(d, v, nb, beta, norm, seed=0, steps=400):
    k1.configure(d=d, v=v, na=50, nd=50, nb=nb)
    p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, beta, norm, 1.0)
    kB, vB = B
    ilr = lr / k1.INJECT_RATIO
    p = p0
    rng = np.random.default_rng(seed + 5)
    ts, Ms, hs, accs = [], [], [], []
    for t in range(steps + 1):
        if t % 2 == 0:
            M, hn, ac = margins(p, kB, vB, norm, 1.0)
            ts.append(t); Ms.append(M); hs.append(hn); accs.append(ac)
            if ac >= 0.9 and t > 20:
                break
        i = rng.integers(0, k1.NB, 32)
        p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, 0.01, norm, 1.0)
    ts = np.array(ts, float); Ms = np.array(Ms); accs = np.array(accs)
    # early-phase slope: from step 0 to where B first reaches 0.5 (the linear regime)
    j = int(np.argmax(accs >= 0.5)) if (accs >= 0.5).any() else len(ts) - 1
    j = max(j, 4)
    slope = float(np.polyfit(ts[:j + 1], Ms[:j + 1], 1)[0])
    tb = float(ts[int(np.argmax(accs >= 0.5))]) if (accs >= 0.5).any() else np.nan
    Mat = float(np.interp(tb, ts, Ms)) if np.isfinite(tb) else np.nan
    return dict(d=d, v=v, nb=nb, beta=beta, norm=norm, seed=seed, ilr=ilr, lr=lr,
                M0=float(Ms[0]), dMdt=slope, t_B=tb, M_at_tB=Mat,
                h0=float(hs[0]), h_end=float(hs[-1]))


if __name__ == "__main__":
    d = int(sys.argv[1]); out = []
    cfgs = ([(d, 32, 50, 0.5)] if d != 128 else
            [(128, 32, 50, b) for b in (0.0, 0.25, 0.5, 0.75, 0.9)] +
            [(128, 32, n, 0.5) for n in (5, 12, 25, 100, 200)] +
            [(128, vv, 50, 0.5) for vv in (8, 16, 64, 128)])
    for (dd, vv, nb, beta) in cfgs:
        for norm in (0, 1):
            try:
                r = probe(dd, vv, nb, beta, norm)
            except SystemExit as e:
                print("skip", dd, vv, nb, beta, norm, e, flush=True); continue
            R = (1 - beta) * (1 - 1.0 / vv) / nb + beta / vv
            Rind = (1 - beta) * (1 - 1.0 / vv) / nb
            r["R_sum"] = R; r["R_ind"] = Rind
            r["pred_lin"] = r["ilr"] * dd * R
            out.append(r)
            print(f"d={dd:4d} V={vv:4d} nB={nb:4d} b={beta:.2f} norm={norm} | "
                  f"eta {r['ilr']:.3e} | dM/dt {r['dMdt']:+.4f} | pred eta*d*R "
                  f"{r['pred_lin']:.4f} | ratio {r['dMdt']/r['pred_lin']:+.2f} | "
                  f"||h|| {r['h0']:.2f} | t_B {r['t_B']:.0f} M(t_B) {r['M_at_tB']:.2f} "
                  f"M0 {r['M0']:.2f}", flush=True)
    json.dump(out, open(f"timing_force_d{d}.json", "w"))
