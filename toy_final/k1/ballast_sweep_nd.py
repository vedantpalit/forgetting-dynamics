"""Task 3: gate-matched injection across n_D in {0, 5, 12, 25, 50}, both arms, 3 seeds.

Uses sweeps.match_gate unchanged (it only needs p0 and B, so the empty ballast never reaches
it). The injection loop is a copy of k1.inject with three extra columns, because the ballast
question is about the ABSOLUTE shared row, not only its change:

    s_abs_on_w   <mu W, w>          w = the unit X->Y separation of the pretrained readout;
                                    k1.inject logs only <mu dW, w>, which starts at 0 by
                                    construction and so hides the pretrained prior.
    m_cross      mean over A of z[correct] - max over the OTHER half      the headroom the
                                                                          Y-ward bias eats
    m_own        mean over A of z[correct] - max over the OWN half        the rank-sparing one
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import numpy as np
import jax.numpy as jnp

import k1
import sweeps
import ballast_k1 as bk

BETA, GAIN, USCALE, STEPS, TARGET = 0.5, 1.0, 0.01, 5000, 200
NDS = [0, 5, 12, 25, 50]
SEEDS = [0, 1, 2]


def margins(p, kA, vA, norm, gain):
    z = np.asarray(k1.fwd(p, jnp.asarray(kA), norm, gain)[0])
    i = np.arange(len(vA))
    own = np.full(k1.V, -np.inf); own[k1.X] = 0.0
    oth = np.full(k1.V, -np.inf); oth[k1.Y] = 0.0
    zo = z + own[None, :]; zo[i, vA] = -np.inf
    return float((z[i, vA] - (z + oth[None, :]).max(1)).mean()), float((z[i, vA] - zo.max(1)).mean())


def inject_plus(p0, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr):
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
    _, hA0 = [np.asarray(x) for x in k1.fwd(p0, jnp.asarray(kA), norm, gain)]
    postA0 = np.asarray(k1.rms(jnp.asarray(hA0))) if norm else hA0
    w = U0[:, k1.Y].mean(1) - U0[:, k1.X].mean(1)
    w /= np.linalg.norm(w)

    p = p0
    rng = np.random.default_rng(seed + 5)
    gi = set(k1.grid(steps)); rows = []
    for t in range(steps + 1):
        if t in gi:
            _, hA = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(kA), norm, gain)]
            postA = np.asarray(k1.rms(jnp.asarray(hA))) if norm else hA
            W = np.asarray(p["W"]); U = np.asarray(p["U"]); dW = W - W0
            s = mu @ dW
            dpost = (postA - postA0).mean(0)
            eps = (postA - postA0) - dpost
            mc, mo = margins(p, kA, vA, norm, gain)
            rows.append(dict(
                step=t,
                A=bk.acc(p, kA, vA, norm, gain),
                A_own=bk.acc(p, kA, vA, norm, gain, restrict=k1.X),
                Dacc=bk.acc(p, kD, vD, norm, gain), B=bk.acc(p, kB, vB, norm, gain),
                s_norm=float(np.linalg.norm(s)),
                s_share=float(np.linalg.norm(s) ** 2 / max(np.linalg.norm(dW) ** 2, 1e-30)),
                s_on_w=float(s @ w),
                s_abs_on_w=float((mu @ W) @ w),
                s_abs_norm=float(np.linalg.norm(mu @ W)),
                m_cross=mc, m_own=mo,
                dW=float(np.linalg.norm(dW)),
                dU_rel=float(np.linalg.norm(U - U0) / np.linalg.norm(U0)),
                delta=float(np.linalg.norm(dpost)),
                eps=float(np.sqrt((eps ** 2).sum(-1)).mean()),
                post_norm=float(np.linalg.norm(postA, axis=-1).mean()),
                a_frac=float(np.linalg.norm(dpost) / np.linalg.norm(postA, axis=-1).mean()),
            ))
        i = rng.integers(0, k1.NB, 32)
        p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm, gain)
    return rows, p


def main():
    k1.configure(d=128, v=32, na=50, nb=50)
    out = {}
    for nd in NDS:
        for norm in (1, 0):
            for seed in SEEDS:
                bk.configure(nd=nd)
                p0, lr, pg, mu, A, Dd, B = bk.pretrain(seed, BETA, norm, GAIN, nd)
                base = lr / k1.INJECT_RATIO
                ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, GAIN, USCALE, base, seed, TARGET)
                rows, _ = inject_plus(p0, mu, A, Dd, B, norm, GAIN, USCALE, STEPS, seed, ilr)
                r = dict(sweep="nd", value=nd, norm=norm, seed=seed, beta=BETA, d=k1.D, v=k1.V,
                         na=k1.NA, nd=nd, nb=k1.NB, gain=GAIN, u_scale=USCALE, pretrain_lr=lr,
                         pretrain_gate=pg, inject_lr=ilr, matched=bool(ok), probe_gate=g, rows=rows)
                q = k1.summarise(r)
                out[f"nd{nd}-n{norm}-s{seed}"] = r
                flag = "" if ok else "  [GATE NOT MATCHED]"
                print(f"nd={nd:2d} norm={norm} s{seed} | ilr {ilr:.2e} Bgate {q['B_gate']} | "
                      f"A trough {q['trough']:.2f}@{q['trough_step']} -> after {q['after']:.2f} "
                      f"(rec {q['recovery']:+.2f}) end {q['end']:.2f} | s.w {q['sw_peak']:.2f}"
                      f"->{q['sw_end']:.2f} ({-100 * q['sw_drop']:+.0f}%) | "
                      f"abs.w {rows[0]['s_abs_on_w']:+.2f}->{rows[-1]['s_abs_on_w']:+.2f} | "
                      f"mcross {rows[0]['m_cross']:+.2f}->{rows[-1]['m_cross']:+.2f} | "
                      f"eps {q['eps_end']:.2f} | a {q['a_peak']:.2f} | "
                      f"Aown {q['A_own_end']:.2f}{flag}", flush=True)
                json.dump(out, open("ballast_nd_sweep.json", "w"))
    print(f"wrote ballast_nd_sweep.json ({len(out)} runs)")


if __name__ == "__main__":
    main()
