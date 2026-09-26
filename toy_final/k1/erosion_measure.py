"""Ingredient logging for the erosion law (THEORY.md section 10, items 3 and 5).

The law under test is eq. (4):

    eps_a = - eta g^2 d (1-b) sum_b <g_a,g_b> r_b

with the FITTED size eps/state ~ sqrt(n_B)/d.  The derivation only supplies
sqrt(n_B) (incoherence), one 1/sqrt(d) from var<g_a,g_b> = 1/d, and one
1/sqrt(d) from dividing by a state of norm sqrt(d).  Everything else -- how
||r|| and the accumulated dose behave as functions of d, beta, n_A -- is an
assumption.  This file measures those ingredients instead of assuming them.

Nothing here modifies k1.py / sweeps.py; both are imported read-only.

Per checkpoint we log
    eps_post / post_norm      the fitted observable (post = rms(h), norm = sqrt(d) exactly)
    eps_h    / h_norm         the same on the raw pre-readout state
    r_mean, r_rms             mean_b ||r_b||,  sqrt(mean_b ||r_b||^2)   (r carries the 1/n_B)
    kappa_r                   ||sum_b r_b|| / (n_B r_mean)              coherence of B's errors
    E_mean                    mean_a || sum_b <g_a,g_b> r_b ||          the EXACT incoherent sum
    E_pred                    sqrt(n_B) * gg_rms * r_rms                its incoherent estimate
    dW                        ||W(t)-W(0)||_F
    dose                      sum over steps of ilr*||grad_W||_F        (accumulated dose)
    inc                       ilr * gain^2 * d * (1-b) * E_mean         predicted per-step size
    cum_lin, cum_rw           trapezoid of inc,  sqrt(trapezoid of inc^2)  linear / random walk
and once per run the realised key crosstalk gg_mean = mean|<g_a,g_b>|, gg_rms.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import json
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import k1
import sweeps


@partial(jax.jit, static_argnames=("norm",))
def step_diag(p, k, v, lr, u_scale, norm, gain):
    g = jax.grad(k1.loss_fn)(p, k, v, norm, gain)
    return ({"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]},
            jnp.linalg.norm(g["W"]))


def compute_r(p, kB, vB, norm, gain):
    """r_b = dL/dh_b by autodiff through the readout only (the 1/n_B is included)."""
    _, h = k1.fwd(p, jnp.asarray(kB), norm, gain)
    vj = jnp.asarray(vB)

    def L(hh):
        zz = (k1.rms(hh) if norm else hh) @ p["U"]
        return -jnp.mean(jax.nn.log_softmax(zz, -1)[jnp.arange(hh.shape[0]), vj])

    r = jax.grad(L)(h)
    zz = np.asarray(((k1.rms(h) if norm else h) @ p["U"]))
    pr = np.asarray(jax.nn.softmax(jnp.asarray(zz), -1))
    margin = zz[np.arange(len(vB)), np.asarray(vB)] - (pr * zz).sum(-1)
    return np.asarray(r), np.asarray(h), margin


def g_of(keys, mu, beta):
    """Recover the individual key parts: k = sqrt(b) mu + sqrt(1-b) g."""
    if beta >= 1.0:
        return np.zeros_like(keys)
    return (keys - np.sqrt(beta) * mu[None]) / np.sqrt(1.0 - beta)


def measure(p0, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr, beta):
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    d = k1.D
    W0 = np.asarray(p0["W"])
    _, hA0 = [np.asarray(x) for x in k1.fwd(p0, jnp.asarray(kA), norm, gain)]
    postA0 = np.asarray(k1.rms(jnp.asarray(hA0))) if norm else hA0

    gA, gB = g_of(kA, mu, beta), g_of(kB, mu, beta)
    GG = gA @ gB.T                                   # (n_A, n_B) realised crosstalk
    gg_mean, gg_rms = float(np.abs(GG).mean()), float(np.sqrt((GG ** 2).mean()))

    p = p0
    rng = np.random.default_rng(seed + 5)
    gi = set(k1.grid(steps))
    rows, dose = [], 0.0
    prev = None                                      # (step, inc) for the trapezoid
    cum_lin, cum_sq = 0.0, 0.0
    for t in range(steps + 1):
        if t in gi:
            _, hA = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(kA), norm, gain)]
            postA = np.asarray(k1.rms(jnp.asarray(hA))) if norm else hA
            dpost = (postA - postA0).mean(0)
            eps_post = (postA - postA0) - dpost
            dh = (hA - hA0).mean(0)
            eps_h = (hA - hA0) - dh

            r, hB, margin = compute_r(p, kB, vB, norm, gain)
            rn = np.linalg.norm(r, axis=-1)
            E = GG @ r                               # (n_A, d): sum_b <g_a,g_b> r_b
            E_mean = float(np.linalg.norm(E, axis=-1).mean())
            r_rms = float(np.sqrt((rn ** 2).mean()))
            inc = ilr * gain ** 2 * d * (1.0 - beta) * E_mean
            if prev is not None:
                dt = t - prev[0]
                cum_lin += 0.5 * (inc + prev[1]) * dt
                cum_sq += 0.5 * (inc ** 2 + prev[1] ** 2) * dt
            prev = (t, inc)

            W = np.asarray(p["W"])
            rows.append(dict(
                step=t,
                A=k1.accuracy(p, kA, vA, norm, gain),
                A_own=k1.accuracy(p, kA, vA, norm, gain, restrict=k1.X),
                B=k1.accuracy(p, kB, vB, norm, gain),
                margin=float(margin.mean()),
                eps_post=float(np.sqrt((eps_post ** 2).sum(-1)).mean()),
                post_norm=float(np.linalg.norm(postA, axis=-1).mean()),
                eps_h=float(np.sqrt((eps_h ** 2).sum(-1)).mean()),
                h_norm=float(np.linalg.norm(hA, axis=-1).mean()),
                delta_post=float(np.linalg.norm(dpost)),
                r_mean=float(rn.mean()), r_rms=r_rms,
                r_sum=float(np.linalg.norm(r.sum(0))),
                kappa_r=float(np.linalg.norm(r.sum(0)) / max(len(rn) * rn.mean(), 1e-30)),
                hB_norm=float(np.linalg.norm(hB, axis=-1).mean()),
                E_mean=E_mean,
                E_pred=float(np.sqrt(len(vB)) * gg_rms * r_rms),
                inc=inc, cum_lin=cum_lin, cum_rw=float(np.sqrt(cum_sq)),
                dW=float(np.linalg.norm(W - W0)), dose=dose,
                s_norm=float(np.linalg.norm(mu @ (W - W0))),
            ))
        i = rng.integers(0, k1.NB, 32)
        p, gwn = step_diag(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm, gain)
        dose += ilr * float(gwn)
    return rows, dict(gg_mean=gg_mean, gg_rms=gg_rms, n_pairs=int(GG.size))


def one(seed, beta, norm, gain, u_scale, steps, match, ilr_fixed):
    p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, beta, norm, gain)
    base = lr / k1.INJECT_RATIO
    if ilr_fixed is not None:
        ilr, g, ok = float(ilr_fixed), -1, True
    elif match:
        ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, gain, u_scale, base, seed, match)
    else:
        ilr, g, ok = base, -1, True
    rows, extra = measure(p0, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr, beta)
    return dict(seed=seed, beta=beta, norm=norm, gain=gain, u_scale=u_scale,
                d=k1.D, v=k1.V, na=k1.NA, nd=k1.ND, nb=k1.NB,
                pretrain_lr=lr, pretrain_gate=pg, inject_lr=ilr, matched=bool(ok),
                probe_gate=g, steps=steps, rows=rows, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--na", default="50")
    ap.add_argument("--nb", type=int, default=50)
    ap.add_argument("--betas", default="0.5")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--norms", default="1")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--u_scale", type=float, default=0.01)
    ap.add_argument("--match_gate", type=int, default=200)
    ap.add_argument("--ilr", default="", help="fixed injection rate; disables matching. "
                                              "comma list -> one per seed")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {}
    for na in [int(x) for x in a.na.split(",")]:
        k1.configure(d=a.d, v=a.vocab, na=na, nd=na, nb=a.nb)
        for beta in [float(x) for x in a.betas.split(",")]:
            for norm in [int(x) for x in a.norms.split(",")]:
                for j, seed in enumerate([int(x) for x in a.seeds.split(",")]):
                    fix = None
                    if a.ilr:
                        v = [float(x) for x in a.ilr.split(",")]
                        fix = v[j] if len(v) > 1 else v[0]
                    t0 = time.time()
                    try:
                        r = one(seed, beta, norm, a.gain, a.u_scale, a.steps, a.match_gate, fix)
                    except SystemExit as e:
                        print(f"  PRETRAIN FAILED d={a.d} na={na} beta={beta} s={seed}: {e}",
                              flush=True)
                        continue
                    last = r["rows"][-1]
                    ratio = last["eps_post"] / last["post_norm"]
                    out[f"{a.tag}-d{a.d}-na{na}-b{beta}-n{norm}-s{seed}"] = r
                    print(f"d={a.d} na={na} nb={a.nb} b={beta} n={norm} s={seed} | "
                          f"ilr {r['inject_lr']:.3e} Bgate~{r['probe_gate']} | "
                          f"eps/state {ratio:.4f} | r_rms {last['r_rms']:.3e} | "
                          f"dW {last['dW']:.3f} dose {last['dose']:.3f} | "
                          f"gg_rms {r['gg_rms']:.4f} (1/sqrt(d)={1/np.sqrt(a.d):.4f}) | "
                          f"A {last['A']:.2f} B {last['B']:.2f} | {time.time()-t0:.0f}s",
                          flush=True)
                    json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out} ({len(out)} runs)")


if __name__ == "__main__":
    main()
