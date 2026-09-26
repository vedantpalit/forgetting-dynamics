"""Instrumented re-run of the K=1 toy, logging the exact force balance that governs the
shared write's between-half content x_w := <s, w_hat>, s = mu^T (W - W0).

THE IDENTITY BEING LOGGED (exact, from verify_math claims 2a + 3):

    d<s,w>/dt  =  (eta g sqrt(d) / n_B) * sum_b  omega_b * [ sqrt(d) Phi_b  -  M_b * C_b ]

    omega_b := <mu, k_hat_b> / ||h_b||          > 0
    C_b     := <h_hat_b, w_hat>                  the state's own between-half content
    M_b     := z_{b,y_b} - E_{p_b}[z_b]          the confidence margin (the anti-state term)
    Phi_b   := phi_{y_b} - E_{p_b}[phi],  phi_v := <u_v, w_hat>     the HALF-margin

Phi_b is the plain term's only surviving projection on w_hat: it is a margin measured in the
between-half direction, so it is carried almost entirely by the probability mass B still puts
on the WRONG HALF. When that mass dies, the writing force dies, but the anti-state force does
not -- it lives on the margin against the WITHIN-HALF runner-up. Hence stationarity:

    STATIONARY  <=>  weighted mean of C_b  =  sqrt(d) * Phibar / Mbar                    (*)

which holds at BOTH the peak and the end of the fall. Everything logged here is either an
exact ingredient of (*) or a direct check of it.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, json, time

import numpy as np
import jax
import jax.numpy as jnp

import k1
from k1 import fwd, rms, accuracy, GATE
import sweeps as sw

jax.config.update("jax_enable_x64", True)


def diag(p, kB, vB, mu, W0, w, norm, gain, d, span=True):
    """All per-checkpoint diagnostics. Full batch, no minibatch noise."""
    z, h = [np.asarray(x) for x in fwd(p, jnp.asarray(kB), norm, gain)]
    U = np.asarray(p["U"])
    pr = np.asarray(jax.nn.softmax(jnp.asarray(z), -1))
    n = len(vB)
    ar = np.arange(n)
    M = z[ar, vB] - (pr * z).sum(-1)                       # margins
    phi = w @ U                                            # per-class half-score
    Phi = phi[vB] - pr @ phi                               # half-margins
    hn = np.linalg.norm(h, axis=-1)
    hhat = h / hn[:, None]
    C = hhat @ w
    khat = kB / np.linalg.norm(kB, axis=-1)[:, None]
    coef = khat @ mu
    om = coef / hn                                         # omega_b > 0
    dW = np.asarray(p["W"]) - W0
    s = mu @ dW
    # probability mass by region: own class / wrong-but-same-half / other half
    Yset = set(k1.Y.tolist())
    inY = np.array([v in Yset for v in range(k1.V)])
    own = pr[ar, vB]
    p_same = pr[:, inY].sum(-1) - own if vB[0] in Yset else pr[:, ~inY].sum(-1) - own
    p_other = pr[:, ~inY].sum(-1) if vB[0] in Yset else pr[:, inY].sum(-1)
    out = dict(
        B=float((z.argmax(-1) == vB).mean()),
        s_on_w=float(s @ w), s_norm=float(np.linalg.norm(s)),
        M=float(M.mean()), Phi=float(Phi.mean()),
        # the weighted versions that enter (*) exactly
        Mbar=float((om * M).sum() / om.sum()), Phibar=float((om * Phi).sum() / om.sum()),
        Cbar=float((om * M * C).sum() / max((om * M).sum(), 1e-300)),
        C_mean=float(C.mean()), hnorm=float(hn.mean()),
        p_same=float(p_same.mean()), p_other=float(p_other.mean()),
        # the two forces, in the units of the identity above
        F_plain=float((om * np.sqrt(d) * Phi).sum() / n),
        F_anti=float(-(om * M * C).sum() / n),
        # x_b = <h_b, w> and its shared part, for converting C_b back to s
        x_h=float((h @ w).mean()), x_k=float((kB @ w).mean()),
    )
    out["Q"] = np.sqrt(d) * out["Phibar"] / out["Mbar"] if out["Mbar"] != 0 else np.nan
    if span:
        # bound: the anti-state force lies in span{h_b}. Component of s orthogonal to it.
        q, _ = np.linalg.qr(h.T)                           # d x n_B orthonormal basis
        s_perp = s - (s @ q) @ q.T
        out["s_perp"] = float(np.linalg.norm(s_perp))
        out["s_perp_on_w"] = float(s_perp @ w)
    return out


def run(seed, beta, d, v, na, nb, steps, u_scale=0.01, gain=1.0, norm=1,
        match=200, grid_steps=None):
    k1.configure(d=d, v=v, na=na, nd=na, nb=nb)
    p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, beta, norm, gain)
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    base = lr / k1.INJECT_RATIO
    if match:
        ilr, g, ok = sw.match_gate(p0, A, Dd, B, norm, gain, u_scale, base, seed, match)
    else:
        ilr, g, ok = base, -1, True
    W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
    w = U0[:, k1.Y].mean(1) - U0[:, k1.X].mean(1); w /= np.linalg.norm(w)
    _, hA0 = [np.asarray(x) for x in fwd(p0, jnp.asarray(kA), norm, gain)]
    postA0 = np.asarray(rms(jnp.asarray(hA0)))

    gi = set(k1.grid(steps)) | {int(x) for x in
                                np.unique(np.round(np.logspace(0, np.log10(max(steps, 10)), 120)))}
    rows = []
    p = p0
    rng = np.random.default_rng(seed + 5)
    t0 = time.time()
    for t in range(steps + 1):
        if t in gi:
            r = diag(p, kB, vB, mu, W0, w, norm, gain, k1.D)
            _, hA = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), norm, gain)]
            postA = np.asarray(rms(jnp.asarray(hA)))
            dpost = (postA - postA0).mean(0)
            r.update(step=t, A=accuracy(p, kA, vA, norm, gain),
                     A_own=accuracy(p, kA, vA, norm, gain, restrict=k1.X),
                     a_frac=float(np.linalg.norm(dpost) /
                                  np.linalg.norm(postA, axis=-1).mean()),
                     dU_rel=float(np.linalg.norm(np.asarray(p["U"]) - U0) /
                                  np.linalg.norm(U0)))
            rows.append(r)
        i = rng.integers(0, k1.NB, 32)
        p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm, gain)
    return dict(seed=seed, beta=beta, d=k1.D, v=k1.V, na=k1.NA, nb=k1.NB, norm=norm,
                u_scale=u_scale, gain=gain, inject_lr=ilr, matched=bool(ok),
                pretrain_lr=lr, steps=steps, secs=time.time() - t0, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", default="0.5")
    ap.add_argument("--nb", default="50")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--v", type=int, default=32)
    ap.add_argument("--na", type=int, default=50)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--match", type=int, default=200)
    ap.add_argument("--u_scale", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for beta in [float(x) for x in a.beta.split(",")]:
        for nb in [int(x) for x in a.nb.split(",")]:
            for seed in [int(x) for x in a.seeds.split(",")]:
                r = run(seed, beta, a.d, a.v, a.na, nb, a.steps,
                        u_scale=a.u_scale, match=a.match)
                key = f"b{beta}-nb{nb}-d{a.d}-v{a.v}-s{seed}"
                out[key] = r
                sw_ = np.array([x["s_on_w"] for x in r["rows"]])
                pk = int(sw_.argmax())
                print(f"{key}: ilr={r['inject_lr']:.2e} matched={r['matched']} "
                      f"sw peak {sw_[pk]:.4f}@{r['rows'][pk]['step']} end {sw_[-1]:.4f} "
                      f"f={1 - sw_[-1] / sw_[pk]:.3f}  ({r['secs']:.0f}s)", flush=True)
                json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
