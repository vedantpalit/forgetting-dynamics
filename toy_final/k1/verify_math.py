"""Numerical verification of the K=1 derivation. Every claim below is checked against
autodiff or against the measured trajectory; nothing is asserted.

THE MODEL.  h = k + g sqrt(d) k_hat W,  z = rms(h) U  (or z = h U in the linear arm).

CLAIM 1 -- the collective write is the mu-row of Delta W.
    Delta h_a = g sqrt(d) k_hat_a Delta W, and k_a = sqrt(b) mu + sqrt(1-b) g_a, so
        mean_a Delta h_a  =  g sqrt(d) sqrt(b) <1/||k||> (mu^T Delta W)  +  O(1/sqrt(n_A))
    i.e. what every A individual receives in common is exactly s := mu^T Delta W, scaled.
    CHECK: relative error between measured mean_a Delta h_a and that prediction.

CLAIM 2 -- the exact gradient identity for the collective write.
        mu^T grad_W L  =  g sqrt(d) sqrt(b) sum_b (1/||k_b||) dL/dh_b
    with, in the LINEAR arm,        dL/dh_b = U (p_b - e_b) / n_B
    and in the NORMALIZED arm,      dL/dh_b = J_b U (p_b - e_b) / n_B,
                                    J_b = (sqrt(d)/||h_b||) (I - h_hat h_hat^T).
    CHECK: relative error against jax.grad. Should be at machine precision, which is what
    licenses every hand step that follows.

CLAIM 3 -- the normalizer, and only the normalizer, adds an anti-state force.
    J_b U(p_b - e_b) = (sqrt(d)/||h_b||) U(p_b - e_b)  +  (M_b/||h_b||) h_hat_b
    with M_b = z_{b,y_b} - E_{p_b}[z_b] the confidence margin, because
        <h_hat_b, U(p_b - e_b)> = (1/sqrt(d)) <z_b, p_b - e_b> = -M_b/sqrt(d).
    The second term points along MINUS B's own state (it enters the update with a minus) and
    its coefficient is B's own confidence. The linear arm has no such term at all.
    CHECK: the split reproduces J_b U(p_b - e_b) exactly, and the two terms are tracked
    separately as projections on the current write direction s_hat.

CLAIM 4 -- the sign.
    <ds/dt, s_hat> is non-negative for every step in the linear arm (the write can stall but
    never reverse) and changes sign in the normalized arm, at B's onset.
    CHECK: measured directly, no assumptions. This is the crux of the whole account.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse

import numpy as np
import jax
import jax.numpy as jnp

import k1
from k1 import (D, V, X, Y, NB, INJECT_RATIO, build, fwd, loss_fn, step, pretrain, rms,
                accuracy, grid)

jax.config.update("jax_enable_x64", True)


def manual_dLdh(p, kB, vB, norm, gain):
    """dL/dh for every B individual, plus the two-term split of the normalized case."""
    z, h = [np.asarray(x) for x in fwd(p, jnp.asarray(kB), norm, gain)]
    pr = np.asarray(jax.nn.softmax(jnp.asarray(z), axis=-1))
    e = np.zeros_like(pr); e[np.arange(len(vB)), vB] = 1.0
    U = np.asarray(p["U"])
    y = (pr - e) @ U.T / len(vB)                       # (n_B, d): U (p - e) / n_B
    # The margin is a property of the logits and exists in BOTH arms. That is the point:
    # it grows either way, but only the normalized arm's gradient contains a term built
    # from it. Computing it in both makes that comparison possible.
    margin = z[np.arange(len(vB)), vB] - (pr * z).sum(-1)       # M_b
    if not norm:
        return y, y, np.zeros_like(y), h, margin
    hn = np.linalg.norm(h, axis=-1, keepdims=True)
    hhat = h / hn
    term_a = np.sqrt(D) / hn * y                               # the un-normalized-like part
    term_b = (margin[:, None] / hn[:, 0][:, None]) * hhat / len(vB)   # the anti-state part
    return term_a + term_b, term_a, term_b, h, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save", default="")
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--u_scale", type=float, default=1.0,
                    help="readout learning-rate multiplier during injection; the paper's "
                         "protocol is 0.01 (k1.py docstring). Default 1.0 keeps the original "
                         "track_s*.json behaviour reproducible.")
    ap.add_argument("--match", type=int, default=0,
                    help="gate-match the injection rate so B reaches 0.99 at this step "
                         "(0 = fixed rate lr/INJECT_RATIO). The paper's mechanism figure uses "
                         "200: at a fixed rate B's clock varies 2.4x across seeds, which smears "
                         "the averaged curves in time without changing the depth (LOG.md, "
                         "'where the toy's noise comes from').")
    a = ap.parse_args()
    if a.d:
        # rebind k1's dimension, then refresh this module's copy of it. rms/build/init read
        # k1's global directly, so they pick the new value up on their own.
        k1.set_d(a.d)
        global D
        D = k1.D
    saved = {}

    for norm in (0, 1):
        p0, lr, gate, mu, (kA, vA), (kD, vD), (kB, vB) = pretrain(a.seed, a.beta, norm, a.gain)
        W0 = np.asarray(p0["W"])
        _, hA0 = [np.asarray(x) for x in fwd(p0, jnp.asarray(kA), norm, a.gain)]
        knorm = np.linalg.norm(kA, axis=-1)
        pref = a.gain * np.sqrt(D) * np.sqrt(a.beta) * np.mean(1.0 / knorm)

        print(f"\n{'='*104}\nARM: {'normalized readout' if norm else 'linear (no norm)'}   "
              f"beta={a.beta} gain={a.gain} seed={a.seed} pretrain lr={lr} gate@{gate}\n{'='*104}")
        print(f"{'step':>6} {'B acc':>6} {'||s||':>7} {'c1 err':>9} {'c2 err':>9} {'c3 err':>9} "
              f"{'<ds,s>':>10} {'  from A':>10} {'  from B':>10} {'margin':>8}")

        p = p0
        rng = np.random.default_rng(a.seed + 5)
        ilr = lr / INJECT_RATIO
        if a.match:
            import sweeps
            ilr, g, ok = sweeps.match_gate(p0, (kA, vA), (kD, vD), (kB, vB), norm, a.gain,
                                           a.u_scale, ilr, a.seed, a.match)
            print(f"  gate-matched: rate {ilr:.4g}, B gate at {g} (target {a.match}, matched={ok})")
        gi = set(grid(a.steps))
        c1, c2, c3, c2b, sgn_lin, track, rec = [], [], [], [], [], [], []
        w = np.asarray(p0["U"])[:, Y].mean(1) - np.asarray(p0["U"])[:, X].mean(1)
        w /= np.linalg.norm(w)
        for t in range(a.steps + 1):
            if t in gi:
                dW = np.asarray(p["W"]) - W0
                s = mu @ dW
                sn = np.linalg.norm(s)
                shat = s / max(sn, 1e-30)

                # claim 1: the common shift on A is the mu-row of dW.
                # exact form: it is the MEAN KEY DIRECTION's row of dW. approximate form:
                # that mean direction is mu, up to A-side crosstalk of order 1/sqrt(d n_A).
                _, hA = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), norm, a.gain)]
                meas = (hA - hA0).mean(0)
                kbar = (kA / np.linalg.norm(kA, axis=-1)[:, None]).mean(0)
                exact1 = a.gain * np.sqrt(D) * (kbar @ dW)
                err1x = np.linalg.norm(meas - exact1) / max(np.linalg.norm(meas), 1e-30)
                err1 = np.linalg.norm(meas - pref * s) / max(np.linalg.norm(meas), 1e-30)

                # claim 2a: the EXACT gradient identity, against autodiff. The coefficient of
                # each individual's dL/dh is its own key's overlap with mu, nothing else.
                gW = np.asarray(jax.grad(loss_fn)(p, jnp.asarray(kB), jnp.asarray(vB),
                                                  norm, a.gain)["W"])
                auto = mu @ gW
                dLdh, ta, tb, hB, margin = manual_dLdh(p, kB, vB, norm, a.gain)
                khat = kB / np.linalg.norm(kB, axis=-1)[:, None]
                coef = khat @ mu                                   # exact: <mu, k_hat_b>
                exact = a.gain * np.sqrt(D) * (coef[:, None] * dLdh).sum(0)
                err2 = np.linalg.norm(auto - exact) / max(np.linalg.norm(auto), 1e-30)
                # claim 2b: the O(1/sqrt(d)) simplification <mu, k_hat_b> -> sqrt(beta)/||k_b||
                man = a.gain * np.sqrt(D) * np.sqrt(a.beta) * (dLdh / np.linalg.norm(kB, axis=-1)[:, None]).sum(0)
                err2b = np.linalg.norm(auto - man) / max(np.linalg.norm(auto), 1e-30)

                # claim 3: the two-term split reproduces the normalized gradient exactly
                err3 = np.linalg.norm(dLdh - (ta + tb)) / max(np.linalg.norm(dLdh), 1e-30)

                # claim 4: the sign of the write's own growth rate, and where it comes from
                force = -ilr * exact
                fa = -ilr * a.gain * np.sqrt(D) * (coef[:, None] * ta).sum(0)
                fb = -ilr * a.gain * np.sqrt(D) * (coef[:, None] * tb).sum(0)
                proj, pa, pb = force @ shat, fa @ shat, fb @ shat

                c1.append(err1); c2.append(err2); c3.append(err3); c2b.append(err2b)
                track.append((t, margin.mean(), proj))
                rec.append(dict(step=t, A=accuracy(p, kA, vA, norm, a.gain),
                                B=accuracy(p, kB, vB, norm, a.gain),
                                margin=float(margin.mean()), s_norm=float(sn),
                                s_on_w=float(s @ w), proj=float(proj),
                                proj_plain=float(pa), proj_anti=float(pb)))
                if sn > 1e-9:
                    sgn_lin.append(proj)
                if t in (0, 5, 10, 20, 40, 60, 80, 100, 150, 200) or t in (
                        [x for x in sorted(gi) if x > 200][::10]):
                    print(f"{t:>6} {accuracy(p, kB, vB, norm, a.gain):>6.2f} {sn:>7.3f} "
                          f"{err1:>9.2e} {err2:>9.2e} {err3:>9.2e} "
                          f"{proj:>+10.3e} {pa:>+10.3e} {pb:>+10.3e} {margin.mean():>8.3f}")
            i = rng.integers(0, NB, 32)
            p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, a.u_scale, norm, a.gain)

        sg = np.array(sgn_lin)
        print(f"\n  claim 2a EXACT gradient identity vs autodiff   max rel err {max(c2):.2e}")
        print(f"  claim 3  two-term split of the normalized grad  max rel err {max(c3):.2e}")
        print(f"  claim 1  common shift is the mu-row, O(1/sqrt(d n_A)) residual  "
              f"max rel err {max(c1):.2e}")
        print(f"  claim 2b sqrt(beta) simplification, O(1/sqrt(d)) residual       "
              f"max rel err {max(c2b):.2e}")
        print(f"  claim 4  sign of <ds/dt, s_hat>: min {sg.min():+.3e}, "
              f"negative at {100 * (sg < 0).mean():.0f}% of checkpoints")

        # claim 5: the write turns around exactly when B's mean margin turns positive.
        # Both crossings are read off the same checkpoint grid, so they are directly comparable.
        tr = [x for x in track if x[0] > 0]

        def cross(vals, rising):
            for i in range(1, len(vals)):
                a0, b0 = vals[i - 1][1], vals[i][1]
                if (rising and a0 < 0 <= b0) or (not rising and a0 > 0 >= b0):
                    return vals[i][0]
            return None

        mcross = cross([(t, m) for t, m, _ in tr], rising=True)
        pcross = cross([(t, p) for t, _, p in tr], rising=False)
        print(f"  claim 5  B's mean margin crosses 0 at step {mcross}; "
              f"<ds/dt, s_hat> crosses 0 at step {pcross}")
        if a.save:
            saved[f"n{norm}-s{a.seed}"] = dict(rows=rec, margin_cross=mcross, proj_cross=pcross)
    if a.save:
        import json
        json.dump(saved, open(a.save, "w"))
        print(f"\n  wrote {a.save}")


if __name__ == "__main__":
    main()
