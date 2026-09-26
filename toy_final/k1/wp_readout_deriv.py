"""Task 1.  dL/dU for the normalized model, verified against autodiff, and the sign of the
readout bias's drift.

THE POINT.  In  z = rms(h) U  the readout enters LINEARLY and rms(h) does not depend on U, so

    dL/dU  =  (1/n_B) sum_b  rms(h_b)^T (p_b - e_{y_b})                                    (U1)

with NO second term.  Contrast (7) of THEORY.md: dL/dh_b carries the normalizer's Jacobian,
which contributes  +(M_b/(n_B||h_b||)) h_hat_b  -- the anti-state force that lets B's own
confidence un-write the shared row in W.  There is no such term in (U1); the only thing that
multiplies the error signal is a positive scalar times the state itself.

CONSEQUENCE.  Define the READOUT BIAS that A receives,

    RB(t) = < sbar_A(0),  mean_{v in Y} U_v(t) - mean_{v in X} U_v(t) >                    (U2)

with sbar_A(0) the pretrained mean readout state of A.  Since y_b in Y for every b, and with
|X| = |Y| = V/2, one line of algebra on (U1) gives the SGD drift

    dRB/dt  =  (2 rho eta / (n_B |Y|)) * sum_b  P_b^X  < sbar_A(0), rms(h_b) >             (U3)

where P_b^X = sum_{v in X} p_{b,v} is B's residual softmax mass on the OTHER half.  Every
factor is non-negative as long as the states of A and B are positively aligned (they share
sqrt(beta) mu), so RB is non-decreasing while B has any mass left on X, and stationary only
when that mass is gone.  A Y-favouring class bias in U is monotone-good for B's loss: nothing
in the dynamics removes it.  That is the asymmetry this file verifies.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import jax.numpy as jnp

import k1
import ballast_k1 as bk
import wp_readout_lib as wl

k1.configure(d=128, v=32, na=50, nb=50)
NORM, GAIN, BETA = 1, 1.0, 0.5


def hand_gradU(p, k, v, norm, gain):
    """(U1) written out by hand."""
    z, h = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(k), norm, gain)]
    post = np.asarray(k1.rms(jnp.asarray(h))) if norm else h
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); pr = e / e.sum(1, keepdims=True)
    err = pr.copy(); err[np.arange(len(v)), v] -= 1.0
    return post.T @ err / len(v)


def main():
    print("=" * 78)
    print("1a.  dL/dU against autodiff  (normalized arm, the arm with the Jacobian in dL/dh)")
    print("=" * 78)
    worst = 0.0
    for seed in (0, 1, 2):
        for nd in (0, 50):
            bk.configure(nd=nd)
            p0, lr, pg, mu, A, Dd, B = bk.pretrain(seed, BETA, NORM, GAIN, nd)
            kB, vB = B
            ga = np.asarray(jax.grad(k1.loss_fn)(p0, jnp.asarray(kB), jnp.asarray(vB),
                                                 NORM, GAIN)["U"])
            gh = hand_gradU(p0, kB, vB, NORM, GAIN)
            rel = np.abs(ga - gh).max() / np.abs(ga).max()
            worst = max(worst, rel)
            print(f"  seed {seed} nd {nd:2d}:  max |autodiff - hand| / max|autodiff| = {rel:.3e}")
    print(f"  worst over 6 configurations: {worst:.3e}   -> (U1) is exact, no extra term.")

    print()
    print("=" * 78)
    print("1b.  sign of dRB/dt along a run, and (U3) against the measured one-step change")
    print("=" * 78)
    for nd in (0, 50):
        for us in (1.0, 0.1):
            bk.configure(nd=nd)
            p0, lr, pg, mu, A, Dd, B = bk.pretrain(0, BETA, NORM, GAIN, nd)
            (kA, vA), (kB, vB) = A, B
            ilr, g, ok = wl.match_gate2(p0, B, NORM, GAIN, us, 1.0,
                                        lr / k1.INJECT_RATIO, 0, 200)
            post0 = wl._state(p0, kA, NORM, GAIN)
            sbar0 = post0.mean(0)
            U0 = np.asarray(p0["U"])
            ug0 = wl.u_gap(U0)

            p = p0
            rng = np.random.default_rng(5)
            preds, meas, aligns, pxs, rbs = [], [], [], [], []
            nY = len(k1.Y)
            for t in range(2001):
                U = np.asarray(p["U"])
                rb = float(sbar0 @ (wl.u_gap(U) - ug0))
                rbs.append(rb)
                if t % 25 == 0:
                    # (U3) evaluated on the FULL B population (the SGD batch is a sample of it)
                    z, h = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(kB), NORM, GAIN)]
                    post = np.asarray(k1.rms(jnp.asarray(h)))
                    zz = z - z.max(1, keepdims=True)
                    e = np.exp(zz); pr = e / e.sum(1, keepdims=True)
                    PX = pr[:, k1.X].sum(1)
                    al = post @ sbar0
                    preds.append(float(2 * ilr * us / (len(vB) * nY) * (PX * al).sum()))
                    aligns.append(float(al.min()))
                    pxs.append(float(PX.mean()))
                i = rng.integers(0, k1.NB, 32)
                p = wl.step2(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, us, 1.0,
                             NORM, GAIN)
            rbs = np.array(rbs)
            d = np.diff(rbs)
            # the exact per-step batch drift, compared against (U3) on the full population
            print(f"  nd={nd:2d} u_scale={us}  ilr={ilr:.2e} (gate {g})")
            print(f"     RB: {rbs[0]:+.3f} -> {rbs[-1]:+.3f};  min over run {rbs.min():+.3f}")
            print(f"     per-step dRB: negative at {100*(d<0).mean():.1f}% of 2000 SGD steps "
                  f"(batch noise), min {d.min():+.2e}, mean {d.mean():+.2e}")
            # 25-step block means, which average the batch sampling out
            blk = rbs[::25]; dblk = np.diff(blk)
            print(f"     25-step blocks: negative at {100*(dblk<0).mean():.1f}%, "
                  f"min {dblk.min():+.2e}")
            pr_blk = 25 * np.array(preds[:-1])
            ok_mask = np.abs(dblk) > 1e-9
            ratio = (dblk[ok_mask] / pr_blk[ok_mask])
            print(f"     (U3)*25 vs measured block change: ratio median {np.median(ratio):.3f}, "
                  f"IQR [{np.percentile(ratio,25):.3f}, {np.percentile(ratio,75):.3f}]")
            print(f"     min_b <sbar_A(0), rms(h_b)> over the run: {min(aligns):+.2f}  "
                  f"(positive => every term in (U3) is >= 0)")
            print(f"     B's residual mass on X: {pxs[0]:.3f} -> {pxs[-1]:.3e}")


if __name__ == "__main__":
    main()
