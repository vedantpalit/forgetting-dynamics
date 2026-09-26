"""Untransferred item (II): the READOUT has no anti-state term, so the class bias it
carries is monotone and never withdrawn.

    z = rms(h) U,  L = -mean log softmax(z)_y

R1  dL/dU = rms(h)^T (p - e)/n_B  EXACTLY.  U sits DOWNSTREAM of the normalizer, so no
    Jacobian factor, no projector, and -- unlike dL/dW -- no term built from the margin
    along -hhat.  Checked against jax.grad.

R2  With the between-half readout direction  c := U w,  w_v = +2/V on Y, -2/V on X,
    and a FROZEN probe direction hhat_B0 (B's mean state direction at injection step 0),
    the class bias  b_U := <c, hhat_B0>  obeys EXACTLY

        d b_U/dt  =  + (4 rho eta sqrt(d) / (n_B V)) * sum_b <hhat_b, hhat_B0> * PX_b

    with PX_b = sum_{v in X} p_{b,v} = B's residual mass on the WRONG half.  Every factor
    is non-negative while <hhat_b, hhat_B0> >= 0 (true: B's states all share mu), so b_U is
    MONOTONE NON-DECREASING and reaches zero rate only as PX -> 0.  Nothing removes it.

    Contrast: the store's shared row has the anti-state term -(M_b/||h_b||) hhat_b, which is
    what produces the balance of THEORY section 7.  The readout has no counterpart.

usage: python wp_theory_readout.py [u_scale] [nd] [steps]
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import numpy as np
import jax
import jax.numpy as jnp

import k1
import ballast_k1 as bk

jax.config.update("jax_enable_x64", True)


def main():
    u_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    beta, gain, norm, seed = 0.5, 1.0, 1, 0
    bk.configure(nd=nd)
    p0, lr, gate, mu, (kA, vA), (kD, vD), (kB, vB) = bk.pretrain(seed, beta, norm, gain, nd)
    d, V = k1.D, k1.V
    X, Y = k1.X, k1.Y
    ilr = lr / k1.INJECT_RATIO
    print(f"nd={nd} u_scale={u_scale} d={d} V={V} n_B={len(vB)} pretrain lr={lr} gate@{gate} "
          f"ilr={ilr:.5f}")

    # the between-half readout weight vector and the FROZEN probe direction
    w = np.zeros(V); w[Y] = 2.0 / V; w[X] = -2.0 / V
    _, hB0 = [np.asarray(x) for x in k1.fwd(p0, jnp.asarray(kB), norm, gain)]
    hB0_hat = hB0.mean(0); hB0_hat /= np.linalg.norm(hB0_hat)
    U0 = np.asarray(p0["U"])

    p = p0
    rng = np.random.default_rng(seed + 5)
    print(f"\n{'step':>6} {'A':>5} {'B':>5} {'b_U':>10} {'db_U meas':>11} {'db_U pred':>11} "
          f"{'rel err':>9} {'mean PX':>8} {'<hb,hB0>':>9} {'store s.w':>10} {'dU/U':>7}")
    rows = []
    prev = None
    for t in range(steps + 1):
        Un = np.asarray(p["U"]); Wn = np.asarray(p["W"])
        bU = float((Un @ w) @ hB0_hat)
        # --- the exact predicted rate for THIS step's minibatch, before taking it
        i = rng.integers(0, k1.NB, 32)
        kb, vb = kB[i], vB[i]
        zb, hb = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(kb), norm, gain)]
        pr = np.asarray(jax.nn.softmax(jnp.asarray(zb), -1))
        PX = pr[:, X].sum(-1)
        hbh = hb / np.linalg.norm(hb, axis=-1, keepdims=True)
        ov = hbh @ hB0_hat
        pred = (4.0 * u_scale * ilr * np.sqrt(d) / (len(vb) * V)) * float((ov * PX).sum())
        # --- the exact gradient identity for dL/dU
        gr = jax.grad(k1.loss_fn)(p, jnp.asarray(kb), jnp.asarray(vb), norm, gain)
        hr = np.sqrt(d) * hbh
        e = np.zeros_like(pr); e[np.arange(len(vb)), vb] = 1.0
        man = hr.T @ ((pr - e) / len(vb))
        errU = float(np.linalg.norm(np.asarray(gr["U"]) - man) /
                     np.linalg.norm(np.asarray(gr["U"])))
        p = k1.step(p, jnp.asarray(kb), jnp.asarray(vb), ilr, u_scale, norm, gain)
        bU_new = float((np.asarray(p["U"]) @ w) @ hB0_hat)
        meas = bU_new - bU
        rel = abs(meas - pred) / max(abs(meas), 1e-300)
        s = mu @ (Wn - np.asarray(p0["W"]))
        wdir = U0[:, Y].mean(1) - U0[:, X].mean(1); wdir /= np.linalg.norm(wdir)
        rows.append(dict(t=t, bU=bU, meas=meas, pred=pred, rel=rel, errU=errU,
                         PX=float(PX.mean()), ov=float(ov.mean()),
                         A=bk.acc(p, kA, vA, norm, gain), B=bk.acc(p, kB, vB, norm, gain),
                         sw=float(s @ wdir),
                         dU=float(np.linalg.norm(np.asarray(p["U"]) - U0) /
                                  np.linalg.norm(U0))))
        if t in (0, 5, 10, 20, 40, 80, 150, 300, 600, 1000, 2000, steps):
            r = rows[-1]
            print(f"{t:>6} {r['A']:>5.2f} {r['B']:>5.2f} {r['bU']:>10.5f} {r['meas']:>11.3e} "
                  f"{r['pred']:>11.3e} {r['rel']:>9.2e} {r['PX']:>8.4f} {r['ov']:>9.4f} "
                  f"{r['sw']:>10.4f} {r['dU']:>7.4f}")

    _rr = [r["rel"] for r in rows if abs(r["meas"]) > 1e-14]
    mrel = max(_rr) if _rr else 0.0
    meU = max(r["errU"] for r in rows)
    neg = [r for r in rows if r["meas"] < 0]
    negp = [r for r in rows if r["pred"] < 0]
    bs = np.array([r["bU"] for r in rows])
    sw = np.array([r["sw"] for r in rows])
    print(f"\n  R1 dL/dU = rms(h)^T (p-e)/n vs jax.grad   max rel err {meU:.3e}")
    print(f"  R2 db_U/dt formula vs measured increment  max rel err {mrel:.3e}")
    print(f"  b_U increments negative at {len(neg)}/{len(rows)} steps "
          f"(predicted negative at {len(negp)}); min increment {min(r['meas'] for r in rows):+.3e}")
    print(f"  b_U: start {bs[0]:+.4f} -> max {bs.max():+.4f} -> end {bs[-1]:+.4f}   "
          f"(end/max = {bs[-1]/max(bs.max(),1e-30):.4f})")
    print(f"  store s.w (for contrast): peak {sw.max():+.4f}@{int(sw.argmax())} "
          f"end {sw[-1]:+.4f}  (end/peak = {sw[-1]/max(sw.max(),1e-30):.4f})")
    # the honest monotonicity statement is about the SMOOTHED (full-population) rate;
    # minibatch noise can make single steps negative. Report both.
    win = 50
    sm = np.convolve(np.diff(bs), np.ones(win) / win, mode="valid")
    print(f"  {win}-step smoothed db_U: negative at {100*(sm<0).mean():.1f}% of windows, "
          f"min {sm.min():+.3e}")
    Aa = np.array([r["A"] for r in rows]); tr = int(Aa.argmin())
    print(f"  A: trough {Aa.min():.3f}@{tr} -> end {Aa[-1]:.3f} "
          f"(recovery fraction f = {(Aa[-1]-Aa.min())/max(1.0-Aa.min(),1e-30):.3f} of the crash)")
    smw = np.convolve(np.diff(sw), np.ones(win) / win, mode="valid")
    print(f"  {win}-step smoothed d(s.w): negative at {100*(smw<0).mean():.1f}% of windows, "
          f"min {smw.min():+.3e}")


if __name__ == "__main__":
    main()
