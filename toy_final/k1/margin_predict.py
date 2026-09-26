"""Open item 1 of THEORY.md section 10: turn eq. (6) into a zero-parameter prediction.

Eq. (6) says an old individual a fails iff its pretrained margin m_a is smaller than the
per-class bias gap it receives.  The bias is IDENTICAL for every a (it is `delta U` in the
linear arm), so the entire crash is a threshold on the margin DISTRIBUTION.  Nothing has
been logged to test that; this does.

At pretrain end, for every A individual:
    m_a        correct logit minus the best wrong logit over the FULL vocabulary
    m_own      correct logit minus the best wrong logit inside X (a's own half)
    m_cross    correct logit minus the best logit in Y (the half B writes toward)

At every injection checkpoint, with z_a(0) the pretrained logits and z_a(t) the current ones:
    dz_bar     mean_a (z_a(t) - z_a(0))          the EMPIRICAL per-class bias
    actual     argmax z_a(t)                == y_a
    const      argmax (z_a(0) + dz_bar)      == y_a     the threshold / bias-only model
    vary       argmax (z_a(0) + dz_a - dz_bar) == y_a   the individual-component-only model
    thresh     1 - fraction(m_cross_a < gap(t)),  gap(t) = max_Y dz_bar - dz_bar[y_a]

In the linear arm dz_bar equals `delta U` exactly when U is frozen; U is NOT frozen (u_scale
= 0.01), so `bias_dev` logs how far dz_bar is from delta @ U(t).  That is reported, not hidden.

One process per configuration: k1's jitted functions close over D at trace time.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import json

import numpy as np
import jax.numpy as jnp

import k1
import sweeps


def margins(z, v, X, Y):
    """Full-vocabulary, own-half and cross-half margins for each row of z."""
    n, V = z.shape
    correct = z[np.arange(n), v]
    zf = z.copy(); zf[np.arange(n), v] = -np.inf
    m_full = correct - zf.max(1)
    own = np.full(V, -np.inf); own[X] = 0.0
    m_own = correct - (zf + own[None]).max(1)
    cross = np.full(V, -np.inf); cross[Y] = 0.0
    m_cross = correct - (z + cross[None]).max(1)
    return m_full, m_own, m_cross


def track(p0, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr):
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    X, Y = k1.X, k1.Y
    kAj = jnp.asarray(kA)
    z0, h0 = [np.asarray(x) for x in k1.fwd(p0, kAj, norm, gain)]
    m_full, m_own, m_cross = margins(z0, vA, X, Y)
    U0 = np.asarray(p0["U"])

    p = p0
    rng = np.random.default_rng(seed + 5)
    gi = set(k1.grid(steps))
    rows = []
    for t in range(steps + 1):
        if t in gi:
            z, h = [np.asarray(x) for x in k1.fwd(p, kAj, norm, gain)]
            dz = z - z0
            dzb = dz.mean(0)
            ok_act = (z.argmax(1) == vA)
            ok_con = ((z0 + dzb[None]).argmax(1) == vA)
            ok_var = ((z0 + dz - dzb[None]).argmax(1) == vA)
            # eq. (6)'s gap, per individual: best Y bias minus the bias on a's own class
            gap_a = dzb[Y].max() - dzb[vA]
            gap_mean = float(dzb[Y].mean() - dzb[X].mean())
            thresh = float((m_cross > gap_a).mean())
            # size of eq. (6)'s eps-dependent term, in the SAME units as m_a and gap:
            # the per-individual jitter in the critical logit gap (correct vs the class the
            # bias-only model says is a's strongest competitor).
            zc = z0 + dzb[None]
            zc_masked = zc.copy(); zc_masked[np.arange(len(vA)), vA] = -np.inf
            vstar = zc_masked.argmax(1)
            dev = dz - dzb[None]
            ar = np.arange(len(vA))
            dev_gap = dev[ar, vstar] - dev[ar, vA]
            # linear-arm identity check: is dz_bar really delta @ U ?
            U = np.asarray(p["U"])
            post = np.asarray(k1.rms(jnp.asarray(h))) if norm else h
            post0 = np.asarray(k1.rms(jnp.asarray(h0))) if norm else h0
            delta = (post - post0).mean(0)
            bias_dev = float(np.abs(delta @ U - dzb).max())
            bias_dev0 = float(np.abs(delta @ U0 - dzb).max())
            rows.append(dict(
                step=t,
                A=float(ok_act.mean()), A_const=float(ok_con.mean()),
                A_vary=float(ok_var.mean()), A_thresh=thresh,
                B=k1.accuracy(p, kB, vB, norm, gain),
                Dacc=k1.accuracy(p, kD, vD, norm, gain),
                A_own=k1.accuracy(p, kA, vA, norm, gain, restrict=X),
                gap_max=float(gap_a.mean()), gap_mean=gap_mean,
                dev_gap_std=float(dev_gap.std()), dev_gap_mean=float(dev_gap.mean()),
                dev_std=float(dev.std()),
                m_slack=float(np.mean(np.array(m_cross) - gap_a)),
                bias_dev=bias_dev, bias_dev_U0=bias_dev0,
                dU_rel=float(np.linalg.norm(U - U0) / np.linalg.norm(U0)),
                ok=ok_act.astype(int).tolist(),
                ok_const=ok_con.astype(int).tolist(),
            ))
        i = rng.integers(0, len(kB), 32)
        p = k1.step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm, gain)
    return rows, dict(m_full=m_full.tolist(), m_own=m_own.tolist(),
                      m_cross=m_cross.tolist(), vA=np.asarray(vA).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--na", type=int, default=50)
    ap.add_argument("--nb", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--u_scale", type=float, default=0.01)
    ap.add_argument("--norms", default="1,0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--match_gate", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k1.configure(d=a.d, v=a.vocab, na=a.na, nd=a.na, nb=a.nb)
    out = {}
    for norm in [int(x) for x in a.norms.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, a.beta, norm, a.gain)
            base = lr / k1.INJECT_RATIO
            if a.match_gate:
                ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, a.gain, a.u_scale,
                                               base, seed, a.match_gate)
            else:
                ilr, g, ok = base, -1, True
            rows, marg = track(p0, mu, A, Dd, B, norm, a.gain, a.u_scale, a.steps, seed, ilr)
            key = f"{a.tag}-n{norm}-s{seed}"
            out[key] = dict(tag=a.tag, norm=norm, seed=seed, d=k1.D, v=k1.V, na=k1.NA,
                            nb=k1.NB, beta=a.beta, gain=a.gain, u_scale=a.u_scale,
                            pretrain_lr=lr, pretrain_gate=pg, inject_lr=ilr,
                            matched=bool(ok), probe_gate=g, rows=rows, margins=marg)
            Aa = np.array([r["A"] for r in rows])
            tr = int(Aa.argmin())
            mf = np.array(marg["m_full"])
            print(f"{key} d={k1.D} V={k1.V} nA={k1.NA} | ilr {ilr:.2e} gate {g} "
                  f"| m_full mean {mf.mean():.2f} p10 {np.percentile(mf, 10):.2f} "
                  f"p50 {np.percentile(mf, 50):.2f} p90 {np.percentile(mf, 90):.2f} "
                  f"| trough {Aa[tr]:.2f}@{rows[tr]['step']} const {rows[tr]['A_const']:.2f} "
                  f"vary {rows[tr]['A_vary']:.2f} thresh {rows[tr]['A_thresh']:.2f} "
                  f"| end {Aa[-1]:.2f} const {rows[-1]['A_const']:.2f} "
                  f"vary {rows[-1]['A_vary']:.2f} | biasdev {rows[tr]['bias_dev']:.2e}",
                  flush=True)
            json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out} ({len(out)} runs)")


if __name__ == "__main__":
    main()
