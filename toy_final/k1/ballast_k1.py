"""Ballast as a knob for the K=1 toy: n_D in {0, 5, 12, 25, 50}.

Nothing in k1.py is edited. Two things there do not survive n_D = 0 and are fixed here:

  1. `k1.configure(nd=0)` is a no-op -- it guards with `if nd:`, and 0 is falsy. We set the
     module globals directly instead.
  2. `k1.pretrain`'s gate calls `k1.accuracy` on an empty ballast set, which is a mean over an
     empty array: NaN plus a RuntimeWarning. (`min(accA, nan)` happens to return accA, so the
     gate would still work by accident; we do not rely on that.) `pretrain` here takes the
     min over the populations that actually exist.

KEY DRAW.  `k1.build` draws A, then D, then B from ONE rng stream, so shrinking n_D would
also change B's keys and labels -- a confound on every n_D column. `build` here always draws
50 ballast keys and truncates to the first n_D, so A and B are bit-identical across the n_D
sweep and n_D = 50 reproduces `k1.build` exactly (asserted in `_selftest`).
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp

import k1

ND_MAX = 50


def configure(d=None, v=None, na=None, nd=None, nb=None):
    """Like k1.configure but 0 is a legal value for every size."""
    if d is not None:
        k1.D = int(d)
    if v is not None:
        k1.V = int(v)
        k1.X = np.arange(0, k1.V // 2)
        k1.Y = np.arange(k1.V // 2, k1.V)
    if na is not None:
        k1.NA = int(na)
    if nd is not None:
        k1.ND = int(nd)
    if nb is not None:
        k1.NB = int(nb)


def build(seed, beta, nd):
    """A / ballast / B, with the ballast truncated rather than re-drawn (see module docstring)."""
    r = np.random.default_rng(seed)
    mu = r.normal(size=k1.D); mu /= np.linalg.norm(mu)

    def keys(n):
        g = r.normal(size=(n, k1.D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(beta) * mu[None] + np.sqrt(1.0 - beta) * g

    kA, vA = keys(k1.NA), k1.X[r.integers(0, len(k1.X), k1.NA)]
    kD, vD = keys(ND_MAX), k1.Y[r.integers(0, len(k1.Y), ND_MAX)]
    kB, vB = keys(k1.NB), k1.Y[r.integers(0, len(k1.Y), k1.NB)]
    return mu, (kA, vA), (kD[:nd], vD[:nd]), (kB, vB)


def acc(p, k, v, norm, gain, restrict=None):
    """k1.accuracy, but an empty population scores 1.0 (vacuously satisfied) instead of NaN."""
    if len(k) == 0:
        return 1.0
    return k1.accuracy(p, k, v, norm, gain, restrict=restrict)


def pretrain(seed, beta, norm, gain, nd, cap=20000):
    """k1.pretrain with the n_D = 0 case handled: pretrain on A (+ ballast if any) to the gate."""
    mu, (kA, vA), (kD, vD), (kB, vB) = build(1000 + seed, beta, nd)
    kp = np.concatenate([kA, kD]) if nd else kA
    vp = np.concatenate([vA, vD]) if nd else vA
    for lr in k1.LR_GRID:
        rng = np.random.default_rng(seed)
        p = k1.init(seed)
        for t in range(cap):
            i = rng.integers(0, len(kp), 32)
            p = k1.step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0, norm, gain)
            if t % 100 == 0 and min(acc(p, kA, vA, norm, gain),
                                    acc(p, kD, vD, norm, gain)) >= k1.GATE:
                return p, lr, t, mu, (kA, vA), (kD, vD), (kB, vB)
    raise SystemExit(f"pretraining never reached {k1.GATE} (nd={nd}, norm={norm}, seed={seed})")


def inject(p0, lr, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr=None):
    """k1.inject with the empty-ballast accuracy patched out. Same rows otherwise."""
    if len(Dd[0]) == 0:
        real_accuracy = k1.accuracy
        k1.accuracy = lambda p, k, v, n, g, restrict=None: (
            1.0 if len(k) == 0 else real_accuracy(p, k, v, n, g, restrict=restrict))
        try:
            return k1.inject(p0, lr, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr=ilr)
        finally:
            k1.accuracy = real_accuracy
    return k1.inject(p0, lr, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr=ilr)


# ----------------------------------------------------------------------------------------
# The shared / individual split of A's margin.
#
# The store is linear in the key and the key splits as k_a = sqrt(b) mu + sqrt(1-b) g_a, so
#     h_a = sqrt(b) [ mu + gain*c_a * (mu W) ]  +  sqrt(1-b) [ g_a + gain*c_a * (g_a W) ]
#           ------------- shared, same for all a ----        ------ individual ------
# with c_a = sqrt(d)/||k_a|| a per-individual scalar. In the linear arm z = h U splits the
# same way, exactly. In the normalized arm z = rms(h) U multiplies both parts by the same
# positive scalar sqrt(d)/||h_a||, so every logit GAP scales and the shared FRACTION of a gap
# is identical in the two arms for the same parameters.
#
# We report three routes into the margin gap:
#   mu_row   sqrt(b) gain c_a (mu W) U       -- the shared write, the hypothesis's quantity
#   mu_res   sqrt(b) mu U                    -- mu carried by the residual key, no W
#   ind      sqrt(1-b) (g_a + gain c_a g_a W) U
# ----------------------------------------------------------------------------------------
def margin_split(p, kA, vA, mu, beta, gain, norm):
    W = np.asarray(p["W"]); U = np.asarray(p["U"])
    kA = np.asarray(kA)
    c = np.sqrt(k1.D) / np.linalg.norm(kA, axis=1)              # (nA,)
    g = (kA - np.sqrt(beta) * mu[None]) / np.sqrt(1.0 - beta)    # (nA, d)

    z_mu_row = np.outer(np.sqrt(beta) * gain * c, (mu @ W) @ U)  # (nA, V)
    z_mu_res = np.tile(np.sqrt(beta) * (mu @ U), (len(kA), 1))
    z_ind = np.sqrt(1.0 - beta) * ((g + gain * c[:, None] * (g @ W)) @ U)
    z = z_mu_row + z_mu_res + z_ind

    scale = np.ones(len(kA))
    if norm:
        h = kA + gain * (c[:, None] * kA) @ W
        scale = np.sqrt(k1.D) / np.linalg.norm(h, axis=1)
    z = z * scale[:, None]
    z_mu_row = z_mu_row * scale[:, None]
    z_mu_res = z_mu_res * scale[:, None]
    z_ind = z_ind * scale[:, None]

    masked = z.copy()
    masked[np.arange(len(vA)), vA] = -np.inf
    best_wrong = masked.argmax(1)
    idx = np.arange(len(vA))

    def gap(zz):
        return zz[idx, vA] - zz[idx, best_wrong]

    tot = gap(z)
    return dict(total=tot, mu_row=gap(z_mu_row), mu_res=gap(z_mu_res), ind=gap(z_ind),
                correct=(z.argmax(1) == vA))


def zero_mu_row(p, mu):
    """W -> (I - mu mu^T) W: the store keeps every direction except the one mu drives."""
    W = np.asarray(p["W"])
    return {"W": jnp.asarray(W - np.outer(mu, mu @ W)), "U": p["U"]}


def _selftest():
    k1.configure(d=128, v=32, na=50, nb=50)
    configure(nd=50)
    mu0, A0, D0, B0 = k1.build(1234, 0.5)
    mu1, A1, D1, B1 = build(1234, 0.5, 50)
    assert np.allclose(mu0, mu1)
    for x, y in ((A0, A1), (D0, D1), (B0, B1)):
        assert np.allclose(x[0], y[0]) and (x[1] == y[1]).all()
    mu2, A2, D2, B2 = build(1234, 0.5, 0)
    assert np.allclose(A2[0], A1[0]) and np.allclose(B2[0], B1[0]) and len(D2[0]) == 0
    print("selftest ok: nd=50 reproduces k1.build; A and B invariant to nd")


if __name__ == "__main__":
    _selftest()
