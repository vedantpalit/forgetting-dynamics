"""The K=1 toy: a linear store, a scale-invariant readout, a softmax.

    h = k + k W                 W: (d, d)   the store, LINEAR in the key
    z = rms(h) U                U: (d, V)   the readout; rms(x) = x/||x|| * sqrt(d)
    p = softmax(z)

`--norms 0` drops the rms and gives the Zucchet toy exactly (keys -> one matrix -> softmax).
`--norms 1` adds the ONLY nonlinearity in the model. That one change is what turns the
monotone collective write into a reversal; everything else is identical.

WHY K=1 AND FINAL-NORM-ONLY. A per-block norm at K=1 reads the key, which never changes
during training, so it is a fixed per-individual rescale absorbed into W -- inert by
construction. Dropping it makes the model the one the hand derivation is written for.

KEYS.  k_i = sqrt(beta) mu + sqrt(1-beta) g_i,  g_i a random unit vector, mu shared by all.
beta is the single knob for how related the populations are. beta = 0 is three unrelated
populations; beta -> 1 makes every individual the same key.

READOUT LEARNING RATE.  The transformer unembedding rows move 0.51% of their own norm
during injection (FINDINGS 3.21.4). A toy whose readout moves 35% is not a model of it, so
--u_scales multiplies the U gradient DURING INJECTION ONLY (pretraining is untouched,
exactly as the transformer readout is built by pretraining and then barely moves).
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import json
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

D, V = 128, 32
NA = ND = NB = 50


def set_d(d):
    """Rebind the key/residual dimension. Call BEFORE any jitted function is traced, and use
    one process per d: the jitted functions close over D at trace time."""
    global D
    D = int(d)


STORE_SCALE = None            # None: store input is rms(k). A float c: store input is c * k.


def configure(d=None, v=None, na=None, nd=None, nb=None, store_scale=None):
    """Rebind any of the problem sizes. Same rule as set_d: one process per configuration,
    called before the first traced call. X and Y are recomputed from V.

    `store_scale`: replace the store's input normalizer rms(k) = sqrt(d) k/||k|| with a plain
    constant, c * k (the string "sqrt_d" gives c = sqrt(d), the scale rms would produce on a
    unit key). Since the keys are within O(1/sqrt(d)) of unit norm this differs from rms(k)
    only by that per-individual factor; the check that the two agree is the ii figure."""
    global D, V, NA, ND, NB, X, Y, STORE_SCALE
    if store_scale is not None:
        STORE_SCALE = float(np.sqrt(D if not d else int(d))) if store_scale == "sqrt_d" else float(store_scale)
    if d:
        D = int(d)
    if v:
        V = int(v)
        X = np.arange(0, V // 2)
        Y = np.arange(V // 2, V)
    if na:
        NA = int(na)
    if nd:
        ND = int(nd)
    if nb:
        NB = int(nb)
X = np.arange(0, V // 2)
Y = np.arange(V // 2, V)
GATE = 0.99
INJECT_RATIO = 13.33          # the transformer pretrain:inject learning-rate ratio
LR_GRID = [0.5, 0.3, 0.2, 0.1, 0.05]


def rms(h):
    return h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)


def build(seed, beta):
    """A and the ballast D are pretrained; B is injected. All three share mu."""
    r = np.random.default_rng(seed)
    mu = r.normal(size=D); mu /= np.linalg.norm(mu)

    def keys(n):
        g = r.normal(size=(n, D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(beta) * mu[None] + np.sqrt(1.0 - beta) * g

    return (mu,
            (keys(NA), X[r.integers(0, len(X), NA)]),
            (keys(ND), Y[r.integers(0, len(Y), ND)]),
            (keys(NB), Y[r.integers(0, len(Y), NB)]))


def init(seed):
    r = np.random.default_rng(seed + 77)
    return {"W": jnp.asarray(r.normal(size=(D, D)) * 0.02 / np.sqrt(D)),
            "U": jnp.asarray(r.normal(size=(D, V)) / np.sqrt(D))}


@partial(jax.jit, static_argnames=("norm",))
def fwd(p, k, norm, gain=1.0):
    """h = k + gain * rms(k) W.  At K = 1 the store input is the KEY, which never changes
    during training, so rms(k) is a fixed per-individual rescale and the store stays exactly
    linear. Its only role is to set the SIZE of the write relative to the key, i.e. how much
    of the readout state the collective shift occupies. That fraction is a precondition for
    the reversal, not an architectural choice: the force removing the shift is
    -(margin / ||h||) h_hat, so it acts on the shift only in proportion to the shift's share
    of h. The transformer sits at 82% (FINDINGS 3.21.5); gain = 1 here puts the toy in the
    same regime, gain = 1/sqrt(d) puts it in the weak-write regime where nothing reverses."""
    h = k + gain * (STORE_SCALE * k if STORE_SCALE is not None else rms(k)) @ p["W"]
    z = (rms(h) if norm else h) @ p["U"]
    return z, h


def loss_fn(p, k, v, norm, gain):
    z, _ = fwd(p, k, norm, gain)
    return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])


@partial(jax.jit, static_argnames=("norm",))
def step(p, k, v, lr, u_scale, norm, gain):
    g = jax.grad(loss_fn)(p, k, v, norm, gain)
    return {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]}


def accuracy(p, k, v, norm, gain, restrict=None):
    z = np.asarray(fwd(p, jnp.asarray(k), norm, gain)[0])
    if restrict is not None:
        m = np.full(V, -np.inf); m[restrict] = 0.0
        z = z + m[None, :]
    return float((z.argmax(-1) == v).mean())


def pretrain(seed, beta, norm, gain, cap=20000):
    """LR probe, then pretrain A + ballast to the gate. Raises if no LR reaches it."""
    mu, (kA, vA), (kD, vD), (kB, vB) = build(1000 + seed, beta)
    kp = np.concatenate([kA, kD]); vp = np.concatenate([vA, vD])
    for lr in LR_GRID:
        rng = np.random.default_rng(seed)
        p = init(seed)
        for t in range(cap):
            i = rng.integers(0, len(kp), 32)
            p = step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0, norm, gain)
            if t % 100 == 0 and min(accuracy(p, kA, vA, norm, gain),
                                    accuracy(p, kD, vD, norm, gain)) >= GATE:
                return p, lr, t, mu, (kA, vA), (kD, vD), (kB, vB)
    raise SystemExit(f"pretraining never reached {GATE} (beta={beta}, norm={norm}, seed={seed})")


def grid(steps):
    g = set(range(0, 201, 5))
    g |= {int(x) for x in np.unique(np.round(np.logspace(np.log10(200), np.log10(steps), 60)))}
    return sorted(s for s in g if s <= steps)


def inject(p0, lr, mu, A, Dd, B, norm, gain, u_scale, steps, seed, ilr=None):
    """Fine-tune on B alone. Returns one row per checkpoint.

    `ilr` overrides the default lr/INJECT_RATIO. Gate-matched sweeps use it: they bisect the
    injection rate per configuration so B reaches 0.99 at the same step everywhere, which is
    what makes crash depths comparable across columns."""
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
    _, hA0 = [np.asarray(x) for x in fwd(p0, jnp.asarray(kA), norm, gain)]
    postA0 = np.asarray(rms(jnp.asarray(hA0))) if norm else hA0
    w = U0[:, Y].mean(1) - U0[:, X].mean(1)
    w /= np.linalg.norm(w)                       # the X -> Y separation direction

    p = p0
    rng = np.random.default_rng(seed + 5)
    if ilr is None:
        ilr = lr / INJECT_RATIO
    gi = set(grid(steps)); rows = []
    for t in range(steps + 1):
        if t in gi:
            _, hA = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), norm, gain)]
            postA = np.asarray(rms(jnp.asarray(hA))) if norm else hA
            W = np.asarray(p["W"]); U = np.asarray(p["U"]); dW = W - W0
            s = mu @ dW                                   # the collective write
            dpost = (postA - postA0).mean(0)
            eps = (postA - postA0) - dpost
            rows.append(dict(
                step=t,
                A=accuracy(p, kA, vA, norm, gain),
                A_own=accuracy(p, kA, vA, norm, gain, restrict=X),
                Dacc=accuracy(p, kD, vD, norm, gain), B=accuracy(p, kB, vB, norm, gain),
                s_norm=float(np.linalg.norm(s)),
                s_share=float(np.linalg.norm(s) ** 2 / max(np.linalg.norm(dW) ** 2, 1e-30)),
                s_on_w=float(s @ w),                      # the write X->Y content
                dW=float(np.linalg.norm(dW)),
                dU_rel=float(np.linalg.norm(U - U0) / np.linalg.norm(U0)),
                delta=float(np.linalg.norm(dpost)),
                eps=float(np.sqrt((eps ** 2).sum(-1)).mean()),
                # norms are d-dependent, so log the state's own norm and report the two
                # damage channels as FRACTIONS of it when comparing across d.
                post_norm=float(np.linalg.norm(postA, axis=-1).mean()),
                # the shift's share of the readout state: the transformer sits at 0.82
                a_frac=float(np.linalg.norm(dpost) / np.linalg.norm(postA, axis=-1).mean()),
            ))
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, norm, gain)
    return rows, p


def run(seed, beta, norm, u_scale, steps, gain=1.0):
    p0, lr, gate, mu, A, Dd, B = pretrain(seed, beta, norm, gain)
    rows, _ = inject(p0, lr, mu, A, Dd, B, norm, gain, u_scale, steps, seed)
    return dict(seed=seed, beta=beta, norm=norm, u_scale=u_scale, gain=gain,
                pretrain_lr=lr, pretrain_gate=gate, rows=rows)


def summarise(r):
    rows = r["rows"]
    st = np.array([x["step"] for x in rows])
    A = np.array([x["A"] for x in rows]); B = np.array([x["B"] for x in rows])
    s = np.array([x["s_norm"] for x in rows])
    af = np.array([x["a_frac"] for x in rows])
    # PRIMARY OBSERVABLE: the write's content along the X -> Y value separation. ||s|| is the
    # wrong one -- it also counts directions that move A's state without moving any logit gap
    # between the halves, so a write that has stopped hurting A can still grow in norm.
    # s_on_w is exactly the per-class bias the derivation is about.
    sw = np.array([x["s_on_w"] for x in rows])
    tr = int(A.argmin()); pk = int(s.argmax()); pw = int(sw.argmax())
    bgate = int(st[np.argmax(B >= GATE)]) if (B >= GATE).any() else -1
    return dict(trough=float(A[tr]), trough_step=int(st[tr]),
                after=float(A[tr:].max()), end=float(A[-1]),
                recovery=float(A[tr:].max() - A[tr]),
                s_peak=float(s[pk]), s_peak_step=int(st[pk]), s_end=float(s[-1]),
                s_drop=float(1 - s[-1] / s[pk]) if s[pk] > 0 else 0.0,
                sw_peak=float(sw[pw]), sw_peak_step=int(st[pw]), sw_end=float(sw[-1]),
                sw_drop=float(1 - sw[-1] / sw[pw]) if sw[pw] > 1e-9 else 0.0,
                B_at_sw_peak=float(B[pw]),
                B_at_s_peak=float(B[pk]), B_gate=bgate, a_peak=float(af.max()),
                dU_rel=float(rows[-1]["dU_rel"]), A_own_end=float(rows[-1]["A_own"]),
                eps_end=float(rows[-1]["eps"]), D_end=float(rows[-1]["Dacc"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="0.5")
    ap.add_argument("--norms", default="0,1")
    ap.add_argument("--u_scales", default="1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gains", default="1.0")
    ap.add_argument("--d", type=int, default=0, help="override the key dimension")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.d:
        set_d(a.d)

    out = {}
    for beta in [float(x) for x in a.betas.split(",")]:
        for norm in [int(x) for x in a.norms.split(",")]:
            for gn in [float(x) for x in a.gains.split(",")]:
                for us in [float(x) for x in a.u_scales.split(",")]:
                    for seed in [int(x) for x in a.seeds.split(",")]:
                        r = run(seed, beta, norm, us, a.steps, gain=gn)
                        q = summarise(r)
                        r["d"] = D
                        out[f"b{beta}-n{norm}-g{gn}-u{us}-s{seed}"] = r
                        print(f"d={D} beta={beta} norm={norm} gain={gn} u={us} seed={seed} "
                              f"lr={r['pretrain_lr']} | A trough {q['trough']:.2f}@{q['trough_step']} "
                              f"-> after {q['after']:.2f} (rec {q['recovery']:+.2f}) end {q['end']:.2f} "
                              f"| s.w peak {q['sw_peak']:.3f}@{q['sw_peak_step']} "
                              f"end {q['sw_end']:.3f} ({-100 * q['sw_drop']:+.0f}%) "
                              f"B there {q['B_at_sw_peak']:.2f} | Bgate {q['B_gate']} "
                              f"| dU/U {q['dU_rel']:.4f} | a {q['a_peak']:.2f} "
                              f"| eps {q['eps_end']:.2f} | Aown {q['A_own_end']:.2f}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
