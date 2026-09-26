"""Two-channel (store / readout) instrumentation for the K=1 toy.

Nothing in k1.py, sweeps.py or ballast_k1.py is edited. Three additions live here:

  1. `step2` -- a step with SEPARATE gradient multipliers on W and U during injection.
     k1.step already has `u_scale`; `w_scale` is the new knob, the store's counterpart.
  2. `inject2` -- k1.inject's row plus the two-channel diagnostics:
        * the crash-carrier counterfactuals  A(W_t, U_0)  and  A(W_0, U_t);
        * the EXACT split of A's mean Y-vs-X class-bias gap into a store part, a
          readout part and a cross term (see `bias_split`);
        * B's mean correct-class margin under all four (W, U) combinations, which
          gives the store / readout shares of B's acquisition;
        * the X-row and Y-row norms of the readout (the "cold rows" question).
  3. `match_gate2` -- sweeps.match_gate with the extra knob threaded through.

THE BIAS SPLIT, exactly.  Let sbar(t) = mean over A of the readout state rms(h_a), and
u_gap(t) = mean_{v in Y} U[:,v] - mean_{v in X} U[:,v].  A's mean class-bias gap is
G(t) = <sbar(t), u_gap(t)>, and

    G(t) - G(0) = <sbar(t)-sbar(0), u_gap(0)>        store   -- the collective state shift
                + <sbar(0), u_gap(t)-u_gap(0)>       readout -- the class bias written into U
                + <sbar(t)-sbar(0), u_gap(t)-u_gap(0)>   cross

with no remainder.  The middle term is the "readout bias" of the derivation in
`wp_readout_deriv.py`; the first is the shared write read through the pretrained readout.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import k1
import ballast_k1 as bk


@partial(jax.jit, static_argnames=("norm",))
def step2(p, k, v, lr, u_scale, w_scale, norm, gain):
    """k1.step with an independent multiplier on the STORE gradient as well."""
    g = jax.grad(k1.loss_fn)(p, k, v, norm, gain)
    return {"W": p["W"] - lr * w_scale * g["W"],
            "U": p["U"] - lr * u_scale * g["U"]}


def _margin(p, kB, vB, norm, gain):
    """mean over B of z[correct] - max over the rest.  Sign gives correctness."""
    z = np.asarray(k1.fwd(p, jnp.asarray(kB), norm, gain)[0])
    i = np.arange(len(vB))
    zz = z.copy(); zz[i, vB] = -np.inf
    return float((z[i, vB] - zz.max(1)).mean())


def _mass_on_X(p, kB, norm, gain):
    """mean over B of the softmax mass still sitting on the OTHER half."""
    z = np.asarray(k1.fwd(p, jnp.asarray(kB), norm, gain)[0])
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); pr = e / e.sum(1, keepdims=True)
    return float(pr[:, k1.X].sum(1).mean())


def _state(p, k, norm, gain):
    _, h = [np.asarray(x) for x in k1.fwd(p, jnp.asarray(k), norm, gain)]
    return np.asarray(k1.rms(jnp.asarray(h))) if norm else h


def margins_A(p, kA, vA, norm, gain):
    """(cross-half headroom, own-half headroom), as in ballast_sweep_nd.margins."""
    z = np.asarray(k1.fwd(p, jnp.asarray(kA), norm, gain)[0])
    i = np.arange(len(vA))
    own = np.full(k1.V, -np.inf); own[k1.X] = 0.0
    oth = np.full(k1.V, -np.inf); oth[k1.Y] = 0.0
    zo = z + own[None, :]; zo[i, vA] = -np.inf
    return (float((z[i, vA] - (z + oth[None, :]).max(1)).mean()),
            float((z[i, vA] - zo.max(1)).mean()))


def u_gap(U):
    return U[:, k1.Y].mean(1) - U[:, k1.X].mean(1)


def gate_of2(p0, B, norm, gain, u_scale, w_scale, ilr, seed, cap=4000, every=5):
    (kB, vB) = B
    p = p0
    rng = np.random.default_rng(seed + 5)
    for t in range(cap + 1):
        if t % every == 0 and k1.accuracy(p, kB, vB, norm, gain) >= k1.GATE:
            return t
        i = rng.integers(0, k1.NB, 32)
        p = step2(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, w_scale, norm, gain)
    return cap


def match_gate2(p0, B, norm, gain, u_scale, w_scale, base_ilr, seed, target,
                iters=14, tol=0.15):
    """sweeps.match_gate, with w_scale threaded through.  The bracket is widened to 2^13
    because the store-only cells (u_scale = 0, w_scale = 0.01) need ~10^3 x the base rate to
    put B on the common clock; the gate is still monotone in the rate there (checked)."""
    lo, hi = -6.0, 13.0
    g0 = gate_of2(p0, B, norm, gain, u_scale, w_scale, base_ilr, seed)
    best = (base_ilr, g0, abs(g0 - target) <= tol * target)
    if best[2]:
        return best
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ilr = base_ilr * 2.0 ** mid
        g = gate_of2(p0, B, norm, gain, u_scale, w_scale, ilr, seed)
        if abs(g - target) < abs(best[1] - target):
            best = (ilr, g, abs(g - target) <= tol * target)
        if best[2]:
            break
        if g > target:
            lo = mid
        else:
            hi = mid
    return best


def inject2(p0, mu, A, Dd, B, norm, gain, u_scale, w_scale, steps, seed, ilr):
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    W0 = jnp.asarray(p0["W"]); U0 = jnp.asarray(p0["U"])
    W0n = np.asarray(W0); U0n = np.asarray(U0)
    postA0 = _state(p0, kA, norm, gain)
    sbar0 = postA0.mean(0)
    ug0 = u_gap(U0n)
    w = ug0 / np.linalg.norm(ug0)                 # the pretrained X -> Y separation direction
    G0 = float(sbar0 @ ug0)
    M_B_00 = _margin(p0, kB, vB, norm, gain)
    uX0 = float(np.linalg.norm(U0n[:, k1.X], axis=0).mean())
    uY0 = float(np.linalg.norm(U0n[:, k1.Y], axis=0).mean())

    p = p0
    rng = np.random.default_rng(seed + 5)
    gi = set(k1.grid(steps)); rows = []
    for t in range(steps + 1):
        if t in gi:
            W = np.asarray(p["W"]); U = np.asarray(p["U"])
            p_WU0 = {"W": p["W"], "U": U0}        # store trained, readout reverted
            p_W0U = {"W": W0, "U": p["U"]}        # readout trained, store reverted
            postA = _state(p, kA, norm, gain)
            sbar = postA.mean(0)
            ug = u_gap(U)
            dsbar = sbar - sbar0; dug = ug - ug0
            dpost = (postA - postA0).mean(0)
            eps = (postA - postA0) - dpost
            dW = W - W0n
            s = mu @ dW
            mc, mo = margins_A(p, kA, vA, norm, gain)
            rows.append(dict(
                step=t,
                A=bk.acc(p, kA, vA, norm, gain),
                A_own=bk.acc(p, kA, vA, norm, gain, restrict=k1.X),
                Dacc=bk.acc(p, kD, vD, norm, gain),
                B=bk.acc(p, kB, vB, norm, gain),
                # --- crash-carrier counterfactuals -------------------------------
                A_storeonly=bk.acc(p_WU0, kA, vA, norm, gain),   # U reverted, W trained
                A_readonly=bk.acc(p_W0U, kA, vA, norm, gain),    # W reverted, U trained
                # --- the exact class-bias split ----------------------------------
                G=float(sbar @ ug), G0=G0,
                G_store=float(dsbar @ ug0),
                G_read=float(sbar0 @ dug),
                G_cross=float(dsbar @ dug),
                # --- the store's shared write ------------------------------------
                s_on_w=float(s @ w),
                s_abs_on_w=float((mu @ W) @ w),
                s_norm=float(np.linalg.norm(s)),
                # --- B's margin under all four (W, U) combinations ----------------
                M_B=_margin(p, kB, vB, norm, gain),
                M_B_store=_margin(p_WU0, kB, vB, norm, gain),
                M_B_read=_margin(p_W0U, kB, vB, norm, gain),
                M_B_0=M_B_00,
                B_store=bk.acc(p_WU0, kB, vB, norm, gain),
                B_read=bk.acc(p_W0U, kB, vB, norm, gain),
                PX=_mass_on_X(p, kB, norm, gain),
                # --- movement ----------------------------------------------------
                dU_rel=float(np.linalg.norm(U - U0n) / np.linalg.norm(U0n)),
                dW_rel=float(np.linalg.norm(dW) / max(np.linalg.norm(W0n), 1e-30)),
                dW=float(np.linalg.norm(dW)),
                uX=float(np.linalg.norm(U[:, k1.X], axis=0).mean()),
                uY=float(np.linalg.norm(U[:, k1.Y], axis=0).mean()),
                uX0=uX0, uY0=uY0,
                m_cross=mc, m_own=mo,
                delta=float(np.linalg.norm(dpost)),
                eps=float(np.sqrt((eps ** 2).sum(-1)).mean()),
                post_norm=float(np.linalg.norm(postA, axis=-1).mean()),
                a_frac=float(np.linalg.norm(dpost) / np.linalg.norm(postA, axis=-1).mean()),
            ))
        i = rng.integers(0, k1.NB, 32)
        p = step2(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, w_scale, norm, gain)
    return rows, p


def summarise2(rows):
    st = np.array([r["step"] for r in rows])
    A = np.array([r["A"] for r in rows]); B = np.array([r["B"] for r in rows])
    tr = int(A.argmin())
    bg = int(st[np.argmax(B >= k1.GATE)]) if (B >= k1.GATE).any() else -1
    # attribution at B's gate (or at the last checkpoint if the gate is never reached)
    gi = int(np.argmax(B >= k1.GATE)) if (B >= k1.GATE).any() else len(rows) - 1
    r = rows[gi]
    tot = r["M_B"] - r["M_B_0"]
    st_sh = (r["M_B_store"] - r["M_B_0"]) / tot if abs(tot) > 1e-12 else float("nan")
    rd_sh = (r["M_B_read"] - r["M_B_0"]) / tot if abs(tot) > 1e-12 else float("nan")
    depth = float(A[0] - A[tr])
    rec = float(A[tr:].max() - A[tr])
    return dict(
        A0=float(A[0]), trough=float(A[tr]), trough_step=int(st[tr]),
        depth=depth, recovery=rec,
        rec_frac=float(rec / depth) if depth > 1e-9 else float("nan"),
        end=float(A[-1]), B_end=float(B[-1]), B_gate=bg,
        gate_idx=gi, gate_step=int(st[gi]),
        store_share=float(st_sh), read_share=float(rd_sh),
        M_B_gain=float(tot),
        A_storeonly_min=float(min(x["A_storeonly"] for x in rows)),
        A_readonly_min=float(min(x["A_readonly"] for x in rows)),
        A_storeonly_at_trough=float(rows[tr]["A_storeonly"]),
        A_readonly_at_trough=float(rows[tr]["A_readonly"]),
        G_store_trough=float(rows[tr]["G_store"]),
        G_read_trough=float(rows[tr]["G_read"]),
        G_read_end=float(rows[-1]["G_read"]),
        G_store_end=float(rows[-1]["G_store"]),
        s_on_w_peak=float(max(x["s_on_w"] for x in rows)),
        s_on_w_end=float(rows[-1]["s_on_w"]),
        s_abs_on_w_0=float(rows[0]["s_abs_on_w"]),
        s_abs_on_w_end=float(rows[-1]["s_abs_on_w"]),
        dU_rel=float(rows[-1]["dU_rel"]), dW_rel=float(rows[-1]["dW_rel"]),
        uX0=float(rows[0]["uX0"]), uY0=float(rows[0]["uY0"]),
        uX_end=float(rows[-1]["uX"]), uY_end=float(rows[-1]["uY"]),
        A_own_end=float(rows[-1]["A_own"]), eps_end=float(rows[-1]["eps"]),
    )
