"""A K-block residual stack, to ask whether DE-COHERENCE is what depth buys.

The K=1 toy recovers by pure shrinkage: one shared write grows and is then withdrawn along
its own direction. The transformer recovers differently -- the per-block contributions P_j to
the shift on A keep or GROW their norms while their pairwise cosines fall (0.49 -> 0.23), so
the SUM shrinks by de-coherence. This script builds the intermediate object.

    h_0     = k
    h_{j+1} = h_j + g * rms(h_j) W_j        j = 0 .. K-1     (norm arm, --norm 1)
    h_{j+1} = h_j + g * h_j     W_j                          (linear arm, --norm 0, no norms anywhere)
    z       = rms(h_K) U   /   h_K U
    --norm 2: per-block rms kept, FINAL rms removed (z = h_K U). The toy version of the
    transformer's no-final-norm ablation (src/experiments/nofinalnorm_*): the transformer
    still recovered without its final LayerNorm (0.32 -> 0.67, 3 seeds), so this asks
    whether per-block normalizers alone produce a recovery in the stack.

W_j init N(0, 0.02^2/d) -- NOT zero, which would keep every block identical forever.
U is trained at the full rate during pretraining and at `u_scale` (0.01, the transformer's
readout movement) during injection, exactly as in k1.py.

Injection rate is GATE MATCHED (bisection, reusing sweeps.match_gate's logic) so B reaches
0.99 at step 200 in every cell: the crash is caused by B's acquisition, so unmatched columns
differ in dose, not in geometry.

Per-block contributions to A's shift are measured BOTH ways:
    direct    P_j^dir = mean_a[ g rms(h_j,a) W_j ]  minus its step-0 value  (residual space)
    knockout  P_j^ko  = mean_a[ post(theta_t) - post(theta_t | W_j <- W_j(0)) ]
                                                             (post-final-norm space; this is
                                                              what the transformer measured)

    python wp_depth_stack.py --K 4 --seeds 0,1,2 --out wp_depth_K4.json
"""
import os; os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import json
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

D, V, NA, ND, NB, BETA = 128, 32, 50, 50, 50, 0.5
X = np.arange(0, V // 2)
Y = np.arange(V // 2, V)
GATE = 0.99
INJECT_RATIO = 13.33
LR_GRID = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


def rms(h):
    return h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)


def build(seed):
    r = np.random.default_rng(seed)
    mu = r.normal(size=D); mu /= np.linalg.norm(mu)

    def keys(n):
        g = r.normal(size=(n, D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(BETA) * mu[None] + np.sqrt(1.0 - BETA) * g

    return (mu, (keys(NA), X[r.integers(0, len(X), NA)]),
            (keys(ND), Y[r.integers(0, len(Y), ND)]),
            (keys(NB), Y[r.integers(0, len(Y), NB)]))


def init(seed, K):
    r = np.random.default_rng(seed + 77)
    return {"W": jnp.asarray(r.normal(size=(K, D, D)) * 0.02 / np.sqrt(D)),
            "U": jnp.asarray(r.normal(size=(D, V)) / np.sqrt(D))}


@partial(jax.jit, static_argnames=("K", "norm"))
def fwd(p, k, K, norm, gain):
    h = k
    P, H = [], []
    for j in range(K):
        H.append(h)
        inp = rms(h) if norm else h            # norm 1 and 2: per-block input normalizer
        pj = gain * (inp @ p["W"][j])
        P.append(pj)
        h = h + pj
    post = rms(h) if norm == 1 else h       # norm 2: no final normalizer
    return post @ p["U"], jnp.stack(P), post, jnp.stack(H)


def loss_fn(p, k, v, K, norm, gain):
    z = fwd(p, k, K, norm, gain)[0]
    return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])


@partial(jax.jit, static_argnames=("K", "norm"))
def step(p, k, v, lr, u_scale, K, norm, gain):
    g = jax.grad(loss_fn)(p, k, v, K, norm, gain)
    return {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]}


def accuracy(p, k, v, K, norm, gain, restrict=None):
    if len(k) == 0:
        return 1.0
    z = np.asarray(fwd(p, jnp.asarray(k), K, norm, gain)[0])
    if restrict is not None:
        m = np.full(V, -np.inf); m[restrict] = 0.0
        z = z + m[None, :]
    return float((z.argmax(-1) == v).mean())


def pretrain(seed, K, norm, gain, cap=20000):
    mu, (kA, vA), (kD, vD), (kB, vB) = build(1000 + seed)
    kp = np.concatenate([kA, kD]); vp = np.concatenate([vA, vD])
    for lr in LR_GRID:
        rng = np.random.default_rng(seed)
        p = init(seed, K)
        for t in range(cap):
            i = rng.integers(0, len(kp), 32)
            p = step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0, K, norm, gain)
            if t % 100 == 0 and min(accuracy(p, kA, vA, K, norm, gain),
                                    accuracy(p, kD, vD, K, norm, gain)) >= GATE:
                return p, lr, t, mu, (kA, vA), (kD, vD), (kB, vB)
    raise SystemExit(f"pretraining never reached {GATE} (K={K} norm={norm} seed={seed})")


def gate_of(p0, B, K, norm, gain, u_scale, ilr, seed, cap=1200, every=5):
    kB, vB = B
    p = p0
    rng = np.random.default_rng(seed + 5)
    for t in range(cap + 1):
        if t % every == 0 and accuracy(p, kB, vB, K, norm, gain) >= GATE:
            return t
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, K, norm, gain)
    return cap


def match_gate(p0, B, K, norm, gain, u_scale, base_ilr, seed, target=200, iters=11, tol=0.15):
    """Bisect the injection rate in log space so B's gate lands on `target` (sweeps.py)."""
    lo, hi = -6.0, 6.0
    best = (base_ilr, gate_of(p0, B, K, norm, gain, u_scale, base_ilr, seed), False)
    if abs(best[1] - target) <= tol * target:
        return best[0], best[1], True
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ilr = base_ilr * 2.0 ** mid
        g = gate_of(p0, B, K, norm, gain, u_scale, ilr, seed)
        if abs(g - target) < abs(best[1] - target):
            best = (ilr, g, abs(g - target) <= tol * target)
        if best[2]:
            break
        if g > target:
            lo = mid
        else:
            hi = mid
    return best


def grid(steps):
    g = set(range(0, 301, 5))
    g |= {int(x) for x in np.unique(np.round(np.logspace(np.log10(300), np.log10(steps), 50)))}
    return sorted(s for s in g if s <= steps)


def unit(v, axis=-1):
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), 1e-30, None)


def inject(p0, K, norm, gain, u_scale, ilr, A, Dd, B, steps, seed):
    (kA, vA), (kD, vD), (kB, vB) = A, Dd, B
    W0 = np.asarray(p0["W"])
    _, P0, post0, H0 = [np.asarray(x) for x in fwd(p0, jnp.asarray(kA), K, norm, gain)]
    P0m = P0.mean(1)                 # (K, d)   block contribution, A-mean, at step 0
    post0m = post0.mean(0)

    def postdecomp(P, H, k):
        """The EXACTLY additive post-final-norm decomposition.

            post_a = c_a (k_a + sum_j P_{j,a}),   c_a = sqrt(d)/||h_K,a||   (c_a = 1 linear)

        so the A-mean post-norm state splits with no remainder into one term per block plus a
        key term, and every such term is a genuine share of `delta` once its step-0 value is
        subtracted. The knockout is a marginal effect and does NOT sum to delta; this does."""
        hK = H[-1] + P[-1]
        c = (np.sqrt(D) / np.linalg.norm(hK, axis=-1)) if norm == 1 else np.ones(len(hK))
        return (P * c[None, :, None]).mean(1), (k * c[:, None]).mean(0)

    Ppost0, key0 = postdecomp(P0, H0, kA)
    U0 = np.asarray(p0["U"])
    wdir = U0[:, Y].mean(1) - U0[:, X].mean(1)
    wdir /= np.linalg.norm(wdir)     # the X -> Y readout separation (the per-class bias axis)

    p = p0
    rng = np.random.default_rng(seed + 5)
    gi = set(grid(steps))
    rec = dict(step=[], A=[], A_own=[], B=[], Dacc=[], delta=[], eps=[], post_norm=[],
               delta_on_w=[], dcos=[], drecon=[], dvec=[], Pdir=[], Pko=[], hhatA=[],
               hhatB=[], dW=[], blk_norm=[], h_norm=[], Ppost=[], keyterm=[])
    for t in range(steps + 1):
        if t in gi:
            _, P, post, H = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), K, norm, gain)]
            _, _, _, HB = [np.asarray(x) for x in fwd(p, jnp.asarray(kB), K, norm, gain)]
            postm = post.mean(0)
            dpost = postm - post0m
            epsv = (post - post0) - dpost[None, :]
            W = np.asarray(p["W"])
            Pko = np.zeros((K, D))
            for j in range(K):
                pj = {"W": jnp.asarray(np.concatenate(
                    [W[:j], W0[j:j + 1], W[j + 1:]], axis=0)), "U": p["U"]}
                post_ko = np.asarray(fwd(pj, jnp.asarray(kA), K, norm, gain)[2]).mean(0)
                Pko[j] = postm - post_ko
            rec["step"].append(t)
            rec["A"].append(accuracy(p, kA, vA, K, norm, gain))
            rec["A_own"].append(accuracy(p, kA, vA, K, norm, gain, restrict=X))
            rec["B"].append(accuracy(p, kB, vB, K, norm, gain))
            rec["Dacc"].append(accuracy(p, kD, vD, K, norm, gain))
            rec["delta"].append(float(np.linalg.norm(dpost)))
            rec["eps"].append(float(np.sqrt((epsv ** 2).sum(-1)).mean()))
            rec["post_norm"].append(float(np.linalg.norm(post, axis=-1).mean()))
            rec["delta_on_w"].append(float(dpost @ wdir))
            # how well the knockouts reconstruct the shift they are supposed to decompose
            sko = Pko.sum(0)
            rec["dcos"].append(float(sko @ dpost /
                                     max(np.linalg.norm(sko) * np.linalg.norm(dpost), 1e-30)))
            rec["drecon"].append(float(np.linalg.norm(sko) / max(np.linalg.norm(dpost), 1e-30)))
            rec["dvec"].append(dpost)
            Ppost, keyt = postdecomp(P, H, kA)
            rec["Ppost"].append(Ppost - Ppost0)
            rec["keyterm"].append(float(np.linalg.norm(keyt - key0)))
            rec["Pdir"].append(P.mean(1) - P0m)
            rec["Pko"].append(Pko)
            rec["hhatA"].append(unit(H, axis=-1).mean(1))      # (K, d)
            rec["hhatB"].append(unit(HB, axis=-1).mean(1))
            # the write's size relative to the state THE BLOCK READS, per block
            rec["blk_norm"].append(np.linalg.norm(P, axis=-1).mean(1))
            rec["h_norm"].append(np.linalg.norm(H, axis=-1).mean(1))
            rec["dW"].append([float(np.linalg.norm(W[j] - W0[j])) for j in range(K)])
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, u_scale, K, norm, gain)
    rec["dvec"] = np.stack(rec["dvec"])
    rec["blk_norm"] = np.stack(rec["blk_norm"]); rec["h_norm"] = np.stack(rec["h_norm"])
    for key in ("Pdir", "Pko", "Ppost", "hhatA", "hhatB"):
        rec[key] = np.stack(rec[key])                          # (T, K, d)
    return rec, p


def coh(P):
    """Norms, mean pairwise cosine and the norm of the sum, per checkpoint. P: (T, K, d)."""
    T, K, _ = P.shape
    n = np.linalg.norm(P, axis=-1)                              # (T, K)
    U = P / np.clip(n[:, :, None], 1e-30, None)
    G = np.einsum("tkd,tld->tkl", U, U)                         # (T, K, K)
    iu = np.triu_indices(K, 1)
    mc = G[:, iu[0], iu[1]].mean(1) if K > 1 else np.zeros(T)
    s = np.linalg.norm(P.sum(1), axis=-1)
    return n, G, mc, s


def sum_from(n, G):
    """||sum_j P_j|| rebuilt from norms n (K,) and cosine Gram G (K,K)."""
    return float(np.sqrt(max(n @ G @ n, 0.0)))


def share(n_pk, G_pk, n_rc, G_rc):
    """Split the fall of ||sum P|| into a norm part and a de-coherence part.

    counterfactual 'cos frozen at peak'   -> norms moved, cosines held: the NORM channel
    counterfactual 'norms frozen at peak' -> cosines moved, norms held: the DE-COHERENCE one
    Shares are each channel's fall over the actual fall; they need not sum to 1 (interaction).
    """
    s_pk, s_rc = sum_from(n_pk, G_pk), sum_from(n_rc, G_rc)
    fall = s_pk - s_rc
    s_cosfrozen = sum_from(n_rc, G_pk)
    s_normfrozen = sum_from(n_pk, G_rc)
    if abs(fall) < 1e-12:
        return dict(fall=fall, norm_share=float("nan"), decoh_share=float("nan"),
                    s_peak=s_pk, s_rec=s_rc)
    return dict(fall=float(fall),
                norm_share=float((s_pk - s_cosfrozen) / fall),
                decoh_share=float((s_pk - s_normfrozen) / fall),
                s_peak=float(s_pk), s_rec=float(s_rc))


def summarise(rec, K, early=600):
    """Reference checkpoints.

    pk  the DELTA PEAK inside the crash window (steps <= `early`). Restricted on purpose:
        ||delta|| also counts directions that move A's state without moving any X/Y logit
        gap, and in a few runs that part drifts up for thousands of steps, which would put
        the 'peak' at the last checkpoint and make the peak -> recovery comparison vacuous.
    rc  A's best checkpoint AFTER pk -- the recovery peak. (Not `argmin A` then `argmax`
        after it: when A decays slowly at late times the global argmin lands at the end.)
    """
    st = np.array(rec["step"]); A = np.array(rec["A"]); Bacc = np.array(rec["B"])
    dl = np.array(rec["delta"])
    w = st <= early
    pk = int(np.argmax(np.where(w, dl, -np.inf)))
    tr = int(np.argmin(np.where(w, A, np.inf)))
    rc = pk + int(A[pk:].argmax())
    out = dict(K=K, steps=st.tolist(),
               trough=float(A[tr]), trough_step=int(st[tr]),
               rec_acc=float(A[rc]), rec_step=int(st[rc]), end=float(A[-1]),
               recovery=float(A[rc] - A[tr]),
               A_own_end=float(rec["A_own"][-1]), D_end=float(rec["Dacc"][-1]),
               B_gate=int(st[np.argmax(Bacc >= GATE)]) if (Bacc >= GATE).any() else -1,
               delta_peak=float(dl[pk]), delta_peak_step=int(st[pk]), delta_end=float(dl[-1]),
               delta_rec=float(dl[rc]),
               dw_peak=float(rec["delta_on_w"][pk]), dw_rec=float(rec["delta_on_w"][rc]),
               dw_end=float(rec["delta_on_w"][-1]),
               ko_recon_peak=float(rec["drecon"][pk]), ko_cos_peak=float(rec["dcos"][pk]),
               ko_recon_rec=float(rec["drecon"][rc]), ko_cos_rec=float(rec["dcos"][rc]),
               eps_end=float(rec["eps"][-1]), i_peak=pk, i_rec=rc, i_trough=tr)

    for tag in ("dir", "ko", "post"):
        P = {"dir": rec["Pdir"], "ko": rec["Pko"], "post": rec["Ppost"]}[tag]
        n, G, mc, s = coh(P)
        out[f"n_{tag}"] = n.tolist()
        out[f"sumn_{tag}"] = n.sum(1).tolist()
        out[f"snorm_{tag}"] = s.tolist()
        out[f"meancos_{tag}"] = mc.tolist()
        out[f"gram_{tag}_peak"] = G[pk].tolist()
        out[f"gram_{tag}_rec"] = G[rc].tolist()
        out[f"share_{tag}"] = share(n[pk], G[pk], n[rc], G[rc])
        out[f"share_{tag}_end"] = share(n[pk], G[pk], n[-1], G[-1])
        # per-block rotation from the delta-peak reference
        Uv = P / np.clip(n[:, :, None], 1e-30, None)
        cosref = np.einsum("tkd,kd->tk", Uv, Uv[pk])
        out[f"rot_{tag}"] = np.degrees(np.arccos(np.clip(cosref, -1, 1))).tolist()
        # each block's signed share of the shift it is decomposing, <P_j, delta_hat>/||delta||
        dh = unit(rec["dvec"])
        frac = np.einsum("tkd,td->tk", P, dh) / np.clip(
            np.linalg.norm(rec["dvec"], axis=-1)[:, None], 1e-30, None)
        out[f"frac_{tag}"] = frac.tolist()
        for nm, i in (("peak", pk), ("rec", rc), ("end", len(st) - 1)):
            out[f"{nm}_{tag}"] = dict(sumn=float(n[i].sum()), snorm=float(s[i]),
                                      meancos=float(mc[i]), norms=n[i].tolist(),
                                      rot=out[f"rot_{tag}"][i], frac=frac[i].tolist())

    # anti-state directions: the restoring force on block j points along -h_hat_j
    for who in ("A", "B"):
        Hh = unit(rec["hhat" + who])                           # (T, K, d)
        G = np.einsum("tkd,tld->tkl", Hh, Hh)
        iu = np.triu_indices(K, 1)
        mc = G[:, iu[0], iu[1]].mean(1) if K > 1 else np.ones(len(G))
        out[f"hhat{who}_meancos"] = mc.tolist()
        out[f"hhat{who}_meanang_peak"] = float(np.degrees(np.arccos(np.clip(mc[pk], -1, 1))))
        out[f"hhat{who}_gram_peak"] = G[pk].tolist()
    cross = np.einsum("tkd,tkd->tk", unit(rec["hhatA"]), unit(rec["hhatB"]))
    out["hhatAB_cos_peak"] = cross[pk].tolist()
    # how far each block's own input direction TURNS during the recovery: the second half of
    # the mechanism (the restoring force is along -h_hat_j, so a moving h_hat_j is a moving
    # restoring direction)
    for who in ("A", "B"):
        Hh = unit(rec["hhat" + who])
        rot = np.degrees(np.arccos(np.clip(np.einsum("tkd,kd->tk", Hh, Hh[pk]), -1, 1)))
        out[f"hhat{who}_rot"] = rot.tolist()
        out[f"hhat{who}_rot_rec"] = float(rot[rc].mean())
        out[f"hhat{who}_rot_end"] = float(rot[-1].mean())
    out["wfrac"] = (rec["blk_norm"] / np.clip(rec["h_norm"], 1e-30, None)).tolist()
    out["wfrac_peak"] = float((rec["blk_norm"][pk] / rec["h_norm"][pk]).mean())
    out["afrac_peak"] = float(dl[pk] / rec["post_norm"][pk])
    for key in ("A", "A_own", "B", "Dacc", "delta", "eps", "post_norm", "dW",
                "delta_on_w", "dcos", "drecon", "keyterm"):
        out[key] = rec[key] if isinstance(rec[key], list) else rec[key].tolist()
    return out


def one(seed, K, norm, gain, u_scale, steps, target, base_mult=1.0):
    t0 = time.time()
    p0, lr, pg, mu, A, Dd, B = pretrain(seed, K, norm, gain)
    base = lr / INJECT_RATIO
    if target:
        ilr, g, ok = match_gate(p0, B, K, norm, gain, u_scale, base, seed, target)
    else:
        ilr, g, ok = base * base_mult, -1, True
    rec, _ = inject(p0, K, norm, gain, u_scale, ilr, A, Dd, B, steps, seed)
    q = summarise(rec, K)
    q.update(seed=seed, norm=norm, gain=gain, u_scale=u_scale, pretrain_lr=lr,
             pretrain_gate=pg, inject_lr=float(ilr), matched=bool(ok), probe_gate=int(g),
             secs=round(time.time() - t0, 1))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--norm", type=int, default=1)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--u_scale", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--target", type=int, default=200, help="0 disables gate matching")
    ap.add_argument("--base_mult", type=float, default=1.0,
                    help="with --target 0: injection rate = base_mult * pretrain_lr / INJECT_RATIO")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {}
    for seed in [int(x) for x in a.seeds.split(",")]:
        q = one(seed, a.K, a.norm, a.gain, a.u_scale, a.steps, a.target, a.base_mult)
        out[f"K{a.K}-n{a.norm}-g{a.gain}-s{seed}"] = q
        sd = q["share_dir"]; sk = q["share_ko"]
        print(f"K={a.K} norm={a.norm} g={a.gain} s{seed} lr={q['pretrain_lr']} "
              f"ilr={q['inject_lr']:.2e} Bgate={q['B_gate']}{'' if q['matched'] else ' [UNMATCHED]'}"
              f" | A {q['trough']:.2f}@{q['trough_step']} -> {q['rec_acc']:.2f}@{q['rec_step']}"
              f" end {q['end']:.2f} | delta {q['delta_peak']:.2f}@{q['delta_peak_step']}"
              f" -> {q['delta_end']:.2f}"
              f" | dir sumn {q['peak_dir']['sumn']:.2f}->{q['rec_dir']['sumn']:.2f}"
              f" cos {q['peak_dir']['meancos']:+.2f}->{q['rec_dir']['meancos']:+.2f}"
              f" decoh {sd['decoh_share']:+.2f}"
              f" | ko sumn {q['peak_ko']['sumn']:.2f}->{q['rec_ko']['sumn']:.2f}"
              f" cos {q['peak_ko']['meancos']:+.2f}->{q['rec_ko']['meancos']:+.2f}"
              f" decoh {sk['decoh_share']:+.2f} | {q['secs']}s", flush=True)
        json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out} ({len(out)} runs)")


if __name__ == "__main__":
    main()
