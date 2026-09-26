"""Is per-block recovery in the K-stack toy INPUT DRIFT or WEIGHT CHANGE?

A copy of ballast_kmech.py's K-block residual stack

    h_{j+1} = h_j + rms(h_j) W_j ,   z = rms(h_K) U

with three changes, all additive:

  * u_scale = 0.01 during injection only (the transformer readout barely moves),
  * the injection rate is bisected so B reaches the gate at step 200 (sweeps.match_gate),
  * per block, per checkpoint, the A-mean contribution P_j is split EXACTLY into

        m_j(t) := mean_a rms(h_{j,a}(t))            the block's A-mean input
        P_j(t)  = m_j(t) W_j(t)
        dP_j(t) = P_j(t) - P_j(0)
                = (m_j(t) - m_j(0)) W_j(0)   +   m_j(t) (W_j(t) - W_j(0))
                =        DRIFT_j(t)          +          WRITE_j(t)

    DRIFT is what the block's contribution does when only its INPUT moves (upstream blocks
    having rewritten the residual stream); WRITE is what it does when only its OWN weights
    move. At j = 0 the input is rms(k), frozen, so DRIFT_0 == 0 by construction: that is the
    K=1 case, and it is the control for the identity.

The dW snapshots are kept in memory (float32) for the counterfactual rotations, which need
W_j at the delta-peak checkpoint; only derived numbers are written out.

    python wp_drift_stack.py <K> [steps] [seeds]
"""
import os; os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax, jax.numpy as jnp, json, sys, time
from functools import partial
jax.config.update("jax_enable_x64", True)

D, V, NA, ND, NB, BETA = 128, 32, 50, 50, 50, 0.5
GATE = 0.99
U_SCALE = 0.01
GATE_TARGET = 200
K = int(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SEEDS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
X = np.arange(0, V // 2); Y = np.arange(V // 2, V)


def rms(h): return h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)


def build(seed):
    r = np.random.default_rng(seed); mu = r.normal(size=D); mu /= np.linalg.norm(mu)
    def keys(n):
        g = r.normal(size=(n, D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(BETA) * mu[None] + np.sqrt(1 - BETA) * g
    return (mu, (keys(NA), X[r.integers(0, len(X), NA)]),
            (keys(ND), Y[r.integers(0, len(Y), ND)]),
            (keys(NB), Y[r.integers(0, len(Y), NB)]))


def init(seed):
    r = np.random.default_rng(seed + 77)
    return {"W": jnp.asarray(r.normal(size=(K, D, D)) * 0.02 / np.sqrt(D)),
            "U": jnp.asarray(r.normal(size=(D, V)) / np.sqrt(D))}


@jax.jit
def fwd(p, k):
    """returns logits, per-block contributions P (K,n,D), readout state, per-block inputs."""
    h = k; P = []; INP = []
    for j in range(K):
        inp = rms(h); INP.append(inp)
        pj = inp @ p["W"][j]; P.append(pj); h = h + pj
    hr = rms(h)
    return hr @ p["U"], jnp.stack(P), hr, jnp.stack(INP)


@jax.jit
def step(p, k, v, lr, u_scale):
    def L(p):
        z = fwd(p, k)[0]
        return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])
    g = jax.grad(L)(p)
    return {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]}


def acc(p, k, v):
    if len(k) == 0: return 1.0
    z = np.asarray(fwd(p, jnp.asarray(k))[0]); return float((z.argmax(-1) == v).mean())


def pretrain(seed):
    mu, (kA, vA), (kD, vD), (kB, vB) = build(1000 + seed)
    kp = np.concatenate([kA, kD]); vp = np.concatenate([vA, vD])
    for lr in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
        rng = np.random.default_rng(seed); p = init(seed)
        for t in range(20000):
            i = rng.integers(0, len(kp), 32)
            p = step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0)
            if t % 200 == 0 and min(acc(p, kA, vA), acc(p, kD, vD)) >= GATE:
                return p, lr, t, mu, (kA, vA), (kD, vD), (kB, vB)
    raise SystemExit(f"pretraining never reached {GATE} (K={K}, seed={seed})")


def gate_of(p0, kB, vB, ilr, seed, cap=2000, every=5):
    p = p0; rng = np.random.default_rng(seed + 5)
    for t in range(cap + 1):
        if t % every == 0 and acc(p, kB, vB) >= GATE: return t
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, U_SCALE)
    return cap


def match_gate(p0, kB, vB, base, seed, target=GATE_TARGET, iters=11, tol=0.15):
    """Bisection on log2 of a rate multiplier; gate is monotone decreasing in the rate."""
    lo, hi = -6.0, 6.0
    best = (base, gate_of(p0, kB, vB, base, seed), False)
    if abs(best[1] - target) <= tol * target: return best
    for _ in range(iters):
        mid = 0.5 * (lo + hi); ilr = base * 2.0 ** mid
        g = gate_of(p0, kB, vB, ilr, seed)
        if abs(g - target) < abs(best[1] - target):
            best = (ilr, g, abs(g - target) <= tol * target)
        if best[2]: break
        if g > target: lo = mid
        else: hi = mid
    return best


def grid(steps):
    g = set(range(0, 301, 5))
    g |= {int(x) for x in np.unique(np.round(np.logspace(np.log10(300), np.log10(steps), 45)))}
    return sorted(s for s in g if s <= steps)


def run(seed):
    t0 = time.time()
    p0, lr, pg, mu, (kA, vA), (kD, vD), (kB, vB) = pretrain(seed)
    ilr, bg, ok = match_gate(p0, kB, vB, lr / 13.33, seed)
    W0 = np.asarray(p0["W"])
    _, PA0, _, INP0 = [np.asarray(x) for x in fwd(p0, jnp.asarray(kA))]
    m0 = INP0.mean(1)                    # (K, D)   A-mean input per block at step 0
    P0 = PA0.mean(1)                     # (K, D)

    p = p0; rng = np.random.default_rng(seed + 5)
    gi = set(grid(STEPS))
    steps, Aa, Ba, Da = [], [], [], []
    M, DR, WR, PP = [], [], [], []
    dWs = []                             # float32 snapshots, memory only
    maxid = 0.0
    for t in range(STEPS + 1):
        if t in gi:
            _, PA, _, INP = [np.asarray(x) for x in fwd(p, jnp.asarray(kA))]
            W = np.asarray(p["W"]); dW = W - W0
            m = INP.mean(1); P = PA.mean(1)
            drift = np.einsum('kd,kde->ke', m - m0, W0)
            write = np.einsum('kd,kde->ke', m, dW)
            maxid = max(maxid, float(np.abs(P - P0 - drift - write).max()))
            steps.append(t); Aa.append(acc(p, kA, vA)); Ba.append(acc(p, kB, vB))
            Da.append(acc(p, kD, vD))
            M.append(m); DR.append(drift); WR.append(write); PP.append(P)
            dWs.append(dW.astype(np.float32))
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, U_SCALE)

    steps = np.array(steps); Aa = np.array(Aa); Ba = np.array(Ba); Da = np.array(Da)
    M = np.array(M); DR = np.array(DR); WR = np.array(WR); PP = np.array(PP)
    dP = PP - P0[None]
    delta = np.linalg.norm(dP.sum(1), axis=-1)

    pk = int(delta.argmax())                       # delta peak
    tr = int(Aa.argmin())                          # A trough
    rp = tr + int(Aa[tr:].argmax())                # recovery peak (first max after trough)
    if rp <= pk: rp = len(steps) - 1

    # counterfactual contributions relative to the delta-peak checkpoint
    Wpk = W0 + dWs[pk].astype(np.float64)
    P_drift_only = np.einsum('tkd,kde->tke', M, Wpk)                       # W frozen at peak
    P_write_only = np.stack([np.einsum('kd,kde->ke', M[pk], W0 + dWs[t].astype(np.float64))
                             for t in range(len(steps))])                  # input frozen at peak

    out = dict(K=K, seed=seed, pretrain_lr=lr, pretrain_gate=pg, inject_lr=float(ilr),
               B_gate=int(bg), matched=bool(ok), identity_max_abs=maxid,
               steps=steps, A=Aa, B=Ba, Dacc=Da, delta=delta,
               pk=pk, tr=tr, rp=rp, P0=P0, m0=m0, M=M, DRIFT=DR, WRITE=WR, P=PP,
               P_drift_only=P_drift_only, P_write_only=P_write_only,
               secs=time.time() - t0)
    print(f"K={K} s{seed} lr={lr} ilr={ilr:.3e} Bgate={bg}{'' if ok else ' [UNMATCHED]'} "
          f"| identity {maxid:.2e} | A trough {Aa[tr]:.2f}@{steps[tr]} -> {Aa[rp]:.2f}@{steps[rp]} "
          f"end {Aa[-1]:.2f} | delta peak {delta[pk]:.2f}@{steps[pk]} end {delta[-1]:.2f} "
          f"| B end {Ba[-1]:.2f} | {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for s in SEEDS:
        r = run(s)
        for k, v in r.items(): res[f"s{s}/{k}"] = v
    np.savez_compressed(f"wp_drift_K{K}.npz", **{k: np.asarray(v) for k, v in res.items()})
    print(f"  wrote wp_drift_K{K}.npz")
