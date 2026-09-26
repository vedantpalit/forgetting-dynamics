"""Task 5: does DEPTH restore the recovery that removing the ballast takes away?

A copy of the scratchpad's kmech.py (K-block residual stack, per-block + final RMS norm)
with one knob added: ND. Everything else -- learning-rate probe, ilr = lr/13.33, U trained
at the full rate, logging -- is kept as it was so the ND = 50 column reproduces the existing
kmech{1,2,4}.json runs.

As in ballast_k1.py, the ballast keys are drawn at full size and truncated, so A and B are
identical across the ND columns.

    python ballast_kmech.py <K> <ND> [steps] [seeds] [ln]
"""
import os; os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax, jax.numpy as jnp, json, sys
from functools import partial
jax.config.update("jax_enable_x64", True)

D, V, NA, NB, BETA = 128, 32, 50, 50, 0.5
K = int(sys.argv[1])
ND = int(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
SEEDS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1]
LNS = [bool(int(x)) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [True]
ND_MAX = 50
X = np.arange(0, V // 2); Y = np.arange(V // 2, V)


def rms(h): return h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)


def build(seed):
    r = np.random.default_rng(seed); mu = r.normal(size=D); mu /= np.linalg.norm(mu)
    def keys(n):
        g = r.normal(size=(n, D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(BETA) * mu[None] + np.sqrt(1 - BETA) * g
    kA, vA = keys(NA), X[r.integers(0, len(X), NA)]
    kD, vD = keys(ND_MAX), Y[r.integers(0, len(Y), ND_MAX)]
    kB, vB = keys(NB), Y[r.integers(0, len(Y), NB)]
    return mu, (kA, vA), (kD[:ND], vD[:ND]), (kB, vB)


def init(seed):
    r = np.random.default_rng(seed + 77)
    return {"W": jnp.asarray(r.normal(size=(K, D, D)) * 0.02 / np.sqrt(D)),
            "U": jnp.asarray(r.normal(size=(D, V)) / np.sqrt(D))}


@partial(jax.jit, static_argnames=("ln",))
def fwd(p, k, ln):
    h = k; P = []; G = []; INP = []
    for j in range(K):
        inp = rms(h) if ln else h
        G.append(jnp.sqrt(D) / jnp.linalg.norm(h, axis=-1)); INP.append(inp)
        pj = inp @ p["W"][j]; P.append(pj); h = h + pj
    hr = rms(h) if ln else h
    G.append(jnp.sqrt(D) / jnp.linalg.norm(h, axis=-1))
    return hr @ p["U"], jnp.stack(P), hr, jnp.stack(G), jnp.stack(INP)


@partial(jax.jit, static_argnames=("ln",))
def step(p, k, v, lr, ln):
    def L(p):
        z = fwd(p, k, ln)[0]
        return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])
    g = jax.grad(L)(p); return jax.tree_util.tree_map(lambda a, b: a - lr * b, p, g)


def acc(p, k, v, ln):
    if len(k) == 0: return 1.0
    z = np.asarray(fwd(p, jnp.asarray(k), ln)[0]); return float((z.argmax(-1) == v).mean())


def prank(M):
    s = np.linalg.svd(M, compute_uv=False); s2 = s ** 2; return float(s2.sum() ** 2 / (s2 ** 2).sum())


def run(seed, ln):
    mu, (kA, vA), (kD, vD), (kB, vB) = build(1000 + seed)
    kp = np.concatenate([kA, kD]) if ND else kA
    vp = np.concatenate([vA, vD]) if ND else vA
    rng = np.random.default_rng(seed)
    for lr in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
        p = init(seed); ok = False
        for t in range(20000):
            i = rng.integers(0, len(kp), 32); p = step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, ln)
            if t % 200 == 0 and min(acc(p, kA, vA, ln), acc(p, kD, vD, ln)) >= 0.99: ok = True; break
        if ok: break
    p0 = {k: np.asarray(v) for k, v in p.items()}
    _, PA0, hrA0, GA0, INPA0 = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), ln)]
    _, PB0, _, _, INPB0 = [np.asarray(x) for x in fwd(p, jnp.asarray(kB), ln)]
    sh = np.concatenate([INPA0, INPB0], axis=1).mean(1); sh /= np.linalg.norm(sh, axis=-1, keepdims=True)
    # the X -> Y separation of the pretrained readout, for the absolute per-class bias
    wdir = p0["U"][:, Y].mean(1) - p0["U"][:, X].mean(1); wdir /= np.linalg.norm(wdir)
    ilr = lr / 13.33; rows = []
    grid = sorted(set(list(range(0, 201, 5)) +
                      [int(x) for x in np.unique(np.round(np.logspace(2.3, np.log10(STEPS), 40)))]))
    gi = set(grid)
    for t in range(STEPS + 1):
        if t in gi:
            _, PA, hrA, GA, INPA = [np.asarray(x) for x in fwd(p, jnp.asarray(kA), ln)]
            _, PB, _, _, _ = [np.asarray(x) for x in fwd(p, jnp.asarray(kB), ln)]
            W = np.asarray(p["W"]); U = np.asarray(p["U"]); dW = W - p0["W"]; dU = U - p0["U"]
            dPA = (PA - PA0); dPAm = dPA.mean(1)
            n = np.linalg.norm(dPAm, axis=-1); u = dPAm / np.clip(n[:, None], 1e-12, None)
            delta = dPAm.sum(0); dn = np.linalg.norm(delta); dhat = delta / max(dn, 1e-12)
            dh = (hrA - hrA0); dpost = dh.mean(0); eps = dh - dpost
            dPB = (PB - PB0); tot = dPB.sum(0)
            kapB = float(np.linalg.norm(tot.mean(0)) ** 2 / np.mean(np.linalg.norm(tot, axis=-1) ** 2))
            z = np.asarray(fwd(p, jnp.asarray(kA), ln)[0]); ii = np.arange(NA)
            oth = np.full(V, -np.inf); oth[Y] = 0.0
            r = dict(step=t, A=acc(p, kA, vA, ln), B=acc(p, kB, vB, ln), Dacc=acc(p, kD, vD, ln),
                     delta=float(dn), eps=float(np.sqrt((eps ** 2).sum(-1)).mean()),
                     dU=float(np.linalg.norm(dU) / np.linalg.norm(p0["U"])),
                     # the collective shift's content along the X->Y readout separation, and
                     # the absolute per-class bias that the A-mean state carries there
                     delta_on_w=float(delta @ wdir),
                     bias_abs_on_w=float(hrA.mean(0) @ wdir),
                     m_cross=float((z[ii, vA] - (z + oth[None, :]).max(1)).mean()),
                     dW=[float(np.linalg.norm(dW[j])) for j in range(K)],
                     shared_share=[float(np.linalg.norm(sh[j] @ dW[j]) ** 2 /
                                         max(np.linalg.norm(dW[j]) ** 2, 1e-24)) for j in range(K)],
                     prank=[prank(dW[j]) for j in range(K)],
                     cos_pairs=[float(u[i] @ u[j]) for i in range(K) for j in range(i + 1, K)],
                     cos_block_delta=[float(u[j] @ dhat) for j in range(K)],
                     kappaB=kapB,
                     gain_mean=[float(GA[j].mean()) for j in range(K + 1)])
            rows.append(r)
        i = rng.integers(0, NB, 32); p = step(p, jnp.asarray(kB[i]), jnp.asarray(vB[i]), ilr, ln)
    return lr, rows


out = {}
for ln in LNS:
    for s in SEEDS:
        lr, rows = run(s, ln); out[f'ln{int(ln)}_s{s}'] = rows
        A = np.array([r['A'] for r in rows]); d = np.array([r['delta'] for r in rows])
        st = [r['step'] for r in rows]; tr = int(A.argmin())
        cp = np.array([r['cos_pairs'] for r in rows]) if K > 1 else None
        print(f"K={K} ND={ND} LN={int(ln)} s{s} lr={lr} | A trough {A.min():.2f}@{st[tr]} "
              f"after {A[tr:].max():.2f} (rec {A[tr:].max()-A.min():+.2f}) end {A[-1]:.2f} | "
              f"delta peak {d.max():.2f}@{st[int(d.argmax())]} end {d[-1]:.2f} "
              f"(drop {100*(1-d[-1]/max(d.max(),1e-12)):.0f}%) | dw {rows[tr]['delta_on_w']:+.2f}"
              f"->{rows[-1]['delta_on_w']:+.2f} | mcross {rows[0]['m_cross']:+.2f}"
              f"->{rows[-1]['m_cross']:+.2f} | dU/U {rows[-1]['dU']:.3f}"
              + (f" | cos_pairs end {np.round(cp[-1],2).tolist()} trough {np.round(cp[tr],2).tolist()}"
                 if cp is not None else ""), flush=True)
json.dump(out, open(f'ballast_kmech_K{K}_nd{ND}.json', 'w'))
