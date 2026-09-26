"""The minimal toy (toy_minimal.ipynb: h = Wk, z = U rms(h), normal init, one rate), with
the two claims from THEORY sections 6-7 checked on it directly:

  1. remove the norm  -> the shared write's growth rate along its own direction never goes
                         negative; no recovery                        (THEORY 6, exact)
  2. remove mu (beta=0) -> the write has no common component; delta ~ sqrt(beta) n_B -> 0;
                         nothing collective to crash, nothing to recover (THEORY 3, 5)
  3. with both        -> the normalizer's term flips sign when B's mean margin crosses zero,
                         and the write turns over there                (THEORY 7, claim 5)

Same construction as the notebook, 3 seeds, every 10 steps.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

D, V, N = 128, 32, 128
LR = 0.5


def population(key, mu, lo, hi, beta):
    kg, kv = jax.random.split(key)
    g = jax.random.normal(kg, (N, D)); g /= jnp.linalg.norm(g, axis=1, keepdims=True)
    return jnp.sqrt(beta) * mu + jnp.sqrt(1 - beta) * g, jax.random.randint(kv, (N,), lo, hi)


def fwd(p, k, norm):
    h = k @ p["W"]
    if norm:
        h = h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)
    return h @ p["U"]


def loss(p, k, v, norm):
    return -jnp.mean(jax.nn.log_softmax(fwd(p, k, norm), -1)[jnp.arange(k.shape[0]), v])


grad = jax.jit(jax.grad(loss), static_argnums=3)


def run(seed, norm, beta, steps=3000, every=10):
    k0, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(jax.random.key(seed), 8)
    mu = jax.random.normal(k0, (D,)); mu /= jnp.linalg.norm(mu)
    A = population(k1, mu, 0, V // 2, beta); Dd = population(k2, mu, V // 2, V, beta)
    B = population(k3, mu, V // 2, V, beta)
    p = {"W": jax.random.normal(k4, (D, D)) / jnp.sqrt(D), "U": jax.random.normal(k5, (D, V)) / jnp.sqrt(D)}
    kp, vp = jnp.concatenate([A[0], Dd[0]]), jnp.concatenate([A[1], Dd[1]])
    idx = jax.random.randint(k6, (2000, 32), 0, 2 * N)
    for i in idx:
        g = grad(p, kp[i], vp[i], norm); p = {"W": p["W"] - LR * g["W"], "U": p["U"] - LR * g["U"]}
    W0 = np.asarray(p["W"]); U0 = np.asarray(p["U"]); mu_ = np.asarray(mu)
    X, Y = np.arange(V // 2), np.arange(V // 2, V)
    w = U0[:, Y].mean(1) - U0[:, X].mean(1); w /= np.linalg.norm(w)
    idx = jax.random.randint(k7, (steps, 32), 0, N); lr = LR / 13.33
    rows = []
    for t in range(steps + 1):
        if t % every == 0:
            s = mu_ @ (np.asarray(p["W"]) - W0)
            g = grad(p, B[0], B[1], norm)                       # full-batch gradient: the force
            ds = -lr * (mu_ @ np.asarray(g["W"]))               # step of the shared row
            shat = s / max(np.linalg.norm(s), 1e-30)
            z = np.asarray(fwd(p, B[0], norm)); pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            M = z[np.arange(N), np.asarray(B[1])] - (pr * z).sum(1)
            accA = float((np.asarray(fwd(p, A[0], norm)).argmax(-1) == np.asarray(A[1])).mean())
            accB = float((z.argmax(-1) == np.asarray(B[1])).mean())
            rows.append((t, accA, accB, s @ w, float(ds @ shat), float(M.mean()), float(np.linalg.norm(s))))
        if t < steps:
            i = idx[t]; g = grad(p, B[0][i], B[1][i], norm)
            p = {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * g["U"]}
    return np.array(rows)


def cross(st, v, rising):
    for i in range(1, len(v)):
        if (rising and v[i - 1] < 0 <= v[i]) or (not rising and v[i - 1] > 0 >= v[i]):
            return int(st[i])
    return None


if __name__ == "__main__":
    for label, norm, beta in (("norm, beta=0.5 (the phenomenon)", True, 0.5),
                              ("NO norm, beta=0.5", False, 0.5),
                              ("norm, beta=0 (no shared mu)", True, 0.0)):
        print(f"== {label}")
        for seed in range(3):
            r = run(seed, norm, beta)
            st, A, B, sw, proj, M, sn = r.T
            tr = int(A[:100].argmin()); after = A[tr:].max()
            mc = cross(st, M, True); pc = cross(st[1:], proj[1:], False)
            neg = 100 * (proj[1:] < 0).mean()
            print(f"  seed {seed}: A trough {A[tr]:.2f}@{int(st[tr])} -> max after {after:.2f}, end {A[-1]:.2f} | "
                  f"||s|| peak {sn.max():.2f} | write growth <0 at {neg:.0f}% of checkpoints | "
                  f"B margin crosses 0 at {mc}, write turns at {pc}")
