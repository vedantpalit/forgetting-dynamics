"""math.tex to-do (a): on the minimal model, is the removing force proportional to the shift?

The claim (math.tex, "The removing force is proportional to the shift"): the shared write s is
in every B state, so C_b n_b = <h_b, w_hat> tracks <s, w_hat>. With h_b = W k_b and
k_b = sqrt(beta) mu + ..., the shared part of h_b is sqrt(beta) s, so the prediction is

    mean_b <h_b - h_b^0, w_hat>  =  sqrt(beta) <s, w_hat>          (up to the eps_b noise)

and the removing force  sum_b M_b C_b / n_b  has the sign of the mean margin and, after onset,
grows with ||s||. Same construction as wp_minimal_check.py, 3 seeds, every 10 steps.
"""
import numpy as np
import jax, jax.numpy as jnp
from wp_minimal_check import D, V, N, LR, population, fwd, grad, cross

BETA = 0.5


def run(seed, steps=1500, every=10):
    k0, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(jax.random.key(seed), 8)
    mu = jax.random.normal(k0, (D,)); mu /= jnp.linalg.norm(mu)
    A = population(k1, mu, 0, V // 2, BETA); Dd = population(k2, mu, V // 2, V, BETA)
    B = population(k3, mu, V // 2, V, BETA)
    p = {"W": jax.random.normal(k4, (D, D)) / jnp.sqrt(D), "U": jax.random.normal(k5, (D, V)) / jnp.sqrt(D)}
    kp, vp = jnp.concatenate([A[0], Dd[0]]), jnp.concatenate([A[1], Dd[1]])
    for i in jax.random.randint(k6, (2000, 32), 0, 2 * N):
        g = grad(p, kp[i], vp[i], True); p = {"W": p["W"] - LR * g["W"], "U": p["U"] - LR * g["U"]}
    W0 = np.asarray(p["W"]); U0 = np.asarray(p["U"]); mu_ = np.asarray(mu); kB = np.asarray(B[0]); yB = np.asarray(B[1])
    X, Y = np.arange(V // 2), np.arange(V // 2, V)
    w = U0[:, Y].mean(1) - U0[:, X].mean(1); w /= np.linalg.norm(w)
    h0 = kB @ W0
    idx = jax.random.randint(k7, (steps, 32), 0, N); lr = LR / 13.33
    rows = []
    for t in range(steps + 1):
        if t % every == 0:
            W = np.asarray(p["W"]); s = mu_ @ (W - W0)
            h = kB @ W; n = np.linalg.norm(h, axis=1); C = (h / n[:, None]) @ w
            z = np.asarray(fwd(p, B[0], True)); pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            M = z[np.arange(N), yB] - (pr * z).sum(1)
            accA = float((np.asarray(fwd(p, A[0], True)).argmax(-1) == np.asarray(A[1])).mean())
            rows.append((t, accA, s @ w, np.sqrt(BETA) * (s @ w), ((h - h0) @ w).mean(), (C * n).mean(),
                         M.mean(), (M * C / n).sum(), np.linalg.norm(s)))
        if t < steps:
            i = idx[t]; g = grad(p, B[0][i], B[1][i], True)
            p = {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * g["U"]}
    return np.array(rows)


if __name__ == "__main__":
    for seed in range(3):
        r = run(seed)
        st, accA, sw, pred, meas, Cn, M, force, sn = r.T
        k = pred != 0
        ratio = meas[k] / pred[k]
        tr = int(accA[:100].argmin())
        print(f"seed {seed}: A trough {accA[tr]:.2f}@{int(st[tr])}, end {accA[-1]:.2f}")
        print(f"  mean_b<h_b-h_b0,w> / sqrt(beta)<s,w>: mean {ratio.mean():.3f}, sd {ratio.std():.3f}, "
              f"corr {np.corrcoef(meas[k], pred[k])[0,1]:.4f}")
        print(f"  corr(mean_b C_b n_b, ||s||) {np.corrcoef(Cn, sn)[0,1]:.4f}; "
              f"corr(mean_b C_b n_b - at t=0, <s,w>) {np.corrcoef(Cn - Cn[0], sw)[0,1]:.4f}")
        mc, fc = cross(st, M, True), cross(st, force, True)
        sign_agree = 100 * (np.sign(M) == np.sign(force)).mean()
        print(f"  removing force sum_b M_b C_b/n_b: sign agrees with mean margin at {sign_agree:.0f}% of checkpoints; "
              f"margin crosses 0 at {mc}, force at {fc}")
        post = st > (mc or 0)
        print(f"  after onset: corr(force, M_mean * ||s||) {np.corrcoef(force[post], (M * sn)[post])[0,1]:.4f}")
        for j in (0, 5, 10, 15, 20, 40, 80, 150):
            print(f"    t={int(st[j]):5d} A {accA[j]:.2f} <s,w> {sw[j]:7.2f} meas {meas[j]:7.2f} pred {pred[j]:7.2f} "
                  f"M {M[j]:6.2f} force {force[j]:7.2f} ||s|| {sn[j]:6.2f}")
