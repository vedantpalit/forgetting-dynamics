"""Derivation checks for the K-BLOCK residual stack (untransferred item I: DEPTH).

    h_0 = k
    h_{j+1} = h_j + g rms(h_j) W_j       rms(x) = sqrt(d) x / ||x||
    z = rms(h_K) U,   L = CE

Everything below is checked against jax.grad. The claims:

 C1  dL/dW_j = inp_j^T r_{j+1}  with inp_j = g rms(h_j) and r_{j+1} = dL/dh_{j+1}.
 C2  r_K = y J_K  with y = (p-e)U^T/n and J_K = (sqrt(d)/||h_K||)(I - hhat_K hhat_K^T),
     and it splits EXACTLY as   r_K = (sqrt(d)/||h_K||) y  +  (M/(n||h_K||)) hhat_K.
 C3  r_{j+1} = r_K (I + g W_{K-1}^T J_{K-1}) ... (I + g W_{j+1}^T J_{j+1}).
 C4  the restoring (anti-state) force on block j's shared row is
       -eta sum_b <sh_j, inp_{j,b}> a_b^{(j+1)},  a_b^{(j+1)} = a_b prod_{m>j}(I + g W_m^T J_m)
     where a_b = (M_b/(n||h_{K,b}||)) hhat_{K,b}.  FIRST ORDER:
       a^{(j+1)} ~ ||a|| [ hhat_K + sum_{m=j+1}^{K-1} delta_m ],  delta_m = g (hhat_K W_m^T) J_m
     so delta_m is ORTHOGONAL TO hhat_m but is ADDED, not projected out.
 C5  the conjectured form "hhat_K projected orthogonal to hhat_{j+1},...,hhat_{K-1}" is
     tested directly against the exact a^{(j+1)}.

usage:  python wp_theory_kstack.py point     # exact identities at a random point
        python wp_theory_kstack.py traj      # identities + angles along an injection run
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------ model
def fwd_np(W, U, k, g):
    d = k.shape[-1]; K = W.shape[0]
    H = [k]; INP = []
    h = k
    for j in range(K):
        inp = g * np.sqrt(d) * h / np.linalg.norm(h, axis=-1, keepdims=True)
        INP.append(inp)
        h = h + inp @ W[j]
        H.append(h)
    hr = np.sqrt(d) * h / np.linalg.norm(h, axis=-1, keepdims=True)
    return H, INP, hr, hr @ U


def loss_jnp(p, k, v, g, K):
    d = k.shape[-1]
    h = k
    for j in range(K):
        inp = g * jnp.sqrt(d) * h / jnp.linalg.norm(h, axis=-1, keepdims=True)
        h = h + inp @ p["W"][j]
    hr = jnp.sqrt(d) * h / jnp.linalg.norm(h, axis=-1, keepdims=True)
    z = hr @ p["U"]
    return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])


def Japply(x, h):
    """x (n,d) row vectors, right-multiplied by J(h) = (sqrt(d)/||h||)(I - hhat hhat^T)."""
    d = h.shape[-1]
    hn = np.linalg.norm(h, axis=-1, keepdims=True)
    hh = h / hn
    return (np.sqrt(d) / hn) * (x - (x * hh).sum(-1, keepdims=True) * hh)


def backprop(W, U, k, v, g):
    """Manual backward pass. Returns everything the derivation names."""
    d = k.shape[-1]; K = W.shape[0]; n = len(v)
    H, INP, hr, z = fwd_np(W, U, k, g)
    pr = np.asarray(jax.nn.softmax(jnp.asarray(z), -1))
    e = np.zeros_like(pr); e[np.arange(n), v] = 1.0
    q = (pr - e) / n
    y = q @ U.T                                        # (n,d)
    hK = H[K]; hn = np.linalg.norm(hK, axis=-1, keepdims=True); hh = hK / hn
    M = z[np.arange(n), v] - (pr * z).sum(-1)          # confidence margin
    plain = np.sqrt(d) / hn * y                        # the plain term
    anti = (M[:, None] / hn) * hh / n                  # the ANTI-STATE term (enters with -eta)
    rK = Japply(y, hK)
    # propagate both pieces down; R[m] = dL/dh_m
    R = [None] * (K + 1); RA = [None] * (K + 1)
    R[K] = rK; RA[K] = anti
    for m in range(K - 1, -1, -1):
        R[m] = R[m + 1] + g * Japply(R[m + 1] @ W[m].T, H[m])
        RA[m] = RA[m + 1] + g * Japply(RA[m + 1] @ W[m].T, H[m])
    gW = np.stack([INP[j].T @ R[j + 1] for j in range(K)])
    gU = hr.T @ q
    return dict(H=H, INP=INP, hr=hr, z=z, M=M, y=y, plain=plain, anti=anti,
                rK=rK, R=R, RA=RA, gW=gW, gU=gU, hh=hh, hn=hn[:, 0])


def relerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


# ------------------------------------------------------------------ C1-C3 at a point
def check_point(seed, K, d, V, n, g, scale, verbose=True):
    r = np.random.default_rng(seed)
    mu = r.normal(size=d); mu /= np.linalg.norm(mu)
    gg = r.normal(size=(n, d)); gg /= np.linalg.norm(gg, axis=1, keepdims=True)
    k = np.sqrt(0.5) * mu[None] + np.sqrt(0.5) * gg
    W = r.normal(size=(K, d, d)) * scale / np.sqrt(d)
    U = r.normal(size=(d, V)) / np.sqrt(d)
    v = r.integers(0, V, n)
    B = backprop(W, U, k, v, g)
    gr = jax.grad(loss_jnp)({"W": jnp.asarray(W), "U": jnp.asarray(U)},
                            jnp.asarray(k), jnp.asarray(v), g, K)
    e_W = relerr(np.asarray(gr["W"]), B["gW"])
    e_U = relerr(np.asarray(gr["U"]), B["gU"])
    e_split = relerr(B["rK"], B["plain"] + B["anti"])
    # C3 as an explicit matrix product, independent of the recursion above
    e_prod = 0.0
    for j in range(K):
        acc = B["rK"].copy()
        for m in range(K - 1, j, -1):
            acc = acc + g * Japply(acc @ W[m].T, B["H"][m])
        e_prod = max(e_prod, relerr(B["R"][j + 1], acc))
    if verbose:
        print(f"  C1 dL/dW_j = inp_j^T r_(j+1)      vs jax.grad : rel err {e_W:.3e}")
        print(f"  C1 dL/dU   = rms(h_K)^T (p-e)/n   vs jax.grad : rel err {e_U:.3e}")
        print(f"  C2 r_K = plain + anti-state split            : rel err {e_split:.3e}")
        print(f"  C3 r_(j+1) = r_K prod(I + g W_m^T J_m)       : rel err {e_prod:.3e}")
    return B, W, U, k, v, dict(W=e_W, U=e_U, split=e_split, prod=e_prod)


# ------------------------------------------------------------------ C4/C5 restoring dirs
def restoring_dirs(B, W, g):
    """Block j's restoring force on its shared row: -eta sum_b <sh_j,inp_jb> a_b^(j+1).
    Returns (exact dirs, first-order dirs, conjectured 'projected hhat_K' dirs)."""
    K = W.shape[0]
    exact, first, conj = [], [], []
    hhK = B["hh"]
    for j in range(K):
        sh = B["INP"][j].mean(0); sh /= np.linalg.norm(sh)
        c = B["INP"][j] @ sh                                   # (n,) coefficients
        f = c @ B["RA"][j + 1]                                 # EXACT restoring force
        exact.append(f / np.linalg.norm(f))
        # first order: a^(j+1) ~ a + sum_{m>j} g (a W_m^T) J_m
        a1 = B["anti"].copy()
        for m in range(j + 1, K):
            a1 = a1 + g * Japply(B["anti"] @ W[m].T, B["H"][m])
        f1 = c @ a1
        first.append(f1 / np.linalg.norm(f1))
        # conjecture: hhat_K projected orthogonal to hhat_{j+1..K-1}, per individual
        pv = hhK.copy()
        for m in range(j + 1, K):
            hm = B["H"][m] / np.linalg.norm(B["H"][m], axis=-1, keepdims=True)
            pv = pv - (pv * hm).sum(-1, keepdims=True) * hm
        nrm = np.linalg.norm(pv, axis=-1).mean()
        fc = c @ (pv * (B["M"][:, None] / B["hn"][:, None]) / len(c))
        conj.append((fc / max(np.linalg.norm(fc), 1e-300), nrm))
    return np.array(exact), np.array(first), conj


def rho_blocks(B, W, g):
    """rho_m = ||P_m|| / ||h_m||, the relative write size of block m (A-mean)."""
    K = W.shape[0]
    out = []
    for m in range(K):
        P = B["INP"][m] @ W[m]
        out.append(float((np.linalg.norm(P, axis=-1) /
                          np.linalg.norm(B["H"][m], axis=-1)).mean()))
    return np.array(out)


def report_dirs(B, W, g, tag=""):
    K = W.shape[0]
    ex, fi, cj = restoring_dirs(B, W, g)
    rho = rho_blocks(B, W, g)
    print(f"\n  {tag} rho_m = ||P_m||/||h_m|| : {np.round(rho, 4).tolist()}")
    print(f"  {'i,j':>6} {'cos exact':>10} {'cos 1st-ord':>12} {'pred 1-r^2|i-j|/2':>18} "
          f"{'cos conj':>10}")
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            ce = float(ex[i] @ ex[j]); cf = float(fi[i] @ fi[j])
            # first-order prediction with delta_m incoherent across m
            rbar = rho[i + 1:j + 1].mean() if j > i else 0.0
            pred = 1 - 0.5 * (rbar ** 2) * (j - i)
            cc = float(cj[i][0] @ cj[j][0])
            rows.append((i, j, ce, cf, pred, cc))
            print(f"  {i},{j:>3} {ce:>10.5f} {cf:>12.5f} {pred:>18.5f} {cc:>10.5f}")
    # the conjecture's own norms: how much of hhat_K survives the projections
    print(f"  conjecture: ||hhat_K after projecting out hhat_(j+1..K-1)|| per block j = "
          f"{[round(c[1], 4) for c in cj]}")
    print(f"  cos(exact_j, hhat_K-direction of block K-1) = "
          f"{[round(float(ex[j] @ ex[K-1]), 5) for j in range(K)]}")
    return rows, rho


# ------------------------------------------------------------------ trajectory
def run_traj(seed, K, d, V, nA, nB, g, steps, inject_steps):
    """Minimal pretrain (A only, plus a ballast half) then inject B; log block contributions."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=d); mu /= np.linalg.norm(mu)

    def keys(nn):
        gg = rng.normal(size=(nn, d)); gg /= np.linalg.norm(gg, axis=1, keepdims=True)
        return np.sqrt(0.5) * mu[None] + np.sqrt(0.5) * gg
    X = np.arange(0, V // 2); Y = np.arange(V // 2, V)
    kA, vA = keys(nA), X[rng.integers(0, len(X), nA)]
    kD, vD = keys(nA), Y[rng.integers(0, len(Y), nA)]
    kB, vB = keys(nB), Y[rng.integers(0, len(Y), nB)]
    p = {"W": jnp.asarray(rng.normal(size=(K, d, d)) * 0.02 / np.sqrt(d)),
         "U": jnp.asarray(rng.normal(size=(d, V)) / np.sqrt(d))}
    gfn = jax.jit(jax.grad(loss_jnp), static_argnums=(4,))

    def sgd(p, k, v, lr):
        gr = gfn(p, jnp.asarray(k), jnp.asarray(v), g, K)
        return jax.tree_util.tree_map(lambda a, b: a - lr * b, p, gr)

    def acc(p, k, v):
        z = fwd_np(np.asarray(p["W"]), np.asarray(p["U"]), k, g)[3]
        return float((z.argmax(-1) == v).mean())

    kp = np.concatenate([kA, kD]); vp = np.concatenate([vA, vD])
    lr = 0.2
    for t in range(steps):
        i = rng.integers(0, len(kp), 32)
        p = sgd(p, kp[i], vp[i], lr)
        if t % 200 == 0 and min(acc(p, kA, vA), acc(p, kD, vD)) >= 0.99:
            break
    print(f"  K={K}: pretrained to A={acc(p, kA, vA):.2f} D={acc(p, kD, vD):.2f} at step {t}")
    ilr = lr / 13.33
    log = []
    Wn = np.asarray(p["W"]); Un = np.asarray(p["U"])
    H, INP, _, _ = fwd_np(Wn, Un, kA, g)
    P0 = np.stack([INP[j] @ Wn[j] for j in range(K)]).mean(1)
    for t in range(inject_steps + 1):
        if t % 10 == 0 or t < 20:
            Wn = np.asarray(p["W"]); Un = np.asarray(p["U"])
            H, INP, _, _ = fwd_np(Wn, Un, kA, g)
            P = np.stack([INP[j] @ Wn[j] for j in range(K)]).mean(1)
            m = np.stack([INP[j].mean(0) for j in range(K)])          # A-mean input
            Hs = np.stack([H[j] for j in range(K)])                   # per-block A states
            BB = backprop(Wn, Un, kB, vB, g)
            ex, fi, cj = restoring_dirs(BB, Wn, g)
            dP = P - P0; u = dP / np.clip(np.linalg.norm(dP, axis=-1, keepdims=True), 1e-30, None)
            Pu = P / np.linalg.norm(P, axis=-1, keepdims=True)
            log.append(dict(t=t, A=acc(p, kA, vA), B=acc(p, kB, vB),
                            W=Wn.copy(), U=Un.copy(), P=P.copy(), m=m.copy(), H=Hs.copy(),
                            rho=rho_blocks(BB, Wn, g),
                            cos_rest=[float(ex[i] @ ex[j]) for i in range(K) for j in range(i + 1, K)],
                            cos_dP=[float(u[i] @ u[j]) for i in range(K) for j in range(i + 1, K)],
                            cos_P=[float(Pu[i] @ Pu[j]) for i in range(K) for j in range(i + 1, K)],
                            Pnorm=np.linalg.norm(P, axis=-1)))
        i = rng.integers(0, nB, 32)
        p = sgd(p, kB[i], vB[i], ilr)
    return log, (kA, vA), (kB, vB), g


def drift_vs_weight(log, g, d):
    """d P_j/dt = g sqrt(d) [ (dm_j/dt) W_j + m_j (dW_j/dt) ] + O(dt^2)."""
    K = log[0]["P"].shape[0]
    print(f"\n  {'t':>6} {'blk':>4} {'|dP|':>10} {'|drift|':>10} {'|weight|':>10} "
          f"{'drift/wt':>9} {'resid':>9} {'rho_j':>7} {'ang/wgrow':>10} "
          f"{'|dm| err':>9} {'d|P|/|P|':>9} {'dang(P)':>8}")
    out = []
    for a, b in zip(log[:-1], log[1:]):
        if a["t"] not in (0, 10, 40, 100, 200, 400) and a["t"] % 200:
            continue
        for j in range(K):
            dm = b["m"][j] - a["m"][j]
            dW = b["W"][j] - a["W"][j]
            dP = b["P"][j] - a["P"][j]
            drift = dm @ a["W"][j]
            wgt = a["m"][j] @ dW
            res = relerr(dP, drift + wgt)
            nd_, nw = np.linalg.norm(drift), np.linalg.norm(wgt)
            angvel = np.linalg.norm(dm) / max(np.linalg.norm(a["m"][j]), 1e-30)
            wgrow = np.linalg.norm(dW) / max(np.linalg.norm(a["W"][j]), 1e-30)
            # is the input drift exactly the ORTHOGONAL part of the upstream write, /||h_j||?
            Ha, Hb = a["H"][j], b["H"][j]
            hn = np.linalg.norm(Ha, axis=-1, keepdims=True); hh = Ha / hn
            dh = Hb - Ha
            dm_pred = (g * np.sqrt(d) *
                       ((dh - (dh * hh).sum(-1, keepdims=True) * hh) / hn).mean(0))
            e_dm = relerr(dm, dm_pred) if np.linalg.norm(dm) > 0 else 0.0
            # norm preservation vs rotation of the contribution itself
            nP0, nP1 = np.linalg.norm(a["P"][j]), np.linalg.norm(b["P"][j])
            dnorm = (nP1 - nP0) / max(nP0, 1e-30)
            dang = float(np.arccos(np.clip(a["P"][j] @ b["P"][j] / max(nP0 * nP1, 1e-30),
                                           -1, 1)))
            print(f"  {a['t']:>6} {j:>4} {np.linalg.norm(dP):>10.3e} {nd_:>10.3e} {nw:>10.3e} "
                  f"{nd_/max(nw,1e-30):>9.3f} {res:>9.2e} {a['rho'][j]:>7.4f} "
                  f"{angvel/max(wgrow,1e-30):>10.3f} {e_dm:>9.2e} {dnorm:>+9.2e} {dang:>8.2e}")
            out.append((a["t"], j, nd_, nw, res, a["rho"][j], angvel, wgrow))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "point"
    if mode == "point":
        for (K, d, V, n, g, sc) in [(4, 32, 16, 8, 1.0, 0.2), (4, 32, 16, 8, 1.0, 1.0),
                                    (8, 32, 16, 8, 1.0, 0.5), (2, 32, 16, 8, 1.0, 0.5),
                                    (4, 64, 32, 50, 1.0, 0.3)]:
            print(f"\n=== K={K} d={d} V={V} n={n} g={g} W-scale={sc} ===")
            B, W, U, k, v, errs = check_point(0, K, d, V, n, g, sc)
            report_dirs(B, W, g, tag=f"K={K} sc={sc}")
    elif mode == "traj":
        d, V, nA, nB, g = 64, 32, 40, 40, 1.0
        for K in (2, 4, 8):
            print(f"\n=== trajectory K={K} d={d} ===")
            log, A, Bp, _ = run_traj(0, K, d, V, nA, nB, g, 6000, 600)
            print(f"  {'t':>6} {'A':>5} {'B':>5} {'mean rho':>9} "
                  f"{'mean cos(rest)':>15} {'mean cos(dP)':>13} {'mean cos(P)':>12}")
            for r in log:
                if r["t"] in (0, 10, 40, 100, 200, 400, 600):
                    print(f"  {r['t']:>6} {r['A']:>5.2f} {r['B']:>5.2f} {r['rho'].mean():>9.4f} "
                          f"{np.mean(r['cos_rest']):>15.5f} {np.mean(r['cos_dP']):>13.5f} "
                          f"{np.mean(r['cos_P']):>12.5f}")
            drift_vs_weight(log, g, d)
    elif mode == "scale":
        # Does the FIRST-ORDER formula converge as the write shrinks, and does the
        # conjectured "projected hhat_K" form ever match the exact restoring direction?
        K, d, V, n, g = 4, 32, 16, 8, 1.0
        print(f"{'W-scale':>8} {'rho_1..K-1':>26} {'1-cos(0,K-1) exact':>19} "
              f"{'1st-ord':>9} {'pred r^2(K-1)/2':>16} {'cos(exact,conj)':>16} "
              f"{'cos(exact,1st)':>15}")
        for sc in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            B, W, U, k, v, errs = check_point(0, K, d, V, n, g, sc, verbose=False)
            assert max(errs.values()) < 1e-12
            ex, fi, cj = restoring_dirs(B, W, g)
            rho = rho_blocks(B, W, g)
            rb = rho[1:].mean()
            ce = 1 - float(ex[0] @ ex[K - 1]); cf = 1 - float(fi[0] @ fi[K - 1])
            print(f"{sc:>8} {np.round(rho[1:], 5).tolist()!s:>26} {ce:>19.3e} "
                  f"{cf:>9.3e} {0.5*rb**2*(K-1):>16.3e} "
                  f"{np.mean([float(ex[j] @ cj[j][0]) for j in range(K)]):>16.5f} "
                  f"{np.mean([float(ex[j] @ fi[j]) for j in range(K)]):>15.7f}")
    elif mode == "gradtraj":
        # C1-C3 held not just at a random point but at every step of an SGD trajectory.
        for K in (2, 4, 8):
            d, V, n, g = 32, 16, 16, 1.0
            r = np.random.default_rng(3)
            mu = r.normal(size=d); mu /= np.linalg.norm(mu)
            gg = r.normal(size=(n, d)); gg /= np.linalg.norm(gg, axis=1, keepdims=True)
            k = np.sqrt(.5) * mu[None] + np.sqrt(.5) * gg
            v = r.integers(0, V, n)
            p = {"W": jnp.asarray(r.normal(size=(K, d, d)) * 0.02 / np.sqrt(d)),
                 "U": jnp.asarray(r.normal(size=(d, V)) / np.sqrt(d))}
            gfn = jax.jit(jax.grad(loss_jnp), static_argnums=(4,))
            eW = eU = esp = epr = 0.0
            for t in range(400):
                Wn = np.asarray(p["W"]); Un = np.asarray(p["U"])
                B = backprop(Wn, Un, k, v, g)
                gr = gfn(p, jnp.asarray(k), jnp.asarray(v), g, K)
                eW = max(eW, relerr(np.asarray(gr["W"]), B["gW"]))
                eU = max(eU, relerr(np.asarray(gr["U"]), B["gU"]))
                esp = max(esp, relerr(B["rK"], B["plain"] + B["anti"]))
                for j in range(K):
                    acc = B["rK"].copy()
                    for m in range(K - 1, j, -1):
                        acc = acc + g * Japply(acc @ Wn[m].T, B["H"][m])
                    epr = max(epr, relerr(B["R"][j + 1], acc))
                p = jax.tree_util.tree_map(lambda a, b: a - 0.05 * b, p, gr)
            print(f"K={K} over 400 SGD steps: max rel err  dL/dW {eW:.2e}  dL/dU {eU:.2e}  "
                  f"anti-split {esp:.2e}  Jacobian-product {epr:.2e}")
    else:
        raise SystemExit("mode: point | traj | scale | gradtraj")


if __name__ == "__main__":
    main()
