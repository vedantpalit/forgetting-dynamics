"""The smallest system that reverses the shared write: 3 classes, 3 coordinates, 2 rates.

WHY THREE CLASSES.  Two are not enough. With a single competitor the shared bias solves B's
problem outright and the loss is monotone in it, so it can never be asked to shrink. The
reversal needs the competitor set to SPLIT: one competitor the shared bias beats (A's value,
in the other half) and one it cannot touch (the runner-up inside B's own half, which only b's
individual signal can separate).

WHY THREE COORDINATES.  With only two the loss depends on the state's DIRECTION, i.e. on the
single angle phi = atan2(y, x), so

    xdot = lambda_x L'(phi) sin(phi) / rho

changes sign only when L'(phi) does, and a 1-D gradient flow never passes its optimum. x would
be monotone for ANY anisotropy lambda_x/lambda_y. The third coordinate zeta -- the part of B's
pretrained state that already points at A's value -- is what makes the optimal x/y ratio
itself move: while zeta is there, x is the only fast way to hold A's value down; once the
slow individual channel has removed zeta and built y, x is pure dilution and is withdrawn.
Freezing zeta (`lam_z = 0`) removes the reversal entirely, which is the check that it is
load-bearing and not decoration.

GEOMETRY (orthonormal axes a = between-half, e = b's own value, n = A's value):

    h   = x a + y e + zeta n
    u_1 = alpha a + beta_u e       b's own value, in B's half           (target)
    u_2 = alpha a - beta_u e       the runner-up inside B's half        (m_Y of them)
    u_3 = -alpha a + beta_u n      A's value, in the other half         (m_X of them)
    z   = c h_hat U,   p ~ m_v exp(z_v),   L = -log p_1,   c = sqrt(d) ||u||

All three columns have the same norm, so no class is favoured by construction. Multiplicities
m_X, m_Y enter as log-count offsets, which is what having ~16 wrong classes per half does.

RATES.  From the full toy: the shared row S = mu^T W receives every B individual's error
signal with weight sqrt(beta_key), each individual row G_b receives only its own with weight
sqrt(1-beta_key), and h_b reads both back through k_hat_b. The a axis is fed by both, the e
and n axes only by the individual channel, so with time rescaled to lambda_y = 1

    lambda_x  =  1 + beta_key n_B / (1 - beta_key)                                    (LAM)

is the ONLY place n_B and the key overlap enter. x_S/x = lambda_x/(1+lambda_x) is constant in
time (both channels see the same partial derivative), so the SHARED write is proportional to
the a coordinate and has the same reversal fraction -- which is why one coordinate suffices
for both.

THE FORCE, hand-derived and checked against autodiff in verify():

    dL/dh = (c/rho) [ (E_p[u] - u_1) + (M/c) h_hat ],       M = z_1 - E_p[z]
    xdot  = (lambda_x/rho) [ c Phi - M x/rho ],             Phi := alpha - E_p[u_a] = 2 alpha p_3

Phi, the HALF-margin, is the plain term's whole projection on the a axis and is EXACTLY twice
alpha times the probability mass left on the other half. So: the writing force is the residual
other-half mass; the removing force is the confidence margin times the write's own share of
the state. Stationarity of x is therefore

    X := x/rho  =  c Phi / M                                                            (BAL)

and expanding p_3/p_2 in (BAL) with Y := y/rho, Z := zeta/rho gives the closed form

    2 alpha X  =  beta_u (Y + Z)  +  (1/c) ln( alpha m_X / (beta_u m_Y X Y) )            (FP)

to first order in the mass on the other half. Leading order in 1/c: z_3 = z_2. THE WRITE IS
HELD AT EXACTLY THE LEVEL THAT KEEPS A'S VALUE LEVEL WITH B'S OWN RUNNER-UP. The reversal is
the difference between that level with Z (at the turnover) and without it (at the end).
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)


def U_of(alpha, beta_u):
    """columns: u_1 target, u_2 within-half runner-up, u_3 other half. Rows are (a, e, n)."""
    return np.array([[alpha, alpha, -alpha],
                     [beta_u, -beta_u, 0.0],
                     [0.0, 0.0, beta_u]])


def probs(h, U, c, logm):
    z = c * (h / np.linalg.norm(h)) @ U
    a = z + logm
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum(), z


def hand_grad(h, U, c, logm):
    """dL/dh = (c/rho) [ (E_p[u] - u_1) + (M/c) h_hat ]."""
    rho = np.linalg.norm(h)
    hh = h / rho
    p, z = probs(h, U, c, logm)
    M = z[0] - p @ z
    return (c / rho) * ((U @ p - U[:, 0]) + (M / c) * hh), M, p, z


def forces(h, U, c, logm, lam):
    """xdot split into its plain and anti-state pieces, hand-derived (not autodiffed)."""
    rho = np.linalg.norm(h)
    p, z = probs(h, U, c, logm)
    M = z[0] - p @ z
    Phi = U[0, 0] - U[0] @ p                       # half-margin on the a axis = 2 alpha p_3
    plain_x = lam * c * Phi / rho
    anti_x = -lam * M * h[0] / rho ** 2
    return plain_x, anti_x, M, Phi, p, z


def rhs(t, u, U, c, logm, lam, lam_z):
    """Three coordinates at rates (lam, 1, lam_z). lam_z = 0 freezes zeta."""
    g = hand_grad(np.array(u), U, c, logm)[0]
    return [-lam * g[0], -g[1], -lam_z * g[2]]


def integrate(lam=51.0, zeta=1.0, c=11.3, alpha=0.7, beta_u=0.7, mX=16, mY=15,
              x0=0.0, y0=0.0, T=4000.0, n_out=3000, lam_z=1.0):
    U = U_of(alpha, beta_u)
    logm = np.log(np.array([1.0, mY, mX]))
    ts = np.concatenate([np.linspace(0, 20, n_out // 3, endpoint=False),
                         np.linspace(20, T, n_out - n_out // 3)])
    sol = solve_ivp(rhs, (0, T), [x0, y0, zeta], args=(U, c, logm, lam, lam_z),
                    t_eval=ts, rtol=1e-11, atol=1e-13, method="LSODA")
    x, y, zt = sol.y
    out = dict(t=sol.t, x=x, y=y, z=zt, c=c, lam=lam, zeta=zeta, alpha=alpha, beta_u=beta_u,
               mX=mX, mY=mY, lam_z=lam_z, x0=x0)
    rec = []
    for xi, yi, zi in zip(x, y, zt):
        h = np.array([xi, yi, zi])
        px, ax, M, Phi, p, z = forces(h, U, c, logm, lam)
        rec.append((M, Phi, p[0], p[1], p[2], np.linalg.norm(h), z[0], z[1], z[2], px, ax))
    rec = np.array(rec)
    for i, k in enumerate(["M", "Phi", "p1", "p2", "p3", "rho", "z1", "z2", "z3",
                           "F_plain", "F_anti"]):
        out[k] = rec[:, i]
    pk = int(np.argmax(x))
    out["peak"] = pk
    out["x_peak"] = float(x[pk])
    out["x_end"] = float(x[-1])
    # the reversal fraction is measured on the WRITE, i.e. on x - x(0)
    out["f"] = float(1 - (x[-1] - x0) / (x[pk] - x0)) if x[pk] > x0 else np.nan
    out["X_end"] = float(x[-1] / out["rho"][-1])
    out["Y_end"] = float(y[-1] / out["rho"][-1])
    out["Z_pk"] = float(zt[pk] / out["rho"][pk])
    out["Y_pk"] = float(y[pk] / out["rho"][pk])
    out["X_pk"] = float(x[pk] / out["rho"][pk])
    # (FP), evaluated at the end: prediction of X from Y, Z and the multiplicities
    out["X_end_FP"] = float(fp_predict(out["Y_end"], 0.0, alpha, beta_u, c, mX, mY))
    out["X_pk_FP"] = float(fp_predict(out["Y_pk"], out["Z_pk"], alpha, beta_u, c, mX, mY))
    return out


def fp_predict(Y, Z, alpha, beta_u, c, mX, mY, iters=200):
    """Solve (FP) for X by fixed-point iteration. Derived, no fitted constant."""
    X = beta_u * (Y + Z) / (2 * alpha)
    for _ in range(iters):
        arg = alpha * mX / (beta_u * mY * max(X, 1e-12) * max(Y, 1e-12))
        X = max(beta_u * (Y + Z) / (2 * alpha) + np.log(max(arg, 1e-12)) / (2 * c * alpha),
                1e-12)
    return X


def verify(seed=0):
    """The hand gradient and the plain/anti-state split against autodiff."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(200):
        alpha, beta_u = rng.uniform(0.3, 1.2, 2)
        c = rng.uniform(1.0, 20.0)
        h = rng.normal(size=3) * rng.uniform(0.1, 5)
        U = U_of(alpha, beta_u)
        logm = np.log(rng.uniform(1, 20, 3))
        logm[0] = 0.0

        def L(hh):
            z = c * (hh / jnp.linalg.norm(hh)) @ jnp.asarray(U)
            a = z + jnp.asarray(logm)
            return -(a[0] - jax.scipy.special.logsumexp(a))

        g = np.asarray(jax.grad(L)(jnp.asarray(h)))
        worst = max(worst, np.linalg.norm(g - hand_grad(h, U, c, logm)[0]) /
                    np.linalg.norm(g))
        px, ax = forces(h, U, c, logm, 1.0)[:2]
        worst = max(worst, abs((px + ax) + g[0]) / max(abs(g[0]), 1e-12))
    return worst


def show(r, tag):
    idx = sorted(set([0, max(r["peak"] // 4, 1), r["peak"] // 2, r["peak"],
                      min(r["peak"] * 2, len(r["t"]) - 1), len(r["t"]) // 4,
                      len(r["t"]) // 2, len(r["t"]) - 1]))
    print(f"\n{tag}: x peak {r['x_peak']:.4f} at t={r['t'][r['peak']]:.3g}, "
          f"end {r['x_end']:.4f}, f={r['f']:.3f}")
    print(f"{'t':>9} {'x':>8} {'y':>8} {'zeta':>8} {'rho':>7} {'M':>8} {'Phi':>10} "
          f"{'p1':>7} {'p3/p2':>7} {'z3-z2':>7} {'cPhi/M':>8} {'x/rho':>7}")
    for i in idx:
        print(f"{r['t'][i]:9.3g} {r['x'][i]:8.4f} {r['y'][i]:8.4f} {r['z'][i]:8.4f} "
              f"{r['rho'][i]:7.3f} {r['M'][i]:8.4f} {r['Phi'][i]:10.3e} {r['p1'][i]:7.4f} "
              f"{r['p3'][i] / r['p2'][i]:7.3f} {r['z3'][i] - r['z2'][i]:+7.3f} "
              f"{r['c'] * r['Phi'][i] / r['M'][i]:8.4f} {r['x'][i] / r['rho'][i]:7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="revfrac_reduced.json")
    a = ap.parse_args()

    err = verify()
    print(f"hand ODE vs autodiff (200 random points): max rel err {err:.2e}")
    assert err < 1e-8

    base = dict(lam=51.0, zeta=1.0, c=11.3, alpha=0.7, beta_u=0.7, mX=16, mY=15)
    scans = {"base": integrate(**base)}
    show(scans["base"], "base")

    print("\nABLATION: freezing zeta (lam_z = 0) removes the reversal.")
    print(f"  {'lam_z':>6} {'x_peak':>8} {'x_end':>8} {'f':>7} {'zeta_end':>9}")
    for lz in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        q = integrate(**{**base, "lam_z": lz})
        scans[f"lamz{lz}"] = q
        print(f"  {lz:6.2f} {q['x_peak']:8.4f} {q['x_end']:8.4f} {q['f']:+7.3f} "
              f"{q['z'][-1]:9.4f}")

    hdr = (f"  {'value':>8} {'x_peak':>8} {'x_end':>8} {'f':>7} | {'X_pk':>6} {'X_end':>6} "
           f"{'Z_pk':>6} {'Y_pk':>6} {'Y_end':>6} | {'FPpk':>6} {'FPend':>6} "
           f"{'f_FP':>6} | {'z3-z2':>6}")
    for nm, vals in [("zeta", [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
                     ("lam", [2.0, 6.0, 11.0, 26.0, 51.0, 101.0, 201.0]),
                     ("c", [4.0, 8.0, 11.3, 16.0, 22.6, 32.0]),
                     ("mY", [1, 3, 7, 15, 31]),
                     ("mX", [1, 4, 16, 64]),
                     ("alpha", [0.3, 0.5, 0.7, 0.9, 1.1]),
                     ("beta_u", [0.3, 0.5, 0.7, 0.9, 1.1]),
                     ("x0", [-1.0, -0.5, -0.2, 0.0, 0.2])]:
        print(f"\n{nm} scan:")
        print(hdr)
        for v in vals:
            q = integrate(**{**base, nm: v})
            scans[f"{nm}{v}"] = q
            # f predicted from (FP) alone: rho ratio measured, X from the fixed point
            f_fp = 1 - (q["rho"][-1] * q["X_end_FP"] - q["x0"]) / \
                       (q["rho"][q["peak"]] * q["X_pk_FP"] - q["x0"])
            print(f"  {v:8.3g} {q['x_peak']:8.4f} {q['x_end']:8.4f} {q['f']:+7.3f} | "
                  f"{q['X_pk']:6.3f} {q['X_end']:6.3f} {q['Z_pk']:6.3f} {q['Y_pk']:6.3f} "
                  f"{q['Y_end']:6.3f} | {q['X_pk_FP']:6.3f} {q['X_end_FP']:6.3f} "
                  f"{f_fp:+6.3f} | {q['z3'][-1] - q['z2'][-1]:+6.3f}")

    json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in v.items()} for k, v in scans.items()},
              open(a.out, "w"))
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
