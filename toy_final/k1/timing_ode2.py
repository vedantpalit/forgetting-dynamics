"""TASK 3 (corrected): the reduced ODE for the crash-recovery cycle.

WHY THE FIRST ATTEMPT FAILED, and it is worth recording. Driving the system with m itself
does not work because M_b = <z_b, e_y - p_b> is NOT monotone. Measured, it starts at -9.4,
crosses zero at step 67, peaks at +0.63 near step 140, and decays back to ~0.01 by step
6000. There is no "B is learned, so m is large and positive" state: eq (7)'s anti-state
force is a TRANSIENT, on only while B is partly-but-not-fully confident.

So the state is (x, G) with G = B's true logit gap, which IS monotone, and m is derived:

    u(G)  = 1 / (1 + (V-1) e^{-G})                  confidence, V-1 equally placed rivals
    m(G)  = G (1 - u)                               eq (7)'s coefficient -- DERIVED
    n(x)  = sqrt(h0^2 + d x^2)                      ||h_b||, from h = k + g sqrt(d) k_hat W

    dG/dtau = c1 (1 - u) d / n(x)^2                          two Jacobian factors, eq (7)
    dx/dtau = a1 (1 - u) sqrt(d) / n(x)  -  a2 m(G) x / n(x)^2
              the plain term of (7)          the anti-state term of (7)

tau = eta t, so eta appears NOWHERE else: rate-independence is imposed by construction, and
the out-of-sample test is whether the SHAPE is right at rates the fit never saw.
THREE fitted constants (a1, a2, c1), fitted once on dose=1 seed=0. h0 and G(0) are measured
per run.
"""
import json
import numpy as np
from scipy.optimize import minimize

D = json.load(open("timing_ode_data.json"))
VV, DD = 32.0, 128.0
NSTEP = 4000


def u_of(G):
    return 1.0 / (1.0 + (VV - 1.0) * np.exp(-np.clip(G, -60, 60)))


def deriv(x, G, a1, a2, c1, h0):
    u = u_of(G); e = 1.0 - u; m = G * e
    n2 = h0 * h0 + DD * x * x
    return a1 * e * np.sqrt(DD) / np.sqrt(n2) - a2 * m * x / n2, c1 * e * DD / n2


def sim(p, ilr, ts, G0, h0):
    """Fixed-step RK4 in tau = eta*t, then sampled onto the checkpoint grid."""
    a1, a2, c1 = p
    tau = np.asarray(ts, float) * ilr
    T = float(tau[-1]); dt = T / NSTEP
    x, G = 0.0, float(G0)
    xs = np.empty(NSTEP + 1); Gs = np.empty(NSTEP + 1)
    xs[0] = x; Gs[0] = G
    for i in range(NSTEP):
        k1x, k1g = deriv(x, G, a1, a2, c1, h0)
        k2x, k2g = deriv(x + .5 * dt * k1x, G + .5 * dt * k1g, a1, a2, c1, h0)
        k3x, k3g = deriv(x + .5 * dt * k2x, G + .5 * dt * k2g, a1, a2, c1, h0)
        k4x, k4g = deriv(x + dt * k3x, G + dt * k3g, a1, a2, c1, h0)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        G += dt / 6 * (k1g + 2 * k2g + 2 * k3g + k4g)
        xs[i + 1] = x; Gs[i + 1] = G
    g = np.linspace(0.0, T, NSTEP + 1)
    X = np.interp(tau, g, xs); GG = np.interp(tau, g, Gs)
    return X, GG * (1 - u_of(GG))


def curve(k):
    r = D[k]
    return (np.array([q["step"] for q in r["rows"]], float),
            np.array([q["x"] for q in r["rows"]], float),
            np.array([q["m"] for q in r["rows"]], float),
            np.array([q["hn"] for q in r["rows"]], float), r["ilr"])


FIT = "dose1.0-s0"
ts, xo, mo, hno, ilr = curve(FIT)
h0, G0 = hno[0], mo[0]


def obj(lp):
    try:
        xs, ms = sim(np.exp(lp), ilr, ts, G0, h0)
    except Exception:
        return 1e9
    if not np.isfinite(xs).all():
        return 1e9
    return float(np.mean((xs - xo) ** 2) / np.var(xo) + np.mean((ms - mo) ** 2) / np.var(mo))


best, bv = None, 1e18
for g in [(5.0, 50.0, 5.0), (50.0, 2000.0, 50.0)]:
    r = minimize(obj, np.log(g), method="Nelder-Mead",
                 options=dict(maxiter=300, maxfev=300, xatol=1e-3, fatol=1e-5))
    if r.fun < bv:
        bv, best = r.fun, np.exp(r.x)
print("=" * 88)
print(f"FIT on {FIT} ONLY.  a1 = {best[0]:.4g}  a2 = {best[1]:.4g}  c1 = {best[2]:.4g}"
      f"   (normalized SSE {bv:.3f})")
print(f"measured, not fitted: h0 and G(0) per run, d = {DD:.0f}, V = {VV:.0f}\n")


def summ(t, x):
    j = int(np.argmax(x)); peak, end = x[j], x[-1]
    tgt = peak - 0.5 * (peak - end)
    st_, sx = t[j:], x[j:]
    k = int(np.argmax(sx <= tgt)) if (sx <= tgt).any() else -1
    tr = (st_[k - 1] + (tgt - sx[k - 1]) * (st_[k] - st_[k - 1]) / (sx[k] - sx[k - 1])
          - t[j]) if k > 0 else np.nan
    return float(t[j]), float(peak), float(end), float(tr)


print("OUT-OF-SAMPLE (no refit; only eta, G(0), h0 change)")
print(f"  {'run':14s} | {'t_peak  m / p':>15s} | {'x_peak  m / p':>15s} | "
      f"{'x_end  m / p':>15s} | {'t_rec50 m / p':>16s} | {'rmse/sd':>7s}")
rows = []
for k in sorted(D):
    t, x_, m_, hn_, il = curve(k)
    xs, ms = sim(best, il, t, m_[0], hn_[0])
    a, b = summ(t, x_), summ(t, xs)
    rm = float(np.sqrt(np.mean((xs - x_) ** 2)) / np.std(x_))
    rows.append((k, a, b, rm, t, m_, ms))
    print(f"  {k:14s} | {a[0]:7.0f} {b[0]:7.0f} | {a[1]:7.3f} {b[1]:7.3f} | "
          f"{a[2]:7.3f} {b[2]:7.3f} | {a[3]:7.0f} {b[3]:8.0f} | {rm:7.2f}"
          + ("  <-- FITTED" if k == FIT else ""))
oos = [r for r in rows if r[0] != FIT]
for i, nm in [(0, "t_peak"), (1, "x_peak"), (2, "x_end"), (3, "t_rec50")]:
    rr = np.array([r[2][i] / r[1][i] for r in oos
                   if np.isfinite(r[1][i]) and np.isfinite(r[2][i]) and r[1][i] != 0])
    print(f"    out-of-sample pred/meas {nm:8s}: median {np.median(rr):.2f}  "
          f"range [{rr.min():.2f}, {rr.max():.2f}]")
print(f"    out-of-sample rmse/sd of x(t): median {np.median([r[3] for r in oos]):.2f}"
      f"  max {max(r[3] for r in oos):.2f}")
print()
print("the margin transient (what eq (7) turns on):")
print(f"  {'run':14s} {'t(m=0) m/p':>14s} {'m_peak m/p':>14s} {'t(m_peak) m/p':>15s}")
for k, a, b, rm, t, m_, ms in rows:
    half = len(t) // 2
    zm = float(np.interp(0.0, m_[:half], t[:half]))
    zp = float(np.interp(0.0, ms[:half], t[:half]))
    print(f"  {k:14s} {zm:6.0f} {zp:7.0f} {m_.max():7.2f} {ms.max():6.2f} "
          f"{t[int(m_.argmax())]:8.0f} {t[int(ms.argmax())]:6.0f}")
