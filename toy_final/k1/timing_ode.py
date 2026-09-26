"""TASK 3: a 2-variable reduced ODE for the crash-recovery cycle.

STATE   x = the shared write's between-half content  (s . w, THEORY's primary observable)
        m = B's mean confidence margin               (the quantity eq (7)'s second term uses)

FORCES, read off THEORY eq (3) and eq (7).  Write tau = eta * t; eta then appears NOWHERE
else, which IS the derived statement of Section 8's dose invariance.

  dx/dtau =  a1 * (1 - sigma(m))        the plain term of (7): proportional to the mass B
                                        still puts on the wrong half, which is 1 - sigma(m)
           -  a2 * m * x                the anti-state term of (7): coefficient M_b, acting
                                        along h_hat, i.e. proportional to the write itself

  dm/dtau =  c1 * (1 - sigma(m))        B's own learning; the force is the error (p - e),
                                        which is also 1 - sigma(m)

THREE constants (a1, a2, c1). m(0) is MEASURED per seed, not fitted. Fitted ONCE on
dose = 1, seed = 0; every other curve is a prediction with no refit.

Late-time behaviour is not imposed: m ~ log(tau) falls out of dm/dtau ~ e^{-m}, and then
x* = a1 (1-sigma(m)) / (a2 m) decays slowly rather than to zero -- the measured plateau.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

D = json.load(open("timing_ode_data.json"))


def rhs(tau, y, a1, a2, c1):
    x, m = y
    e = 1.0 / (1.0 + np.exp(np.clip(m, -50, 50)))      # 1 - sigma(m)
    return [a1 * e - a2 * m * x, c1 * e]


def simulate(p, ilr, ts, m0):
    a1, a2, c1 = p
    tau = np.asarray(ts) * ilr
    s = solve_ivp(rhs, (0, tau[-1]), [0.0, m0], t_eval=tau, args=(a1, a2, c1),
                  method="LSODA", rtol=1e-7, atol=1e-9, max_step=np.inf)
    return s.y[0], s.y[1]


def curve(key):
    r = D[key]
    ts = np.array([q["step"] for q in r["rows"]], float)
    x = np.array([q["x"] for q in r["rows"]], float)
    m = np.array([q["m"] for q in r["rows"]], float)
    return ts, x, m, r["ilr"]


FIT = "dose1.0-s0"
ts, xo, mo, ilr = curve(FIT)


def obj(lp):
    p = np.exp(lp)
    try:
        xs, ms = simulate(p, ilr, ts, mo[0])
    except Exception:
        return 1e9
    if len(xs) != len(ts):
        return 1e9
    return float(np.mean((xs - xo) ** 2) / np.var(xo) +
                 np.mean((ms - mo) ** 2) / np.var(mo))


best, bv = None, 1e18
for g in [(2, .2, 2), (10, 1, 10), (50, .5, 50), (200, 2, 200), (500, 5, 500)]:
    r = minimize(obj, np.log(g), method="Nelder-Mead",
                 options=dict(maxiter=3000, xatol=1e-6, fatol=1e-9))
    if r.fun < bv:
        bv, best = r.fun, np.exp(r.x)
a1, a2, c1 = best
print("=" * 84)
print(f"FIT on {FIT} only.  a1 = {a1:.4g}   a2 = {a2:.4g}   c1 = {c1:.4g}   "
      f"(normalized SSE {bv:.4f})")
print(f"m(0) taken from data per run (not fitted).")
print()


def summar(t, x):
    j = int(np.argmax(x))
    peak, end = x[j], x[-1]
    tgt = peak - 0.5 * (peak - end)
    seg_t, seg_x = t[j:], x[j:]
    k = int(np.argmax(seg_x <= tgt)) if (seg_x <= tgt).any() else -1
    tr = (seg_t[k - 1] + (tgt - seg_x[k - 1]) * (seg_t[k] - seg_t[k - 1]) /
          (seg_x[k] - seg_x[k - 1]) - t[j]) if k > 0 else np.nan
    return float(t[j]), float(peak), float(end), float(tr)


print("OUT-OF-SAMPLE PREDICTION (no refit; the ODE is the same in tau = eta*t)")
print(f"  {'run':14s} | {'t_peak m/p':>16s} | {'x_peak m/p':>15s} | {'x_end m/p':>15s} "
      f"| {'t_rec50 m/p':>16s} | {'rmse/sd':>7s}")
for key in sorted(D):
    t, xo_, mo_, il = curve(key)
    xs, ms = simulate(best, il, t, mo_[0])
    a = summar(t, xo_); b = summar(t, xs)
    rm = np.sqrt(np.mean((xs - xo_) ** 2)) / np.std(xo_)
    print(f"  {key:14s} | {a[0]:7.0f} {b[0]:8.0f} | {a[1]:7.3f} {b[1]:7.3f} | "
          f"{a[2]:7.3f} {b[2]:7.3f} | {a[3]:7.0f} {b[3]:8.0f} | {rm:7.2f}"
          + ("   <-- FITTED" if key == FIT else ""))
print()
print("m(t) check (margin crossing zero, which is what turns the write over):")
print(f"  {'run':14s} {'t(m=0) meas':>12s} {'t(m=0) pred':>12s} {'m_end meas':>11s} "
      f"{'m_end pred':>11s}")
for key in sorted(D):
    t, xo_, mo_, il = curve(key)
    xs, ms = simulate(best, il, t, mo_[0])
    def zc(mm):
        return float(np.interp(0.0, mm, t)) if (mm > 0).any() else np.nan
    print(f"  {key:14s} {zc(mo_):12.0f} {zc(ms):12.0f} {mo_[-1]:11.2f} {ms[-1]:11.2f}")
