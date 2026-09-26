"""TASK 2: a derived expression for B's learning time, tested against all six sweeps.

DERIVATION (linear arm, early phase, p_b near uniform).

  dW   = -eta g sqrt(d) sum_c k_hat_c^T r_c ,    r_c = (p_c - e_c) U^T / n_B     [eq (1)]
  dh_b = g sqrt(d) k_hat_b dW = -eta g^2 d sum_c <k_hat_b, k_hat_c> r_c          [sec 3]
  dz_b = dh_b U  and  U^T U = I_V + O(1/sqrt(d))  (unit-norm readout columns), so

      dz_b = -(eta g^2 d / n_B) sum_c <k_hat_b,k_hat_c> (p_c - e_c)

  <k_hat_b,k_hat_b> = 1 and <k_hat_b,k_hat_c> -> beta in the mean (THEORY sec 3), so split
  the sum into b's own term with weight (1-beta) and a coherent term with weight beta over
  all n_B individuals.  The margin M_b = z_{b,y_b} - E_{p_b}[z_b] at uniform p picks up

      own term       (1 - p_{y_b}) + (sum_v p_v^2 - p_{y_b})  ->  1 - 1/V
      shared term    (1/n_B) sum_c (e_c - p_c)_{y_b} = (2/V) - (1/V) = 1/V
                     (B's labels are uniform on the half Y of size V/2)

  Both forces are constant while p is far from e, so M grows LINEARLY in t:

      dM/dt  =  eta g^2 d * R,      R := (1-beta)(1 - 1/V)/n_B  +  beta/V            (D1)
      t_B    =  M* / (eta g^2 d R)                                                    (D2)

  with M* the O(1) margin at which B's accuracy reaches 1/2 -- the ONE fitted constant.

NORMALIZED ARM.  The error signal carries J_c = (sqrt(d)/||h_c||)(I - h_hat h_hat^T) and the
readout carries the same Jacobian again on the way out, so to leading order (dropping the
anti-state term of eq (7), which is O(M) and therefore negligible while M ~ 0)

      dM/dt|norm = (d / ||h||^2) * dM/dt|linear      =>   t_B|norm = (||h||^2/d) * (D2)

  ||h|| is not logged, and pretraining is gate-matched, so (||h||^2/d) is treated as one
  further constant absorbed into M*.  That is a FITTED step, not a derived one.
"""
import numpy as np
from timing_extract import load_all2
from timing_fits import loglog

rows = load_all2()


def R_of(r):
    b, V, nB = r["beta"], r["v"], r["nb"]
    return (1 - b) * (1 - 1.0 / V) / nB + b / V


def report(arm, label):
    R = [r for r in rows if r["norm"] == arm and np.isfinite(r["t_B_on"])]
    x = np.array([1.0 / (r["ilr"] * r["d"] * R_of(r)) for r in R])   # t_B / M*
    y = np.array([r["t_B_on"] for r in R])
    Mstar = float(np.exp(np.mean(np.log(y) - np.log(x))))            # ONE fitted constant
    pred = Mstar * x
    f = loglog(x, y)
    rel = pred / y
    print("=" * 84)
    print(f"{label}   n = {len(R)}")
    print(f"  fitted M* = {Mstar:.3f}   (the only free constant)")
    print(f"  log-log measured vs predicted: slope {f['slope']:+.3f}  R2 {f['r2']:.3f}")
    lr = np.log(rel)
    print(f"  pred/meas: median {np.median(rel):.2f}  geo-sd {np.exp(lr.std()):.2f}x  "
          f"range [{rel.min():.2f}, {rel.max():.2f}]   "
          f"({100*np.mean(np.abs(lr) < np.log(1.5)):.0f}% within 1.5x)")
    print()
    print(f"  {'sweep':7s} {'value':>7s} {'d':>4s} {'nB':>4s} {'V':>4s} {'beta':>5s} "
          f"{'eta':>9s} {'R':>7s} {'t_B meas':>9s} {'t_B pred':>9s} {'p/m':>5s}")
    order = sorted(range(len(R)), key=lambda i: (R[i]["file"], R[i]["value"], R[i]["seed"]))
    seen = {}
    for i in order:
        r = R[i]
        k = (r["file"], r["value"])
        seen.setdefault(k, []).append((y[i], pred[i]))
    for k, v in seen.items():
        v = np.array(v)
        r = next(r for j, r in enumerate(R) if (r["file"], r["value"]) == k)
        print(f"  {k[0][3:-5]:7s} {k[1]:7g} {r['d']:4d} {r['nb']:4d} {r['v']:4d} "
              f"{r['beta']:5.2f} {r['ilr']:9.2e} {R_of(r):7.4f} {v[:,0].mean():9.1f} "
              f"{v[:,1].mean():9.1f} {v[:,1].mean()/v[:,0].mean():5.2f}")
    print()
    # which factor carries the residual?
    resid = np.log(y / pred)
    for nm, vv in [("log d", [np.log(r["d"]) for r in R]),
                   ("log n_B", [np.log(r["nb"]) for r in R]),
                   ("log V", [np.log(r["v"]) for r in R]),
                   ("beta", [r["beta"] for r in R]),
                   ("log eta", [np.log(r["ilr"]) for r in R])]:
        vv = np.array(vv, float)
        if vv.std() < 1e-9:
            continue
        c = np.corrcoef(vv, resid)[0, 1]
        s = np.polyfit(vv, resid, 1)[0]
        print(f"    residual vs {nm:8s}: corr {c:+.2f}  slope {s:+.3f}")
    print()


report(0, "LINEAR ARM (norm=0) -- where (D1)-(D2) is derived")
report(1, "NORMALIZED ARM (norm=1)")
