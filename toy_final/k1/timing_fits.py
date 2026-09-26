"""TASK 1: do the crash and the write's peak happen on B's clock?"""
import numpy as np
from timing_extract import load_all

np.set_printoptions(suppress=True)


def loglog(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = np.log(x[m]), np.log(y[m])
    if len(x) < 3:
        return dict(n=len(x), slope=np.nan, icpt=np.nan, r2=np.nan)
    s, c = np.polyfit(x, y, 1)
    pred = s * x + c
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return dict(n=int(m.sum()), slope=float(s), icpt=float(c), r2=float(r2),
                mult=float(np.exp(c)))


def ratio_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    r = y[m] / x[m]
    return float(np.median(r)), float(r.mean()), float(r.std()), r.min(), r.max()


rows = load_all()
print(f"loaded {len(rows)} runs\n")

for arm, lbl in [(1, "NORMALIZED (norm=1)"), (0, "LINEAR (norm=0)")]:
    R = [r for r in rows if r["norm"] == arm]
    print("=" * 78)
    print(f"{lbl}   n = {len(R)} runs over {len(set(r['file'] for r in R))} sweeps")
    tB = [r["t_B_on"] for r in R]
    tBg = [r["t_B_gate"] for r in R]
    for name, y in [("t_trough", [r["t_trough"] for r in R]),
                    ("t_peak(s.w)", [r["t_peak"] for r in R]),
                    ("t_turn", [r["t_turn"] for r in R]),
                    ("t_peak(|s|)", [r["t_s_peak"] for r in R])]:
        f = loglog(tB, y)
        med, mn, sd, lo, hi = ratio_stats(tB, y)
        print(f"  {name:12s} vs t_B(0.5):  slope {f['slope']:+.3f}  R2 {f['r2']:.3f}  "
              f"n {f['n']:3d} | ratio med {med:.2f} mean {mn:.2f}+-{sd:.2f} "
              f"[{lo:.2f},{hi:.2f}]")
    for name, y in [("t_trough", [r["t_trough"] for r in R]),
                    ("t_peak(s.w)", [r["t_peak"] for r in R])]:
        f = loglog(tBg, y)
        med, mn, sd, lo, hi = ratio_stats(tBg, y)
        print(f"  {name:12s} vs t_B(0.99): slope {f['slope']:+.3f}  R2 {f['r2']:.3f}  "
              f"n {f['n']:3d} | ratio med {med:.2f} mean {mn:.2f}+-{sd:.2f}")
    print()

# ---- per-sweep breakdown, normalized arm --------------------------------------
print("=" * 78)
print("PER-SWEEP, normalized arm: t_trough vs t_B(0.5)")
print(f"{'sweep':8s} {'n':>3s} {'slope':>7s} {'R2':>6s} {'c=med(t_tr/t_B)':>16s} "
      f"{'t_B range':>18s}")
for f in sorted(set(r["file"] for r in rows)):
    R = [r for r in rows if r["norm"] == 1 and r["file"] == f]
    fit = loglog([r["t_B_on"] for r in R], [r["t_trough"] for r in R])
    med, *_ = ratio_stats([r["t_B_on"] for r in R], [r["t_trough"] for r in R])
    tb = [r["t_B_on"] for r in R if np.isfinite(r["t_B_on"])]
    print(f"{f[3:-5]:8s} {fit['n']:3d} {fit['slope']:+7.3f} {fit['r2']:6.3f} "
          f"{med:16.2f} {min(tb):8.0f}-{max(tb):<8.0f}")

# ---- dose sweep: t_B vs rate --------------------------------------------------
print()
print("=" * 78)
print("DOSE SWEEP: B's clock against the injection rate")
for arm in (1, 0):
    R = sorted([r for r in rows if r["file"] == "sw_dose.json" and r["norm"] == arm],
               key=lambda r: (r["dose"], r["seed"]))
    print(f"\n  arm norm={arm}")
    print(f"  {'dose':>5s} {'ilr':>9s} {'seed':>4s} {'t_B.5':>7s} {'t_B.99':>7s} "
          f"{'t_trough':>8s} {'t_peak':>7s} {'trough':>7s} {'sw_peak':>7s}")
    for r in R:
        print(f"  {r['dose']:5.2f} {r['ilr']:9.2e} {r['seed']:4d} {r['t_B_on']:7.1f} "
              f"{r['t_B_gate']:7.1f} {r['t_trough']:8.0f} {r['t_peak']:7.0f} "
              f"{r['trough']:7.2f} {r['sw_peak']:7.3f}")
    for lab, key in [("t_B(0.5)", "t_B_on"), ("t_B(0.99)", "t_B_gate"),
                     ("t_trough", "t_trough"), ("t_peak", "t_peak")]:
        fit = loglog([1.0 / r["ilr"] for r in R], [r[key] for r in R])
        print(f"    log {lab:10s} vs log(1/eta): slope {fit['slope']:+.3f} "
              f"(eta-exponent {-fit['slope']:+.3f})  R2 {fit['r2']:.3f}  n {fit['n']}")
