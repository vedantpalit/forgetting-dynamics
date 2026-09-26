"""TASK 1, cleaned: the crash clock against B's clock."""
import numpy as np
from timing_extract import load_all2
from timing_fits import loglog, ratio_stats

rows = load_all2()
N1 = [r for r in rows if r["norm"] == 1]
U1 = [r for r in N1 if r["usable"]]
print(f"normalized arm: {len(N1)} runs, {len(U1)} usable (0.15 <= crash depth <= 0.97)\n")

print("=" * 84)
print("A. POOLED, all six sweeps, normalized arm, usable runs")
for name, key in [("t_trough(win)", "t_trough_w"), ("t_A_half", "t_A_half"),
                  ("t_peak(s.w)", "t_peak"), ("t_turn", "t_turn")]:
    for bk, bl in [("t_B_on", "B=0.5"), ("t_B_gate", "B=0.99")]:
        f = loglog([r[bk] for r in U1], [r[key] for r in U1])
        med, mn, sd, lo, hi = ratio_stats([r[bk] for r in U1], [r[key] for r in U1])
        print(f"  {name:14s} vs t_B({bl:5s}): slope {f['slope']:+.3f} R2 {f['r2']:+.3f} "
              f"n {f['n']:3d} | c med {med:5.2f} mean {mn:5.2f}+-{sd:.2f} "
              f"[{lo:.2f},{hi:.2f}]")

print()
print("=" * 84)
print("B. WHY THE POOLED R2 IS LOW: the gate-matched sweeps pin t_B by construction.")
print("   Leverage = spread of the predictor. CV = sd/mean in log space (dex).")
print(f"  {'sweep':7s} {'n':>3s} {'t_B .5 dex':>11s} {'t_trough dex':>13s} "
      f"{'t_peak dex':>11s} {'slope':>7s} {'R2':>6s} {'c':>6s}")
for f in sorted(set(r["file"] for r in U1)):
    R = [r for r in U1 if r["file"] == f]
    if len(R) < 4:
        continue
    def dex(k):
        v = np.array([r[k] for r in R], float)
        v = v[np.isfinite(v) & (v > 0)]
        return np.std(np.log10(v))
    fit = loglog([r["t_B_on"] for r in R], [r["t_trough_w"] for r in R])
    med, *_ = ratio_stats([r["t_B_on"] for r in R], [r["t_trough_w"] for r in R])
    print(f"  {f[3:-5]:7s} {len(R):3d} {dex('t_B_on'):11.3f} {dex('t_trough_w'):13.3f} "
          f"{dex('t_peak'):11.3f} {fit['slope']:+7.3f} {fit['r2']:+6.3f} {med:6.2f}")

print()
print("=" * 84)
print("C. THE ONE SWEEP WITH LEVERAGE: dose (16x rate range), normalized arm")
R = [r for r in U1 if r["file"] == "sw_dose.json"]
for name, key in [("t_B(0.5)", "t_B_on"), ("t_B(0.99)", "t_B_gate"),
                  ("t_trough", "t_trough_w"), ("t_A_half", "t_A_half"),
                  ("t_peak(s.w)", "t_peak"), ("t_turn", "t_turn")]:
    f = loglog([1.0 / r["ilr"] for r in R], [r[key] for r in R])
    print(f"  {name:12s} ~ eta^{-f['slope']:+.3f}   R2 {f['r2']:.3f}  n {f['n']}")
print()
for name, key in [("t_trough", "t_trough_w"), ("t_A_half", "t_A_half"),
                  ("t_peak(s.w)", "t_peak"), ("t_turn", "t_turn")]:
    f = loglog([r["t_B_on"] for r in R], [r[key] for r in R])
    med, mn, sd, lo, hi = ratio_stats([r["t_B_on"] for r in R], [r[key] for r in R])
    print(f"  {name:12s} vs t_B(0.5): slope {f['slope']:+.3f} R2 {f['r2']:.3f} | "
          f"c med {med:.2f} mean {mn:.2f}+-{sd:.2f} [{lo:.2f},{hi:.2f}]")
