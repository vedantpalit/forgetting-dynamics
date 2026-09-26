import numpy as np
from timing_extract import load_all
rows = [r for r in load_all() if r["norm"] == 1]
rows.sort(key=lambda r: -(r["t_trough"] / max(r["t_B_on"], 1e-9)))
print(f"{'key':22s} {'file':12s} {'t_B.5':>7s} {'t_tr':>6s} {'ratio':>6s} {'depth':>6s} "
      f"{'t_pk':>6s} {'swpk':>6s} {'Bpk':>5s}")
for r in rows[:14] + rows[-6:]:
    print(f"{str(r['key']):22s} {r['file'][3:-5]:12s} {r['t_B_on']:7.1f} {r['t_trough']:6.0f} "
          f"{r['t_trough']/max(r['t_B_on'],1e-9):6.2f} {1-r['trough']:6.2f} {r['t_peak']:6.0f} "
          f"{r['sw_peak']:6.3f} {r['B_at_peak']:5.2f}")
print()
d = np.array([1 - r["trough"] for r in rows])
print("crash depth quantiles:", np.round(np.quantile(d, [0, .1, .25, .5, .75, 1]), 3))
