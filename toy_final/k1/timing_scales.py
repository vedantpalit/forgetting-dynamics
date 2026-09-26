"""TASK 4: the three time scales, from the 20000-step dense run and from the sweeps."""
import json
import numpy as np
from timing_extract import load_all2, first_cross
from timing_fits import loglog


def scales(rows, label):
    st = np.array([r["step"] for r in rows], float)
    A = np.array([r["A"] for r in rows], float)
    B = np.array([r["B"] for r in rows], float)
    x = np.array([r["s_on_w"] for r in rows], float)
    eps = np.array([r["eps"] for r in rows], float) / np.array([r["post_norm"] for r in rows])
    dlt = np.array([r["delta"] for r in rows], float) / np.array([r["post_norm"] for r in rows])
    pk = int(np.argmax(x))
    peak, end = x[pk], x[-1]
    t10 = first_cross(st[:pk + 1], x[:pk + 1], 0.1 * peak)
    t90 = first_cross(st[:pk + 1], x[:pk + 1], 0.9 * peak)
    # fall: from the peak to 50% / 90% of the total fall
    def fall(f):
        tgt = peak - f * (peak - end)
        seg_s, seg_y = st[pk:], x[pk:]
        j = int(np.argmax(seg_y <= tgt)) if (seg_y <= tgt).any() else -1
        if j <= 0:
            return np.nan
        return float(np.interp(-tgt, -seg_y[j - 1:j + 1][::-1] * 0 - seg_y[j - 1:j + 1],
                               seg_s[j - 1:j + 1]) - st[pk]) if False else float(
            seg_s[j - 1] + (tgt - seg_y[j - 1]) * (seg_s[j] - seg_s[j - 1]) /
            (seg_y[j] - seg_y[j - 1]) - st[pk])
    t_r50, t_r90 = fall(0.5), fall(0.9)
    e_end = eps[-1]
    te50 = first_cross(st, eps, 0.5 * e_end)
    te90 = first_cross(st, eps, 0.9 * e_end)
    # late-phase power law for eps
    m = (st > 500) & (eps > 0)
    p_eps = np.polyfit(np.log(st[m]), np.log(eps[m]), 1)[0] if m.sum() > 5 else np.nan
    tB = first_cross(st, B, 0.5); tBg = first_cross(st, B, 0.99)
    tr = int(np.argmin(A))
    print(f"--- {label}")
    print(f"    t_B(0.5) {tB:7.1f}   t_B(0.99) {tBg:7.1f}")
    print(f"    t_crash : x 10%->90% = {t90 - t10:7.1f}  (t10 {t10:.0f}, t90 {t90:.0f}), "
          f"peak {peak:.3f} @ {st[pk]:.0f}")
    print(f"    t_recover: peak->50% of fall = {t_r50:7.1f}   ->90% = {t_r90:7.1f}   "
          f"x_end {end:.3f} (fall {100*(1-end/peak):.0f}%)")
    print(f"    t_erode  : eps/|h| -> 50% of final = {te50:7.1f}   ->90% = {te90:7.1f}   "
          f"final {e_end:.3f}   late slope d log eps/d log t = {p_eps:+.2f}")
    print(f"    A: trough {A[tr]:.2f} @ {st[tr]:.0f}, max after trough {A[tr:].max():.2f} "
          f"@ {st[tr + int(np.argmax(A[tr:]))]:.0f}, end {A[-1]:.2f}")
    print(f"    RATIOS   t_recover/t_crash = {t_r50 / (t90 - t10):.2f}   "
          f"t_erode/t_recover = {te50 / t_r50:.2f}   t_crash/t_B(0.5) = "
          f"{(t90 - t10) / tB:.2f}")
    return dict(tB=tB, tBg=tBg, t_crash=t90 - t10, t_peak=float(st[pk]), peak=float(peak),
                t_r50=t_r50, t_r90=t_r90, te50=te50, te90=te90, p_eps=p_eps,
                end=float(end), A_trough=float(A[tr]))


print("=" * 84)
print("LONG RUN: d=128, beta=0.5, n_B=50, |V|=32, normalized arm, 20000 steps, dose 1")
L = json.load(open("timing_long_n1.json"))
res = {k: scales(v["rows"], k) for k, v in L.items()}
print()
print("=" * 84)
print("RATE INDEPENDENCE of the three scales (sw_dose.json, normalized arm, 5000 steps)")
rows = [r for r in load_all2() if r["norm"] == 1 and r["file"] == "sw_dose.json"]
raw = json.load(open("sw_dose.json"))
tab = []
for k, v in raw.items():
    if v["norm"] != 1:
        continue
    st = np.array([r["step"] for r in v["rows"]], float)
    x = np.array([r["s_on_w"] for r in v["rows"]], float)
    B = np.array([r["B"] for r in v["rows"]], float)
    e = np.array([r["eps"] for r in v["rows"]], float) / np.array(
        [r["post_norm"] for r in v["rows"]])
    pk = int(np.argmax(x)); peak, end = x[pk], x[-1]
    t10 = first_cross(st[:pk + 1], x[:pk + 1], 0.1 * peak)
    t90 = first_cross(st[:pk + 1], x[:pk + 1], 0.9 * peak)
    tgt = peak - 0.5 * (peak - end)
    seg_s, seg_y = st[pk:], x[pk:]
    j = int(np.argmax(seg_y <= tgt)) if (seg_y <= tgt).any() else -1
    tr50 = (seg_s[j - 1] + (tgt - seg_y[j - 1]) * (seg_s[j] - seg_s[j - 1]) /
            (seg_y[j] - seg_y[j - 1]) - st[pk]) if j > 0 else np.nan
    te = first_cross(st, e, 0.5 * e[-1])
    tab.append((v["dose"], v["seed"], v["inject_lr"], first_cross(st, B, 0.5),
                t90 - t10, tr50, te, peak, end))
tab.sort()
print(f"  {'dose':>5s} {'seed':>4s} {'t_B.5':>7s} {'t_crash':>8s} {'t_rec50':>8s} "
      f"{'t_erode':>8s} {'rec/crash':>9s} {'ero/rec':>8s} {'x_peak':>7s} {'x_end':>6s}")
for d_, s_, ilr, tb, tc, trr, te, pkv, ev in tab:
    print(f"  {d_:5.2f} {s_:4d} {tb:7.1f} {tc:8.1f} {trr:8.1f} {te:8.1f} "
          f"{trr / tc:9.2f} {te / trr:8.2f} {pkv:7.3f} {ev:6.3f}")
arr = np.array([(t[0], t[4], t[5], t[6], t[5] / t[4]) for t in tab if np.isfinite(t[5])])
for i, nm in [(1, "t_crash"), (2, "t_recover"), (3, "t_erode"), (4, "t_rec/t_crash")]:
    f = loglog(1.0 / arr[:, 0], arr[:, i])
    print(f"    {nm:14s} ~ eta^{-f['slope']:+.3f}  R2 {f['r2']:+.3f}   "
          f"(rate-independent would be 0)")

print()
print("=" * 84)
print("HOW THE RATIO MOVES WITH beta AND n_B (gate-matched sweeps, normalized arm)")
for fn, axis in [("sw_beta.json", "beta"), ("sw_nb.json", "n_B")]:
    raw = json.load(open(fn))
    print(f"  -- {axis}")
    print(f"  {'val':>6s} {'t_crash':>8s} {'t_rec50':>8s} {'rec/crash':>9s} "
          f"{'x_peak':>7s} {'fall%':>6s} {'t_erode':>8s}")
    agg = {}
    for k, v in raw.items():
        if v["norm"] != 1:
            continue
        st = np.array([r["step"] for r in v["rows"]], float)
        x = np.array([r["s_on_w"] for r in v["rows"]], float)
        e = np.array([r["eps"] for r in v["rows"]], float) / np.array(
            [r["post_norm"] for r in v["rows"]])
        pk = int(np.argmax(x)); peak, end = x[pk], x[-1]
        if peak < 0.1 or end >= peak:
            continue
        t10 = first_cross(st[:pk + 1], x[:pk + 1], 0.1 * peak)
        t90 = first_cross(st[:pk + 1], x[:pk + 1], 0.9 * peak)
        tgt = peak - 0.5 * (peak - end)
        seg_s, seg_y = st[pk:], x[pk:]
        j = int(np.argmax(seg_y <= tgt)) if (seg_y <= tgt).any() else -1
        if j <= 0 or not np.isfinite(t10) or t90 <= t10:
            continue
        tr50 = (seg_s[j - 1] + (tgt - seg_y[j - 1]) * (seg_s[j] - seg_s[j - 1]) /
                (seg_y[j] - seg_y[j - 1]) - st[pk])
        agg.setdefault(v["value"], []).append(
            (t90 - t10, tr50, peak, 1 - end / peak, first_cross(st, e, 0.5 * e[-1])))
    for val in sorted(agg):
        a = np.array(agg[val])
        print(f"  {val:6g} {a[:,0].mean():8.1f} {a[:,1].mean():8.1f} "
              f"{(a[:,1]/a[:,0]).mean():9.2f} {a[:,2].mean():7.3f} "
              f"{100*a[:,3].mean():6.0f} {np.nanmean(a[:,4]):8.1f}")
