"""Analysis of the four injection arms (4L/8L x ballast/no-ballast) from the run logs.

Answers tasks 1-6 of the depth-of-recovery question:
  why does 8L no-ballast recover 39% of its crash and 4L no-ballast 3%?

Everything is parsed from the .out eval lines; nothing is read off a plot.
Run:  .venv/bin/python toy_final/k1/wp_logs_analysis.py
"""
import json
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wp_logs_parse import ARMS, ROOT, load_arm, stack, common_grid  # noqa: E402

MAIN = ["4L_ballast", "4L_noballast", "8L_ballast", "8L_noballast"]
LBL = {k: ARMS[k]["label"] for k in ARMS}
OUT = os.environ.get("WP_OUT", "/tmp/wp_logs")


# ---------------------------------------------------------------- helpers
def interp_cross(x, y, level, rising=True):
    """First crossing of `level` by y, linearly interpolated in x. NaN if never."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    for i in range(1, len(y)):
        if rising and y[i - 1] < level <= y[i]:
            t = (level - y[i - 1]) / (y[i] - y[i - 1])
            return x[i - 1] + t * (x[i] - x[i - 1])
        if not rising and y[i - 1] > level >= y[i]:
            t = (y[i - 1] - level) / (y[i - 1] - y[i])
            return x[i - 1] + t * (x[i] - x[i - 1])
    return np.nan


def at_clock(clock, y, levels):
    """y interpolated at given levels of a monotonised `clock` (cummax, rising part)."""
    clock = np.asarray(clock, float)
    y = np.asarray(y, float)
    ok = np.isfinite(clock) & np.isfinite(y)
    c, yy = clock[ok], y[ok]
    cm = np.maximum.accumulate(c)
    # keep strictly increasing samples so np.interp is well posed
    keep = np.concatenate([[True], np.diff(cm) > 0])
    cm, yy = cm[keep], yy[keep]
    out = []
    for lv in levels:
        out.append(np.interp(lv, cm, yy) if cm[0] <= lv <= cm[-1] else np.nan)
    return np.array(out)


TROUGH_WINDOW = 150   # the transient trough always falls inside the first 150 steps
PEAK_WINDOW = 400     # the recovery peak always falls inside the first 400


def trough_peak(grid, y):
    """(trough_idx, peak_idx) with the paper's convention: the transient minimum in
    the first TROUGH_WINDOW steps, then the recovery maximum before PEAK_WINDOW.
    A global argmin would instead return the endpoint of the slow post-recovery
    decline in the no-ballast arms, which is a different quantity."""
    m = grid <= TROUGH_WINDOW
    ti = int(np.nanargmin(np.where(m, y, np.inf)))
    m2 = (grid > grid[ti]) & (grid <= PEAK_WINDOW)
    if not m2.any():
        return ti, ti
    pi = int(np.nanargmax(np.where(m2, y, -np.inf)))
    return ti, pi


def m_sd(a, axis=0):
    return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)


def fmt(v, n=3):
    return "nan" if not np.isfinite(v) else f"{v:.{n}f}"


# ---------------------------------------------------------------- load
DATA = {}
for k in MAIN + ["4L_noballast_v2"]:
    runs = load_arm(k)
    grid = common_grid(runs)
    DATA[k] = dict(runs=runs, grid=grid)


def arr(k, pop, field):
    a, seeds = stack(DATA[k]["runs"], pop, field, DATA[k]["grid"])
    return a, seeds


# =================================================================== 1. tables
def task1():
    print("=" * 100)
    print("TASK 1 -- per-arm trajectory tables (5-seed mean +- sd)")
    print("=" * 100)
    key_steps = [0, 10, 20, 30, 40, 50, 70, 100, 130, 160, 200, 300, 400, 600, 800, 1000, 1200]
    os.makedirs(OUT, exist_ok=True)
    for k in MAIN:
        grid = DATA[k]["grid"]
        pops = [p for p in ["dataA", "dataB", "ballast", "dataC"] if p in next(iter(DATA[k]["runs"].values()))["pops"]]
        # full CSV dump
        cols = ["step"]
        block = [grid.astype(float)]
        for p in pops:
            for f in ["first_acc", "attr_acc", "ret_own", "rank_own_top1", "halluc_gap"]:
                a, seeds = arr(k, p, f)
                mu, sd = m_sd(a)
                cols += [f"{p}.{f}.mean", f"{p}.{f}.sd"]
                block += [mu, sd]
        M = np.vstack(block).T
        path = os.path.join(OUT, f"traj_{k}.csv")
        np.savetxt(path, M, delimiter=",", header=",".join(cols), comments="", fmt="%.5f")
        print(f"\n--- {LBL[k]}  ({len(DATA[k]['runs'])} seeds; full table -> {path}) ---")
        hdr = f"{'step':>5}"
        for p in pops:
            short = {"dataA": "A", "dataB": "B", "ballast": "Bal", "dataC": "C"}[p]
            hdr += f" | {short+' first':>11} {short+' attr':>11} {short+' rk_own':>11} {short+' hgap':>11}"
        print(hdr)
        idx = {s: i for i, s in enumerate(grid.tolist())}
        for s in key_steps:
            if s not in idx:
                continue
            j = idx[s]
            line = f"{s:>5}"
            for p in pops:
                for f in ["first_acc", "attr_acc", "rank_own_top1", "halluc_gap"]:
                    a, _ = arr(k, p, f)
                    mu, sd = np.nanmean(a[:, j]), np.nanstd(a[:, j])
                    line += f" | {mu:6.3f}±{sd:4.3f}" if False else f" {mu:6.3f}±{sd:.3f}"
            print(line)

    # per-seed summary of the headline numbers
    print("\n--- per-seed headline numbers (A first_acc) ---")
    print(f"{'arm':>16} {'seed':>5} {'trough':>8} {'t_step':>7} {'peak':>8} {'p_step':>7} {'end1200':>8} {'rise':>7}")
    SUMM = {}
    for k in MAIN:
        grid = DATA[k]["grid"]
        a, seeds = arr(k, "dataA", "first_acc")
        rows = []
        for i, sd_ in enumerate(seeds):
            y = a[i]
            ti, pi = trough_peak(grid, y)
            rows.append((sd_, y[ti], grid[ti], y[pi], grid[pi], y[-1], y[pi] - y[ti]))
            print(f"{LBL[k]:>16} {sd_:>5} {y[ti]:>8.3f} {grid[ti]:>7} {y[pi]:>8.3f} "
                  f"{grid[pi]:>7} {y[-1]:>8.3f} {y[pi]-y[ti]:>7.3f}")
        R = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], float)
        SUMM[k] = dict(trough=R[:, 0], trough_step=R[:, 1], peak=R[:, 2],
                       peak_step=R[:, 3], end=R[:, 4], rise=R[:, 5])
        print(f"{LBL[k]:>16} {'MEAN':>5} {R[:,0].mean():>8.3f} {R[:,1].mean():>7.0f} "
              f"{R[:,2].mean():>8.3f} {R[:,3].mean():>7.0f} {R[:,4].mean():>8.3f} {R[:,5].mean():>7.3f}")
    return SUMM


# =================================================================== 2. B acquisition
def task2():
    print("\n" + "=" * 100)
    print("TASK 2 -- B's acquisition on three clocks")
    print("=" * 100)
    print("\n(a) crossing steps, per-seed mean +- sd")
    print(f"{'arm':>16} | {'B first .5':>11} {'B first .9':>11} {'B first .99':>11} | "
          f"{'B attr .5':>11} {'B attr .9':>11} | {'hgap<0':>9}")
    res = {}
    for k in MAIN:
        grid = DATA[k]["grid"]
        bf, seeds = arr(k, "dataB", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        bh, _ = arr(k, "dataB", "halluc_gap")
        rec = {n: [] for n in ["f50", "f90", "f99", "a50", "a90", "h0",
                               "bf_at_a50", "bf_at_a90", "a_at_f50"]}
        for i in range(len(seeds)):
            s_f50 = interp_cross(grid, bf[i], 0.5)
            s_f90 = interp_cross(grid, bf[i], 0.9)
            s_f99 = interp_cross(grid, bf[i], 0.99)
            s_a50 = interp_cross(grid, ba[i], 0.5)
            s_a90 = interp_cross(grid, ba[i], 0.9)
            s_h0 = interp_cross(grid, -bh[i], 0.0)  # halluc gap crossing zero downward
            rec["f50"].append(s_f50); rec["f90"].append(s_f90); rec["f99"].append(s_f99)
            rec["a50"].append(s_a50); rec["a90"].append(s_a90); rec["h0"].append(s_h0)
            rec["bf_at_a50"].append(np.interp(s_a50, grid, bf[i]))
            rec["bf_at_a90"].append(np.interp(s_a90, grid, bf[i]))
            rec["a_at_f50"].append(np.interp(s_f50, grid, ba[i]))
        res[k] = {n: np.array(v, float) for n, v in rec.items()}
        r = res[k]
        print(f"{LBL[k]:>16} | {r['f50'].mean():6.1f}±{r['f50'].std():4.1f} "
              f"{r['f90'].mean():6.1f}±{r['f90'].std():4.1f} "
              f"{r['f99'].mean():6.1f}±{r['f99'].std():4.1f} | "
              f"{r['a50'].mean():6.1f}±{r['a50'].std():4.1f} "
              f"{r['a90'].mean():6.1f}±{r['a90'].std():4.1f} | "
              f"{r['h0'].mean():6.1f}±{r['h0'].std():4.1f}")

    print("\n(b) B's first_acc AT THE MOMENT B's attr_acc crosses 0.5 / 0.9   "
          "(low = cold start closes before individuation)")
    print(f"{'arm':>16} | {'attr0':>7} {'B first @attr=.5':>17} {'B first @attr=.9':>17} "
          f"{'B attr @first=.5':>17}")
    for k in MAIN:
        ba, _ = arr(k, "dataB", "attr_acc")
        r = res[k]
        print(f"{LBL[k]:>16} | {np.nanmean(ba[:,0]):7.3f} "
              f"{r['bf_at_a50'].mean():9.3f}±{r['bf_at_a50'].std():5.3f} "
              f"{r['bf_at_a90'].mean():9.3f}±{r['bf_at_a90'].std():5.3f} "
              f"{r['a_at_f50'].mean():9.3f}±{r['a_at_f50'].std():5.3f}")

    print("\n(c) B's attr_acc and halluc_gap as a function of B's OWN first_acc (dose clock)")
    levels = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    for field, name in [("attr_acc", "B attr_acc"), ("halluc_gap", "B halluc_gap")]:
        print(f"\n  {name} at B first_acc =")
        print(f"{'arm':>16} | " + " ".join(f"{lv:>8.2f}" for lv in levels))
        for k in MAIN:
            bf, seeds = arr(k, "dataB", "first_acc")
            yv, _ = arr(k, "dataB", field)
            vals = np.array([at_clock(bf[i], yv[i], levels) for i in range(len(seeds))])
            mu = np.nanmean(vals, axis=0)
            print(f"{LBL[k]:>16} | " + " ".join(f"{v:8.3f}" for v in mu))

    print("\n(d) cold-start closure fraction: (attr_acc - attr0)/(1 - attr0) at B first_acc =")
    print(f"{'arm':>16} | " + " ".join(f"{lv:>8.2f}" for lv in levels))
    for k in MAIN:
        bf, seeds = arr(k, "dataB", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        vals = []
        for i in range(len(seeds)):
            a0 = ba[i, 0]
            vals.append((at_clock(bf[i], ba[i], levels) - a0) / (1 - a0))
        print(f"{LBL[k]:>16} | " + " ".join(f"{v:8.3f}" for v in np.nanmean(vals, axis=0)))
    return res


# =================================================================== 3. A damage
def task3(SUMM):
    print("\n" + "=" * 100)
    print("TASK 3 -- A's damage on three clocks")
    print("=" * 100)
    print("\n(a) where A's trough sits on B's clocks (per-seed, mean +- sd)")
    print(f"{'arm':>16} | {'trough':>7} {'step':>6} | {'B first @trough':>16} "
          f"{'B attr @trough':>15} {'B hgap @trough':>15}")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        bf, _ = arr(k, "dataB", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        bh, _ = arr(k, "dataB", "halluc_gap")
        tv, ts, v1, v2, v3 = [], [], [], [], []
        for i in range(len(seeds)):
            ti, _ = trough_peak(grid, af[i])
            tv.append(af[i, ti]); ts.append(grid[ti])
            v1.append(bf[i, ti]); v2.append(ba[i, ti]); v3.append(bh[i, ti])
        tv, ts, v1, v2, v3 = map(np.array, (tv, ts, v1, v2, v3))
        print(f"{LBL[k]:>16} | {tv.mean():7.3f} {ts.mean():6.0f} | "
              f"{v1.mean():8.3f}±{v1.std():5.3f} {v2.mean():7.3f}±{v2.std():5.3f} "
              f"{v3.mean():7.2f}±{v3.std():4.2f}")

    levels_f = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99]
    print("\n(b) A first_acc vs B first_acc (dose clock), 5-seed mean")
    print(f"{'arm':>16} | " + " ".join(f"{lv:>7.2f}" for lv in levels_f))
    for k in MAIN:
        af, seeds = arr(k, "dataA", "first_acc")
        bf, _ = arr(k, "dataB", "first_acc")
        vals = np.array([at_clock(bf[i], af[i], levels_f) for i in range(len(seeds))])
        print(f"{LBL[k]:>16} | " + " ".join(f"{v:7.3f}" for v in np.nanmean(vals, 0)))

    print("\n(c) A first_acc vs B attr_acc (cold-start clock), 5-seed mean")
    levels_a = [0.11, 0.20, 0.30, 0.40, 0.51, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(f"{'arm':>16} | " + " ".join(f"{lv:>7.2f}" for lv in levels_a))
    for k in MAIN:
        af, seeds = arr(k, "dataA", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        vals = np.array([at_clock(ba[i], af[i], levels_a) for i in range(len(seeds))])
        print(f"{LBL[k]:>16} | " + " ".join(f"{v:7.3f}" for v in np.nanmean(vals, 0)))

    print("\n(d) fraction of A's total crash (1.000 -> trough) already completed when:")
    print(f"{'arm':>16} | {'B attr half-closed':>19} {'B first=0.05':>13} {'B first=0.50':>13} "
          f"{'B hgap<0':>10}")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        bf, _ = arr(k, "dataB", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        bh, _ = arr(k, "dataB", "halluc_gap")
        f1, f2, f3, f4 = [], [], [], []
        for i in range(len(seeds)):
            ti, _ = trough_peak(grid, af[i])
            trough = af[i, ti]
            denom = af[i, 0] - trough
            a0 = ba[i, 0]
            half = a0 + 0.5 * (1 - a0)
            s_half = interp_cross(grid, ba[i], half)
            s_h0 = interp_cross(grid, -bh[i], 0.0)
            s_f05 = interp_cross(grid, bf[i], 0.05)
            s_f50 = interp_cross(grid, bf[i], 0.5)
            for s_, acc in ((s_half, f1), (s_f05, f2), (s_f50, f3), (s_h0, f4)):
                if not np.isfinite(s_) or denom <= 0:
                    acc.append(np.nan)
                else:
                    acc.append((af[i, 0] - np.interp(s_, grid, af[i])) / denom)
        print(f"{LBL[k]:>16} | {np.nanmean(f1):19.3f} {np.nanmean(f2):13.3f} "
              f"{np.nanmean(f3):13.3f} {np.nanmean(f4):10.3f}")

    print("\n(e) flatness check, 4L no ballast: A first_acc at successive B first_acc doses")
    k = "4L_noballast"
    af, seeds = arr(k, "dataA", "first_acc")
    bf, _ = arr(k, "dataB", "first_acc")
    ba, _ = arr(k, "dataB", "attr_acc")
    lv = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    v_dose = np.nanmean([at_clock(bf[i], af[i], lv) for i in range(len(seeds))], 0)
    print("  B first:", " ".join(f"{x:6.2f}" for x in lv))
    print("  A first:", " ".join(f"{x:6.3f}" for x in v_dose))
    lv2 = [0.11, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    v_attr = np.nanmean([at_clock(ba[i], af[i], lv2) for i in range(len(seeds))], 0)
    print("  B attr :", " ".join(f"{x:6.2f}" for x in lv2))
    print("  A first:", " ".join(f"{x:6.3f}" for x in v_attr))


# =================================================================== 4. suppression signature
def task4(SUMM):
    print("\n" + "=" * 100)
    print("TASK 4 -- A's rank_own_top1 minus first_acc (between-half suppression signature)")
    print("=" * 100)
    print(f"{'arm':>16} | {'@trough: first':>14} {'rk_own':>8} {'gap':>7} | "
          f"{'@peak: first':>13} {'rk_own':>8} {'gap':>7} | {'@1200: first':>13} {'rk_own':>8} {'gap':>7}")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        ar, _ = arr(k, "dataA", "rank_own_top1")
        row = []
        for i in range(len(seeds)):
            ti, pi = trough_peak(grid, af[i])
            row.append([af[i, ti], ar[i, ti], ar[i, ti] - af[i, ti],
                        af[i, pi], ar[i, pi], ar[i, pi] - af[i, pi],
                        af[i, -1], ar[i, -1], ar[i, -1] - af[i, -1]])
        R = np.array(row).mean(0)
        S = np.array(row).std(0)
        print(f"{LBL[k]:>16} | {R[0]:14.3f} {R[1]:8.3f} {R[2]:7.3f} | {R[3]:13.3f} "
              f"{R[4]:8.3f} {R[5]:7.3f} | {R[6]:13.3f} {R[7]:8.3f} {R[8]:7.3f}")
        print(f"{'':>16} | {'(sd)':>14} {'':>8} {S[2]:7.3f} | {'':>13} {'':>8} {S[5]:7.3f} "
              f"| {'':>13} {'':>8} {S[8]:7.3f}")
    print("\n  max gap over the whole run, and the step it occurs:")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        ar, _ = arr(k, "dataA", "rank_own_top1")
        g = ar - af
        mi = [int(np.nanargmax(g[i])) for i in range(len(seeds))]
        print(f"{LBL[k]:>16}: max gap {np.mean([g[i, mi[i]] for i in range(len(seeds))]):.3f} "
              f"at step {np.mean([grid[m] for m in mi]):.0f}; "
              f"ret_own@trough {np.nanmean([arr(k,'dataA','ret_own')[0][i, trough_peak(grid, af[i])[0]] for i in range(len(seeds))]):.3f}")


# =================================================================== 5. double hump
def task5():
    print("\n" + "=" * 100)
    print("TASK 5 -- the 8L no-ballast double hump")
    print("=" * 100)
    k = "8L_noballast"
    grid = DATA[k]["grid"]
    af, seeds = arr(k, "dataA", "first_acc")
    print("\nper-seed A first_acc around the hump (steps 80-400):")
    js = [j for j, s in enumerate(grid) if 80 <= s <= 400]
    print("  step :", " ".join(f"{grid[j]:6d}" for j in js))
    for i, s_ in enumerate(seeds):
        print(f"  seed{s_}:", " ".join(f"{af[i,j]:6.3f}" for j in js))
    print("   mean:", " ".join(f"{np.nanmean(af[:,j]):6.3f}" for j in js))
    print("\nper-seed local peak / dip / second peak (searching 60-180, 150-260, 220-450):")
    def loc(lo, hi, mode):
        out = []
        for i in range(len(seeds)):
            m = (grid >= lo) & (grid <= hi)
            idx = np.where(m)[0]
            j = idx[np.nanargmin(af[i, idx])] if mode == "min" else idx[np.nanargmax(af[i, idx])]
            out.append((grid[j], af[i, j]))
        return out
    for name, (lo, hi, mode) in [("peak1", (60, 180, "max")), ("dip", (150, 260, "min")),
                                 ("peak2", (220, 450, "max"))]:
        v = loc(lo, hi, mode)
        print(f"  {name}: " + " ".join(f"s{seeds[i]}=({v[i][0]},{v[i][1]:.3f})" for i in range(len(v)))
              + f"  mean step {np.mean([x[0] for x in v]):.0f}, acc {np.mean([x[1] for x in v]):.3f}")
    print("\nB's metrics at the hump landmarks (mean over seeds):")
    print(f"{'step':>6} | {'A first':>8} {'B first':>8} {'B attr':>8} {'B rk_own':>9} "
          f"{'B hgap':>8} {'B ret_own':>10} {'C first':>8} {'C attr':>8} {'C hgap':>8}")
    for s in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 250, 280, 300, 350, 400]:
        if s not in grid.tolist():
            continue
        j = grid.tolist().index(s)
        vals = [np.nanmean(arr(k, "dataA", "first_acc")[0][:, j])]
        for f in ["first_acc", "attr_acc", "rank_own_top1", "halluc_gap", "ret_own"]:
            vals.append(np.nanmean(arr(k, "dataB", f)[0][:, j]))
        for f in ["first_acc", "attr_acc", "halluc_gap"]:
            vals.append(np.nanmean(arr(k, "dataC", f)[0][:, j]))
        print(f"{s:>6} | " + " ".join(f"{v:8.3f}" for v in vals))
    print("\nfor comparison, 8L ballast A first_acc over the same window (mean):")
    gk = DATA["8L_ballast"]["grid"]
    a2, _ = arr("8L_ballast", "dataA", "first_acc")
    js2 = [j for j, s in enumerate(gk) if 80 <= s <= 400]
    print("  step :", " ".join(f"{gk[j]:6d}" for j in js2))
    print("  mean :", " ".join(f"{np.nanmean(a2[:,j]):6.3f}" for j in js2))


# =================================================================== 6. exposure confound
def task6(SUMM):
    print("\n" + "=" * 100)
    print("TASK 6 -- pretraining-exposure confound (FINDINGS 3.14, 8L with ballast)")
    print("=" * 100)
    sw = json.load(open(os.path.join(ROOT, "pretrain_sweep_results.json")))
    print(f"{'pstep':>7} {'trough':>8} {'sd':>6} {'peak':>8} {'end':>8} {'rise':>7} {'norm.rec':>9}")
    for r in sw:
        nr = (r['peak'] - r['trough']) / (1.0 - r['trough'])
        print(f"{r['pstep']:>7} {r['trough']:>8.3f} {r['trough_std']:>6.3f} {r['peak']:>8.3f} "
              f"{r['end']:>8.3f} {r['peak']-r['trough']:>7.3f} {nr:>9.3f}")
    print("  norm.rec = (peak - trough) / (1 - trough): the fraction of the crash bought back,")
    print("  which is the quantity the 39%-vs-3% question is about.")
    d = {r["pstep"]: r for r in sw}
    print("\n  effect of doubling pretraining 8000 -> 16000 (both WITH ballast):")
    print(f"    trough {d[8000]['trough']:.3f} -> {d[16000]['trough']:.3f}  "
          f"(delta {d[16000]['trough']-d[8000]['trough']:+.3f})")
    print(f"    peak   {d[8000]['peak']:.3f} -> {d[16000]['peak']:.3f}  "
          f"(delta {d[16000]['peak']-d[8000]['peak']:+.3f})")
    print(f"    end    {d[8000]['end']:.3f} -> {d[16000]['end']:.3f}  "
          f"(delta {d[16000]['end']-d[8000]['end']:+.3f})")
    print(f"    rise   {d[8000]['peak']-d[8000]['trough']:.3f} -> "
          f"{d[16000]['peak']-d[16000]['trough']:.3f}")
    print("\n  the same quantity in our four arms:")
    print(f"{'arm':>16} {'trough':>8} {'peak':>8} {'end':>8} {'rise':>7} {'norm.rec':>9}")
    for k in MAIN:
        t = SUMM[k]['trough'].mean(); pk = SUMM[k]['peak'].mean()
        nr = (pk - t) / (1.0 - t)
        nrs = np.std((SUMM[k]['peak'] - SUMM[k]['trough']) / (1.0 - SUMM[k]['trough']))
        print(f"{LBL[k]:>16} {t:8.3f} {pk:8.3f} {SUMM[k]['end'].mean():8.3f} "
              f"{SUMM[k]['rise'].mean():7.3f} {nr:9.3f} (sd {nrs:.3f})")
    print("\n  observed ballast -> no-ballast deltas in our four arms:")
    for a, b in [("4L_ballast", "4L_noballast"), ("8L_ballast", "8L_noballast")]:
        print(f"    {LBL[a]} -> {LBL[b]}: trough {SUMM[a]['trough'].mean():.3f} -> "
              f"{SUMM[b]['trough'].mean():.3f} (delta {SUMM[b]['trough'].mean()-SUMM[a]['trough'].mean():+.3f}); "
              f"peak {SUMM[a]['peak'].mean():.3f} -> {SUMM[b]['peak'].mean():.3f}; "
              f"end {SUMM[a]['end'].mean():.3f} -> {SUMM[b]['end'].mean():.3f}; "
              f"rise {SUMM[a]['rise'].mean():.3f} -> {SUMM[b]['rise'].mean():.3f}")


# =================================================================== extras
def extra():
    print("\n" + "=" * 100)
    print("EXTRA -- 4L no-ballast long horizon (v2, 3600-step schedule; jobs died early)")
    print("=" * 100)
    runs = DATA["4L_noballast_v2"]["runs"]
    for s in sorted(runs):
        r = runs[s]
        y = r["pops"]["dataA"]["first_acc"]
        g = r["step"]
        ti, pi = trough_peak(g, y)
        a1200 = f"{y[g.tolist().index(1200)]:.3f}" if 1200 in g.tolist() else "n/a"
        print(f"  seed{s}: last step {g[-1]}, early-trough {y[ti]:.3f}@{g[ti]}, "
              f"max-after {y[pi]:.3f}@{g[pi]}, final {y[-1]:.3f}, A@1200 {a1200}")
    print("\n  A first_acc mean over seeds still alive, at late steps:")
    for s in [1200, 1500, 1800, 2100, 2400]:
        vals = [r["pops"]["dataA"]["first_acc"][r["step"].tolist().index(s)]
                for r in runs.values() if s in r["step"].tolist()]
        if vals:
            print(f"    step {s}: {np.mean(vals):.3f} (n={len(vals)})")

    print("\nEXTRA -- dataC (never trained) and ballast trajectories at landmarks")
    print(f"{'arm':>16} {'pop':>8} | {'step0 first':>11} {'step0 hgap':>11} "
          f"{'@200 first':>11} {'@1200 first':>12} {'@1200 hgap':>11}")
    for k in MAIN:
        grid = DATA[k]["grid"].tolist()
        for p in ["ballast", "dataC"]:
            if p not in next(iter(DATA[k]["runs"].values()))["pops"]:
                continue
            f0 = np.nanmean(arr(k, p, "first_acc")[0][:, 0])
            h0 = np.nanmean(arr(k, p, "halluc_gap")[0][:, 0])
            f2 = np.nanmean(arr(k, p, "first_acc")[0][:, grid.index(200)])
            f12 = np.nanmean(arr(k, p, "first_acc")[0][:, -1])
            h12 = np.nanmean(arr(k, p, "halluc_gap")[0][:, -1])
            print(f"{LBL[k]:>16} {p:>8} | {f0:11.3f} {h0:11.3f} {f2:11.3f} {f12:12.3f} {h12:11.3f}")


def task7(SUMM):
    """Alignment of A's recovery landmarks with B's milestones, and the
    own-half / between-half decomposition of crash and recovery."""
    print("\n" + "=" * 100)
    print("TASK 7 -- what A's recovery peak lines up with, and what the crash is made of")
    print("=" * 100)
    print("\n(a) A's recovery peak step vs B's milestone steps (5-seed means)")
    print(f"{'arm':>16} | {'A peak step':>11} | {'B attr=.5':>10} {'B first=.5':>11} "
          f"{'B first=.9':>11} {'B first=.99':>12} | {'peak/attr.5':>12} {'peak/first.99':>14}")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        bf, _ = arr(k, "dataB", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        pk = np.mean([grid[trough_peak(grid, af[i])[1]] for i in range(len(seeds))])
        a50 = np.nanmean([interp_cross(grid, ba[i], 0.5) for i in range(len(seeds))])
        f50 = np.nanmean([interp_cross(grid, bf[i], 0.5) for i in range(len(seeds))])
        f90 = np.nanmean([interp_cross(grid, bf[i], 0.9) for i in range(len(seeds))])
        f99 = np.nanmean([interp_cross(grid, bf[i], 0.99) for i in range(len(seeds))])
        r1 = pk / a50 if np.isfinite(a50) else np.nan
        print(f"{LBL[k]:>16} | {pk:11.0f} | {a50:10.1f} {f50:11.1f} {f90:11.1f} {f99:12.1f} | "
              f"{fmt(r1,2):>12} {pk/f99:14.2f}")
    print("  (the ballast arms start with B attr_acc ~0.51, i.e. above 0.5 at step 0 -> no crossing)")

    print("\n(b) 8L no-ballast: both humps against B's milestones (per-seed)")
    k = "8L_noballast"
    grid = DATA[k]["grid"]
    af, seeds = arr(k, "dataA", "first_acc")
    bf, _ = arr(k, "dataB", "first_acc")
    ba, _ = arr(k, "dataB", "attr_acc")
    for i in range(len(seeds)):
        m1 = (grid >= 60) & (grid <= 180)
        m2 = (grid >= 220) & (grid <= 450)
        p1 = grid[np.where(m1)[0][np.nanargmax(af[i][m1])]]
        p2 = grid[np.where(m2)[0][np.nanargmax(af[i][m2])]]
        print(f"  seed{seeds[i]}: hump1 {p1:4d} vs B attr=0.5 at "
              f"{interp_cross(grid, ba[i], 0.5):6.1f} | hump2 {p2:4d} vs B first=0.9 at "
              f"{interp_cross(grid, bf[i], 0.9):6.1f}")

    print("\n(c) 4L no-ballast fine structure, steps 100-700 (5-seed mean A first_acc)")
    k = "4L_noballast"
    grid = DATA[k]["grid"]
    af, _ = arr(k, "dataA", "first_acc")
    js = [j for j, s in enumerate(grid) if 100 <= s <= 700 and s % 20 == 0]
    print("  step:", " ".join(f"{grid[j]:6d}" for j in js))
    print("  A   :", " ".join(f"{np.nanmean(af[:,j]):6.3f}" for j in js))
    bf, _ = arr(k, "dataB", "first_acc")
    print("  B   :", " ".join(f"{np.nanmean(bf[:,j]):6.3f}" for j in js))

    print("\n(d) decomposition: A first_acc = rank_own_top1 - gap; change from step 0")
    print(f"{'arm':>16} | {'d(first) crash':>14} {'d(rk_own)':>10} {'d(gap)':>8} | "
          f"{'d(first) rec':>12} {'d(rk_own)':>10} {'d(gap)':>8}")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        ar, _ = arr(k, "dataA", "rank_own_top1")
        rows = []
        for i in range(len(seeds)):
            ti, pi = trough_peak(grid, af[i])
            g = ar[i] - af[i]
            rows.append([af[i, ti] - af[i, 0], ar[i, ti] - ar[i, 0], g[ti] - g[0],
                         af[i, pi] - af[i, ti], ar[i, pi] - ar[i, ti], g[pi] - g[ti]])
        R = np.array(rows).mean(0)
        print(f"{LBL[k]:>16} | {R[0]:14.3f} {R[1]:10.3f} {R[2]:8.3f} | {R[3]:12.3f} "
              f"{R[4]:10.3f} {R[5]:8.3f}")
    print("  (crash: step0 -> trough; rec: trough -> peak. d(first) = d(rk_own) - d(gap).)")

    print("\n(e) dataC (never trained, draws from half X like A): attr_acc drift")
    print(f"{'arm':>16} | {'C attr @0':>10} {'@100':>7} {'@400':>7} {'@1200':>7} | "
          f"{'C hgap @0':>10} {'@1200':>7}")
    for k in MAIN:
        grid = DATA[k]["grid"].tolist()
        ca, _ = arr(k, "dataC", "attr_acc")
        ch, _ = arr(k, "dataC", "halluc_gap")
        print(f"{LBL[k]:>16} | {np.nanmean(ca[:,0]):10.3f} "
              f"{np.nanmean(ca[:,grid.index(100)]):7.3f} {np.nanmean(ca[:,grid.index(400)]):7.3f} "
              f"{np.nanmean(ca[:,-1]):7.3f} | {np.nanmean(ch[:,0]):10.3f} {np.nanmean(ch[:,-1]):7.3f}")

    print("\n(f) A's attr_acc (attribute-type integrity) at the same landmarks")
    print(f"{'arm':>16} | {'@0':>6} {'@trough':>8} {'@peak':>7} {'@1200':>7}")
    for k in MAIN:
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        aa, _ = arr(k, "dataA", "attr_acc")
        v = []
        for i in range(len(seeds)):
            ti, pi = trough_peak(grid, af[i])
            v.append([aa[i, 0], aa[i, ti], aa[i, pi], aa[i, -1]])
        V = np.array(v).mean(0)
        print(f"{LBL[k]:>16} | " + " ".join(f"{x:7.3f}" for x in V))


if __name__ == "__main__":
    SUMM = task1()
    task2()
    task3(SUMM)
    task4(SUMM)
    task5()
    task6(SUMM)
    task7(SUMM)
    extra()
