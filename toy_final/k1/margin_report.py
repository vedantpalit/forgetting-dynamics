"""Tables for the margin-threshold test.  Reads margin_*.json written by margin_predict.py."""
import glob
import json
import sys

import numpy as np


def phases(rows):
    A = np.array([r["A"] for r in rows])
    st = np.array([r["step"] for r in rows])
    tr = int(A.argmin())
    pk = tr + int(A[tr:].argmax())
    return tr, pk, len(rows) - 1, st


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        out.update(json.load(open(f)))
    return out


def task12(runs):
    print("\n=== TASK 1-2: actual vs const-only vs vary-only (default config) ===")
    print(f"{'run':<16}{'phase':<10}{'step':>6}{'actual':>8}{'const':>8}{'vary':>8}"
          f"{'thresh':>8}{'gap':>8}{'epsSD':>8}{'B':>6}{'biasShare':>10}")
    for k, r in runs.items():
        rows = r["rows"]; tr, pk, en, st = phases(rows)
        a0 = rows[0]["A"]
        for nm, i in (("trough", tr), ("recovery", pk), ("end", en)):
            x = rows[i]
            tot = a0 - x["A"]
            sh = (a0 - x["A_const"]) / tot if tot > 1e-9 else float("nan")
            print(f"{k:<16}{nm:<10}{x['step']:>6}{x['A']:>8.2f}{x['A_const']:>8.2f}"
                  f"{x['A_vary']:>8.2f}{x['A_thresh']:>8.2f}{x['gap_max']:>8.2f}"
                  f"{x['dev_gap_std']:>8.2f}{x['B']:>6.2f}{sh:>10.2f}")
    print("\n--- deviations over the whole trajectory ---")
    print(f"{'run':<16}{'max|const-act|':>15}{'@step':>7}{'max|vary-act|':>15}"
          f"{'meanAbs const':>15}{'vary<0.99 @':>12}{'trough@':>9}{'rec@':>7}")
    for k, r in runs.items():
        rows = r["rows"]; tr, pk, en, st = phases(rows)
        A = np.array([x["A"] for x in rows])
        C = np.array([x["A_const"] for x in rows])
        Vv = np.array([x["A_vary"] for x in rows])
        dc = np.abs(C - A); dv = np.abs(Vv - A)
        j = int(dc.argmax())
        first = st[np.argmax(Vv < 0.99)] if (Vv < 0.99).any() else -1
        print(f"{k:<16}{dc.max():>15.2f}{st[j]:>7}{dv.max():>15.2f}"
              f"{dc.mean():>15.3f}{first:>12}{st[tr]:>9}{st[pk]:>7}")


def task3(runs):
    print("\n=== TASK 3: cross-configuration (normalized arm, gate-matched) ===")
    print(f"{'cfg':<14}{'m_full mean':>12}{'p10':>7}{'p50':>7}{'p90':>7}"
          f"{'gap@tr':>8}{'crashAct':>9}{'crashCon':>9}{'crashThr':>9}{'epsSD':>7}{'match':>7}")
    for k, r in sorted(runs.items()):
        rows = r["rows"]; tr, pk, en, st = phases(rows)
        m = np.array(r["margins"]["m_full"])
        x = rows[tr]
        a0 = rows[0]["A"]
        print(f"{k:<14}{m.mean():>12.2f}{np.percentile(m,10):>7.2f}"
              f"{np.percentile(m,50):>7.2f}{np.percentile(m,90):>7.2f}"
              f"{x['gap_max']:>8.2f}{a0-x['A']:>9.2f}{a0-x['A_const']:>9.2f}"
              f"{a0-x['A_thresh']:>9.2f}{x['dev_gap_std']:>7.2f}{str(r['matched']):>7}")
    # does const-only explain crash depth ACROSS configurations?
    act, con, thr, gap = [], [], [], []
    for k, r in sorted(runs.items()):
        rows = r["rows"]; tr, _, _, _ = phases(rows)
        a0 = rows[0]["A"]
        act.append(a0 - rows[tr]["A"]); con.append(a0 - rows[tr]["A_const"])
        thr.append(a0 - rows[tr]["A_thresh"]); gap.append(rows[tr]["gap_max"])
    act, con, thr, gap = map(np.array, (act, con, thr, gap))
    def r2(pred, y):
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\nacross {len(act)} cells, crash depth: "
          f"R^2(const-only, zero-param) = {r2(con, act):.2f}   "
          f"R^2(threshold m_cross<gap) = {r2(thr, act):.2f}   "
          f"corr(gap, actual crash) = {np.corrcoef(gap, act)[0,1]:.2f}")
    print(f"R^2(isotonic in gap alone, n-1 free params, the pooled-collapse baseline) = "
          f"{r2(isotonic(gap, act), act):.2f}")
    print(f"mean signed error const-only = {np.mean(con - act):+.2f} "
          f"(negative = under-predicts the crash); "
          f"mean |err| = {np.mean(np.abs(con - act)):.2f}")


def isotonic(x, y):
    """PAVA fit of y monotone non-decreasing in x, evaluated at x."""
    o = np.argsort(x); ys = y[o].astype(float).copy()
    w = np.ones_like(ys); n = len(ys); i = 0
    lvl = list(ys); wt = list(w)
    out = []
    for v, ww in zip(lvl, wt):
        out.append([v, ww])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop(); v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    fit = np.concatenate([np.full(int(w), v) for v, w in out])
    res = np.empty(n); res[o] = fit
    return res


def task4(runs):
    print("\n=== TASK 4: terciles of pretrained margin m_a ===")
    print(f"{'run':<16}{'phase':<10}{'bottom':>8}{'middle':>8}{'top':>8}"
          f"{'m_bot':>8}{'m_mid':>8}{'m_top':>8}")
    for k, r in runs.items():
        m = np.array(r["margins"]["m_full"])
        order = np.argsort(m)
        thirds = np.array_split(order, 3)
        rows = r["rows"]; tr, pk, en, st = phases(rows)
        for nm, i in (("trough", tr), ("recovery", pk), ("end", en)):
            ok = np.array(rows[i]["ok"])
            print(f"{k:<16}{nm:<10}" + "".join(f"{ok[t].mean():>8.2f}" for t in thirds)
                  + "".join(f"{m[t].mean():>8.2f}" for t in thirds))
    print(f"\n{'run':<16}{'phase':<10}{'corr(m_a, survives)':>21}{'corr(m_cross, surv)':>21}"
          f"{'m spread p90-p10':>18}{'epsSD':>7}")
    for k, r in runs.items():
        m = np.array(r["margins"]["m_full"]); mc = np.array(r["margins"]["m_cross"])
        rows = r["rows"]; tr, pk, en, st = phases(rows)
        for nm, i in (("trough", tr), ("recovery", pk)):
            ok = np.array(rows[i]["ok"]).astype(float)
            c1 = np.corrcoef(m, ok)[0, 1] if ok.std() > 0 else np.nan
            c2 = np.corrcoef(mc, ok)[0, 1] if ok.std() > 0 else np.nan
            print(f"{k:<16}{nm:<10}{c1:>21.2f}{c2:>21.2f}"
                  f"{np.percentile(m,90)-np.percentile(m,10):>18.2f}"
                  f"{rows[i]['dev_gap_std']:>7.2f}")


if __name__ == "__main__":
    which = sys.argv[1]
    runs = load(sys.argv[2])
    if which == "12":
        task12(runs); task4(runs)
    else:
        task3(runs); task4(runs)
