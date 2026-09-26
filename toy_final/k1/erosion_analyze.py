"""Fits for the erosion open items. Reads the erosion_*.json written by erosion_measure.py."""
import glob
import json
import sys

import numpy as np


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        out.update(json.load(open(f)))
    return out


def last(r):
    return r["rows"][-1]


def at(r, step):
    """Row at or just before `step`."""
    best = r["rows"][0]
    for x in r["rows"]:
        if x["step"] <= step:
            best = x
    return best


def loglog(x, y, name=""):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    s, c = np.polyfit(lx, ly, 1)
    pred = s * lx + c
    ss = 1 - ((ly - pred) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-300)
    return float(s), float(ss)


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s, c = np.polyfit(x, y, 1)
    pred = s * x + c
    ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(s), float(c), float(ss)


def table(runs, key, cols, hdr, agg="mean"):
    """Group runs by run[key], average the per-run columns over seeds."""
    keys = sorted({r[key] for r in runs})
    rows = []
    for k in keys:
        sel = [r for r in runs if r[key] == k]
        vals = []
        for c in cols:
            v = [c(r) for r in sel]
            vals.append(np.mean(v))
        rows.append((k, vals, len(sel)))
    w = max(len(hdr[0]), 8)
    print("  " + hdr[0].rjust(w) + "".join(h.rjust(12) for h in hdr[1:]) + "   n")
    for k, v, n in rows:
        print("  " + f"{k:g}".rjust(w) + "".join(f"{x:12.4g}" for x in v) + f"{n:4d}")
    return keys, rows


COLS = [
    ("eps/st", lambda r: last(r)["eps_post"] / last(r)["post_norm"]),
    ("epsh/st", lambda r: last(r)["eps_h"] / last(r)["h_norm"]),
    ("r_rms", lambda r: last(r)["r_rms"]),
    ("r_rms@g", lambda r: at(r, 200)["r_rms"]),
    ("kappa_r", lambda r: last(r)["kappa_r"]),
    ("gg_rms", lambda r: r["gg_rms"]),
    ("E_mean", lambda r: last(r)["E_mean"]),
    ("E/E_pred", lambda r: last(r)["E_mean"] / max(last(r)["E_pred"], 1e-300)),
    ("dW", lambda r: last(r)["dW"]),
    ("dose", lambda r: last(r)["dose"]),
    ("ilr", lambda r: r["inject_lr"]),
    ("h_norm", lambda r: last(r)["h_norm"]),
    ("cum_lin", lambda r: last(r)["cum_lin"]),
    ("cum_rw", lambda r: last(r)["cum_rw"]),
    ("A_end", lambda r: last(r)["A"]),
]


def ingredients(runs, key, label):
    hdr = [label] + [c[0] for c in COLS]
    return table(runs, key, [c[1] for c in COLS], hdr)


def dsweep(runs, title):
    print(f"\n{'='*118}\n{title}\n{'='*118}")
    ingredients(runs, "d", "d")
    ds = np.array([r["d"] for r in runs], float)
    print("\n  log-log slope vs d (R^2):")
    for name, f in COLS:
        if name in ("A_end",):
            continue
        s, ss = loglog(ds, [f(r) for r in runs])
        print(f"    {name:>10}  {s:+7.3f}   R^2 {ss:5.2f}")
    # the chain: eps/state = C sqrt(nB) gg_rms r_rms * (accumulation) / sqrt(d)
    eps = np.array([last(r)["eps_h"] / last(r)["h_norm"] for r in runs])
    print("\n  chain check (each column is eps/state divided by the stated prediction;")
    print("  a d-independent column means the prediction carries all the d-dependence):")
    preds = {
        "cum_lin/h": np.array([last(r)["cum_lin"] / last(r)["h_norm"] for r in runs]),
        "cum_rw/h": np.array([last(r)["cum_rw"] / last(r)["h_norm"] for r in runs]),
        "T*inc/h": np.array([r["steps"] * last(r)["inc"] / last(r)["h_norm"] for r in runs]),
        "sqrtT*inc/h": np.array([np.sqrt(r["steps"]) * last(r)["inc"] / last(r)["h_norm"]
                                 for r in runs]),
        "ilr*d*sqrtnB*gg*r*T/sqrtd": np.array(
            [r["inject_lr"] * r["d"] * (1 - r["beta"]) * np.sqrt(r["nb"]) * r["gg_rms"]
             * last(r)["r_rms"] * r["steps"] / np.sqrt(r["d"]) for r in runs]),
    }
    for nm, p in preds.items():
        s, ss = loglog(ds, eps / p)
        vals = eps / p
        print(f"    {nm:>26}  residual slope {s:+6.3f} (R^2 {ss:4.2f})  "
              f"ratio {vals.min():.3g}..{vals.max():.3g}  spread {vals.max()/max(vals.min(),1e-300):.1f}x")


def main():
    # ---- Task 1/2: gate-matched d sweep ----
    m = list(load("erosion_d[0-9]*.json").values())
    m = [r for r in m if r["seed"] in (0, 1)]
    dsweep(m, "TASK 1/2  d sweep, GATE-MATCHED to 200 (n_B=50, beta=0.5, normalized, 3000 steps)")

    f = list(load("erosion_dfix*.json").values())
    if f:
        dsweep(f, "TASK 2b  d sweep, FIXED injection rate 5e-2 (gate NOT matched)")

    # ---- Task 3: beta ----
    b = list(load("erosion_betafix.json").values())
    if b:
        print(f"\n{'='*118}\nTASK 3  beta at FIXED rate and FIXED steps (d=128, n_B=50, "
              f"normalized, 3000 steps)\n{'='*118}")
        ingredients(b, "beta", "beta")
        bs = np.array([r["beta"] for r in b])
        eps = np.array([last(r)["eps_h"] / last(r)["h_norm"] for r in b])
        print()
        s, c, ss = linfit(1 - bs, eps)
        print(f"    eps/state = {s:+.4f}*(1-b) {c:+.4f}      R^2 {ss:.3f}   "
              f"(pure (1-b) would give c=0)")
        s2, ss2 = loglog(1 - bs, eps)
        print(f"    log-log slope on (1-b): {s2:+.3f}   R^2 {ss2:.3f}   (prediction +1)")
        for nm, p in [("E_mean-accum (cum_lin, incl. (1-b))",
                       np.array([last(r)["cum_lin"] / last(r)["h_norm"] for r in b])),
                      ("cum_rw", np.array([last(r)["cum_rw"] / last(r)["h_norm"] for r in b])),
                      ("(1-b)*r_rms*T", np.array([(1 - r["beta"]) * last(r)["r_rms"]
                                                  * r["steps"] for r in b]))]:
            sl, r2 = loglog(1 - bs, eps / p)
            v = eps / p
            print(f"    eps/state / [{nm}]: residual (1-b) slope {sl:+.3f} (R^2 {r2:.2f}), "
                  f"spread {v.max()/max(v.min(),1e-300):.2f}x")
        # (1-b) divided out of the RAW per-step coefficient only
        eps_over_1b = eps / (1 - bs)
        s3, r3 = loglog(1 - bs, eps_over_1b)
        print(f"    eps/state/(1-b): residual slope {s3:+.3f} (R^2 {r3:.2f})  "
              f"values {eps_over_1b.min():.3g}..{eps_over_1b.max():.3g}")

    # ---- Task 4: n_A ----
    n = list(load("erosion_nafix.json").values())
    if n:
        print(f"\n{'='*118}\nTASK 4  n_A = n_D at FIXED rate and FIXED steps "
              f"(d=128, n_B=50, beta=0.5, 3000 steps)\n{'='*118}")
        ingredients(n, "na", "n_A")
        na = np.array([r["na"] for r in n], float)
        eps = np.array([last(r)["eps_h"] / last(r)["h_norm"] for r in n])
        s, ss = loglog(na, eps)
        print(f"\n    log-log slope of eps/state on n_A: {s:+.3f}  R^2 {ss:.2f}   "
              f"(flat = 0);  values {eps.min():.4f}..{eps.max():.4f}  "
              f"spread {eps.max()/eps.min():.2f}x")
        # per-seed, to separate seed noise from an n_A trend
        for sd in sorted({r["seed"] for r in n}):
            sel = [r for r in n if r["seed"] == sd]
            e = [last(r)["eps_h"] / last(r)["h_norm"] for r in sel]
            print(f"      seed {sd}: " + "  ".join(
                f"n_A={r['na']}: {v:.4f}" for r, v in zip(sel, e)))

    # ---- Task 5: time law ----
    t = list(load("erosion_time.json").values())
    if t:
        print(f"\n{'='*118}\nTASK 5  time law (d=128, beta=0.5, n_B=50, gate-matched to 200, "
              f"10000 steps)\n{'='*118}")
        for r in t:
            rows = [x for x in r["rows"] if x["step"] >= 300]
            st = np.array([x["step"] for x in rows], float)
            e = np.array([x["eps_h"] / x["h_norm"] for x in rows])
            ep = np.array([x["eps_post"] / x["post_norm"] for x in rows])
            s, ss = loglog(st, e)
            print(f"\n  seed {r['seed']}  ilr {r['inject_lr']:.3e}  Bgate~{r['probe_gate']}")
            print(f"    eps_h/state ~ t^{s:+.3f}  R^2 {ss:.3f}   "
                  f"(0.5 = random walk, 1 = linear, <0.5 = saturating)")
            s2, ss2 = loglog(st, ep)
            print(f"    eps_post/state ~ t^{s2:+.3f}  R^2 {ss2:.3f}")
            for lo, hi in ((300, 1000), (1000, 3000), (3000, 10000)):
                w = (st >= lo) & (st <= hi)
                if w.sum() > 3:
                    s3, ss3 = loglog(st[w], e[w])
                    print(f"      window {lo:>5}-{hi:<5}  exponent {s3:+.3f}  R^2 {ss3:.3f}  "
                          f"eps/state {e[w][0]:.4f} -> {e[w][-1]:.4f}")
            # compare to the two accumulation models built from the measured increments
            cl = np.array([x["cum_lin"] / x["h_norm"] for x in rows])
            cr = np.array([x["cum_rw"] / x["h_norm"] for x in rows])
            for nm, p in (("cum_lin", cl), ("cum_rw", cr)):
                sl, r2 = loglog(st, e / p)
                v = e / p
                print(f"      vs {nm}: residual t-slope {sl:+.3f} (R^2 {r2:.2f}), "
                      f"ratio {v.min():.3g}..{v.max():.3g}")
            print("      trajectory: " + "  ".join(
                f"{int(x['step'])}:{x['eps_h']/x['h_norm']:.3f}" for x in rows[::max(1, len(rows)//10)]))


if __name__ == "__main__":
    main()
