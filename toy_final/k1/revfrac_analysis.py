"""Tests of the reduced model's prediction, and of the bounds, on the full K=1 toy.

Reads the instrumented runs written by revfrac_measure.py and, for the descriptive part,
the existing sweeps (sw_*.json), which do not carry Phi or M and so can only be correlated,
not predicted from.

WHAT IS BEING TESTED

 (BAL)  exact, derived:   at any stationary point of <s,w>,   Cbar = sqrt(d) Phibar / Mbar
        Cbar = <h_hat_b, w_hat> (M-weighted), Phibar/Mbar the half-margin and margin.
 (PRO)  derived:          the write's between-half content is proportional to the STATE's,
        so  (x_end - x_0)/(x_peak - x_0)  computed on the state must equal the same ratio
        computed on <s,w>. This is the bridge that lets the 3-coordinate reduced model speak
        about the write.
 (PRE)  one-sided prediction: x_end = rho_end * sqrt(d) Phibar_end / Mbar_end, with the PEAK
        measured, gives f with no fitted constant.
 (LVL)  leading order in 1/c:  z(other half) = z(within-half runner-up) at the fixed point,
        i.e. p_other / p_same = m_X / m_Y with m the class counts. Config-independent.
 (BND1) theorem: the anti-state force on the state's between-half content is -M_b C_b, which
        VANISHES at C_b = 0, while the plain force is + sqrt(d) Phi_b > 0 strictly as long as
        B keeps any mass on the other half. So the STATE's between-half content can never be
        driven to zero, let alone reversed: it is bounded below by the balance point. The
        WRITE <s,w> = state - pretrained level may still go negative, and only if the
        pretrained level was already above the balance.
 (BND2) the span bound suggested for the anti-state force: the component of s_peak orthogonal
        to span{h_b} cannot be removed by it. NOT a theorem (the plain term moves it too),
        so this is reported as measured, with its slack.
"""
import glob
import json
import os

import numpy as np


def load(pats):
    out = {}
    for pat in pats:
        for fn in sorted(glob.glob(pat)):
            for k, v in json.load(open(fn)).items():
                out[f"{os.path.basename(fn)[:-5]}:{k}"] = v
    return out


def arr(rows, k):
    return np.array([r[k] for r in rows], dtype=float)


def summarise(r):
    rows = r["rows"]
    sw = arr(rows, "s_on_w")
    xh = arr(rows, "x_h")
    st = arr(rows, "step")
    pk = int(sw.argmax())
    e = len(rows) - 1
    d = r["d"]
    q = dict(d=d, v=r["v"], nb=r["nb"], beta=r["beta"], seed=r["seed"],
             steps=int(st[-1]), peak_step=int(st[pk]),
             sw_peak=sw[pk], sw_end=sw[e], f=1 - sw[e] / sw[pk] if sw[pk] > 1e-9 else np.nan,
             x0=xh[0], x_pk=xh[pk], x_end=xh[e],
             rho_pk=rows[pk]["hnorm"], rho_end=rows[e]["hnorm"],
             C_pk=rows[pk]["Cbar"], C_end=rows[e]["Cbar"],
             Q_pk=rows[pk]["Q"], Q_end=rows[e]["Q"],
             M_end=rows[e]["Mbar"], Phi_end=rows[e]["Phibar"],
             p_same_end=rows[e]["p_same"], p_other_end=rows[e]["p_other"],
             a_frac_pk=rows[pk]["a_frac"], A_trough=float(arr(rows, "A").min()),
             s_perp_pk=rows[pk]["s_perp"], s_norm_pk=rows[pk]["s_norm"],
             s_norm_end=rows[e]["s_norm"],
             xh_min_post=float(xh[pk:].min()), sw_min_post=float(sw[pk:].min()))
    # (PRO) the same fraction measured on the state's between-half content
    q["f_state"] = 1 - (q["x_end"] - q["x0"]) / (q["x_pk"] - q["x0"])
    # (PRE) predict the END from the balance, take the peak as measured
    q["x_end_pred"] = q["rho_end"] * q["Q_end"]
    q["f_pred"] = 1 - (q["x_end_pred"] - q["x0"]) / (q["x_pk"] - q["x0"])
    # (LVL)
    mX, mY = r["v"] / 2, r["v"] / 2 - 1
    q["lvl"] = np.log(max(q["p_other_end"], 1e-300) / max(q["p_same_end"], 1e-300)) - \
        np.log(mX / mY)
    return q


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    runs = load(["revfrac_beta.json", "revfrac_nb.json", "revfrac_d*.json",
                 "revfrac_v*.json", "revfrac_long.json"])
    S = {k: summarise(v) for k, v in runs.items()}
    keys = sorted(S, key=lambda k: (S[k]["v"], S[k]["d"], S[k]["nb"], S[k]["beta"],
                                    S[k]["seed"]))

    print("=" * 132)
    print("(BAL) + (PRE): the balance Cbar = sqrt(d) Phibar / Mbar at the END, and f "
          "predicted from it with the peak measured")
    print("=" * 132)
    print(f"{'config':>26} {'steps':>6} | {'Cbar_e':>7} {'Q_e':>7} {'relerr':>7} | "
          f"{'x0':>6} {'x_pk':>6} {'x_end':>6} {'pred':>6} | {'f':>6} {'f_state':>7} "
          f"{'f_pred':>7} | {'lvl':>6}")
    for k in keys:
        q = S[k]
        cfg = f"b{q['beta']:g} nb{q['nb']} d{q['d']} v{q['v']} s{q['seed']}"
        re_ = abs(q["Q_end"] - q["C_end"]) / abs(q["C_end"])
        print(f"{cfg:>26} {q['steps']:>6} | {q['C_end']:7.4f} {q['Q_end']:7.4f} "
              f"{re_:7.3f} | {q['x0']:6.2f} {q['x_pk']:6.2f} {q['x_end']:6.2f} "
              f"{q['x_end_pred']:6.2f} | {q['f']:6.3f} {q['f_state']:7.3f} "
              f"{q['f_pred']:7.3f} | {q['lvl']:+6.2f}")

    g = lambda n: np.array([S[k][n] for k in keys], dtype=float)
    print(f"\n(BAL) at the end: median |Cbar-Q|/|Cbar| = "
          f"{np.median(np.abs(g('Q_end') - g('C_end')) / np.abs(g('C_end'))):.3f}   "
          f"(at the PEAK, where Mbar is crossing zero, the same quantity is "
          f"{np.median(np.abs(g('Q_pk') - g('C_pk')) / np.abs(g('C_pk'))):.2f} -- the peak is "
          f"a soft stationary point and (BAL) is ill-conditioned there)")
    print(f"(PRO) f measured on the write vs on the state: r = {corr(g('f'), g('f_state')):.4f}, "
          f"median |diff| = {np.median(np.abs(g('f') - g('f_state'))):.3f}")
    print(f"(PRE) f predicted vs measured: r = {corr(g('f'), g('f_pred')):.4f}, "
          f"median |diff| = {np.median(np.abs(g('f') - g('f_pred'))):.3f}, "
          f"mean signed = {np.mean(g('f_pred') - g('f')):+.3f}")
    print(f"(LVL) log(p_other/p_same) - log(m_X/m_Y) at the end: mean {g('lvl').mean():+.2f}, "
          f"sd {g('lvl').std():.2f}, range [{g('lvl').min():+.2f}, {g('lvl').max():+.2f}] "
          f"logits, over v = {sorted(set(g('v').astype(int)))}")

    print("\n" + "=" * 132)
    print("BOUNDS")
    print("=" * 132)
    xm, sm = g("xh_min_post"), g("sw_min_post")
    print(f"(BND1) min over post-peak checkpoints of the STATE's between-half content x_h: "
          f"min over runs {xm.min():+.3f}; negative in {int((xm < 0).sum())}/{len(xm)} runs")
    print(f"       the same for the WRITE <s,w>: min {sm.min():+.3f}; negative in "
          f"{int((sm < 0).sum())}/{len(sm)} runs")
    neg = [k for k in keys if S[k]["sw_min_post"] < 0]
    for k in neg:
        q = S[k]
        print(f"       write goes negative: b{q['beta']:g} nb{q['nb']} d{q['d']} "
              f"v{q['v']} s{q['seed']}: x0={q['x0']:+.2f} x_pk={q['x_pk']:+.2f} "
              f"x_end={q['x_end']:+.2f} balance={q['x_end_pred']:+.2f} "
              f"-> pretrained level {'ABOVE' if q['x0'] > q['x_end_pred'] else 'below'} "
              f"the balance")
    ratio = g("s_perp_pk") / g("s_norm_end")
    print(f"(BND2) ||P_perp s_peak|| / ||s_end||: median {np.median(ratio):.3f}, "
          f"max {ratio.max():.3f}  -- satisfied in {int((ratio <= 1).sum())}/{len(ratio)} "
          f"runs, but with a factor {1 / np.median(ratio):.1f} of slack, so it is far from "
          f"tight")

    print("\n" + "=" * 132)
    print("DESCRIPTIVE: f against everything logged in the ORIGINAL sweeps (sw_*.json, "
          "norm=1 runs only)")
    print("=" * 132)
    old = load(["sw_beta.json", "sw_nb.json", "sw_d.json", "sw_vocab.json", "sw_load.json",
                "sw_dose.json"])
    rec = []
    for k, r in old.items():
        if r["norm"] != 1:
            continue
        rows = r["rows"]
        s = arr(rows, "s_on_w")
        pk = int(s.argmax())
        if s[pk] < 1e-6:
            continue
        rec.append(dict(f=1 - s[-1] / s[pk], beta=r["beta"], nb=r["nb"], d=r["d"], v=r["v"],
                        na=r["na"], a_frac_pk=rows[pk]["a_frac"],
                        a_frac_end=rows[-1]["a_frac"], sw_peak=s[pk],
                        s_share_pk=rows[pk]["s_share"], eps_end=rows[-1]["eps"],
                        delta_end=rows[-1]["delta"], dU=rows[-1]["dU_rel"],
                        peak_step=rows[pk]["step"],
                        A_trough=float(arr(rows, "A").min()),
                        rec_=float(arr(rows, "A")[int(arr(rows, "A").argmin()):].max() -
                                   arr(rows, "A").min())))
    F = np.array([x["f"] for x in rec])
    print(f"{len(rec)} normalized runs, f = {F.mean():.3f} +/- {F.std():.3f}, "
          f"range [{F.min():.3f}, {F.max():.3f}]")
    for nm in ["beta", "nb", "d", "v", "na", "a_frac_pk", "a_frac_end", "sw_peak",
               "s_share_pk", "eps_end", "delta_end", "dU", "peak_step", "A_trough", "rec_"]:
        x = np.array([q[nm] for q in rec], dtype=float)
        print(f"  corr(f, {nm:>12}) = {corr(F, x):+.3f}    "
              f"corr(f, log {nm:>12}) = "
              f"{corr(F, np.log(np.abs(x) + 1e-12)) if (x > 0).all() else np.nan:+.3f}")


if __name__ == "__main__":
    main()
