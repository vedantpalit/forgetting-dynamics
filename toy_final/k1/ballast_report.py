"""Tables for the ballast report: pretraining measurements and the n_D injection sweep."""
import json
import numpy as np

M1 = json.load(open("ballast_measure.json"))
M2 = json.load(open("ballast_measure2.json"))
SW = json.load(open("ballast_nd_sweep.json"))
NDS = [0, 5, 12, 25, 50]


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return v.mean(), v.std()


print("\n(i)  THE SHARED ROW AFTER PRETRAINING   s0 = mu @ W_pre,  w = u_bar_X - u_bar_Y")
print("     positive s0.w = the shared row is PRO-X.  mean +/- sd over seeds 0,1,2")
for norm in (1, 0):
    print(f"\n  arm = {'normalized' if norm else 'linear'}")
    print(f"  {'n_D':>4} {'||s0||':>14} {'<s0,w>':>16} {'cos(s0,w_hat)':>16} {'<mu,w>':>14}")
    for nd in NDS:
        rs = [r for r in M1 if r["nd"] == nd and r["norm"] == norm]
        a, b = agg(rs, "s0_norm"); c, d = agg(rs, "s0_on_X")
        e, f = agg(rs, "s0_cos"); g, h = agg(rs, "mu_on_X")
        print(f"  {nd:>4} {a:>7.2f}+/-{b:<5.2f} {c:>+8.3f}+/-{d:<6.3f} {e:>+8.3f}+/-{f:<6.3f} "
              f"{g:>+6.3f}+/-{h:<5.3f}")

print("\n(ii) A's CROSS-HALF MARGIN AFTER PRETRAINING, split by route")
print("     m_cross(a) = z[correct] - max over the OTHER half.  exact split (err < 1e-13)")
for norm in (1, 0):
    print(f"\n  arm = {'normalized' if norm else 'linear'}")
    print(f"  {'n_D':>4} {'m_cross':>14} {'mu-row':>14} {'mu-resid':>13} {'individual':>14} "
          f"{'f_mu_row':>15} {'min m_cross':>12}")
    for nd in NDS:
        rs = [r for r in M2 if r["nd"] == nd and r["norm"] == norm]
        cols = []
        for k in ("mc", "mc_murow", "mc_mures", "mc_ind", "f_murow"):
            cols.append(agg(rs, k))
        mn, _ = agg(rs, "mc_min")
        print(f"  {nd:>4} " + " ".join(f"{a:>+7.2f}+/-{b:<5.2f}" for a, b in cols[:4]) +
              f" {cols[4][0]:>+8.3f}+/-{cols[4][1]:<5.3f} {mn:>+11.2f}")

print("\n     same-half margin for contrast (the bias cannot reorder within a half)")
print(f"  {'n_D':>4} {'arm':>11} {'m_own':>14} {'mu-row':>14}")
for norm in (1, 0):
    for nd in NDS:
        rs = [r for r in M2 if r["nd"] == nd and r["norm"] == norm]
        a, b = agg(rs, "mo"); c, d = agg(rs, "mo_murow")
        print(f"  {nd:>4} {'normalized' if norm else 'linear':>11} {a:>+7.2f}+/-{b:<5.2f} "
              f"{c:>+7.2f}+/-{d:<5.2f}")

print("\n(iii) GATE-MATCHED INJECTION (B hits 0.99 at step 200), 5000 steps, 3 seeds")
for norm in (1, 0):
    print(f"\n  arm = {'normalized' if norm else 'linear'}")
    hd = (f"  {'n_D':>4} {'ilr':>9} {'Bgate':>6} {'A trough':>13} {'@step':>7} {'recovery':>13} "
          f"{'A end':>13} {'sw peak':>8} {'sw end':>8} {'sw drop':>8} {'abs.w 0':>8} "
          f"{'abs.w pk':>9} {'abs.w end':>9} {'mcross 0':>9} {'mcross min':>10} {'mcross end':>10}")
    print(hd)
    for nd in NDS:
        ks = [k for k in SW if SW[k]["value"] == nd and SW[k]["norm"] == norm]
        st = []
        for k in sorted(ks):
            r = SW[k]; rows = r["rows"]
            A = np.array([x["A"] for x in rows]); step = np.array([x["step"] for x in rows])
            B = np.array([x["B"] for x in rows])
            sw = np.array([x["s_on_w"] for x in rows])
            aw = np.array([x["s_abs_on_w"] for x in rows])
            mc = np.array([x["m_cross"] for x in rows])
            tr = int(A.argmin()); pw = int(sw.argmax())
            st.append(dict(ilr=r["inject_lr"],
                           bgate=int(step[np.argmax(B >= 0.99)]) if (B >= 0.99).any() else -1,
                           trough=A[tr], tstep=step[tr], rec=A[tr:].max() - A[tr], end=A[-1],
                           swp=sw[pw], swe=sw[-1],
                           swd=1 - sw[-1] / sw[pw] if sw[pw] > 1e-9 else 0.0,
                           aw0=aw[0], awp=aw[int(np.abs(aw).argmax())], awe=aw[-1],
                           mc0=mc[0], mcm=mc.min(), mce=mc[-1], matched=r["matched"]))
        g = lambda k: (np.mean([s[k] for s in st]), np.std([s[k] for s in st]))
        flag = "" if all(s["matched"] for s in st) else "  [some cells unmatched]"
        print(f"  {nd:>4} {g('ilr')[0]:>9.2e} {g('bgate')[0]:>6.0f} "
              f"{g('trough')[0]:>6.2f}+/-{g('trough')[1]:<5.2f} {g('tstep')[0]:>7.0f} "
              f"{g('rec')[0]:>+6.2f}+/-{g('rec')[1]:<5.2f} {g('end')[0]:>6.2f}+/-{g('end')[1]:<5.2f} "
              f"{g('swp')[0]:>8.2f} {g('swe')[0]:>8.2f} {100*g('swd')[0]:>7.0f}% "
              f"{g('aw0')[0]:>+8.2f} {g('awp')[0]:>+9.2f} {g('awe')[0]:>+9.2f} "
              f"{g('mc0')[0]:>+9.2f} {g('mcm')[0]:>+10.2f} {g('mce')[0]:>+10.2f}{flag}")

print("\n  per-seed troughs / recoveries, normalized arm")
for nd in NDS:
    ks = sorted(k for k in SW if SW[k]["value"] == nd and SW[k]["norm"] == 1)
    out = []
    for k in ks:
        A = np.array([x["A"] for x in SW[k]["rows"]]); tr = int(A.argmin())
        out.append(f"{A[tr]:.2f}->{A[tr:].max():.2f}(end {A[-1]:.2f})")
    print(f"  n_D={nd:>3}: " + "  ".join(out))
