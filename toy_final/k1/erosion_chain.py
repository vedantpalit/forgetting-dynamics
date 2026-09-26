"""The corrected erosion law, as a zero-parameter prediction from quantities in the write.

    dh_a = g sqrt(d) k_hat_a dW,  k_hat_a = c_a (sqrt(b) mu + sqrt(1-b) g_a)

The mean over a removes the mu-row of dW (that IS delta).  What is left acts through the
random unit vector g_a, and for a random unit vector  ||g_a M|| = ||M||_F / sqrt(d).  So

    ||eps_a||  ~  sqrt(1-b) * sqrt(d) * ||dW_perp||_F / sqrt(d)  =  sqrt(1-b) ||dW_perp||_F
    ||dW_perp||_F = sqrt( ||dW||_F^2 - ||mu dW||^2 )          (mu is a unit vector)

and the fraction of the state it occupies is that over ||h_a||, which at the start of
injection is the PRETRAINED store's size.  Both sides are already logged.
"""
import numpy as np

from erosion_analyze import load, last, loglog


def pieces(r, row=None):
    x = row or last(r)
    incoh = np.sqrt(max(x["dW"] ** 2 - x["s_norm"] ** 2, 0.0))
    h0 = r["rows"][0]["h_norm"]
    return incoh, h0, x["eps_h"], x["h_norm"], x["eps_post"] / x["post_norm"]


def report(runs, xkey, xlab, xs=None):
    xs = np.asarray(xs if xs is not None else [r[xkey] for r in runs], float)
    P = [pieces(r) for r in runs]
    b = np.array([r["beta"] for r in runs])
    incoh = np.array([p[0] for p in P]); h0 = np.array([p[1] for p in P])
    epsh = np.array([p[2] for p in P]); hend = np.array([p[3] for p in P])
    frac = np.array([p[4] for p in P])
    pred_abs = np.sqrt(1 - b) * incoh
    print(f"\n  {xlab:>7} {'eps_h':>9} {'sqrt(1-b)*|dWperp|':>19} {'ratio':>7} "
          f"{'eps/state':>10} {'pred frac':>10} {'ratio':>7} {'h0':>9} {'h_end':>9}")
    for i in np.argsort(xs):
        print(f"  {xs[i]:7.4g} {epsh[i]:9.4g} {pred_abs[i]:19.4g} "
              f"{epsh[i]/pred_abs[i]:7.3f} {frac[i]:10.4f} {pred_abs[i]/h0[i]:10.4f} "
              f"{frac[i]*h0[i]/pred_abs[i]:7.3f} {h0[i]:9.3f} {hend[i]:9.3f}")
    for nm, y in [("eps_h (absolute)", epsh), ("|dW_perp|_F", incoh),
                  ("h_norm at step 0 (pretrained store)", h0),
                  ("eps_h / [sqrt(1-b)|dW_perp|]", epsh / pred_abs),
                  ("eps/state / [sqrt(1-b)|dW_perp|/h0]", frac * h0 / pred_abs)]:
        s, ss = loglog(xs, y)
        v = np.asarray(y)
        print(f"    slope of {nm:<38} on {xlab:<6} {s:+7.3f} R^2 {ss:5.2f}  "
              f"spread {v.max()/max(v.min(),1e-300):6.2f}x")


def main():
    m = [r for r in load("erosion_d[0-9]*.json").values() if r["seed"] in (0, 1)]
    print("=" * 118 + "\nCORRECTED LAW vs d (gate-matched)\n" + "=" * 118)
    report(m, "d", "d")
    f = list(load("erosion_dfix*.json").values())
    print("\n" + "=" * 118 + "\nCORRECTED LAW vs d (fixed rate)\n" + "=" * 118)
    report(f, "d", "d")
    b = list(load("erosion_betafix.json").values())
    print("\n" + "=" * 118 + "\nCORRECTED LAW vs 1-b (fixed rate, fixed steps)\n" + "=" * 118)
    report(b, "beta", "1-b", xs=[1 - r["beta"] for r in b])
    n = list(load("erosion_nafix.json").values())
    print("\n" + "=" * 118 + "\nCORRECTED LAW vs n_A\n" + "=" * 118)
    report(n, "na", "n_A")

    t = list(load("erosion_time.json").values())
    print("\n" + "=" * 118 + "\nCORRECTED LAW through time (one run, 10000 steps)\n" + "=" * 118)
    for r in t:
        rows = [x for x in r["rows"] if x["step"] >= 50]
        st = np.array([x["step"] for x in rows], float)
        pr = np.array([np.sqrt(1 - r["beta"]) * pieces(r, x)[0] / r["rows"][0]["h_norm"]
                       for x in rows])
        fr = np.array([x["eps_post"] / x["post_norm"] for x in rows])
        s, ss = loglog(st, fr / pr)
        print(f"\n  seed {r['seed']}: measured/predicted over t: {(fr/pr).min():.3f}"
              f"..{(fr/pr).max():.3f}, residual t-slope {s:+.3f} (R^2 {ss:.2f})")
        si, r2 = loglog(st[st >= 300], pr[st >= 300])
        print(f"    |dW_perp|_F itself grows as t^{si:+.3f} (R^2 {r2:.2f}) after B saturates")
        print("    " + "  ".join(f"{int(x['step'])}:{a:.3f}/{p:.3f}"
                                 for x, a, p in list(zip(rows, fr, pr))[::max(1, len(rows)//9)]))


if __name__ == "__main__":
    main()
