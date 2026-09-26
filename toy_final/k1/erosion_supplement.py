"""Second pass: the dose-normalised erosion, the r-integral, and the time law's shape."""
import numpy as np

from erosion_analyze import load, last, at, loglog, linfit


def R_int(r):
    """ilr * integral of r_rms over steps -- the 'how much error signal was written' dose."""
    st = np.array([x["step"] for x in r["rows"]], float)
    rr = np.array([x["r_rms"] for x in r["rows"]], float)
    return float(r["inject_lr"] * np.trapezoid(rr, st))


def frac(r):
    return last(r)["eps_post"] / last(r)["post_norm"]


def block(runs, xkey, xlab, xs=None):
    xs = xs if xs is not None else np.array([r[xkey] for r in runs], float)
    e = np.array([frac(r) for r in runs])
    dose = np.array([last(r)["dose"] for r in runs])
    dW = np.array([last(r)["dW"] for r in runs])
    R = np.array([R_int(r) for r in runs])
    hn = np.array([last(r)["h_norm"] for r in runs])
    d = np.array([r["d"] for r in runs], float)
    nb = np.array([r["nb"] for r in runs], float)
    b = np.array([r["beta"] for r in runs])
    print(f"\n  {xlab:>8} " + "".join(f"{h:>11}" for h in
          ["eps/st", "/dose", "/dW", "R_int", "h_norm", "|W0|h"]))
    for i in np.argsort(xs):
        print(f"  {xs[i]:8.4g} " + "".join(f"{v:11.4g}" for v in
              [e[i], e[i] / dose[i], e[i] / dW[i], R[i], hn[i], 0.0]))
    for nm, y in [("eps/state", e), ("eps/state per unit dose", e / dose),
                  ("eps/state per unit ||dW||", e / dW), ("R_int = ilr*int r_rms dt", R),
                  ("dose", dose), ("||dW||_F", dW), ("h_norm(A)", hn)]:
        s, ss = loglog(xs, y)
        print(f"    slope of {nm:<26} on {xlab:<8} {s:+7.3f}  R^2 {ss:5.2f}")
    # the full chain, dose-normalised
    pred = (1 - b) * np.sqrt(nb) * (1.0 / np.sqrt(d)) * np.sqrt(d) * R * d / hn
    s, ss = loglog(xs, e / pred)
    v = e / pred
    print(f"    FULL CHAIN  eps/state / [(1-b) sqrt(nB) gg_rms * d * R / h_norm]: "
          f"slope {s:+.3f} R^2 {ss:.2f}, spread {v.max()/max(v.min(),1e-30):.2f}x")
    return e, dose, R


def main():
    m = [r for r in load("erosion_d[0-9]*.json").values() if r["seed"] in (0, 1)]
    print("=" * 100 + "\nGATE-MATCHED d SWEEP: dose-normalised accounting\n" + "=" * 100)
    block(m, "d", "d")
    print("\n  per-run pretrain diagnostics (h_norm is the PRE-normalizer A state):")
    for r in sorted(m, key=lambda r: (r["d"], r["seed"])):
        print(f"    d={r['d']:>4} s={r['seed']} pretrain_lr={r['pretrain_lr']:<5} "
              f"gate@{r['pretrain_gate']:<6} ilr={r['inject_lr']:.3e}  "
              f"h_norm={last(r)['h_norm']:9.3f}  sqrt(d)={np.sqrt(r['d']):6.2f}  "
              f"eps/st={frac(r):.4f}")

    f = list(load("erosion_dfix*.json").values())
    print("\n" + "=" * 100 + "\nFIXED-RATE d SWEEP: same accounting\n" + "=" * 100)
    block(f, "d", "d")

    b = list(load("erosion_betafix.json").values())
    print("\n" + "=" * 100 + "\nBETA at fixed rate/steps: is (1-b) there once the dose is out?"
          + "\n" + "=" * 100)
    bs = np.array([1 - r["beta"] for r in b])
    block(b, "beta", "1-b", xs=bs)

    n = list(load("erosion_nafix.json").values())
    print("\n" + "=" * 100 + "\nn_A at fixed rate/steps\n" + "=" * 100)
    block(n, "na", "n_A")

    # ---- time law shape ----
    t = list(load("erosion_time.json").values())
    print("\n" + "=" * 100 + "\nTIME LAW: power vs logarithm, and the increment's own decay\n"
          + "=" * 100)
    for r in t:
        rows = [x for x in r["rows"] if x["step"] >= 300]
        st = np.array([x["step"] for x in rows], float)
        e = np.array([x["eps_post"] / x["post_norm"] for x in rows])
        inc = np.array([x["inc"] for x in rows])
        rr = np.array([x["r_rms"] for x in rows])
        sp, r2p = loglog(st, e)
        sl, cl, r2l = linfit(np.log(st), e)
        si, r2i = loglog(st, inc)
        sr, r2r = loglog(st, rr)
        # saturating alternative: eps = E_inf (1 - exp(-t/tau))  -> check curvature instead
        print(f"\n  seed {r['seed']}: eps/state {e[0]:.4f}@{int(st[0])} -> {e[-1]:.4f}@{int(st[-1])} "
              f"({e[-1]/e[0]:.2f}x over {st[-1]/st[0]:.0f}x in t)")
        print(f"    power law   eps ~ t^{sp:+.3f}        R^2 {r2p:.4f}")
        print(f"    logarithmic eps = {sl:+.4f} ln t {cl:+.4f}  R^2 {r2l:.4f}")
        print(f"    random walk would be +0.500, linear +1.000")
        print(f"    driver: per-step increment inc ~ t^{si:+.3f} (R^2 {r2i:.2f}); "
              f"r_rms ~ t^{sr:+.3f} (R^2 {r2r:.2f})")
        # does eps keep growing or flatten in absolute terms?
        w = st >= 3000
        s2, _ = loglog(st[w], e[w])
        print(f"    late window (t>3000): exponent {s2:+.3f}; "
              f"extrapolated eps/state at t=1e6 would be {e[-1]*(1e6/st[-1])**s2:.3f}")


if __name__ == "__main__":
    main()
