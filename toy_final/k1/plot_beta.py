"""The beta sweep figure: the crash is caused by the shared key component, the recovery by
the normalizer, and neither is an artifact of beta = 0.5."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30"
SEEDS = (0, 1, 2)


def series(d, beta, norm, key):
    runs = [d[f"b{beta}-n{norm}-g1.0-u0.01-s{s}"]["rows"] for s in SEEDS]
    st = np.array([x["step"] for x in runs[0]], float)
    Y = np.array([[x[key] for x in r] for r in runs])
    return st, Y.mean(0), Y.std(0, ddof=1)


def stat(d, beta, norm):
    """crash depth, recovery, and the fall in the write's between-half content."""
    out = []
    for s in SEEDS:
        r = d[f"b{beta}-n{norm}-g1.0-u0.01-s{s}"]["rows"]
        A = np.array([x["A"] for x in r]); sw = np.array([x["s_on_w"] for x in r])
        tr = int(A.argmin()); pw = int(sw.argmax())
        drop = (1 - sw[-1] / sw[pw]) if sw[pw] > 0.05 else np.nan
        out.append((1 - A[tr], A[tr:].max() - A[tr], drop))
    a = np.array(out)
    return a.mean(0), a.std(0, ddof=1)


def main():
    d = json.load(open("beta_sweep.json"))
    betas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 10, "axes.grid": False})
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), dpi=200)

    # top row: example trajectories at three values of beta
    for c, beta in enumerate((0.1, 0.5, 0.9)):
        ax = axes[0, c]
        for norm, ls, lab in ((1, "-", "normalized"), (0, "--", "linear")):
            st, m, sd = series(d, beta, norm, "A")
            ax.fill_between(st, m - sd, m + sd, color=NAVY, alpha=0.14, lw=0)
            ax.plot(st, m, color=NAVY, ls=ls, lw=2.0, label=f"A, {lab}")
            st, m, sd = series(d, beta, norm, "B")
            ax.plot(st, m, color=CRIM, ls=ls, lw=1.4, label=f"B, {lab}")
        ax.set_xscale("symlog", linthresh=10); ax.set_ylim(0, 1.03)
        ax.set_title(f"$\\beta$ = {beta}", fontsize=11)
        ax.set_xlabel("injection step")
        if c == 0:
            ax.set_ylabel("first-token accuracy")
            ax.legend(frameon=False, fontsize=7.5, loc="center left")

    # bottom row: the three summary quantities against beta
    panels = [(0, "crash depth   $1 - A_{\\mathrm{trough}}$", NAVY),
              (1, "recovery   $A_{\\mathrm{after}} - A_{\\mathrm{trough}}$", OLIVE),
              (2, "fall in the collective write", PURP)]
    for c, (idx, label, col) in enumerate(panels):
        ax = axes[1, c]
        for norm, ls, mk, lab in ((1, "-", "o", "normalized readout"),
                                  (0, "--", "s", "linear (no norm)")):
            M = np.array([stat(d, b, norm)[0][idx] for b in betas])
            S = np.array([stat(d, b, norm)[1][idx] for b in betas])
            ok = ~np.isnan(M)
            ax.errorbar(np.array(betas)[ok], M[ok], yerr=S[ok], color=col, ls=ls, marker=mk,
                        ms=4, lw=1.8, capsize=2.5, label=lab)
        ax.set_xlabel("$\\beta$   (shared component of the keys)")
        ax.set_ylabel(label)
        if idx == 2:
            ax.set_ylim(-0.05, 1.0)
            ax.axhline(0, color="0.6", lw=0.8, ls=":")
        else:
            ax.set_ylim(-0.05, 1.05)
        if c == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs("../../plots", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_beta_sweep.{ext}", bbox_inches="tight")
    print("wrote plots/toy_k1_beta_sweep.png")

    print(f"\n{'beta':>5} | {'crash depth':>22} | {'recovery':>22} | {'write falls by':>22}")
    for norm, nm in ((1, "normalized"), (0, "linear")):
        print(f"  --- {nm} readout ---")
        for b in betas:
            m, s = stat(d, b, norm)
            w = "  n/a (no write)" if np.isnan(m[2]) else f"{100*m[2]:>8.0f} +/- {100*s[2]:<4.0f}%"
            print(f"{b:>5} | {m[0]:>10.2f} +/- {s[0]:<8.2f} | {m[1]:>10.2f} +/- {s[1]:<8.2f} | {w:>22}")


if __name__ == "__main__":
    main()
