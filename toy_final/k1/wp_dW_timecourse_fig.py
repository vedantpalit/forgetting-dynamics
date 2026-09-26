"""Does ||W|| grow throughout the phenomenon, or turn over like accuracy?

    plots/toy_plots_working/dW_timecourse.{png,pdf}

K=1, gain=1, beta=0.5, standard config, 3 seeds. ||dW||^2 = ||s||^2 + ||dW_perp||^2 exactly
(s = mu^T dW, the coherent/shared row; dW_perp = dW - mu s^T, the incoherent remainder --
orthogonal by construction, so the squares add). Plots all three against A's accuracy on the
same time axis: the coherent part peaks and is partly withdrawn (this IS the "recovery" seen
in weight space), but the total never turns over because the incoherent part (erosion) keeps
growing and overtakes it. Data: wp_dW_timecourse.json, written inline by the check that
produced this figure's numbers.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
NAVY, TEAL, OLIVE, GREY = "#1B2A4E", "#2F8C7D", "#8A8C30", "0.55"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})


def main():
    data = json.load(open(os.path.join(HERE, "wp_dW_timecourse.json")))
    seeds = sorted(int(s) for s in data)
    st = np.array(data[str(seeds[0])]["step"], float)
    dW = np.array([data[str(s)]["dW"] for s in seeds])
    sN = np.array([data[str(s)]["s"] for s in seeds])
    perp = np.array([data[str(s)]["dWperp"] for s in seeds])
    A = np.array([data[str(s)]["A"] for s in seeds])
    keep = st <= 2000

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.6), dpi=200, sharex=True)

    ax = axes[0]
    ax.plot(st[keep], A.mean(0)[keep], color=NAVY, lw=1.8)
    ax.fill_between(st[keep], A.mean(0)[keep] - A.std(0)[keep], A.mean(0)[keep] + A.std(0)[keep],
                     color=NAVY, alpha=0.15, lw=0)
    ax.set_ylabel("A's first-token accuracy")
    ax.set_ylim(0, 1.04)

    ax = axes[1]
    for arr, c, lab in ((dW, GREY, "$\\|\\Delta W\\|$ (total)"),
                         (sN, TEAL, "$\\|s\\|$ (coherent, shared row)"),
                         (perp, OLIVE, "$\\|\\Delta W_\\perp\\|$ (incoherent, erosion)")):
        m = arr.mean(0)[keep]; sd = arr.std(0)[keep]
        ax.plot(st[keep], m, color=c, lw=1.9, label=lab)
        ax.fill_between(st[keep], m - sd, m + sd, color=c, alpha=0.12, lw=0)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("Weight-change norm")
    ax.legend(frameon=False, fontsize=8, loc="center right", handlelength=1.8)

    fig.tight_layout()
    outdir = os.path.join(ROOT, "plots", "toy_plots_working")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"dW_timecourse.{e}"), bbox_inches="tight")
    print("wrote dW_timecourse")


if __name__ == "__main__":
    main()
