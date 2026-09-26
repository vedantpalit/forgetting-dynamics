"""Figure 1, right panel: OLMo with the low-rank part of the update removed.

    uv run python -m llm.plot_fig1_patch      # plots/fig1_olmo_patch.{png,pdf}

Four curves, the same four as the toy's shared-row figure: old facts, old facts with the top
four singular directions of every weight matrix's update removed at each checkpoint, new
facts, new facts under the same removal. Three seeds, mean and one sd; same size, type and
axes as the left panel (llm/plot_seeds.py --paper). Data: llm/out/patchw_lr1e-05_seed*.json.
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, OLIVE = "#1B2A4E", "#C4245F", "#8A8C30"
XMAX = 1000


def main():
    runs = [json.load(open(f))["curve"] for f in sorted(glob.glob("llm/out/patchw_lr1e-05_seed*.json"))]
    st = [r["step"] for r in runs[0] if r["step"] <= XMAX]
    M = {k: np.array([[r[k] for r in c if r["step"] <= XMAX] for c in runs])
         for k in ("actual/A_noncopy", "patch_r4/A_noncopy", "control_r4/A_noncopy", "actual/B", "patch_r4/B")}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(5.8, 3.75), dpi=200)
    AS_IS, REMOVED = "#1B2A4E", "#8A8C30"
    for k, col, ls, lw, lab in (("actual/A_noncopy", AS_IS, "-", 2.4, "as trained"),
                                ("actual/B", AS_IS, ":", 2.0, None),
                                ("patch_r4/A_noncopy", REMOVED, "-", 2.4, "top directions of the update removed"),
                                ("patch_r4/B", REMOVED, ":", 2.0, None),
                                ("control_r4/A_noncopy", "0.5", "--", 1.2, None)):
        Y = M[k]; mu, sd = Y.mean(0), Y.std(0, ddof=1)
        if lab is not None or ls == ":":
            ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, mu, color=col, ls=ls, lw=lw, label=lab, zorder=3 if col != "0.5" else 4)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("First-token accuracy")
    fig.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=2,
               handlelength=1.8, columnspacing=1.6)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/fig1_olmo_patch.{e}", bbox_inches="tight")
    for k in M:
        v = M[k].mean(0); print(f"{k:>20}: {v[0]:.3f} -> min {v.min():.3f}@{st[int(v.argmin())]} -> end {v[-1]:.3f}")
    print(f"{len(runs)} seeds; wrote plots/fig1_olmo_patch")


if __name__ == "__main__":
    main()
