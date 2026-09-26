"""The no-final-norm arm next to the standard arm: same analyses, three rows.

    plots/toy_plots_working/nofinalnorm_vs_standard.{png,pdf}

Rows: A's accuracy (full vocabulary and own half); ||delta|| and rms||eps|| at the readout
state (delta_structure, last layer); the head splice R_store / R_read. Data: the cluster
outputs in weight_patch/ and delta_structure/ for arms `standard` and `no_final_norm`.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, OLIVE, CRIM = "#1B2A4E", "#2F8C7D", "#8A8C30", "#C4245F"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 9.5, "axes.grid": False})
ARMS = [("standard", "8 layers, standard"), ("no_final_norm", "8 layers, no final LayerNorm")]


def band(ax, x, Y, color, ls="-", lw=1.8, label=None):
    Y = np.asarray(Y); m = Y.mean(0); sd = Y.std(0, ddof=1) if len(Y) > 1 else 0 * m
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, marker="o", ms=2.6, label=label)


def main():
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 8.0), dpi=200, sharex=True)
    for c, (arm, title) in enumerate(ARMS):
        acc = [np.load(f) for f in sorted(glob.glob(os.path.join(ROOT, "weight_patch", f"{arm}-head-acc-*.npz")))]
        ds = [np.load(f) for f in sorted(glob.glob(os.path.join(ROOT, "delta_structure", f"{arm}-p*-seed*.npz")))]
        st = acc[0]["steps"].astype(float); sd_ = ds[0]["steps"].astype(float)
        ax = axes[0, c]
        band(ax, st, [a["acc_t"].mean(-1) for a in acc], NAVY, label="A, full vocabulary")
        band(ax, st, [a["own_t"].mean(-1) for a in acc], NAVY, ls="--", lw=1.3, label="A, own half")
        ax.set_ylim(0, 1.04); ax.set_title(title, fontsize=10.5)
        if c == 0: ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax = axes[1, c]
        band(ax, sd_, [np.linalg.norm(d["delta_layer"][:, :, -1, :].astype(np.float64), axis=-1).mean(1) for d in ds],
             TEAL, label="$\\|\\delta\\|$ (common)")
        band(ax, sd_, [d["eps_layer_rms"][:, :, -1].astype(np.float64).mean(1) for d in ds],
             OLIVE, label="rms$\\|\\varepsilon\\|$ (individual)")
        ax.set_ylim(0, None)
        if c == 0: ax.set_ylabel("readout-state displacement"); ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[2, c]
        keep = st > 0
        band(ax, st[keep], [np.nan_to_num(a["r_body"][keep]) for a in acc], NAVY, label="$R_{store}$")
        band(ax, st[keep], [np.nan_to_num(a["r_group"][keep]) for a in acc], CRIM, label="$R_{read}$")
        ax.set_ylim(-0.05, 1.1); ax.set_xlabel("injection step")
        if c == 0: ax.set_ylabel("share of A's damage"); ax.legend(frameon=False, fontsize=8, loc="center right")
        for r in range(3):
            axes[r, c].set_xscale("symlog", linthresh=10); axes[r, c].set_xlim(0, 1300)
    fig.tight_layout()
    out = os.path.join(ROOT, "plots", "toy_plots_working", "nofinalnorm_vs_standard")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
