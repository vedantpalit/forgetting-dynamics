"""Figures for the four-arm log mining (4L/8L x ballast/no-ballast).

  logs_dose_first  -- A vs B first_acc (dose space), one panel per depth
  logs_dose_attr   -- A vs B attr_acc  (cold-start clock), one panel per depth
  logs_coldstart   -- B attr_acc and B halluc_gap vs step, and halluc_gap vs B first_acc
  logs_suppression -- A rank_own_top1 minus A first_acc vs step (between-half signature)

Run: .venv/bin/python toy_final/k1/wp_logs_figures.py
"""
import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wp_logs_analysis import DATA, LBL, MAIN, arr, at_clock, interp_cross, trough_peak  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "plots", "toy_plots_working")

NAVY = "#1B2A4E"
CRIMSON = "#C4245F"
PURPLE = "#5B3A7A"
OLIVE = "#8A8C30"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
})

STYLE = {
    "4L_ballast": dict(color=NAVY, ls="--", label="4L, ballast"),
    "4L_noballast": dict(color=CRIMSON, ls="--", label="4L, no ballast"),
    "8L_ballast": dict(color=NAVY, ls="-", label="8L, ballast"),
    "8L_noballast": dict(color=CRIMSON, ls="-", label="8L, no ballast"),
}


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
    print("wrote", os.path.join(OUTDIR, name + ".{png,pdf}"))
    plt.close(fig)


def mean_of(k, pop, field):
    a, _ = arr(k, pop, field)
    return np.nanmean(a, axis=0), DATA[k]["grid"]


# ---------------------------------------------------------------- (a),(b) dose space
def fig_dose(clock_field, name, xlabel, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=False)
    for ax, depth in zip(axes, ["4L", "8L"]):
        for k in [f"{depth}_ballast", f"{depth}_noballast"]:
            grid = DATA[k]["grid"]
            af, seeds = arr(k, "dataA", "first_acc")
            cl, _ = arr(k, "dataB", clock_field)
            x = np.nanmean(cl, 0)
            y = np.nanmean(af, 0)
            xm = np.maximum.accumulate(x)
            st = STYLE[k]
            ax.plot(xm, y, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
            # trough and recovery peak markers, from the per-seed convention
            ti, pi = trough_peak(grid, y)
            ax.plot(xm[ti], y[ti], "v", color=st["color"], ms=5, mec="none")
            ax.plot(xm[pi], y[pi], "^", color=st["color"], ms=5, mec="none")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{depth}", loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.03, 1.03)
        ax.legend(loc="upper center")
    axes[0].set_ylabel("A first-token accuracy")
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- (c) cold start
def fig_coldstart():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for k in MAIN:
        st = STYLE[k]
        grid = DATA[k]["grid"]
        ba, _ = mean_of(k, "dataB", "attr_acc")
        bh, _ = mean_of(k, "dataB", "halluc_gap")
        bf, _ = mean_of(k, "dataB", "first_acc")
        axes[0].plot(grid, ba, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
        axes[1].plot(grid, bh, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
        xm = np.maximum.accumulate(bf)
        axes[2].plot(xm, bh, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
    for ax in axes[:2]:
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xlabel("injection step")
        ax.set_xlim(0, 1200)
    axes[0].set_ylabel("B attribute-type accuracy")
    axes[0].axhline(0.5, color="0.6", lw=0.7)
    axes[0].set_title("cold start closing", loc="left")
    axes[1].set_ylabel("B hallucination gap (nats)")
    axes[1].axhline(0.0, color="0.6", lw=0.7)
    axes[1].set_title("B confidently wrong -> right", loc="left")
    axes[2].set_xlabel("B first-token accuracy")
    axes[2].set_ylabel("B hallucination gap (nats)")
    axes[2].axhline(0.0, color="0.6", lw=0.7)
    axes[2].set_title("the same, on B's own dose clock", loc="left")
    axes[2].legend(loc="upper right")
    fig.suptitle("B's acquisition: cold-start closure vs individuation",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "logs_coldstart")


# ---------------------------------------------------------------- (d) suppression signature
def fig_suppression():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for k in MAIN:
        st = STYLE[k]
        grid = DATA[k]["grid"]
        af, _ = mean_of(k, "dataA", "first_acc")
        ar, _ = mean_of(k, "dataA", "rank_own_top1")
        axes[0].plot(grid, ar - af, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
        axes[1].plot(grid, ar, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
    for ax, lab, ttl in [(axes[0], "A rank_own_top1 - first_acc", "between-half suppression"),
                         (axes[1], "A rank_own_top1", "within-half integrity")]:
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xlim(0, 1200)
        ax.set_xlabel("injection step")
        ax.set_ylabel(lab)
        ax.set_title(ttl, loc="left")
    axes[0].axhline(0.0, color="0.6", lw=0.7)
    axes[0].legend(loc="upper right")
    fig.suptitle("A's damage split into its two components", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "logs_suppression")


# ---------------------------------------------------------------- (e) the humps
def fig_humps():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, k in zip(axes, ["4L_noballast", "8L_noballast"]):
        grid = DATA[k]["grid"]
        af, seeds = arr(k, "dataA", "first_acc")
        bf, _ = arr(k, "dataB", "first_acc")
        ba, _ = arr(k, "dataB", "attr_acc")
        for i in range(len(seeds)):
            ax.plot(grid, af[i], color=CRIMSON, lw=0.6, alpha=0.45)
        ax.plot(grid, np.nanmean(af, 0), color=CRIMSON, lw=1.8, label="A (5-seed mean)")
        s1 = np.nanmean([interp_cross(grid, ba[i], 0.5) for i in range(len(seeds))])
        s2 = np.nanmean([interp_cross(grid, bf[i], 0.9) for i in range(len(seeds))])
        ax.axvline(s1, color=PURPLE, lw=1.0, ls=":")
        ax.axvline(s2, color=OLIVE, lw=1.0, ls=":")
        lo, hi = 40, 700
        w = (grid >= lo) & (grid <= hi)
        mu = np.nanmean(af, 0)[w]
        pad = 0.06 * (mu.max() - mu.min())
        ax.set_xlim(lo, hi)
        ax.set_ylim(mu.min() - pad, mu.max() + 3.2 * pad)
        top = ax.get_ylim()[1]
        ax.text(s1, top, "  B attr=0.5", color=PURPLE, fontsize=7,
                rotation=90, va="top", ha="left")
        ax.text(s2, top, "  B first=0.9", color=OLIVE, fontsize=7,
                rotation=90, va="top", ha="left")
        ax.set_xlabel("injection step")
        ax.set_title(LBL[k], loc="left")
    axes[0].set_ylabel("A first-token accuracy")
    fig.suptitle("Two recovery humps, at both depths, locked to B's two milestones (post-crash detail)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "logs_humps")


if __name__ == "__main__":
    fig_dose("first_acc", "logs_dose_first", "B first-token accuracy (dose)",
             "A's damage against B's individuation")
    fig_dose("attr_acc", "logs_dose_attr", "B attribute-type accuracy (cold-start clock)",
             "A's damage against B's cold-start closure")
    fig_coldstart()
    fig_suppression()
    fig_humps()
