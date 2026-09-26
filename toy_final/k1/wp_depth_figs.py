"""Figures for the K-block depth runs (wp_depth_stack.py).

  depth_accuracy     A and B per K, normalized vs the no-norm control
  depth_crossover    sum_j ||P_j||, ||sum_j P_j|| and the mean pairwise cosine vs step
  depth_decoherence  the de-coherence share of the fall, vs K and vs the write gain
  depth_blocks_K8    what each block carries, and how far it turns

The decomposition plotted is the EXACT one: post_a = c_a (k_a + sum_j P_{j,a}), so the per-
block terms sum to the measured shift with no remainder (the key term is 0.01-0.13 of 6-10).
The residual-space ("direct") contributions are shown dashed where they differ, because that
is the measure whose cosines move.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "plots", "toy_plots_working"))
NAVY, CRIM, PURP, OLIVE, GREY = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30", "#9AA0A6"
KS = (1, 2, 4, 8)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 9.5, "axes.grid": False, "figure.dpi": 200})


def load(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        out.update(json.load(open(f)))
    return list(out.values())


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{png,pdf}"))


def band(ax, x, Y, color, ls="-", lw=1.9, label=None, alpha=0.15):
    Y = np.asarray(Y, float)
    m = Y.mean(0)
    if len(Y) > 1:
        sd = Y.std(0, ddof=1)
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=alpha, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
    return m


def series(runs, key):
    n = min(len(r[key]) for r in runs)
    return np.array([r[key][:n] for r in runs]), np.array(runs[0]["steps"][:n], float)


def fig_accuracy():
    lin = load("wp_depth_K8_lin.json")
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.0), sharey=True)
    for ax, K in zip(axes, KS):
        runs = load(f"wp_depth_K{K}.json")
        A, st = series(runs, "A")
        B, _ = series(runs, "B")
        band(ax, st, A, NAVY, label="A  (pretrained)")
        band(ax, st, B, CRIM, lw=1.5, label="B  (injected)")
        m = A.mean(0)
        tr = int(np.argmin(np.where(st <= 600, m, np.inf)))
        ax.plot(st[tr], m[tr], "o", ms=5, mfc="white", mec=NAVY, mew=1.4, zorder=5)
        ax.set_title(f"K = {K}   (normalized)", fontsize=10)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("injection step")
    if lin:
        ax = axes[4]
        A, st = series(lin, "A")
        B, _ = series(lin, "B")
        band(ax, st, A, NAVY, label="A  (pretrained)")
        band(ax, st, B, CRIM, lw=1.5, label="B  (injected)")
        ax.set_title("K = 8   no norms anywhere", fontsize=10)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("injection step")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.04)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Depth deepens the crash and leaves A lower at the end, but the recovery "
                 "survives at every K; removing every normalizer removes it", y=1.06, fontsize=11)
    save(fig, "depth_accuracy")


def fig_crossover():
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.2))
    for c, K in enumerate(KS):
        runs = load(f"wp_depth_K{K}.json")
        st = np.array(runs[0]["steps"], float)
        ax = axes[0, c]
        band(ax, st, [r["sumn_post"] for r in runs], OLIVE, label=r"$\sum_j \|P_j\|$")
        band(ax, st, [r["snorm_post"] for r in runs], NAVY, label=r"$\|\sum_j P_j\|$")
        band(ax, st, [r["delta"] for r in runs], GREY, ls=":", lw=1.4,
             label=r"$\|\delta\|$ (A-mean shift)")
        ax.set_xscale("symlog", linthresh=10); ax.set_title(f"K = {K}", fontsize=10.5)
        if c == 0:
            ax.set_ylabel("post-final-norm space")
            ax.legend(frameon=False, fontsize=8)
        ax = axes[1, c]
        if K > 1:
            band(ax, st, [r["meancos_post"] for r in runs], PURP, label="exact decomposition")
            band(ax, st, [r["meancos_dir"] for r in runs], OLIVE, ls="--", lw=1.5,
                 label="residual-space blocks")
        else:
            ax.text(0.5, 0.5, "one block:\nno pairs", ha="center", va="center", color=GREY,
                    fontsize=9.5, transform=ax.transAxes)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(axes[0, c].get_xlim())
        ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("injection step")
        if c == 0:
            ax.set_ylabel("mean pairwise cosine")
        if c == 1:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("The sum falls because the parts shrink, not because they de-align: "
                 "the cosine is flat or rising through the recovery", y=0.98, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "depth_crossover")


def fig_decoherence():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    cmap = plt.get_cmap("plasma")

    LO, HI = -1.25, 1.25

    def points(ax, xs, groups, tag, color, label, dx=0.0, key="share_{}", ls="-", mfc=None, ytext=6):
        """Shares outside the panel are drawn on the edge and labelled, not clipped away:
        a share of -4 (the cosine channel working AGAINST the fall) is a real reading."""
        mus = []
        for x, runs in zip(xs, groups):
            k = key.format(tag)
            v = [r[k]["decoh_share"] for r in runs
                 if r[k]["s_peak"] > 0 and r[k]["fall"] / r[k]["s_peak"] >= 0.05]
            mus.append(np.mean(v) if v else np.nan)
            ax.plot([x + dx] * len(v), np.clip(v, LO + 0.05, HI - 0.05), "o", ms=3.5,
                    color=color, alpha=0.45, mew=0)
        mus = np.array(mus, float)
        y = np.clip(mus, LO + 0.05, HI - 0.05)
        ax.plot(np.array(xs) + dx, y, ls=ls, marker="o", color=color, ms=5.5, lw=1.8,
                label=label, mfc=mfc or color)
        for x, m, yy in zip(xs, mus, y):
            if np.isfinite(m) and (m < LO or m > HI):
                ax.annotate(f"{m:+.1f}", (x + dx, yy), textcoords="offset points",
                            xytext=(4, ytext if m < LO else -12), fontsize=7.5, color=color)

    ax = axes[0]
    groups = [load(f"wp_depth_K{K}.json") for K in KS]
    points(ax, np.arange(len(KS)), groups, "post", NAVY, "shift peak -> A's recovery peak")
    points(ax, np.arange(len(KS)), groups, "post", PURP, "shift peak -> step 5000", dx=0.10,
           key="share_{}_end", ls="--", mfc="white", ytext=17)
    ax.set_xticks(np.arange(len(KS))); ax.set_xticklabels([f"K = {K}" for K in KS])
    ax.set_ylabel("de-coherence share of the fall")
    ax.set_title("vs depth, write gain 1", fontsize=10.5)

    ax = axes[1]
    gs = (0.5, 1.0, 2.0, 4.0)
    groups = [load(f"wp_depth_K4_g{g}.json") for g in gs]
    points(ax, np.log2(gs), groups, "post", NAVY, "shift peak -> A's recovery peak")
    points(ax, np.log2(gs), groups, "post", PURP, "shift peak -> step 5000", dx=0.08,
           key="share_{}_end", ls="--", mfc="white", ytext=17)
    points(ax, np.log2(gs), groups, "dir", OLIVE, "residual-space blocks, to recovery", dx=0.16)
    ax.set_xticks(np.log2(gs)); ax.set_xticklabels([f"g = {g:g}" for g in gs])
    ax.set_title("vs write gain, K = 4", fontsize=10.5)
    for ax in axes:
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axhline(1, color=GREY, lw=0.8, ls=":")
        ax.set_ylim(LO, HI)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    # the transformer's own number, for scale
    for ax in axes:
        ax.axhspan(0.55, 0.75, color=CRIM, alpha=0.10, lw=0)
    axes[1].annotate("transformer's regime\n(cosine carries most of it)", (np.log2(4.0), 0.65),
                     textcoords="offset points", xytext=(-120, 22), fontsize=8, color=CRIM)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("At A's recovery peak the fall is pure shrinkage at every depth; "
                 "de-coherence only appears later, and with big writes",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "depth_decoherence")


def fig_blocks():
    runs = load("wp_depth_K8.json")
    K = 8
    cmap = plt.get_cmap("plasma")
    cols = [cmap(0.08 + 0.82 * j / (K - 1)) for j in range(K)]
    st = np.array(runs[0]["steps"], float)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))

    ax = axes[0]
    n = np.array([r["n_post"] for r in runs])                 # (seeds, T, K)
    for j in range(K):
        band(ax, st, n[:, :, j], cols[j], lw=1.5, label=f"block {j}", alpha=0.10)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("injection step")
    ax.set_ylabel(r"$\|P_j\|$  (exact, post-norm)")
    ax.set_title("every block shrinks, none backs off alone", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    rot = np.array([r["rot_post"] for r in runs])
    for j in range(K):
        band(ax, st, rot[:, :, j], cols[j], lw=1.5, alpha=0.10)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("injection step")
    ax.set_ylabel("angle from the block's own direction\nat the shift peak (deg)")
    ax.set_title("and every block turns by about the same amount", fontsize=10)

    ax = axes[2]
    x = np.arange(K)
    for nm, idx, mk in (("at the shift peak", "peak", "o"), ("at the recovery peak", "rec", "s")):
        f = np.array([r[f"{idx}_post"]["frac"] for r in runs])
        m, sd = f.mean(0), f.std(0, ddof=1)
        ax.errorbar(x, m, yerr=sd, fmt=mk + "-", color=NAVY if idx == "peak" else CRIM,
                    ms=5, lw=1.6, capsize=2.5, label=nm)
    ax.axhline(0.5, color=GREY, lw=0.9, ls=":")
    ax.set_ylim(0, 0.60)
    ax.annotate("the transformer's last block sits here", (0, 0.5), textcoords="offset points",
                xytext=(2, -13), fontsize=8, color=GREY)
    ax.set_xlabel("block"); ax.set_ylabel(r"share of $\delta$:  $\langle P_j,\hat\delta\rangle/\|\delta\|$")
    ax.set_title("no block carries the shift on its own", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("K = 8: what each block carries (6 seeds, exact post-norm decomposition)",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, "depth_blocks_K8")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "acc"):
        fig_accuracy()
    if which in ("all", "cross"):
        fig_crossover()
    if which in ("all", "decoh"):
        fig_decoherence()
    if which in ("all", "blocks"):
        fig_blocks()
