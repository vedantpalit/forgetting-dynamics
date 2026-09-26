"""Figure 5: suppression in OLMo 2 1B, one figure, three panels, one question each.

    plots/fig5.{png,pdf}

Left    the timing: three learning rates; the old facts' trough moves twofold in step while the
        new facts are still at chance at every trough (circles on both).
Middle  the low-rank removal (lr 1e-5): the top four singular directions of every block
        matrix's update removed at each checkpoint keep the old facts through the collapse and
        leave the new facts unlearned; a random rank-4 removal of the same norm changes nothing.
Right   when it occurs: three new-fact sets on the same model and optimiser -- disjoint answers,
        disjoint answers with fewer facts, answers shared with the old facts -- to step 1000, with
        a magnified window of the troughs and recoveries (old facts only) beside it.

Old facts solid (non-copy stratum), new facts dotted; three seeds, mean and +/- 1 sd.
Data: llm/out/inject_lr*_seed*.json, llm/out/patchw_lr1e-05_seed*.json,
llm/out/inject_{small,same}_lr1e-05_seed*.json.

Run: uv run python -m llm.plot_fig5
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm.plot_timing import ARMS, load as load_rate, phases
from llm.plot_corpora import CORPORA, PLAIN, load as load_corpus, trough_of

NAVY, OLIVE = "#1B2A4E", "#8A8C30"


def band(ax, st, Y, col, ls="-", lw=2.2, lab=None, fill=True, z=3):
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    if fill:
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
    ax.plot(st, m, color=col, ls=ls, lw=lw, label=lab, zorder=z)
    return m


def circle(ax, x, y, col):
    ax.plot([x], [y], marker="o", ms=6.5, mfc="white", mec=col, mew=1.8, zorder=5)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig = plt.figure(figsize=(15.5, 3.7), dpi=200)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.6], wspace=0.12,
                             left=0.05, right=0.99, top=0.78, bottom=0.15)
    a1 = fig.add_subplot(outer[0]); a2 = fig.add_subplot(outer[1], sharey=a1)
    inner = outer[2].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.06)
    a3 = fig.add_subplot(inner[0], sharey=a1); az = fig.add_subplot(inner[1])
    x0, x1, y0, y1 = 60, 300, 0.12, 0.72
    # --- timing across learning rates
    for tag, lab, col in ARMS:
        st, A, B = load_rate(tag, "noncopy"); keep = st <= 1000; st, A, B = st[keep], A[:, keep], B[:, keep]
        m = band(a1, st, A, col, lab=f"learning rate {lab}")
        bm = band(a1, st, B, col, ls=":", lw=1.6, fill=False)
        i = phases(list(m)); circle(a1, st[i], m[i], col); circle(a1, st[i], bm[i], col)
        print(f"timing {lab}: trough {m[i]:.3f}@{st[i]:.0f}, new facts {bm[i]:.3f}")
    a1.set_xlim(0, 1000)
    # --- the low-rank removal
    runs = [json.load(open(f))["curve"] for f in sorted(glob.glob("llm/out/patchw_lr1e-05_seed*.json"))]
    st = np.array([r["step"] for r in runs[0] if r["step"] <= 1000], float)
    get = lambda k: np.array([[r[k] for r in c if r["step"] <= 1000] for c in runs])
    band(a2, st, get("actual/A_noncopy"), NAVY, lab="trained")
    band(a2, st, get("actual/B"), NAVY, ls=":", lw=1.6, fill=False)
    band(a2, st, get("patch_r4/A_noncopy"), OLIVE, lab="top directions removed")
    band(a2, st, get("patch_r4/B"), OLIVE, ls=":", lw=1.6, fill=False)
    band(a2, st, get("control_r4/A_noncopy"), "0.5", ls="--", lw=1.2, fill=False,
         lab="random directions removed", z=4)
    a2.set_xlim(0, 1000)
    # --- the corpora
    for name, pat, col, _ls in CORPORA:
        s3, A = load_corpus(pat, "A/noncopy/acc"); _, B = load_corpus(pat, "B/ALL/acc")
        keep = s3 <= 1000; s3, A, B = s3[keep], A[:, keep], B[:, keep]
        m = band(a3, s3, A, col, lab=PLAIN[name]); bm = band(a3, s3, B, col, ls=":", lw=1.6, fill=False)
        i = trough_of(s3, m); circle(a3, s3[i], m[i], col)
        k = (s3 >= x0) & (s3 <= x1)                       # the magnified window: old facts only
        band(az, s3[k], A[:, k], col); circle(az, s3[i], m[i], col)
        print(f"corpus {name}: trough {m[i]:.3f}@{s3[i]:.0f}, after {m[i:].max():.3f}")
    a3.set_xlim(0, 1000)
    az.set_xlim(x0, x1); az.set_ylim(y0, y1); az.set_xticks([]); az.set_yticks([])
    for sp in az.spines.values():
        sp.set_color("0.5"); sp.set_linestyle("--")
    from matplotlib.patches import Rectangle, ConnectionPatch
    a3.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="0.55", lw=0.9, ls="--", zorder=6))
    for yy in (y0, y1):
        fig.add_artist(ConnectionPatch(xyA=(x1, yy), coordsA=a3.transData, xyB=(x0, yy),
                                       coordsB=az.transData, color="0.55", lw=0.9, ls="--"))
    for ax in (a2, a3):
        plt.setp(ax.get_yticklabels(), visible=False)
    for ax in (a1, a2, a3):
        ax.set_ylim(0, 1.0); ax.set_xlabel("Fine-tuning step")
        ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=1 if ax is not a1 else 1, handlelength=1.8, labelspacing=0.25)
    a1.set_ylabel("First-token accuracy")
    for e in ("png", "pdf"):
        fig.savefig(f"plots/fig5.{e}", bbox_inches="tight")
    print("wrote plots/fig5")


if __name__ == "__main__":
    main()
