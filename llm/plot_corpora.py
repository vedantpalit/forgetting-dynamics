"""Three corpora on OLMo 2 1B, the same model and optimiser: the laws' knobs moved.

    B        1,000 invented people x 4 attributes, answers first-token-disjoint from A's
    B-same   2,000 people x 2 attributes, answers = A's own company / city answers
    B-small    250 people x 4 attributes, B's pools (1,000 facts)

    plots/olmo_corpora_A.{png,pdf}      old facts (non-copy), each corpus's trough marked
    plots/olmo_corpora_B.{png,pdf}      new facts, the same trough steps marked
    plots/olmo_corpora_AB.{png,pdf}     both in one panel: old facts solid, new facts dotted
    plots/olmo_same_strata.{png,pdf}    B-same by stratum: relations whose answers are inside
                                        B's region (lifted) against the rest (suppressed)

Three seeds each, mean and +/-1 sd; lr 1e-5; the first 300 steps.

Run: uv run python llm/plot_corpora.py
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, TEAL, OLIVE, MARK = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "#9A9A9A"
CORPORA = (("B", "llm/out/inject_lr1e-05_seed*.json", "#08306b", "-"),
           ("B-small", "llm/out/inject_small_lr1e-05_seed*.json", "#2171b5", "-."),
           ("B-same", "llm/out/inject_same_lr1e-05_seed*.json", "#6baed6", "--"))
# plain-language names for the combined panel
PLAIN = {"B": "disjoint answers", "B-small": "disjoint answers, fewer facts",
         "B-same": "shared answers"}
XMAX = 1000


TROUGH_WINDOW = 300   # the trough is the minimum inside the event, not the eroded end


def load(pat, key):
    """Mean over seeds on the steps every run has (a run that stopped early is used where it
    exists and the others carry on alone beyond it would be misleading; so: common steps)."""
    runs = [json.load(open(f))["curve"] for f in sorted(glob.glob(pat))]
    common = sorted(set.intersection(*[{r["step"] for r in run} for run in runs]))
    st = np.array(common, float)
    Y = np.array([[{r["step"]: r[key] for r in run}[s] for s in common] for run in runs])
    keep = st <= XMAX
    return st[keep], Y[:, keep]


def trough_of(st, m):
    w = st <= TROUGH_WINDOW
    return int(np.flatnonzero(w)[m[w].argmin()])


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})


def band(ax, st, Y, col, ls, lab, lw=2.4):
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.18, lw=0)
    ax.plot(st, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
    return m


def main():
    style()
    os.makedirs("plots", exist_ok=True)
    troughs = {}
    # --- old facts
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    for name, pat, col, ls in CORPORA:
        st, A = load(pat, "A/noncopy/acc")
        m = band(ax, st, A, col, ls, name)
        tr = trough_of(st, m); troughs[name] = (st[tr], m[tr])
        ax.axvline(st[tr], color=col, ls=":", lw=1.3, alpha=0.8, zorder=1)
        ax.plot([st[tr]], [m[tr]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        print(f"{name:8s} trough {m[tr]:.3f} @ {st[tr]:.0f}   peak after {m[tr:].max():.3f}")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("Old-fact accuracy (non-copy)")
    ax.legend(frameon=False, fontsize=11, loc="lower left", handlelength=1.8, title="new-fact corpus", title_fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"plots/olmo_corpora_A.{e}", bbox_inches="tight")
    plt.close(fig)
    # --- new facts, same markers
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    for name, pat, col, ls in CORPORA:
        st, B = load(pat, "B/ALL/acc")
        m = band(ax, st, B, col, ls, name)
        x_tr, _ = troughs[name]; i = int(np.argmin(abs(st - x_tr)))
        ax.axvline(x_tr, color=col, ls=":", lw=1.3, alpha=0.8, zorder=1)
        ax.plot([x_tr], [m[i]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        print(f"{name:8s} B at trough {m[i]:.3f}")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("New-fact accuracy")
    ax.legend(frameon=False, fontsize=11, loc="upper left", handlelength=1.8, title="new-fact corpus", title_fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"plots/olmo_corpora_B.{e}", bbox_inches="tight")
    plt.close(fig)
    # --- both in one figure: full curves (left), old facts solid and new facts dotted, each
    #     corpus's trough on both; a magnified recovery window (right), old facts only, with
    #     the region boxed on the left and connected to the zoom
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(9.0, 4.15), dpi=200,
                                  gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.08})
    x0, x1, y0, y1 = 60, 300, 0.12, 0.72
    for name, pat, col, ls in CORPORA:
        st, A = load(pat, "A/noncopy/acc"); _, B = load(pat, "B/ALL/acc")
        m = A.mean(0); sd = A.std(0, ddof=1); bm = B.mean(0); tr = trough_of(st, m)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, m, color=col, ls="-", lw=2.4, label=PLAIN[name], zorder=3)
        ax.plot(st, bm, color=col, ls=":", lw=2.0, zorder=3)
        ax.axvline(st[tr], color=col, ls=":", lw=1.2, alpha=0.7, zorder=1)
        ax.plot([st[tr]], [m[tr]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        ax.plot([st[tr]], [bm[tr]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        k = (st >= x0) & (st <= x1)
        axz.fill_between(st[k], (m - sd)[k], (m + sd)[k], color=col, alpha=0.15, lw=0)
        axz.plot(st[k], m[k], color=col, lw=2.4, zorder=3)
        axz.plot([st[tr]], [m[tr]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        pk = tr + int(m[tr:][st[tr:] <= TROUGH_WINDOW].argmax())
        print(f"{name:8s} recovery {m[tr]:.3f}@{st[tr]:.0f} -> {m[pk]:.3f}@{st[pk]:.0f}  (+{m[pk]-m[tr]:.2f}, {(m[pk]-m[tr])/(m[0]-m[tr]):.0%} of the loss)")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("First-token accuracy")
    fig.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, handlelength=1.8, columnspacing=1.6)
    axz.set_xlim(x0, x1); axz.set_ylim(y0, y1)
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_color("0.5"); sp.set_linestyle("--")
    # the zoomed region boxed on the left; two connectors from its right corners to the
    # zoom panel's left corners (drawn by hand so they never cross the curves)
    from matplotlib.patches import Rectangle, ConnectionPatch
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="0.55", lw=0.9, ls="--", zorder=5))
    for yy in (y0, y1):
        fig.add_artist(ConnectionPatch(xyA=(x1, yy), coordsA=ax.transData, xyB=(x0, yy), coordsB=axz.transData,
                                       color="0.55", lw=0.9, ls="--"))
    for e in ("png", "pdf"):
        fig.savefig(f"plots/olmo_corpora_AB.{e}", bbox_inches="tight")
    plt.close(fig)
    # --- B-same by stratum: inside (companies P176/P178, cities are in 'other') vs the rest
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    pat = CORPORA[2][1]
    for key, col, ls, lab in (("Arel/P176/acc", TEAL, "-", "P176 makers (answers inside B's region)"),
                              ("Arel/P178/acc", TEAL, "--", "P178 developers (inside)"),
                              ("Arel/other/acc", CRIM, "-", "other relations (mostly outside)")):
        st, Y = load(pat, key)
        band(ax, st, Y, col, ls, lab)
    st0, Y0 = load(CORPORA[0][1], "Arel/other/acc")
    band(ax, st0, Y0, CRIM, ":", "other relations, corpus B", lw=1.6)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("Old-fact accuracy (all A)")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", handlelength=1.8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"plots/olmo_same_strata.{e}", bbox_inches="tight")
    print("wrote plots/olmo_corpora_{A,B,AB}, plots/olmo_same_strata")


if __name__ == "__main__":
    main()
