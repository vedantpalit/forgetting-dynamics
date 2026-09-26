"""Figure 5 (new): the collapse at scale, and its reversal by removing one update direction.

    plots/fig5_new.{png,pdf}

(a) OLMo-2 1B, old-fact accuracy (solid) and new-fact accuracy (dotted) during finetuning on
    arbitrary facts (synthetic individuals, answers unrelated to the subject) and on real facts
    whose answers share the old facts' types (EntityQuestions P176, manufacturer): both collapse,
    only the arbitrary facts recover. (The occupation control, P106, is not drawn: over 400 steps it
    declines almost as much as P176 -- 0.58 vs 0.55 at 400 -- so it does not isolate competition.)
(b) Arbitrary facts, old facts only: the trained model, the same weights with the top singular
    direction of every matrix's update removed, and with a random direction of the same size removed.
(c) Arbitrary facts, 4,000 new facts (1,000 individuals x 4 attributes) against 1,000 new facts
    (250 individuals x 4; llm/out/inject_small_*): fewer new facts are learned sooner, the shift is
    withdrawn sooner, and the old facts recover more. The not-a-rewind comparison of the edit (old
    facts at equal new-fact accuracy) is reported in the text.

Data: llm/out/patchw_lr1e-05_seed{0,1,2}.json (arbitrary), patchw_eq_P176_lr1e-05_seed{0,1,2}.json
(real), inject_eq_P106_lr1e-05_seed{1,2}.json (occupation). Style of Figures 1-4 (397 pt).

    uv run --frozen python -m llm.plot_fig5_v2
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAVY, CRIM, TEAL, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "0.55"
FIG_WIDTH = 397 / 72
XMAX = 400


def set_default_style():
    matplotlib.rcParams.update({
        "lines.linewidth": 1, "lines.markersize": 3, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "legend.fontsize": 5, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 250, "mathtext.fontset": "dejavuserif",
        "xtick.major.size": 2, "xtick.major.width": 0.5, "ytick.major.size": 2,
        "ytick.major.width": 0.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def load(pattern, keys):
    runs = [json.load(open(f))["curve"] for f in sorted(glob.glob(pattern))]
    st = np.array([r["step"] for r in runs[0]], float)
    for rr in runs:
        assert [r["step"] for r in rr] == list(st), pattern
    return st, {k: np.array([[r[k] for r in rr] for rr in runs]) for k in keys}


def band(ax, st, Y, col, ls="-", lab=None, lw=1.0):
    keep = st <= XMAX
    mu, sd = Y[:, keep].mean(0), Y[:, keep].std(0, ddof=1) if len(Y) > 1 else np.zeros(keep.sum())
    ax.fill_between(st[keep], mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
    ax.plot(st[keep], mu, color=col, ls=ls, lw=lw, label=lab, zorder=3)


def main():
    set_default_style()
    K = ["actual/A_noncopy", "actual/B", "patch_r1/A_noncopy", "patch_r1/B", "control_r1/A_noncopy"]
    # synthetic individuals: the small finetuning set's removal runs when present, else the large set's
    small_patch = sorted(glob.glob("llm/out/patchw_small_lr1e-05_seed*.json"))
    sa, arb = load("llm/out/patchw_small_lr1e-05_seed*.json" if small_patch else "llm/out/patchw_lr1e-05_seed*.json", K)
    ss, sm = load("llm/out/inject_small_lr1e-05_seed*.json", ["A/noncopy/acc", "B/ALL/acc"])
    sr, real = load("llm/out/patchw_eq_P176_lr1e-05_seed*.json", K)
    so, occ = load("llm/out/inject_eq_P106_lr1e-05_seed*.json", ["A/noncopy/acc", "B/ALL/acc"])

    fig = plt.figure(figsize=(FIG_WIDTH, 115.2 / 72))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharey=a1); a3 = fig.add_subplot(gs[2], sharey=a1)
    plt.setp(a2.get_yticklabels(), visible=False)
    title = dict(fontsize=6, pad=4)
    leg = dict(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8,
               handlelength=1.4, borderaxespad=0)
    # (a) the collapse at scale: old facts (solid) and new facts (dotted)
    band(a1, ss, sm["A/noncopy/acc"], NAVY, lab="synthetic individuals")
    band(a1, sr, real["actual/A_noncopy"], CRIM, lab="real entities")
    band(a1, ss, sm["B/ALL/acc"], NAVY, ":", lw=0.8)
    band(a1, sr, real["actual/B"], CRIM, ":", lw=0.8)
    h, l = a1.get_legend_handles_labels()
    h += [Line2D([], [], color="0.3", lw=1), Line2D([], [], color="0.3", lw=0.8, ls=":")]
    l += ["old facts", "new facts"]
    a1.set_ylabel("accuracy")

    # (b) old facts, (c) new facts: trained (light), top direction removed (solid), random (grey dotted)
    def light(c, f=0.45):
        import matplotlib.colors as mc
        r, g, b_ = mc.to_rgb(c); return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b_))
    for st, d, col in ((sa, arb, NAVY), (sr, real, CRIM)):
        band(a2, st, d["actual/A_noncopy"], light(col), lw=1.0)
        band(a2, st, d["patch_r1/A_noncopy"], col, lw=1.1)

    # (c) fewer new facts: the new facts are learned sooner and the old facts recover more
    BLUE = "#9DB2D6"
    sf, full = load("llm/out/inject_lr1e-05_seed*.json", ["A/noncopy/acc", "B/ALL/acc"])
    band(a3, sf, full["A/noncopy/acc"], BLUE)
    band(a3, ss, sm["A/noncopy/acc"], NAVY)
    band(a3, sf, full["B/ALL/acc"], BLUE, ":", lw=0.8)
    band(a3, ss, sm["B/ALL/acc"], NAVY, ":", lw=0.8)
    plt.setp(a3.get_yticklabels(), visible=False)


    small = dict(frameon=False, handlelength=1.6, borderaxespad=0.3, labelspacing=0.35)
    a1.legend([Line2D([], [], color="0.35", lw=1), Line2D([], [], color="0.35", lw=0.8, ls=":")],
              ["old facts", "new facts"], loc="lower right", **small)
    a2.legend([Line2D([], [], color="0.75", lw=1), Line2D([], [], color="0.35", lw=1.1)],
              ["trained", "top direction removed"],
              **(dict(loc="lower right") if small_patch else dict(loc="center right", bbox_to_anchor=(1.0, 0.42))),
              **small)
    lab = dict(fontsize=5, va="center")
    a1.text(215, sm["A/noncopy/acc"].mean(0)[list(ss).index(250)] - 0.13, "synthetic individuals", color=NAVY, **lab)
    a1.text(245, real["actual/A_noncopy"].mean(0)[list(sr).index(250)] + 0.07, "real entities", color=CRIM, **lab)
    a3.text(215, sm["A/noncopy/acc"].mean(0)[list(ss).index(200)] + 0.07, "small finetuning set", color=NAVY, **lab)
    a3.text(215, full["A/noncopy/acc"].mean(0)[list(sf).index(200)] - 0.17, "large finetuning set", color=BLUE, **lab)
    for ax in (a1, a2, a3):
        ax.set_xlim(0, XMAX); ax.set_xlabel("finetuning step"); ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_ylim(-0.03, 1.03)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=0.97)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/fig5_new.{e}")
    base = real["actual/A_noncopy"][:, 0].mean()
    i = int(real["actual/A_noncopy"].mean(0)[sr <= XMAX].argmin())
    print(f"real facts: base {base:.2f}; step {sr[i]:.0f} trained {real['actual/A_noncopy'].mean(0)[i]:.2f} "
          f"removed {real['patch_r1/A_noncopy'].mean(0)[i]:.2f} random {real['control_r1/A_noncopy'].mean(0)[i]:.2f}")
    print('middle panel synthetic removal from', 'SMALL set' if small_patch else 'LARGE set (small-set runs not in yet)')
    j = int(arb["actual/A_noncopy"].mean(0)[sa <= XMAX].argmin())
    print(f"arbitrary: trough step {sa[j]:.0f} trained {arb['actual/A_noncopy'].mean(0)[j]:.2f} "
          f"removed {arb['patch_r1/A_noncopy'].mean(0)[j]:.2f}; occupation min "
          f"{occ['A/noncopy/acc'].mean(0)[so <= XMAX].min():.2f}")
    print("wrote plots/fig5_new")


if __name__ == "__main__":
    main()
