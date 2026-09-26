"""A's trough tracks B's acquisition, not the step count -- the small model's version.

Companion to `llm/plot_timing.py`, same two-panel structure and the same plasma ramp, so the
two dose results can be read as one claim in two models.

THE CLAIM, AND HOW IT DIFFERS FROM THE LLM'S. A's trough step moves 30 -> 100 across five dose
conditions (the SHIFT) while B's accuracy at that step stays inside 0.38-0.45 (the INVARIANCE).
That is the same structure as the OLMo result, but NOT the same level: there the trough falls
where B is still at chance, here it falls just before B crosses 0.5. So the shared claim is that
the turnover is located by B's acquisition; the level at which it happens is model-specific and
should not be carried across.

The right panel therefore marks the measured band rather than a chance line -- a chance line
here sits at 0.014, thirty times below where the troughs actually land, and drawing it would
suggest a parallel the numbers do not support.

Conditions are the four dose arms plus the unfrozen baseline, three seeds each (five for the
baseline). Colour is the plasma ramp ordered by speed, dark for the fastest arm, matching
intermediate_plots/14_dose_accuracy_tracking and the LLM figure.

Parsed from the injection logs, which print an eval line every 10 steps -- no checkpoint pass
and no new runs. The logs carry the GLOBAL step counter, so injection step 0 appears as
[step 16000].

Run: uv run python -m scripts.plot_dose_timing
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots"
PRETRAIN_STEP = 16000
LINE = re.compile(r"\[step (\d+)\].*?dataA: first_acc=([\d.]+).*?dataB: first_acc=([\d.]+)")
MARK = "#9A9A9A"
BAND = "#B9B9B9"

# Slowest first, so the legend reads in the same direction as the LLM figure's.
ARMS = [
    ("logs/mlpfree_dose.lr0.5.bs256dense.seed*.out",  "learning rate ×0.5",  "#fb9f3a"),
    ("logs/mlpfree_dose.lr1.0.bs128dense.seed*.out",  "batch 128",     "#e26a58"),
    ("logs_mlpfree/mlpfree_injection.p16000.disjoint.t1200.seed*.out", "baseline", "#bf3984"),
    ("logs/mlpfree_dose.lr1.0.bs512dense.seed*.out",  "batch 512",     "#8207a7"),
    ("logs/mlpfree_dose.lr2.0.bs256dense.seed*.out",  "learning rate ×2",    "#38049a"),
]
XTICKS = [10, 25, 50, 100, 200, 400]

# The same five conditions on the STANDARD 8-layer arm (scale8_dose_injection.py), all from
# the 3000-step dose runs, including the lr 1.0 / batch 256 reference.
ARMS_SCALE8 = [
    ("logs/scale8_dose.lr0.5.bs256.seed*.out", "LR 0.5×",  "#fb9f3a"),
    ("logs/scale8_dose.lr1.0.bs128.seed*.out", "batch 128", "#e26a58"),
    ("logs/scale8_dose.lr1.0.bs256.seed*.out", "LR 1×",    "#bf3984"),
    ("logs/scale8_dose.lr1.0.bs512.seed*.out", "batch 512", "#8207a7"),
    ("logs/scale8_dose.lr2.0.bs256.seed*.out", "LR 2×",    "#38049a"),
]


def load(pattern, xmax):
    runs = []
    for f in sorted(glob.glob(pattern)):
        r = {}
        for line in open(f, encoding="utf-8", errors="replace"):
            m = LINE.search(line)
            if m:
                r[int(m.group(1)) - PRETRAIN_STEP] = (float(m.group(2)), float(m.group(3)))
        if r:
            runs.append(r)
    if not runs:
        raise SystemExit(f"no logs matching {pattern}")
    st = [s for s in sorted(set.intersection(*(set(r) for r in runs))) if 0 <= s <= xmax]
    return (np.array(st, float),
            np.array([[r[s][0] for s in st] for r in runs]),
            np.array([[r[s][1] for s in st] for r in runs]))


def trough(m):
    """The crashed point with the largest subsequent rebound -- not the global minimum, which
    on a long window is the end of the erosion rather than the crash."""
    cand = [k for k in range(len(m)) if m[k] <= max(m[:k + 1]) - 0.05]
    return max(cand, key=lambda k: max(m[k:]) - m[k]) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=int, default=250)
    ap.add_argument("--xmin", type=int, default=0)
    ap.add_argument("--out", default="dose_timing")
    ap.add_argument("--arm", default="mlp_free", choices=["mlp_free", "scale8"],
                    help="which arm's logs to read; scale8 = the standard 8-layer transformer")
    a = ap.parse_args()
    arms = ARMS_SCALE8 if a.arm == "scale8" else ARMS
    if a.arm == "scale8" and a.out == "dose_timing":
        a.out = "dose_timing_scale8"

    from scripts.plot_fig4 import FIG_WIDTH, set_default_style
    set_default_style()                                  # the paper style of Figures 1-5
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 115.2 / 72))
    fig.subplots_adjust(wspace=0.3, left=0.08, right=0.985, bottom=0.22, top=0.97)

    btr, steps_tr = [], []
    print(f"{'condition':>12} {'n':>3} {'A trough':>9} {'@step':>7} {'B there':>9}")
    for pat, lab, col in arms:
        st, A, B = load(pat, a.xmax)
        keep = st >= a.xmin
        m, sd = A.mean(0), A.std(0, ddof=1)
        bm, bsd = B.mean(0), B.std(0, ddof=1)
        i = trough(list(m))

        axA.fill_between(st[keep], (m - sd)[keep], (m + sd)[keep], color=col, alpha=0.18, lw=0)
        axA.plot(st[keep], m[keep], color=col, lw=1.0, label=lab, zorder=3)
        axB.fill_between(st[keep], (bm - bsd)[keep], (bm + bsd)[keep], color=col,
                         alpha=0.18, lw=0)
        axB.plot(st[keep], bm[keep], color=col, lw=1.0, label=lab, zorder=3)

        # Same step markers on both panels -- the pairing is the argument.
        for ax in (axA, axB):
            ax.axvline(st[i], color=col, ls=":", lw=0.7, alpha=0.75, zorder=1)
        axA.plot([st[i]], [m[i]], marker="o", ms=3, mfc="white", mec=col, mew=0.8, zorder=4)
        axB.plot([st[i]], [bm[i]], marker="o", ms=3, mfc="white", mec=col, mew=0.8, zorder=4)
        btr.append(bm[i]); steps_tr.append(st[i])
        print(f"{lab:>12} {A.shape[0]:>3} {m[i]:>9.3f} {st[i]:>7.0f} {bm[i]:>9.4f}")

    # The measured band, NOT a chance line: chance here is 0.014, thirty times below where the
    # troughs land, so drawing it would imply a parallel to the LLM figure that does not hold.
    axB.axhspan(min(btr), max(btr), color=BAND, alpha=0.35, lw=0, zorder=0)

    # paper conventions: linear steps, old/new facts, the band explained in the caption
    for ax, ylab in ((axA, "old-fact accuracy"), (axB, "new-fact accuracy")):
        ax.set_xlim(a.xmin, a.xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("finetuning step")
        ax.set_ylabel(ylab)
    axA.legend(frameon=False, loc="upper right", handlelength=1.6, labelspacing=0.3)

    print(f"\n  trough steps {[int(s) for s in steps_tr]}  span {max(steps_tr)/min(steps_tr):.1f}x")
    print(f"  B at trough  {min(btr):.4f}-{max(btr):.4f}  span {max(btr)/min(btr):.2f}x")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{a.out}.{ext}")
        fig.savefig(p)                               # exactly 397 pt wide
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
