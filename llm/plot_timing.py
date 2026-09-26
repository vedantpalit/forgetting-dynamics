"""A's trough moves with the learning rate; B is at chance at every one of them.

The claim has two halves and the figure has to show both. A's trough step moves 200 -> 125 -> 100
across a 3.3x learning-rate range (the SHIFT), while B's accuracy at that step is 0.0158 / 0.0173
/ 0.0158 against a chance of 1/66 = 0.0152 (the INVARIANCE). Either half alone is unremarkable:
the shift is what a learning rate does, and B being near zero early is expected. Together they
say the turnover is located by B's acquisition rather than by the step count.

llm_timing_A   A's non-copy accuracy, three arms, each arm's trough marked.
llm_timing_B   B's accuracy, same arms, THE SAME step markers, against the chance line.
Two separate figures (not one two-panel figure), so the paper can place them independently.

WHY THE B FIGURE IS LOG-Y. B at a trough is ~0.016 and B's ceiling is ~0.88. On a linear 0-1
axis every trough value sits inside one pixel of zero and "B is at chance" is unreadable -- the
reader would have to take it from the caption, which is the thing the figure exists to avoid.
Log y resolves the floor and still shows the full rise. Values below the axis floor (B is
identically 0 at step 0) fall off the bottom.

A IS THE NON-COPY STRATUM, as everywhere else: 62% of the gated set is answerable by copying the
answer out of the prompt, and including it dilutes the crash.

Three seeds per arm, mean with a +/-1 sd band. Colour encodes the learning rate as an ordered
ramp, since the series is a dose.

Run: uv run python -m llm.plot_timing
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots"
CHANCE = 1.0 / 66.0
# The plasma ramp used by the small model's dose figure, in the same direction: DARK for the
# fast arm, orange for the slow one. Matching it lets a reader carry the reading across --
# the two dose results are the same claim in two models.
ARMS = [("3e-06", "3e-6", "#fb9f3a"),
        ("5e-06", "5e-6", "#bf3984"),
        ("1e-05", "1e-5", "#38049a")]
MARK = "#9A9A9A"
XTICKS = [25, 50, 100, 200, 400, 800, 1600, 3000]


def phases(acc):
    """Trough index: the crashed point with the largest subsequent rebound.

    Not the global minimum -- the curve crashes, recovers, then erodes for thousands of steps,
    so on this window the global minimum is the last step. Candidates must be a real crash
    (0.05 below the running maximum), because every arm rises ~0.05 before it falls.
    """
    cand = [k for k in range(len(acc)) if acc[k] <= max(acc[:k + 1]) - 0.05]
    if not cand:
        return None
    return max(cand, key=lambda k: max(acc[k:]) - acc[k])


def load(tag, stratum):
    files = sorted(glob.glob(f"llm/out/inject_lr{tag}_seed*.json"))
    if not files:
        raise SystemExit(f"no runs matching llm/out/inject_lr{tag}_seed*.json")
    steps, A, B = None, [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if not d.get("tokens_per_step"):
            print(f"  skipping {os.path.basename(f)}: predates fp32 master weights, voided")
            continue
        c = d["curve"]
        s = [x["step"] for x in c]
        if steps is None:
            steps = s
        elif s != steps:
            raise SystemExit(f"{f} has a different step grid; seeds cannot be averaged")
        A.append([x[f"A/{stratum}/acc"] for x in c])
        B.append([x["B/ALL/acc"] for x in c])
    return np.array(steps, float), np.array(A), np.array(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", default="noncopy", choices=["noncopy", "ALL", "copy"])
    ap.add_argument("--xmin", type=int, default=25)
    ap.add_argument("--ymin_b", type=float, default=0.004)
    ap.add_argument("--out", default="llm_timing")
    ap.add_argument("--superimpose", action="store_true",
                    help="one panel: A (solid) and B (dotted) at the three rates, linear axes, "
                         "each rate's trough marked on both -> <out>_AB")
    a = ap.parse_args()
    if a.superimpose:
        return superimpose(a)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    # two separate figures, the same size and style, so each can be placed on its own;
    # the SAME step markers on both is still the argument.
    figA, axA = plt.subplots(figsize=(5.6, 4.6), dpi=200)
    figB, axB = plt.subplots(figsize=(5.6, 4.6), dpi=200)

    btr = []
    print(f"{'lr':>7} {'n':>3} {'trough':>9} {'@step':>7} {'B there':>9} "
          f"{'B/chance':>9}")
    for tag, lab, col in ARMS:
        st, A, B = load(tag, a.stratum)
        keep = st >= a.xmin
        m, sd = A.mean(0), A.std(0, ddof=1)
        bm, bsd = B.mean(0), B.std(0, ddof=1)
        i = phases(list(m))

        axA.fill_between(st[keep], (m - sd)[keep], (m + sd)[keep], color=col, alpha=0.20, lw=0)
        axA.plot(st[keep], m[keep], color=col, lw=2.4, label=f"lr {lab}", zorder=3)
        axB.fill_between(st[keep], np.maximum(bm - bsd, a.ymin_b / 2)[keep],
                         (bm + bsd)[keep], color=col, alpha=0.20, lw=0)
        axB.plot(st[keep], bm[keep], color=col, lw=2.4, label=f"lr {lab}", zorder=3)

        # The SAME step markers on both panels -- that pairing is the whole argument.
        for ax in (axA, axB):
            ax.axvline(st[i], color=col, ls=":", lw=1.4, alpha=0.75, zorder=1)
        axA.plot([st[i]], [m[i]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        axB.plot([st[i]], [bm[i]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        btr.append(bm[i])
        print(f"{lab:>7} {A.shape[0]:>3} {m[i]:>9.3f} {st[i]:>7.0f} {bm[i]:>9.4f} "
              f"{bm[i]/CHANCE:>9.2f}")

    axB.axhline(CHANCE, color=MARK, ls="--", lw=1.4, zorder=1)
    axB.text(3000, CHANCE * 1.14, "chance, 1/66", color=MARK, fontsize=10.5,
             ha="right", va="bottom")
    # One summary annotation rather than three per-marker labels: at these troughs the values
    # sit within 0.001 of each other, so three labels collide and say the same thing anyway.
    # The invariance is the claim, so it is stated once.
    axB.text(33, 0.30, f"B = {min(btr):.3f}–{max(btr):.3f}\nat all three troughs",
             fontsize=10.5, color=MARK, ha="left", va="center")

    for ax, ylab in ((axA, "A first-token accuracy (non-copy)"),
                     (axB, "B first-token accuracy")):
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(a.xmin, 3000)
        ax.set_xlabel("Injection step")
        ax.set_ylabel(ylab)
    axA.set_ylim(0, 1.0)
    axB.set_yscale("log")
    axB.set_ylim(a.ymin_b, 1.0)
    axA.legend(frameon=False, fontsize=11.5, loc="upper right", handlelength=1.8)
    axB.legend(frameon=False, fontsize=11.5, loc="lower right", handlelength=1.8)

    os.makedirs(OUT_DIR, exist_ok=True)
    for fig, suffix in ((figA, "_A"), (figB, "_B")):
        for ext in ("png", "pdf"):
            p = os.path.join(OUT_DIR, f"{a.out}{suffix}.{ext}")
            fig.savefig(p, bbox_inches="tight")
            print(f"  wrote {p}")


def superimpose(a):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    for tag, lab, col in ARMS:
        st, A, B = load(tag, a.stratum)
        keep = st <= 1000
        m, sd = A.mean(0), A.std(0, ddof=1); bm, bsd = B.mean(0), B.std(0, ddof=1)
        i = phases(list(m))
        ax.fill_between(st[keep], (m - sd)[keep], (m + sd)[keep], color=col, alpha=0.18, lw=0)
        ax.plot(st[keep], m[keep], color=col, lw=2.4, label=f"old facts, lr {lab}", zorder=3)
        ax.plot(st[keep], bm[keep], color=col, lw=2.0, ls=":", label=f"new facts, lr {lab}", zorder=3)
        ax.axvline(st[i], color=col, ls=":", lw=1.2, alpha=0.7, zorder=1)
        ax.plot([st[i]], [m[i]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        ax.plot([st[i]], [bm[i]], marker="o", ms=7, mfc="white", mec=col, mew=2, zorder=4)
        print(f"{lab}: trough {m[i]:.3f} @ {st[i]:.0f}, B there {bm[i]:.3f}")
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Injection step"); ax.set_ylabel("First-token accuracy")
    ax.legend(frameon=False, fontsize=9.5, loc="center right", ncol=1, handlelength=1.8)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{a.out}_AB.{ext}")
        fig.savefig(p, bbox_inches="tight"); print(f"  wrote {p}")


if __name__ == "__main__":
    main()
