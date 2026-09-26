"""A's accuracy with and without the feedforward layers (paper Section 5, move one).

The section performs an exact per-block decomposition, which is only exact because the MLP-free
block is `x + Attn(LN(x))` -- the residual is then the embedding plus the eight attention
outputs and nothing else. That licence is worth nothing if removing the MLPs also removes the
phenomenon, so this shows it does not.

BOTH ARMS CARRY THEIR OWN PHASE MARKERS, in their own colour. The trough and recovery end differ
between them -- roughly 30/110 against 50/200 -- so a single pair of rules would be correct for
one arm and wrong for the other, and a reader would take the wrong pair as applying to both.
The recovery end is the first step reaching 99% of the post-trough maximum, not the argmax,
which on a plateau chases seed noise.

NOT A CLEAN REPLICATION IN MAGNITUDE, AND THE FIGURE SHOULD NOT PRETEND OTHERWISE. The MLP-free
arm crashes to 0.119 against 0.527 and recovers to 0.539 against 0.926, with both phases
arriving later. What replicates is the three-phase shape and its ordering, in a model with 38.1%
of the parameters (10,329,945 against 27,135,833).

Log x: the trough and peak of both arms sit inside the first 200 steps of a 1200-step window, so
a linear axis puts every feature of interest in the left sixth. --linear overrides.

Parsed from the injection logs, which print an eval line every 10 steps; no checkpoint pass and
no new runs. The logs carry the GLOBAL step counter, so injection step 0 appears as [step 16000];
the offset is subtracted explicitly.

Run: uv run python -m scripts.plot_arm_comparison
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
LINE = re.compile(r"\[step (\d+)\].*?dataA: first_acc=([\d.]+)")

ARMS = {
    "standard": dict(
        pattern="logs_scale8/scale8_injection.p16000.disjoint.seed*.out",
        color="#1B2A4E", label="with MLPs"),
    "mlp_free": dict(
        pattern="logs_mlpfree/mlpfree_injection.p16000.disjoint.t1200.seed*.out",
        color="#1F7A7A", label="MLP-free"),
}
XTICKS = [10, 25, 50, 100, 200, 400, 1200]


def load(pattern, xmax):
    runs = []
    for f in sorted(glob.glob(pattern)):
        r = {}
        for line in open(f, encoding="utf-8", errors="replace"):
            m = LINE.search(line)
            if m:
                r[int(m.group(1)) - PRETRAIN_STEP] = float(m.group(2))
        if r:
            runs.append(r)
    if not runs:
        raise SystemExit(f"no logs matching {pattern}")
    steps = [s for s in sorted(set.intersection(*(set(r) for r in runs))) if 0 <= s <= xmax]
    return np.array(steps, float), np.array([[r[s] for s in steps] for r in runs])


def phases(m, frac=0.99):
    """(trough index, recovery-end index). Candidates must be a real crash, 0.05 below the
    running maximum; the recovery end is the first step within `frac` of the post-trough max,
    which does not move with a seed the way the argmax does on a plateau."""
    cand = [k for k in range(len(m)) if m[k] <= max(m[:k + 1]) - 0.05]
    if not cand:
        return None, None
    i = max(cand, key=lambda k: max(m[k:]) - m[k])
    post = m[i:].max()
    return i, next(k for k in range(i, len(m)) if m[k] >= frac * post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=int, default=1200)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out", default="arm_comparison")
    a = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)

    ytop = 1.14
    for k, (tag, spec) in enumerate(ARMS.items()):
        st, A = load(spec["pattern"], a.xmax)
        ns = A.shape[0]
        m = A.mean(0)
        se = A.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(m)
        c = spec["color"]
        ax.fill_between(st, m - se, m + se, color=c, alpha=0.18, lw=0)
        ax.plot(st, m, color=c, lw=2.5, label=spec["label"], zorder=3)

        tr, pk = phases(m)
        for x in (st[tr], st[pk]):
            ax.axvline(x, color=c, ls=":", lw=1.2, alpha=0.55, zorder=1)
        # Each arm's rules are labelled in its own colour, on its own row, so the pairing is
        # unambiguous -- the two arms' troughs and peaks are at different steps.
        y = ytop - 0.035 - 0.055 * k
        ax.text(st[tr], y, f"{st[tr]:.0f}", color=c, fontsize=9.5, ha="center", va="top")
        ax.text(st[pk], y, f"{st[pk]:.0f}", color=c, fontsize=9.5, ha="center", va="top")
        print(f"  {tag:>9} ({ns} seeds): trough {m[tr]:.4f}+/-{se[tr]:.4f} @{st[tr]:.0f}   "
              f"peak {m[pk]:.4f}+/-{se[pk]:.4f} @{st[pk]:.0f}   "
              f"end {m[-1]:.4f}+/-{se[-1]:.4f} @{st[-1]:.0f}   "
              f"recovery {(m[pk]-m[tr])/(m[0]-m[tr]):.3f}")

    if not a.linear:
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(10, a.xmax)
    ax.set_ylim(0, ytop)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Injection step")
    ax.set_ylabel("A first-token accuracy")
    ax.legend(frameon=False, fontsize=11.5, loc="upper right", handlelength=1.8)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{a.out}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
