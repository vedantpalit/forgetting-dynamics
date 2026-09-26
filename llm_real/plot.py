"""Curves for the real-entity runs, in the style of llm/plot_seeds.py: non-copy A (navy) and
B (crimson), mean and one standard deviation over seeds, one panel per (arm, ratio, lr)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM = "#1B2A4E", "#C4245F"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 10, "axes.grid": False})


def load(paths):
    groups = defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        c = d["cfg"]
        key = (c["augment"], c["b_ratio"], c["lr"])
        st = np.array([r["step"] for r in d["curve"]], float)
        A = np.array([r["A/noncopy/acc"] for r in d["curve"]])
        B = np.array([r["B/ALL/acc"] for r in d["curve"]])
        groups[key].append((st, A, B))
    return groups


def main(runs, out):
    paths = runs or sorted(glob.glob("llm_real/out/real_*.json"))
    groups = load(paths)
    n = max(len(groups), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.4), dpi=200, squeeze=False)
    for ax, (key, runs_) in zip(axes[0], sorted(groups.items())):
        st = runs_[0][0]
        for arr, col, lab in ((np.array([r[1] for r in runs_]), NAVY, "A (non-copy)"),
                              (np.array([r[2] for r in runs_]), CRIM, "B")):
            m, sd = arr.mean(0), arr.std(0)
            ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.18, lw=0)
            ax.plot(st, m, color=col, lw=1.9, label=lab)
        aug, ratio, lr = key
        ax.set_title(f"{'augmented' if aug else 'strict'}, B {ratio:g}, lr {lr:g}  (n={len(runs_)})")
        ax.set_xscale("symlog", linthresh=10); ax.set_ylim(0, 1.03)
        ax.set_xlabel("injection step"); ax.set_ylabel("first-token accuracy")
        ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
