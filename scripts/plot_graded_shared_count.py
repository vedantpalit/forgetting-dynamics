"""A's raw first-token accuracy by shared_count, 8-layer standard arm (with MLP).

Reads `graded_scale8/*.npz` (from analyze_graded_key_overlap_scale8) and writes
plots/graded_shared_count_first_acc_scale8.{png,pdf}. Style matches
scripts/make_decomposition_plots.py so the figure sits alongside the others.

Plots the mean over seeds, with the seed range shaded -- four group curves whose ordering is
the claim, so the reader needs to see whether the bands separate or overlap. A monotone
ordering that lies inside its own seed noise is not a dose-response.

Run: uv run python -m scripts.plot_graded_shared_count
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.linewidth"] = 0.6

IN_DIR = "graded_scale8"
OUT_DIR = "plots"
GROUP_COLORS = ["#3a6e8c", "#6e8a6b", "#b08a44", "#b0402e"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="first_acc", choices=["first_acc", "rank_own"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--xscale", default="linear", choices=["linear", "symlog"],
                    help="linear reads as a normal curve; symlog spreads the early steps, "
                         "which are where the trough and peak sit on this checkpoint grid")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(IN_DIR, "*.npz")))
    if not files:
        raise SystemExit(f"no npz in {IN_DIR}/ -- run analyze_graded_key_overlap_scale8 first")

    steps, per_seed, hist = None, [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if steps is None:
            steps = d["steps"]
        elif not np.array_equal(steps, d["steps"]):
            raise SystemExit(f"step grids differ: {f}")
        sc = d["shared_count"]
        hist = {c: int((sc == c).sum()) for c in range(4)}
        m = d[a.metric]                                  # (T, N, 6)
        per_seed.append(np.stack([m[:, sc == c].mean(axis=(1, 2)) for c in range(4)]))
    P = np.stack(per_seed)                               # (S, 4, T)
    mean, lo, hi = P.mean(0), P.min(0), P.max(0)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for c in range(4):
        ax.plot(steps, mean[c], color=GROUP_COLORS[c], lw=1.8,
                label=f"{c} shared (n={hist[c]})")
        if P.shape[0] > 1:
            ax.fill_between(steps, lo[c], hi[c], color=GROUP_COLORS[c], alpha=0.15, lw=0)
    if a.xscale == "symlog":
        ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("injection step")
    ax.set_ylabel("A first-token accuracy" if a.metric == "first_acc"
                  else "A rank_own top-1 rate")
    ax.set_xlim(0, steps.max())
    ax.legend(frameon=False, fontsize=9, title="name parts shared with B",
              title_fontsize=9)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = a.out or (f"graded_shared_count_{a.metric}_scale8"
                     + ("" if a.xscale == "linear" else f"_{a.xscale}"))
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p, dpi=200)
        print(f"  wrote {p}")

    print(f"\n  {len(files)} seeds, groups {hist}")
    print(f"  {'step':>6} " + " ".join(f"{'c='+str(c):>8}" for c in range(4))
          + f" {'c0-c3':>8} {'monotone':>9}")
    for i, s in enumerate(steps):
        mono = all(mean[c, i] >= mean[c + 1, i] for c in range(3))
        print(f"  {s:>6} " + " ".join(f"{mean[c, i]:>8.3f}" for c in range(4))
              + f" {mean[0, i]-mean[3, i]:>8.3f} {str(mono):>9}")


if __name__ == "__main__":
    main()
