"""The recovery is the blocks ceasing to agree, not any block backing off (paper Section 5).

THE FINDING. Eight blocks each add a contribution P_j to the shift. Two things about those
contributions change independently: how large each one is, which grows across the window, and
how much they agree in direction, which falls across it. Contributions pointing the same way add
up; contributions pointing in scattered directions largely cancel. So their sum peaks and turns
over even though neither input does.

THE FIGURE MAKES THAT ONE CLAIM. Two inputs on top, their sum below, and the hump appears only
below. Everything not needed for that claim was removed:

  * ||delta|| and the super-additive shading are gone. The identity
        ||sum_j P_j||^2 = sum_i ||P_i||^2 + 2 sum_{i<j} ||P_i|| ||P_j|| cos_ij
    holds over ||sum_j P_j||, and plotting a fourth curve the two inputs do NOT reconstruct is
    what made the earlier version hard to read. That delta runs about twice ||sum_j P_j||,
    because the cumulative patch is super-additive, is a sentence in the text.
  * The Gram heatmaps are now their own appendix figure (--panel gram). They are a good picture
    of de-coherence but a second finding, and the depth-banding is not load-bearing here.

SUM_j ||P_j|| IS NOT MONOTONE and is neither drawn nor described as if it were: it dips at step
100, ~8 SD below step 50, in all three seeds individually. The claim the panel supports is only
that neither input has a peak where the output does -- which is true, and enough.

LOG X. The checkpoints are log-spaced by design (10, 25, 50, 100, 200, 400); on a linear axis
four of six fall in the left eighth. The rules at 30 and 110 mark A's accuracy trough and
recovery end so the figure can be aligned against the accuracy figure. --linear overrides.

Data: dose_coherence/mlp_free-baseline-* -- the lr 1.0 / batch 256 condition IS the standard
MLP-free injection (||delta|| = 18.892 at step 50 matches weight_patch exactly), evaluated on the
same 512-individual subset as every other mechanistic analysis. Step 0 is excluded: patching
step-0 weights into the step-0 checkpoint is a no-op, every P_j is identically zero and the
cosine is undefined.

Run:
  uv run python -m scripts.plot_crossover                # main figure
  uv run python -m scripts.plot_crossover --panel gram   # appendix figure
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

IN_DIR = "dose_coherence"
OUT_DIR = "plots"

C_OUT = "#1B2A4E"       # ||sum_j P_j||, the output
C_MAG = "#7A8CA3"       # sum_j ||P_j||, input
C_COS = "#2F7A5A"       # mean pairwise cosine, input
C_NEG = "#C4245F"       # negative end of the Gram colormap
MARK = "#9A9A9A"

TROUGH_STEP, RECOVERY_END = 30, 110
XTICKS = [10, 25, 50, 100, 200, 400]


def load(pattern):
    files = sorted(glob.glob(os.path.join(IN_DIR, pattern)))
    if not files:
        raise SystemExit(f"no files matching {pattern} in {IN_DIR}/")
    steps, P, base = None, [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if steps is None:
            steps = d["steps"]
        elif not np.array_equal(steps, d["steps"]):
            raise SystemExit(f"{f} has a different step grid")
        P.append(d["P"].astype(np.float64))              # (T, 8, A, E)
        base.append(d["base_norm"].astype(np.float64))
    return np.array(steps, float), np.stack(P), np.stack(base)


def quantities(P, base):
    N = np.linalg.norm(P, axis=-1)                               # (S, T, 8, A)
    U = P / np.clip(N[..., None], 1e-12, None)
    G = np.einsum("stiae,stjae->staij", U, U)
    iu = np.triu_indices(P.shape[2], 1)
    return dict(
        sum_norms=N.sum(2).mean(-1),                             # sum_j ||P_j||
        norm_sum=np.linalg.norm(P.sum(2), axis=-1).mean(-1),     # ||sum_j P_j||
        cos=G[..., iu[0], iu[1]].mean(-1).mean(-1),
        delta=base.mean(-1),
    )


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })


def main_figure(steps, q, sem, linear, out):
    fig, (ax_in, ax_out) = plt.subplots(
        2, 1, figsize=(7.0, 6.6), dpi=200, sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.13))
    axr = ax_in.twinx()

    m, se = q["sum_norms"].mean(0), sem(q["sum_norms"])
    ax_in.fill_between(steps, m - se, m + se, color=C_MAG, alpha=0.22, lw=0)
    ax_in.plot(steps, m, color=C_MAG, ls="--", lw=1.8, marker="o", ms=5.5,
               mfc="white", mew=1.6, zorder=3)

    c, cse = q["cos"].mean(0), sem(q["cos"])
    axr.fill_between(steps, c - cse, c + cse, color=C_COS, alpha=0.22, lw=0)
    axr.plot(steps, c, color=C_COS, ls="--", lw=1.8, marker="s", ms=5.5,
             mfc="white", mew=1.6, zorder=3)

    s, sse = q["norm_sum"].mean(0), sem(q["norm_sum"])
    ax_out.fill_between(steps, s - sse, s + sse, color=C_OUT, alpha=0.20, lw=0)
    ax_out.plot(steps, s, color=C_OUT, ls="-", lw=3.0, zorder=3)
    i = int(np.argmax(s))
    ax_out.plot([steps[i]], [s[i]], marker="v", ms=8, color=C_OUT, zorder=4)

    # Colour-coded axis labels do the work a legend would, in plain language. The symbols are
    # left to the caption; a reader should be able to take the claim off the axes alone.
    ax_in.set_ylabel("how much each block writes\n(summed)", color=C_MAG, fontsize=11.5)
    ax_in.tick_params(axis="y", colors=C_MAG)
    ax_in.spines["left"].set_color(C_MAG)
    axr.set_ylabel("how much the blocks agree", color=C_COS, fontsize=11.5)
    axr.tick_params(axis="y", colors=C_COS)
    axr.spines["right"].set_color(C_COS)
    axr.spines["left"].set_color(C_MAG)
    ax_out.set_ylabel("what they add up to", color=C_OUT, fontsize=11.5)
    ax_out.tick_params(axis="y", colors=C_OUT)

    ax_in.set_ylim(0, max(m + se) * 1.30)
    axr.set_ylim(0, 1.0)
    # Axis starts at zero -- truncating it would exaggerate the hump, which is the claim.
    # Headroom is trimmed instead, so the curve fills the panel without the scale flattering it.
    ax_out.set_ylim(0, max(s + sse) * 1.08)

    for ax in (ax_in, ax_out):
        for x in (TROUGH_STEP, RECOVERY_END):
            ax.axvline(x, color=MARK, ls=":", lw=1.3, zorder=0)
    ax_in.text(TROUGH_STEP, ax_in.get_ylim()[1] * 0.985, "A trough", color=MARK,
               style="italic", fontsize=10, ha="center", va="top")
    ax_in.text(RECOVERY_END, ax_in.get_ylim()[1] * 0.985, "recovery end", color=MARK,
               style="italic", fontsize=10, ha="center", va="top")

    if not linear:
        ax_out.set_xscale("log")
        ax_out.set_xticks(XTICKS)
        ax_out.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_out.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_out.set_xlim(steps.min(), steps.max())
    ax_out.set_xlabel("Injection step")

    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{out}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")


def gram_figure(steps, P, gsteps, out):
    """Appendix: the 8x8 cosine Gram at two checkpoints."""
    ns, nb = P.shape[0], P.shape[2]
    cmap = LinearSegmentedColormap.from_list("gram", [C_NEG, "#FFFFFF", C_OUT])
    cmap.set_bad("#F2F2F2")
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=1.0)
    fig, axes = plt.subplots(1, len(gsteps), figsize=(6.4, 3.1), dpi=200)
    axes = np.atleast_1d(axes)
    for ax, gstep in zip(axes, gsteps):
        i = int(np.where(steps == gstep)[0][0])
        U = P[:, i] / np.clip(np.linalg.norm(P[:, i], axis=-1, keepdims=True), 1e-12, None)
        # Mean over BOTH seeds and attributes. "siae,sjae->aij" would sum over the seed
        # axis instead of averaging it, inflating every entry by the seed count.
        Gm = np.einsum("siae,sjae->saij", U, U).mean(axis=(0, 1))
        # The diagonal is 1 by construction; leaving it in makes it the darkest thing in the
        # panel and buries the off-diagonal band that is the point.
        Gp = np.array(Gm, dtype=float)
        np.fill_diagonal(Gp, np.nan)
        im = ax.imshow(np.ma.masked_invalid(Gp), cmap=cmap, norm=norm)
        ax.set_xticks(range(nb)); ax.set_yticks(range(nb))
        ax.set_xticklabels(range(nb), fontsize=9); ax.set_yticklabels(range(nb), fontsize=9)
        ax.set_title(f"step {gstep}", fontsize=11, pad=6)
        ax.set_xlabel("block", fontsize=10)
        print(f"  Gram @{gstep}: b0.b1 {Gm[0,1]:.3f}  b6.b7 {Gm[nb-2,nb-1]:.3f}  "
              f"b0.b{nb-1} {Gm[0,nb-1]:.3f}")
    axes[0].set_ylabel("block", fontsize=10)
    fig.subplots_adjust(right=0.86, wspace=0.25)
    pos = axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.03, pos.y0, 0.022, pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.25, 0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label(r"$\cos(P_i, P_j)$", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{out}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="mlp_free-baseline-p16000-disjoint-seed*.npz")
    ap.add_argument("--panel", default="main", choices=["main", "gram"])
    ap.add_argument("--linear", action="store_true", help="linear x instead of log")
    ap.add_argument("--gram_steps", default="50,200")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    steps, P, base = load(a.pattern)
    ns = P.shape[0]
    sem = lambda x: (x.std(0, ddof=1) / np.sqrt(ns) if ns > 1
                     else np.zeros_like(x.mean(0)))
    q = quantities(P, base)

    print(f"{ns} seeds, steps {[int(s) for s in steps]}")
    print(f"  {'step':>6} {'sum||P_j||':>16} {'mean cos':>16} {'||sum P_j||':>16}")
    for i, s in enumerate(steps):
        print(f"  {s:>6.0f} {q['sum_norms'][:, i].mean():>9.3f}+/-{sem(q['sum_norms'])[i]:<5.3f} "
              f"{q['cos'][:, i].mean():>9.4f}+/-{sem(q['cos'])[i]:<5.4f} "
              f"{q['norm_sum'][:, i].mean():>9.3f}+/-{sem(q['norm_sum'])[i]:<5.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    if a.panel == "main":
        main_figure(steps, q, sem, a.linear, a.out or "crossover")
    else:
        gram_figure(steps, P, [int(x) for x in a.gram_steps.split(",")],
                    a.out or "crossover_gram")


if __name__ == "__main__":
    main()
