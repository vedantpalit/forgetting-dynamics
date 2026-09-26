"""Appendix figure: the old facts keep their ranking within their own half through the collapse.

    plots/app_rank.{png,pdf}

Old facts A of the Figure 1 transformer (attention-only, finetuned on B'): accuracy over the full
vocabulary, and the fraction of old facts whose correct value still ranks first among the values of
its own half. Data: shift_decomposition/mlp_free-fig-p16000-disjoint-t1200-seed*.npz (three seeds;
acc_trained and rank_trained, the own-half rank, 1 = first).

Run: uv run --frozen python -m scripts.plot_app_rank
"""
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_fig4 import FIG_WIDTH, NAVY, TEAL, set_default_style

XMAX = 500


def main():
    set_default_style()
    fs = sorted(glob.glob("shift_decomposition/mlp_free-fig-p16000-disjoint-t1200-seed*.npz"))
    ds = [np.load(f) for f in fs]
    st = ds[0]["steps"].astype(float)
    acc = np.stack([d["acc_trained"].astype(float).mean((1, 2)) for d in ds])
    first = np.stack([(d["rank_trained"] == d["rank_trained"][0].min()).mean((1, 2)) for d in ds])
    keep = st <= st[st >= XMAX].min()
    st, acc, first = st[keep], acc[:, keep], first[:, keep]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.5, 115.2 / 72))
    for Y, col, lab in ((first, TEAL, "correct answer first within its half"), (acc, NAVY, "accuracy")):
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
        ax.plot(st, mu, color=col, label=lab, zorder=3)
    i = int(acc.mean(0).argmin())
    ax.axvline(st[i], color="0.55", ls=":", lw=0.8)
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("finetuning step"); ax.set_ylabel("old facts")
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.97)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/app_rank.{e}")
    print(f"{len(fs)} seeds; trough step {st[i]:.0f}: accuracy {acc.mean(0)[i]:.3f}, first within half "
          f"{first.mean(0)[i]:.3f} (seeds {np.round(first[:, i], 3)}); at {st[-1]:.0f}: {acc.mean(0)[-1]:.3f} / "
          f"{first.mean(0)[-1]:.3f}")


if __name__ == "__main__":
    main()
