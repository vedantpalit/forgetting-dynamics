"""Appendix figure: the transformer with MLP blocks (the standard arm, same data, rate and seeds as
the attention-only model of Figures 1 and 4). Old-fact accuracy, with the logit change of the old
facts split into its mean over the facts (c_t) and the rest: common part only (z_a(0) + c_t) and common part
subtracted (z_a(t) - c_t), as in analyze_shift_decomposition.py and Figure 4 (right).

    plots/app_mlp.{png,pdf}        data: shift_decomposition/standard-p16000-disjoint-t1200-seed*.npz

Run: uv run --frozen python -m scripts.plot_app_mlp
"""
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_fig4 import FIG_WIDTH, NAVY, TEAL, OLIVE, set_default_style

XMAX = 1200


def main():
    set_default_style()
    fs = (sorted(glob.glob("shift_decomposition/standard-fig-p16000-disjoint-t1200-seed*.npz"))
          or sorted(glob.glob("shift_decomposition/standard-p16000-disjoint-t1200-seed*.npz")))
    ds = [np.load(f) for f in fs]
    st = ds[0]["steps"].astype(float)
    get = lambda k: np.stack([d[k].astype(np.float64).mean((1, 2)) for d in ds])
    T, C, V = get("acc_trained"), get("acc_const"), get("acc_vary")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.5, 115.2 / 72))
    for Y, col, ls, lab in ((T, NAVY, "-", "trained"), (V, OLIVE, "-", "common shift removed")):
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
        ax.plot(st, mu, color=col, ls=ls, label=lab)
    tr = st[int(T.mean(0).argmin())]
    ax.axvline(tr, color="0.55", ls=":", lw=0.8)
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("finetuning step"); ax.set_ylabel("old-fact accuracy")
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.97)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/app_mlp.{e}")
    i = int(T.mean(0).argmin())
    print(f"{len(fs)} seeds; trough step {st[i]:.0f}: trained {T.mean(0)[i]:.3f}, shift only {C.mean(0)[i]:.3f}, "
          f"removed {V.mean(0)[i]:.3f}; peak after {T.mean(0)[i:].max():.3f}; end {T.mean(0)[-1]:.3f}")


if __name__ == "__main__":
    main()
