"""The real-text arm: continued pretraining of OLMo 2 1B on post-cutoff Wikipedia leads of any
topic (no facts). Three standalone figures, same style as the corpus figures:

    plots/olmo_text_A.{png,pdf}        old facts (non-copy) and their own-pool rank
    plots/olmo_text_B.{png,pdf}        token accuracy on the training documents (B's clock)
    plots/olmo_text_loss.{png,pdf}     loss on the training docs, held-out leads, pre-cutoff prose

Run: uv run python llm_real/plot_corpus.py
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.55"
XT = [10, 25, 50, 100, 200, 400, 1000, 3000]


def load(key):
    runs = [json.load(open(f))["curve"] for f in sorted(glob.glob("llm_real/out/real_corpus_r1_lr1e-05_seed*_wiki.json"))]
    st = np.array([r["step"] for r in runs[0]], float)
    return st, np.array([[r.get(key, np.nan) for r in run] for run in runs])


def style(ax):
    ax.set_xscale("symlog", linthresh=10); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0, 3000); ax.set_xlabel("Injection step")


def band(ax, st, Y, col, ls, lab, lw=2.4):
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.18, lw=0)
    ax.plot(st, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
    return m


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
                         "axes.linewidth": 1.0, "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out"})
    os.makedirs("plots", exist_ok=True)
    st, A = load("A/noncopy/acc"); _, R = load("A/noncopy/rank")
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    m = band(ax, st, A, NAVY, "-", "old facts, accuracy")
    ax.set_ylim(0, 1.0); style(ax); ax.set_ylabel("Old-fact accuracy (non-copy)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    r = band(ax2, st, R, GREY, "--", "own-pool rank", lw=1.8); ax2.set_ylim(1.0, 2.0); ax2.set_ylabel("Own-pool rank of the answer", color="0.4")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=10.5, loc="lower left")
    fig.tight_layout(); [fig.savefig(f"plots/olmo_text_A.{e}", bbox_inches="tight") for e in ("png", "pdf")]; plt.close(fig)
    print(f"A: {m[0]:.2f} -> {m[np.argmin(abs(st-100))]:.2f}@100 -> {m[np.argmin(abs(st-405))]:.2f}@405 -> {m[-1]:.2f}@{st[-1]:.0f}; rank {r[0]:.2f} -> {r[-1]:.2f}")
    _, T = load("B/train/tok_acc")
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    m = band(ax, st, T, CRIM, "-", "new documents, token accuracy")
    ax.set_ylim(0, 1.0); style(ax); ax.set_ylabel("Token accuracy on the training documents")
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    fig.tight_layout(); [fig.savefig(f"plots/olmo_text_B.{e}", bbox_inches="tight") for e in ("png", "pdf")]; plt.close(fig)
    print(f"B token accuracy: {m[0]:.2f} -> {m[np.argmin(abs(st-100))]:.2f}@100 -> {m[np.argmin(abs(st-405))]:.2f}@405 -> {m[-1]:.2f}")
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    for key, col, ls, lab in (("B/train/loss", CRIM, "-", "new documents (trained)"), ("B/held/loss", TEAL, "--", "held-out new leads"),
                              ("B/generic/loss", OLIVE, "-.", "pre-cutoff prose")):
        _, L = load(key); m = band(ax, st, L, col, ls, lab); print(f"{lab}: {m[0]:.2f} -> {m[-1]:.2f}")
    style(ax); ax.set_ylabel("Next-token loss (nats)"); ax.set_ylim(0, 5.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout(); [fig.savefig(f"plots/olmo_text_loss.{e}", bbox_inches="tight") for e in ("png", "pdf")]
    print("wrote plots/olmo_text_{A,B,loss}")


if __name__ == "__main__":
    main()
