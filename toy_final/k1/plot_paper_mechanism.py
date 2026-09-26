"""Paper figures: the mechanism, four separate panels meant for one row.

    plots/toy_k1_mechanism_a.{png,pdf}   A and B accuracy
    plots/toy_k1_mechanism_b.{png,pdf}   the collective write's between-half content <s, w>
    plots/toy_k1_mechanism_c.{png,pdf}   B's mean confidence margin
    plots/toy_k1_mechanism_d.{png,pdf}   growth rate of the write along its own direction,
                                          normalizer's term vs. the total (normalized arm only)

Normalized readout only -- the linear arm never produces the phenomenon (Section
sec:toy), so it is dropped here rather than shown as a null comparison. Data:
track_m{0..9}.json, written by
    verify_math.py --seed S --u_scale 0.01 --match 200 --save track_mS.json
(K=1, d=128, beta=0.5, gain 1, readout rate 0.01, injection rate gate-matched so B reaches
0.99 at step 200, every 5 steps to 200 then a log grid). The earlier track_s{0,1,2}.json
were three seeds at the FULL readout rate and a fixed injection rate; kept for the
verification record, not used here.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
NAVY, TEAL, DEEP, OLIVE, GREY = "#1B2A4E", "#2F8C7D", "#1E5E6B", "#8A8C30", "0.62"
SEEDS = tuple(range(10))
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})


def band(ax, x, Y, color, ls="-", lw=1.9, label=None):
    Y = np.asarray(Y); m = Y.mean(0); sd = Y.std(0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
    return m


def save(fig, name):
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{name}.{e}"), bbox_inches="tight")
    print(f"wrote {name}")


def panel():
    fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=200)
    return fig, ax


def main():
    tr = {s: json.load(open(os.path.join(HERE, f"track_m{s}.json"))) for s in SEEDS}
    st = np.array([x["step"] for x in tr[0]["n1-s0"]["rows"]], float)
    keep = st <= 400
    x = st[keep]

    def grab(key):
        return np.array([[r[key] for r in tr[s]["n1-s" + str(s)]["rows"]] for s in SEEDS])[:, keep]

    mM = grab("margin").mean(0)
    cm = next((x[i] for i in range(1, len(x)) if mM[i - 1] < 0 <= mM[i]), None)

    def mark(ax):
        if cm is not None:
            ax.axvline(cm, color="0.35", lw=1.0, ls="--", alpha=0.85, zorder=0)
        ax.set_xlim(0, 400)
        ax.set_xlabel("Injection step")

    fig, ax = panel()
    band(ax, x, grab("A"), NAVY, label="A")
    band(ax, x, grab("B"), GREY, ls=":", lw=1.7, label="B")
    ax.set_ylim(0, 1.04); ax.set_ylabel("First-token accuracy")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    mark(ax); save(fig, "toy_k1_mechanism_a")

    fig, ax = panel()
    band(ax, x, grab("s_on_w"), TEAL)
    ax.set_ylabel("Collective write,\nbetween-half content")
    ax.set_ylim(0, None)
    mark(ax); save(fig, "toy_k1_mechanism_b")

    fig, ax = panel()
    band(ax, x, grab("margin"), DEEP)
    ax.axhline(0, color="0.45", lw=0.9)
    ax.set_ylabel("B's mean margin $\\bar M$")
    mark(ax); save(fig, "toy_k1_mechanism_c")

    fig, ax = panel()
    band(ax, x, grab("proj") * 1e3, TEAL, label="total")
    band(ax, x, grab("proj_anti") * 1e3, OLIVE, lw=1.5, label="normalizer's term")
    ax.axhline(0, color="0.45", lw=0.9)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Growth rate of the write\n$\\langle \\dot s, \\hat s\\rangle \\times 10^{3}$")
    ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=1.8)
    mark(ax); save(fig, "toy_k1_mechanism_d")

    print(f"margin crosses zero at step {cm}")


if __name__ == "__main__":
    main()
