"""Two figures: the phenomenon, and why it happens.

FIG 1  the accuracy curves. Same store, same data, same optimizer, same learning rate. The
       only difference is whether the readout is scale-invariant. One arm crashes and stays
       dead; the other crashes and comes back.

FIG 2  the mechanism, as four panels on ONE shared time axis, so that every "at the same
       moment" claim is something the reader checks by eye rather than takes on trust.
       (a) A crashes as B is learned, then recovers.
       (b) what reaches A is a single shared write, which rises and then falls.
       (c) B's confidence margin crosses zero.
       (d) at that same step the write's growth rate crosses zero -- but only when the
           readout is normalized. The linear arm's growth rate never goes negative.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE, GREY = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30", "#9AA0A6"
SEEDS = (0, 1, 2)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 10, "axes.grid": False})


def band(ax, x, Y, color, ls="-", lw=2.0, label=None, alpha=0.16):
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=alpha, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
    return m


def fig1():
    d = json.load(open("beta_sweep.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=200, sharey=True)
    panels = ((axes[0], 0, "linear store, plain readout\n(no recovery, ever)"),
              (axes[1], 1, "same store, scale-invariant readout\n(crash, then recovery)"))
    for ax, norm, title in panels:
        runs = [d[f"b0.5-n{norm}-g1.0-u0.01-s{s}"]["rows"] for s in SEEDS]
        st = np.array([x["step"] for x in runs[0]], float)
        A = np.array([[x["A"] for x in r] for r in runs])
        B = np.array([[x["B"] for x in r] for r in runs])
        band(ax, st, A, NAVY, label="A  (old population)")
        band(ax, st, B, CRIM, lw=1.6, label="B  (newly injected)")
        m = A.mean(0)
        tr = int(m.argmin())
        ax.plot(st[tr], m[tr], "o", ms=6, mfc="white", mec=NAVY, mew=1.6, zorder=5)
        ax.annotate(f"trough {m[tr]:.2f}", (st[tr], m[tr]), textcoords="offset points",
                    xytext=(6, -14), fontsize=8.5, color=NAVY)
        if norm == 1:
            ax.annotate(f"recovers to {m[tr:].max():.2f}", (st[-1], m[-1]),
                        textcoords="offset points", xytext=(-74, 10), fontsize=8.5, color=NAVY)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_ylim(0, 1.04)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("injection step")
    axes[0].set_ylabel("first-token accuracy")
    axes[0].legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_phenomenon.{e}", bbox_inches="tight")
    print("wrote plots/toy_k1_phenomenon.png")


def fig2():
    tr = {s: json.load(open(f"track_s{s}.json")) for s in SEEDS}
    st = np.array([x["step"] for x in tr[0]["n1-s0"]["rows"]], float)
    keep = st <= 400
    x = st[keep]

    def grab(norm, key):
        return np.array([[r[key] for r in tr[s][f"n{norm}-s{s}"]["rows"]]
                         for s in SEEDS])[:, keep]

    mM = grab(1, "margin").mean(0)
    mP = grab(1, "proj").mean(0) * 1e3
    cm = next((x[i] for i in range(1, len(x)) if mM[i - 1] < 0 <= mM[i]), None)
    cp = next((x[i] for i in range(1, len(x)) if mP[i - 1] > 0 >= mP[i]), None)

    fig, axes = plt.subplots(4, 1, figsize=(7.6, 10.8), dpi=200, sharex=True)

    ax = axes[0]
    band(ax, x, grab(1, "A"), NAVY, label="A  (old population)")
    band(ax, x, grab(1, "B"), CRIM, lw=1.6, label="B  (newly injected)")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title("(a)  A crashes as B is learned, then comes back", fontsize=10.5, loc="left")

    ax = axes[1]
    band(ax, x, grab(1, "s_on_w"), PURP)
    ax.set_ylabel("collective write on A\n(its X-to-Y content)")
    ax.set_title("(b)  what reaches A is ONE shared write. it rises, then falls",
                 fontsize=10.5, loc="left")

    ax = axes[2]
    band(ax, x, grab(1, "margin"), CRIM)
    ax.axhline(0, color="0.45", lw=1.0)
    ax.set_ylabel("B's confidence\nmargin  $M_b$")
    ax.annotate("B is wrong", (0.03, 0.28), xycoords="axes fraction", fontsize=9, color="0.3")
    ax.annotate("B is confident", (0.30, 0.80), xycoords="axes fraction", fontsize=9,
                color="0.3")
    ax.set_title(f"(c)  B stops being wrong at step {int(cm) if cm else '?'}",
                 fontsize=10.5, loc="left")

    ax = axes[3]
    band(ax, x, grab(1, "proj") * 1e3, PURP, label="normalized readout: total")
    band(ax, x, grab(1, "proj_anti") * 1e3, OLIVE, lw=1.4,
         label="    of which, the term the normalizer adds")
    band(ax, x, grab(0, "proj") * 1e3, NAVY, ls="--", lw=1.6,
         label="linear readout: total (never negative)")
    ax.axhline(0, color="0.45", lw=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("growth rate of the write\n"
                  r"$\langle \dot{s},\ \hat{s} \rangle \times 10^{3}$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title(f"(d)  and at step {int(cp) if cp else '?'} the write turns around",
                 fontsize=10.5, loc="left")
    ax.set_xlabel("injection step")

    for ax in axes:
        if cm is not None:
            ax.axvline(cm, color="0.35", lw=1.2, ls="--", alpha=0.85, zorder=0)
        ax.set_xlim(0, 400)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_mechanism.{e}", bbox_inches="tight")
    print("wrote plots/toy_k1_mechanism.png")


if __name__ == "__main__":
    os.makedirs("../../plots", exist_ok=True)
    fig1()
    fig2()
