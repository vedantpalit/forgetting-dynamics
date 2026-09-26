"""Task 5.  Three figures for the two-channel sweep.

(a) readout_h1_scatter   recovery fraction vs the store's share of B's acquisition
(b) readout_carrier      A's accuracy and the two crash-carrier counterfactuals, for the
                         cell where the store carries the crash and the cell where the
                         readout does
(c) readout_bias_time    the readout bias and the store's shared write over time,
                         n_D = 0 vs n_D = 50, at u_scale = 1
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30"
OUT = "plots/toy_plots_working"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "figure.dpi": 200,
    "savefig.dpi": 200, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
})
MARK = {0.0: "o", 0.01: "s", 0.1: "^", 0.3: "D", 1.0: "v"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT}/{name}.png/.pdf")


def load():
    raw = json.load(open("wp_readout_sweep.json"))
    cells = {}
    for r in raw.values():
        cells.setdefault((r["nd"], r["u_scale"], r["w_scale"]), []).append(r)
    return raw, cells


def mean_curve(runs, key):
    st = np.array([x["step"] for x in runs[0]["rows"]])
    y = np.mean([[x[key] for x in r["rows"]] for r in runs], axis=0)
    return st, y


def fig_a(cells):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for (nd, us, ws), R in sorted(cells.items()):
        s = [r["summary"] for r in R]
        depth = np.mean([x["depth"] for x in s])
        if depth <= 0.05 or np.mean([x["B_end"] for x in s]) < 0.99:
            continue
        x = np.mean([x["store_share"] for x in s])
        y = np.mean([x["rec_frac"] for x in s])
        c = NAVY if nd else CRIM
        ax.scatter(x, y, marker=MARK[us], s=26 + 90 * ws, facecolor=c, edgecolor="white",
                   linewidth=0.5, zorder=3)
    lim = np.array([-0.2, 1.1])
    ax.plot(lim, lim, color="0.55", lw=0.9, ls="--", zorder=1)
    ax.text(0.72, 0.80, "H1: $y=x$", color="0.35", fontsize=8, rotation=38)
    ax.set_xlim(*lim); ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("store's share of B's margin gain at B's gate\n"
                  "$[M(W_t,U_0)-M_0]\\,/\\,[M(W_t,U_t)-M_0]$")
    ax.set_ylabel("recovery fraction   (A regained) / (A lost)")
    ax.set_title("Recovery fraction against the store's share of B\n"
                 "H1 ($y=x$) fails; marker SIZE, the store's rate, does the work",
                 fontsize=9.5)
    h = [plt.Line2D([], [], marker="o", ls="", color=CRIM, label="no ballast ($n_D=0$)"),
         plt.Line2D([], [], marker="o", ls="", color=NAVY, label="ballast ($n_D=50$)")]
    h += [plt.Line2D([], [], marker=MARK[u], ls="", color="0.4",
                     label=f"$u_{{scale}}={u:g}$") for u in sorted(MARK)]
    h += [plt.Line2D([], [], marker="o", ls="", color="0.4",
                     markersize=np.sqrt(26 + 90 * w) / 2.2, label=f"$w_{{scale}}={w:g}$")
          for w in (1.0, 0.1, 0.01)]
    ax.legend(handles=h, fontsize=7, frameon=False, loc="lower right", ncol=2,
              handletextpad=0.4, columnspacing=0.8)
    save(fig, "readout_h1_scatter")


def fig_b(cells):
    """Pick, among crashing cells, the one whose crash the store carries and the one whose
    crash the readout carries, by the counterfactual gap at the trough."""
    cand = []
    for c, R in cells.items():
        s = [r["summary"] for r in R]
        d = np.mean([x["depth"] for x in s])
        if d < 0.25 or np.mean([x["B_end"] for x in s]) < 0.99:
            continue
        cfW = np.mean([x["A_storeonly_at_trough"] for x in s])
        cfU = np.mean([x["A_readonly_at_trough"] for x in s])
        cand.append((c, d, cfW, cfU))
    store_cell = min(cand, key=lambda z: z[2] - z[3])[0]
    read_cell = min(cand, key=lambda z: z[3] - z[2])[0]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, c, ttl in ((axs[0], store_cell, "crash carried by the STORE"),
                       (axs[1], read_cell, "crash carried by the READOUT")):
        R = cells[c]
        st, A = mean_curve(R, "A")
        _, cfW = mean_curve(R, "A_storeonly")
        _, cfU = mean_curve(R, "A_readonly")
        _, Bc = mean_curve(R, "B")
        ax.plot(st, A, color=NAVY, lw=1.8, label="A (both trained)")
        ax.plot(st, cfW, color=OLIVE, lw=1.2, ls="--", label="A, readout reverted ($W_t,U_0$)")
        ax.plot(st, cfU, color=PURP, lw=1.2, ls=":", label="A, store reverted ($W_0,U_t$)")
        ax.plot(st, Bc, color=CRIM, lw=1.0, alpha=0.8, label="B")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("injection step")
        nd, us, ws = c
        ax.set_title(f"{ttl}\n$n_D={nd}$, $u_{{scale}}={us:g}$, $w_{{scale}}={ws:g}$",
                     fontsize=9)
        ax.set_ylim(-0.03, 1.05)
    axs[0].set_ylabel("accuracy")
    axs[0].legend(fontsize=7, frameon=False, loc="center left")
    save(fig, "readout_carrier")
    return store_cell, read_cell


def fig_c(cells):
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for nd, col, lab in ((0, CRIM, "no ballast ($n_D=0$)"), (50, NAVY, "ballast ($n_D=50$)")):
        R = cells[(nd, 1.0, 1.0)]
        st, gr = mean_curve(R, "G_read")
        _, gs = mean_curve(R, "G_store")
        _, A = mean_curve(R, "A")
        axs[0].plot(st, gr, color=col, lw=1.8, label=lab)
        axs[1].plot(st, gs, color=col, lw=1.8, label=lab)
        for ax, y in ((axs[0], gr), (axs[1], gs)):
            j = int(np.argmin(A))
            ax.scatter([st[j]], [y[j]], color=col, s=22, zorder=4, edgecolor="white", lw=0.5)
    for ax, t, yl in ((axs[0], "readout bias written into $U$",
                       "$\\langle \\bar s_A(0),\\ \\bar u_Y(t)-\\bar u_X(t)\\rangle$  (logits)"),
                      (axs[1], "store's shared write, read through $U_0$",
                       "$\\langle \\bar s_A(t)-\\bar s_A(0),\\ \\bar u_Y(0)-\\bar u_X(0)\\rangle$")):
        ax.set_xscale("symlog", linthresh=10)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xlabel("injection step"); ax.set_ylabel(yl, fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("The two channels of A's Y-vs-X class bias, $u_{scale}=1$ "
                 "(dot = A's trough)", fontsize=10)
    save(fig, "readout_bias_time")


def main():
    raw, cells = load()
    print("writing figures:")
    fig_a(cells)
    sc, rc = fig_b(cells)
    fig_c(cells)
    print(f"  (b) store-carried cell {sc}, readout-carried cell {rc}")


if __name__ == "__main__":
    main()
