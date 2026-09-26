"""The four ballast arms, read from the cluster results (CLUSTER_REQUEST.md, all three requests).

Columns: 4L with ballast, 4L without, 8L with, 8L without. Rows:
  1. A's accuracy, full vocabulary and own-half (head-splice files).
  2. the collective shift: its norm ||delta|| and its BETWEEN-HALF CONTENT
     gap = delta . (u_bar_Y - u_bar_X), per attribute, step-0 head. The toy found the norm is
     the wrong observable and the between-half content is the right one; this is the same
     split in the transformer.
  3. the individual component rms||eps|| and the ratio eps/delta.
  4. the head splice: R_store and R_read, and the relative movement of the Y-value
     unembedding rows.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE, GREY = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30", "#8A8F98"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 9.5, "axes.grid": False})
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ARMS = [("demo4", "4 layers, with ballast"), ("demo4_noballast", "4 layers, no ballast"),
        ("standard", "8 layers, with ballast"), ("standard_noballast", "8 layers, no ballast")]


def band(ax, x, Y, color, ls="-", lw=1.9, label=None):
    Y = np.asarray(Y); m = Y.mean(0); sd = Y.std(0, ddof=1) if len(Y) > 1 else 0 * m
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, marker="o", ms=2.8, label=label)


def load(arm):
    acc = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(ROOT, "weight_patch", f"{arm}-head-acc-*.npz")))]
    ug = {int(f.split("seed")[-1].split(".")[0]): np.load(f, allow_pickle=True)
          for f in sorted(glob.glob(os.path.join(ROOT, "unembedding_geometry", f"{arm}-p*-seed*.npz")))}
    ds = {int(f.split("seed")[-1].split(".")[0]): np.load(f)
          for f in sorted(glob.glob(os.path.join(ROOT, "delta_structure", f"{arm}-p*-seed*.npz")))}
    return acc, ug, ds


def gap_series(ds, ug):
    out = {}
    for seed, d in ds.items():
        u = ug[seed]; W0 = u["W0"].astype(np.float64)
        ax_, axi, ay_, ayi = u["attr_x"], u["attr_x_ids"], u["attr_y"], u["attr_y_ids"]
        g = np.zeros((len(d["steps"]), 6)); n = np.zeros_like(g); e = np.zeros_like(g)
        for k in range(6):
            w = W0[:, ayi[ay_ == k]].mean(1) - W0[:, axi[ax_ == k]].mean(1)
            dl = d["delta_layer"][:, k, -1, :].astype(np.float64)
            g[:, k] = dl @ w; n[:, k] = np.linalg.norm(dl, axis=-1); e[:, k] = d["eps_layer_rms"][:, k, -1]
        out[seed] = (g.mean(1), n.mean(1), e.mean(1))
    st = list(ds[next(iter(ds))]["steps"])
    G = np.array([v[0] for v in out.values()]); Nn = np.array([v[1] for v in out.values()]); E = np.array([v[2] for v in out.values()])
    return np.array(st, float), G, Nn, E


def main():
    fig, axes = plt.subplots(4, 4, figsize=(14.5, 11.5), dpi=200)
    for c, (arm, title) in enumerate(ARMS):
        acc, ug, ds = load(arm)
        st = acc[0]["steps"].astype(float)
        # row 1: accuracy
        ax = axes[0, c]
        band(ax, st, [a["acc_t"].mean(-1) for a in acc], NAVY, label="A, full vocabulary")
        band(ax, st, [a["own_t"].mean(-1) for a in acc], NAVY, ls="--", lw=1.4, label="A, own half")
        ax.set_ylim(0, 1.04); ax.set_title(title, fontsize=10.5)
        if c == 0: ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=8, loc="lower left")
        # row 2: norm vs between-half content
        ax = axes[1, c]
        sts, G, Nn, E = gap_series(ds, ug)
        band(ax, sts, G, OLIVE, label="between-half gap (nats)")
        ax.set_ylabel("gap, nats" if c == 0 else "")
        ax2 = ax.twinx(); band(ax2, sts, Nn, GREY, lw=1.3, label="||delta||")
        ax2.set_ylabel("||delta||" if c == 3 else "", color=GREY); ax2.tick_params(axis="y", labelcolor=GREY)
        if c == 0: ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_ylim(0, None); ax2.set_ylim(0, None)
        # row 3: eps and eps/delta
        ax = axes[2, c]
        band(ax, sts, E, PURP, label="rms ||eps||")
        ax.set_ylabel("rms ||eps||" if c == 0 else "")
        ax2 = ax.twinx(); band(ax2, sts, E / Nn, CRIM, lw=1.3, label="eps / delta")
        ax2.axhline(1, color=CRIM, lw=0.7, ls=":")
        ax2.set_ylabel("eps / delta" if c == 3 else "", color=CRIM); ax2.tick_params(axis="y", labelcolor=CRIM)
        if c == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylim(0, None); ax2.set_ylim(0, None)
        # row 4: head splice and head movement
        ax = axes[3, c]
        keep = st > 0
        band(ax, st[keep], [np.nan_to_num(a["r_body"][keep]) for a in acc], NAVY, label="R_store")
        band(ax, st[keep], [np.nan_to_num(a["r_group"][keep]) for a in acc], CRIM, label="R_read")
        ax.set_ylim(-0.05, 1.15); ax.set_ylabel("share of A's damage" if c == 0 else "")
        ax2 = ax.twinx()
        ry = []
        for u in ug.values():
            ry.append((u["norm"][:, u["y_ids"]] / np.linalg.norm(u["W0"][:, u["y_ids"]], axis=0)).mean(1))
        band(ax2, ug[next(iter(ug))]["steps"].astype(float), ry, GREY, lw=1.2, label="Y rows moved (rel.)")
        ax2.set_ylim(0, 0.3); ax2.set_ylabel("Y-row rel. move" if c == 3 else "", color=GREY)
        ax2.tick_params(axis="y", labelcolor=GREY)
        if c == 0:
            ax.legend(frameon=False, fontsize=8, loc="center left"); ax2.legend(frameon=False, fontsize=8, loc="center right")
        for r in range(4):
            axes[r, c].set_xscale("symlog", linthresh=10); axes[r, c].set_xlim(0, 1300)
        axes[3, c].set_xlabel("injection step")
    fig.tight_layout()
    out = os.path.join(ROOT, "plots", "toy_plots_working", "cluster_ballast_arms")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
