"""Memories per parameter: effective rank of the write's individual part, vs n_B and vs |V|.

    plots/toy_k1_sweep_nb_rank.{png,pdf}      rank_perp overlaid across n_B, fixed |V|
    plots/toy_k1_sweep_vocab_rank.{png,pdf}   rank_perp overlaid across |V|, fixed n_B

Same overlay style as the other sweep figures (mint-to-navy ramp, dotted grey B), same
gate-matched harness (B reaches 0.99 at step 200 in every cell). The quantity plotted is the
effective rank (participation ratio) of dW_perp = dW with its shared mu-row removed -- the
individual-only part of the write -- so it isolates "how many distinct directions does B's
writing use up" from the coherent, individual-blind channel already characterised elsewhere.

Also fits, at each cell's final checkpoint, log-log slopes of rank_perp against three
candidates: n_B, n_distinct_B (values actually drawn, with replacement, so <= min(n_B,|Y|)),
and |Y|. This is the scaling-law test: which ceiling the rank tracks.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})
RAMP = LinearSegmentedColormap.from_list("mint_navy",
                                         ["#B9E3CB", "#63BB96", "#2F8C7D", "#1E5E6B", "#1B2A4E"])
B_GREY = "0.62"
GRID = np.array(sorted(set(range(0, 201, 5)) | {int(x) for x in np.unique(np.round(np.logspace(np.log10(200), np.log10(5000), 60)))}), float)


def onto(st, y):
    st = np.asarray(st, float); y = np.asarray(y, float)
    return y if len(st) == len(GRID) and np.allclose(st, GRID) else np.interp(GRID, st, y)


def load(fname):
    """`run_cell` records the pre-sweep DEFAULT nb/vocab in its metadata rather than the
    swept value (a bug in memrank_sweep.py, left uncorrected to avoid a costly re-run) --
    n_distinct_B and every rank number are unaffected, since those come from the actual run,
    but nb/vocab here are reconstructed from `axis` and `value` instead of trusted verbatim.
    """
    d = json.load(open(os.path.join(HERE, fname)))
    out = {}
    for r in d.values():
        st = [x["step"] for x in r["rows"]]
        nb = r["value"] if r["axis"] == "nb" else 50
        vocab = r["value"] if r["axis"] == "vocab" else 32
        out.setdefault(r["value"], []).append(dict(
            A=onto(st, [x["A"] for x in r["rows"]]), B=onto(st, [x["B"] for x in r["rows"]]),
            rank_perp=onto(st, [x["rank_perp"] for x in r["rows"]]),
            rank_full=onto(st, [x["rank_full"] for x in r["rows"]]),
            n_distinct_B=r["n_distinct_B"], vocab=vocab, nb=nb, d=r["d"]))
    return out


def rank_figure(data, out, label, fmt, legend_loc):
    vals = sorted(data)
    cmap = RAMP(np.linspace(0, 1, len(vals)))
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    for i, v in enumerate(vals):
        cells = data[v]
        m = np.array([c["rank_perp"] for c in cells]).mean(0)
        ax.plot(GRID, m, color=cmap[i], lw=1.8, label=f"rank ({fmt(v)})")
    B = np.array([c["B"] for cells in data.values() for c in cells]).mean(0)
    top = max(c["rank_perp"].max() for cells in data.values() for c in cells)
    ax2 = ax.twinx()
    ax2.plot(GRID, B, color=B_GREY, ls=":", lw=1.6, label="B accuracy")
    ax2.set_ylim(0, 1.03)
    ax2.set_ylabel("B accuracy", color=B_GREY)
    ax2.tick_params(axis="y", labelcolor=B_GREY, colors=B_GREY)
    ax2.spines["right"].set_color(B_GREY)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, GRID[-1])
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("Effective rank of $\\Delta W_\\perp$")
    ax.legend(frameon=False, fontsize=7, loc=legend_loc, handlelength=1.8, labelspacing=0.25)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{out}.{e}"), bbox_inches="tight")
    print(f"wrote {out}")


def fit(x, y):
    ok = (np.asarray(x) > 0) & (np.asarray(y) > 0)
    lx, ly = np.log(np.asarray(x)[ok]), np.log(np.asarray(y)[ok])
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return coef[0], r2


def report(data_nb, data_vocab):
    print("\n=== effective rank at the final checkpoint, both sweeps ===")
    print(f"{'axis':>7} {'value':>7} {'n_B':>5} {'|Y|':>5} {'n_distinct_B':>12} "
          f"{'rank_perp':>10} {'rank_full':>10}")
    rows = []
    for axis, data in (("nb", data_nb), ("vocab", data_vocab)):
        for v in sorted(data):
            cells = data[v]
            rp = np.mean([c["rank_perp"][-1] for c in cells])
            rf = np.mean([c["rank_full"][-1] for c in cells])
            nb = cells[0]["nb"]; vocab = cells[0]["vocab"]; nd = np.mean([c["n_distinct_B"] for c in cells])
            print(f"{axis:>7} {v:>7g} {nb:>5} {vocab // 2:>5} {nd:>12.1f} {rp:>10.2f} {rf:>10.2f}")
            rows.append((axis, nb, vocab // 2, nd, rp))

    print("\n=== scaling fits: log(rank_perp) ~ log(x), pooled over both sweeps ===")
    rows = np.array([(r[1], r[2], r[3], r[4]) for r in rows])
    for name, col in (("n_B", 0), ("|Y|", 1), ("n_distinct_B", 2)):
        slope, r2 = fit(rows[:, col], rows[:, 3])
        print(f"  rank_perp ~ {name:>12}^{slope:+.3f}   R^2 = {r2:.3f}")

    print("\n=== within each sweep separately (isolates the axis actually varied) ===")
    for axis, data, key in (("nb", data_nb, 0), ("vocab", data_vocab, 1)):
        sub = np.array([(cells[0]["nb"], cells[0]["vocab"] // 2, np.mean([c["n_distinct_B"] for c in cells]),
                        np.mean([c["rank_perp"][-1] for c in cells])) for cells in [data[v] for v in sorted(data)]])
        for name, col in (("n_B", 0), ("|Y|", 1), ("n_distinct_B", 2)):
            slope, r2 = fit(sub[:, col], sub[:, 3])
            print(f"  [{axis} sweep] rank_perp ~ {name:>12}^{slope:+.3f}   R^2 = {r2:.3f}")


if __name__ == "__main__":
    data_nb = load("memrank_nb.json")
    data_vocab = load("memrank_vocab.json")
    rank_figure(data_nb, "toy_k1_sweep_nb_rank", "n_B", lambda v: f"$n_B$ = {int(v)}", "upper left")
    rank_figure(data_vocab, "toy_k1_sweep_vocab_rank", "|V|", lambda v: f"$|V|$ = {int(v)}", "upper left")
    report(data_nb, data_vocab)
