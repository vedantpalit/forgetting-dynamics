"""Paper figures: the normalized K=1 toy's accuracy curve across four knobs, overlaid.

    plots/toy_k1_sweep_K.{png,pdf}      depth, K blocks             (wp_depth_K*.json, 6 seeds)
    plots/toy_k1_sweep_d.{png,pdf}      key dimension d             (sw_d.json, 3 seeds)
    plots/toy_k1_sweep_beta.{png,pdf}   shared key fraction beta    (sw_beta.json, 3 seeds)
    plots/toy_k1_sweep_dose.{png,pdf}   injection rate              (sw_dose.json, 3 seeds)

A's curves are drawn as a plasma gradient, dark at the smallest value of the knob and light at
the largest. B is drawn once, in crimson, as the mean over every cell of the sweep: the first
three sweeps bisect the injection rate so that B reaches 0.99 at step 200 in every cell (its
gate runs 170 to 236 across all cells), so one B curve is the honest picture. The dose sweep
varies the rate on purpose, so there B is drawn once per rate, thin, with the reference rate
bold; drawing it once would misstate the experiment.

Normalized arm only (z = rms(h) U). Same style as the transformer figures.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The pretrain_sweep_a_vs_b.png palette: A runs from light mint to the project navy across
# the sweep, B is a dotted grey. Reconstructed from the figure; the original script is not
# in the repo.
RAMP = LinearSegmentedColormap.from_list("mint_navy",
                                         ["#B9E3CB", "#63BB96", "#2F8C7D", "#1E5E6B", "#1B2A4E"])
B_GREY = "0.62"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})
GRID = np.array(sorted(set(range(0, 201, 5)) | {int(x) for x in np.unique(np.round(np.logspace(np.log10(200), np.log10(5000), 60)))}), float)


def onto(st, y):
    """Interpolate a run onto the common step grid (runs share it, but do not assume)."""
    st = np.asarray(st, float); y = np.asarray(y, float)
    return y if len(st) == len(GRID) and np.allclose(st, GRID) else np.interp(GRID, st, y)


def load_sw(fname, key="value"):
    d = json.load(open(os.path.join(HERE, fname)))
    out = {}
    for r in d.values():
        if r["norm"] != 1:
            continue
        st = [x["step"] for x in r["rows"]]
        out.setdefault(float(r[key]), []).append((onto(st, [x["A"] for x in r["rows"]]),
                                                  onto(st, [x["B"] for x in r["rows"]])))
    return out


def load_depth():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "wp_depth_K[0-9].json"))):
        for r in json.load(open(f)).values():
            if r.get("norm", 1) != 1:
                continue
            out.setdefault(float(r["K"]), []).append((onto(r["steps"], r["A"]), onto(r["steps"], r["B"])))
    return out


def figure(data, out, label, fmt, dose=False, legend_loc="lower left"):
    vals = sorted(data)
    cmap = RAMP(np.linspace(0, 1, len(vals)))
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    summary = []
    for i, v in enumerate(vals):
        A = np.array([a for a, _ in data[v]]); m = A.mean(0)
        ax.plot(GRID, m, color=cmap[i], lw=1.8, label=f"A ({fmt(v)})")
        tr = int(m.argmin())
        summary.append((v, len(A), m[tr], int(GRID[tr]), m[tr:].max(), m[-1]))
        if dose:
            B = np.array([b for _, b in data[v]]).mean(0)
            ax.plot(GRID, B, color=B_GREY, ls=":", lw=2.0 if v == 1.0 else 0.9,
                    alpha=1.0 if v == 1.0 else 0.45)
    if not dose:
        B = np.array([b for _, b in sum(data.values(), [])]).mean(0)
        ax.plot(GRID, B, color=B_GREY, ls=":", lw=2.0, label="B")
    else:
        ax.plot([], [], color=B_GREY, ls=":", lw=2.0, label="B (bold: ×1)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("First-token accuracy")
    ax.legend(frameon=False, fontsize=7, loc=legend_loc, handlelength=1.8, labelspacing=0.25)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{out}.{e}"), bbox_inches="tight")
    print(f"\n{out}   ({label})")
    print(f"  {'value':>8} {'seeds':>5} {'A trough':>9} {'@step':>6} {'max after':>10} {'A end':>6}")
    for v, n, tr, ts, mx, en in summary:
        print(f"  {v:>8g} {n:>5} {tr:>9.3f} {ts:>6} {mx:>10.3f} {en:>6.3f}")


if __name__ == "__main__":
    figure(load_depth(), "toy_k1_sweep_K", "depth", lambda v: f"K = {int(v)}",
           legend_loc="lower right")
    figure(load_sw("sw_d.json"), "toy_k1_sweep_d", "key dimension", lambda v: f"d = {int(v)}",
           legend_loc="center right")
    figure(load_sw("sw_beta.json"), "toy_k1_sweep_beta", "shared fraction", lambda v: f"β = {v:g}",
           legend_loc="center right")
    figure(load_sw("sw_dose.json"), "toy_k1_sweep_dose", "injection rate", lambda v: f"rate ×{v:g}",
           dose=True, legend_loc="lower right")
