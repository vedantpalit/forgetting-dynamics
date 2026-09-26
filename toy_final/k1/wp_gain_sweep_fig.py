"""Plot: does recovery track ||W||, or the write's share of the state?  (wp_gain_sweep.json)

    plots/toy_plots_working/gain_sweep_share_vs_magnitude.{png,pdf}

Left: mean |dW| at the trough (grey, growing) against mean a_frac at the trough (teal,
non-monotone) across the gate-matched gain sweep, log-x. Right: mean recovered fraction of
the crash against the same axis. If the phenomenon tracked ||W||, recovery would track the
grey curve; it tracks the teal one instead, and both fall together at high gain.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
TEAL, GREY, DEEP = "#2F8C7D", "0.55", "#1E5E6B"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})


def main():
    rows = json.load(open(os.path.join(HERE, "wp_gain_sweep.json")))
    gains = sorted(set(r["gain"] for r in rows))
    af = np.array([np.mean([r["a_frac_trough"] for r in rows if r["gain"] == g]) for g in gains])
    dw = np.array([np.mean([r["dW_trough"] for r in rows if r["gain"] == g]) for g in gains])
    rf = np.array([np.mean([r["recovered_frac"] for r in rows if r["gain"] == g]) for g in gains])
    at = np.array([np.mean([r["A_trough"] for r in rows if r["gain"] == g]) for g in gains])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=200)

    ax = axes[0]
    ax.plot(gains, af, color=TEAL, lw=2.0, marker="o", ms=4, label="write's share of state $a_{frac}$")
    ax.set_ylabel("$a_{frac}$ at the trough", color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL)
    ax.set_xscale("log")
    ax.set_xlabel("gain $\\gamma$")
    ax2 = ax.twinx()
    ax2.plot(gains, dw, color=GREY, lw=2.0, ls="--", marker="s", ms=4, label="$\\|\\Delta W\\|$")
    ax2.set_ylabel("$\\|\\Delta W\\|$ at the trough", color=GREY)
    ax2.tick_params(axis="y", labelcolor=GREY)
    ax.set_title("Share of the state vs. raw weight magnitude")

    ax = axes[1]
    ax.plot(gains, rf, color=DEEP, lw=2.0, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("gain $\\gamma$")
    ax.set_ylabel("Recovered fraction of the crash")
    ax.set_title("Recovery follows the share, not the magnitude")

    fig.tight_layout()
    outdir = os.path.join(ROOT, "plots", "toy_plots_working")
    os.makedirs(outdir, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"gain_sweep_share_vs_magnitude.{e}"), bbox_inches="tight")
    print("wrote gain_sweep_share_vs_magnitude")
    print("gains:", gains)
    print("a_frac:", np.round(af, 3))
    print("dW:", np.round(dw, 2))
    print("recovered:", np.round(rf, 3))
    print("A trough:", np.round(at, 3))


if __name__ == "__main__":
    main()
