"""Figures for the base-model predictors (llm/predict_erosion.py output).

    plots/olmo_predict_erosion.{png,pdf}     old-fact accuracy trajectories by quartile of the
                                             erosion predictor (cos_centered at --tap): the
                                             OLMo analogue of the transformer's graded figure
    plots/olmo_predict_suppression.{png,pdf} the same by quartile of push_ratio
    plots/olmo_predict_table.txt             Spearman table (non-copy), all predictors

Run (after scp of llm/out/predict_erosion_lr1e-05.json):
  uv run python llm/plot_predict.py --tap L8
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCOL = ["#3a6e8c", "#6e8a6b", "#b08a44", "#b0402e"]


def quartile_figure(d, key, title, out):
    steps = np.array(d["steps"], float); C = np.array(d["traj_correct"])
    copy = np.array(d["copy"]); m = copy == "noncopy"
    p = np.array(d["predictors"][key])[m]; Cn = C[:, m]
    q = np.quantile(p, [0.25, 0.5, 0.75]); lab = np.digitize(p, q)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    for k in range(4):
        sel = lab == k
        ax.plot(steps, Cn[:, sel].mean(1), color=QCOL[k], lw=2.2,
                label=f"Q{k + 1} (n={sel.sum()})")
    ax.axvline(d["trough_step"], color="0.6", ls=":", lw=1.2)
    ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, steps.max()); ax.set_ylim(0, 1.0)
    ax.set_xlabel("injection step"); ax.set_ylabel("accuracy on the old facts (non-copy)")
    ax.legend(frameon=False, fontsize=9, title=title, title_fontsize=9)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="llm/out/predict_erosion_lr1e-05.json")
    ap.add_argument("--tap", default="L8")
    ap.add_argument("--src", default="eval")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
                         "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})
    d = json.load(open(a.json))
    os.makedirs("plots", exist_ok=True)
    quartile_figure(d, f"{a.tap}/{a.src}/cos_centered", f"key overlap with the new facts ({a.tap})",
                    "plots/olmo_predict_erosion")
    quartile_figure(d, f"readout/{a.src}/push_ratio", "predicted push / margin",
                    "plots/olmo_predict_suppression")
    heads = ["trough/correct", "trough/margin_drop", "erosion/rank_late_mean", "erosion/margin_drop_last"]
    lines = [f"{'predictor':40s}" + "".join(f"{h:>30s}" for h in heads)]
    for pname in d["predictors"]:
        cells = []
        for h in heads:
            r = d["stats"][f"{pname} -> {h}"]
            seeds = "/".join(f"{r[f'seed{i}/rho']:+.2f}" for i in range(3) if f"seed{i}/rho" in r)
            cells.append(f"{r['noncopy/rho']:+.3f} [{r['noncopy/rho_within_relation']:+.3f}] {seeds}")
        lines.append(f"{pname:40s}" + "".join(f"{c:>30s}" for c in cells))
    lines.append("dataset level: " + json.dumps(d["level"]))
    open("plots/olmo_predict_table.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
