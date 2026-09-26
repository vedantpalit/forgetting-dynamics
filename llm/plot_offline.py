"""The OLMo patch and decomposition at every grain, from decomp_offline.py's output.

    .venv/bin/python -m llm.plot_offline --run llm/out/decomp_lr1e-05_seed1_offline.json

Left: the model (through the pretrained head) against the patch -- subtract the shift
estimated from 4 other old facts per group -- at the three grains. Right: common part alone
and residual alone at each grain. Both panels read every counterfactual through U(0).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, OLIVE, DEEP, CRIM, GREY = "#1B2A4E", "#2F8C7D", "#8A8C30", "#1E5E6B", "#C4245F", "0.55"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="llm/out/decomp_lr1e-05_seed1_offline.json")
    ap.add_argument("--xmax", type=int, default=700)
    ap.add_argument("--out", default="plots/toy_plots_working/olmo_patch_grain")
    a = ap.parse_args()
    d = json.load(open(a.run)); c = [r for r in d["curve"] if r["step"] <= a.xmax]
    st = [r["step"] for r in c]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10.5, "axes.grid": False})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=200, sharey=True)
    axL.plot(st, [r["actual"] for r in c], color=NAVY, lw=2.2, label="old facts (the model)")
    for name, col in (("all", GREY), ("relation", TEAL), ("frame", OLIVE)):
        axL.plot(st, [r.get(f"patch/{name}/k4", float("nan")) for r in c], color=col, lw=1.7,
                 label=f"subtract the shift estimated from 4 facts, per {name}")
    axL.set_title("the patch, at three grains", fontsize=10.5)
    for name, col in (("all", GREY), ("relation", TEAL), ("frame", OLIVE), ("kmeans-8", DEEP), ("kmeans-32", CRIM)):
        axR.plot(st, [r.get(f"{name}/common/acc", float("nan")) for r in c], color=col, lw=1.7, label=f"common alone, {name}")
        axR.plot(st, [r.get(f"{name}/residual/acc", float("nan")) for r in c], color=col, lw=1.2, ls="--")
    axR.plot(st, [r["actual"] for r in c], color=NAVY, lw=2.2)
    axR.set_title("common alone (solid) and residual alone (dashed)", fontsize=10.5)
    for ax in (axL, axR):
        ax.set_xlim(0, a.xmax); ax.set_ylim(0, 1.03); ax.set_xlabel("injection step")
        ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.6)
    axL.set_ylabel("first-token accuracy, old facts")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}.{e}", bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
