"""A and B accuracy during injection, one panel per K, beta as line style. First look only."""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_COLOR, B_COLOR = "#1B2A4E", "#C4245F"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="toy_final/out")
    ap.add_argument("--Ks", default="1,2,4,8")
    ap.add_argument("--betas", default="0.0,0.5")
    ap.add_argument("--fig", default="plots/toy_final_acc")
    a = ap.parse_args()
    Ks = [int(k) for k in a.Ks.split(",")]
    betas = [float(b) for b in a.betas.split(",")]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 11, "axes.grid": False})
    fig, axes = plt.subplots(len(betas), len(Ks), figsize=(3.4 * len(Ks), 3.0 * len(betas)),
                             dpi=200, sharey=True, squeeze=False)
    for r, beta in enumerate(betas):
        for c, K in enumerate(Ks):
            ax = axes[r, c]
            fs = sorted(glob.glob(os.path.join(a.out_dir, f"K{K}-beta{beta}-seed*.json")))
            if not fs:
                ax.set_title(f"K={K}, β={beta}  (no runs)", fontsize=10); continue
            runs = [json.load(open(f)) for f in fs]
            n = min(len(rn["inject"]) for rn in runs)
            st = np.array([x["step"] for x in runs[0]["inject"][:n]], float)
            for key, col, lab in (("A", A_COLOR, "A"), ("B", B_COLOR, "B")):
                Y = np.array([[x[key] for x in rn["inject"][:n]] for rn in runs])
                m, sd = Y.mean(0), Y.std(0, ddof=1) if len(runs) > 1 else np.zeros(n)
                ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.18, lw=0)
                ax.plot(st, m, color=col, lw=2.0, label=lab)
            Ao = np.array([[x["A_own"] for x in rn["inject"][:n]] for rn in runs]).mean(0)
            ax.plot(st, Ao, color=A_COLOR, lw=1.2, ls="--", label="A own-half")
            ax.set_xscale("symlog", linthresh=10)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"K={K}, β={beta}  (n={len(runs)})", fontsize=10)
            if r == len(betas) - 1: ax.set_xlabel("injection step")
            if c == 0: ax.set_ylabel("first-token accuracy")
            if r == 0 and c == 0: ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.fig}.{ext}", bbox_inches="tight")
        print(f"  wrote {a.fig}.{ext}")


if __name__ == "__main__":
    main()
