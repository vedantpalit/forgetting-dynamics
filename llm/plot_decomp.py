"""Figure 1 from OLMo alone: the model's curve with the two counterfactual curves on the same
axes (llm/inject_decomp.py output).

    .venv-llm/bin/python -m llm.plot_decomp --run llm/out/decomp_lr1e-05_seed0.json

Same style as plot_seeds.py --paper: navy actual, teal common-only, olive individual-only,
grey readout-only (thin), crimson dotted new facts; linear x to --xmax.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.55"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="llm/out/decomp_lr1e-05_seed0.json")
    ap.add_argument("--stratum", default="noncopy")
    ap.add_argument("--xmax", type=int, default=1000)
    ap.add_argument("--out", default="plots/fig1_olmo_decomp")
    a = ap.parse_args()
    d = json.load(open(a.run, encoding="utf-8"))
    c = [r for r in d["curve"] if r["step"] <= a.xmax]
    st = [r["step"] for r in c]
    s = a.stratum
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
    for key, col, lw, ls, lab in (("common", TEAL, 2.4, "-", "common shift alone"),
                                  ("individual", OLIVE, 2.4, "-", "individual displacement alone"),
                                  ("readout", GREY, 1.4, "--", "unembedding alone"),
                                  ("actual", NAVY, 2.4, "-", "old facts (the model)")):
        ax.plot(st, [r[f"{s}/{key}/acc"] for r in c], color=col, lw=lw, ls=ls, label=lab, zorder=3)
    ax.plot(st, [r["B/ALL/acc"] for r in c], color=CRIM, lw=2.4, ls=":", label="new facts", zorder=3)
    ax.set_xlim(0, a.xmax); ax.set_ylim(0, 1.14); ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Injection step"); ax.set_ylabel("First-token accuracy")
    ax.legend(frameon=False, fontsize=11, loc="center right", handlelength=1.6)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}.{e}", bbox_inches="tight")
    for key in ("actual", "common", "individual", "readout"):
        v = [r[f"{s}/{key}/acc"] for r in c]; i = min(range(len(v)), key=lambda k: v[k])
        print(f"{key:>10}: {v[0]:.3f} -> trough {v[i]:.3f}@{st[i]} -> max after {max(v[i:]):.3f} "
              f"-> end {v[-1]:.3f} | rank {c[0][f'{s}/{key}/rank']:.2f} -> {c[i][f'{s}/{key}/rank']:.2f}")
    print(f"||delta|| peak {max(r[f'{s}/delta_norm'] for r in c):.2f}, end {c[-1][f'{s}/delta_norm']:.2f}; "
          f"rms||eps|| end {c[-1][f'{s}/eps_rms']:.2f}; ||x|| {c[0][f'{s}/x_norm']:.1f}; "
          f"dU/U end {c[-1]['dU_rel']:.4f}")
    print(f"wrote {a.out}.png/.pdf")


if __name__ == "__main__":
    main()
