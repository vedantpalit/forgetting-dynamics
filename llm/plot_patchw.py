"""OLMo, the two processes separated in weight space (llm/patch_weights.py output).

    uv run python -m llm.plot_patchw            # three seeds, mean and one sd

The model's old-fact curve; the same weights with the top-r singular directions of every
matrix's update removed (r = 1, 4); the random-direction control of the same size; and the
new facts, actual and under the r = 4 patch. Figure-1 style (same size, type, linear x).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, TEAL, OLIVE, GREY = "#1B2A4E", "#C4245F", "#2F8C7D", "#8A8C30", "0.55"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="llm/out/patchw_lr1e-05_seed0.json,llm/out/patchw_lr1e-05_seed1.json,llm/out/patchw_lr1e-05_seed2.json")
    ap.add_argument("--xmax", type=int, default=1000)
    ap.add_argument("--out", default="plots/fig1_olmo_patchw")
    a = ap.parse_args()
    import numpy as np
    runs = [json.load(open(f))["curve"] for f in a.runs.split(",")]
    st = [r["step"] for r in runs[0] if r["step"] <= a.xmax]
    for rr in runs:
        assert [r["step"] for r in rr if r["step"] <= a.xmax] == st, "step grids differ"
    keys = ("actual/A_noncopy", "patch_r1/A_noncopy", "patch_r4/A_noncopy", "control_r4/A_noncopy", "actual/B", "patch_r4/B", "patch_r1/B",
            "actual/generic_loss", "patch_r4/generic_loss", "patch_r1/generic_loss", "control_r4/generic_loss")
    M = {k: np.array([[r[k] for r in rr if r["step"] <= a.xmax] for rr in runs]) for k in keys}
    c = [{k: float(M[k][:, i].mean()) for k in keys} | {"step": s_} for i, s_ in enumerate(st)]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
                         "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 8.4), dpi=200, sharex=True,
                                  gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})

    def band(key, col, lw, ls, lab, z=3, ax=ax):
        Y = M[key]; mu, sd = Y.mean(0), Y.std(0, ddof=1) if len(Y) > 1 else 0 * Y.mean(0)
        ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(st, mu, color=col, lw=lw, ls=ls, label=lab, zorder=z)

    band("actual/A_noncopy", NAVY, 2.4, "-", "old facts")
    band("patch_r4/A_noncopy", OLIVE, 2.4, "-", "old facts, 4 directions/matrix removed")
    band("patch_r1/A_noncopy", OLIVE, 1.4, ":", "old facts, 1 direction removed")
    band("control_r4/A_noncopy", GREY, 1.4, "--", "old facts, random directions removed (control)", z=2)
    band("actual/B", CRIM, 2.4, ":", "new facts")
    band("patch_r1/B", CRIM, 1.4, ":", "new facts, 1 direction removed")
    band("patch_r4/B", CRIM, 1.4, "-.", "new facts, 4 directions removed")
    ax.set_xlim(0, a.xmax); ax.set_ylim(0, 1.14); ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("First-token accuracy")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    # generic text: mean next-token loss on 64 x 512 tokens of pre-cutoff prose
    band("actual/generic_loss", NAVY, 2.4, "-", "model", ax=ax2)
    band("patch_r4/generic_loss", OLIVE, 2.4, "-", "4 directions/matrix removed", ax=ax2)
    band("patch_r1/generic_loss", OLIVE, 1.4, ":", "1 direction removed", ax=ax2)
    band("control_r4/generic_loss", GREY, 1.4, "--", "control", z=2, ax=ax2)
    ax2.set_ylabel("Loss on generic text (nats)"); ax2.set_xlabel("Injection step")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=1.6)
    fig.subplots_adjust(hspace=0.08)
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}.{e}", bbox_inches="tight")
    for k in ("actual/A_noncopy", "patch_r1/A_noncopy", "patch_r4/A_noncopy", "control_r4/A_noncopy", "actual/B", "patch_r4/B"):
        v = [r[k] for r in c]; i = min(range(len(v)), key=lambda j: v[j])
        print(f"{k:>22}: {v[0]:.3f} -> min {v[i]:.3f}@{st[i]} -> end {v[-1]:.3f}")
    import statistics
    for rr in runs:
        sp = [r["spectrum"] for r in rr if r["step"] == 100][0]
        print(f"  top-1 / top-4 / top-16 share of ||dW||^2 per matrix at step 100 (median over matrices): "
              f"{statistics.median(v['top1'] for v in sp.values()):.2f} / {statistics.median(v['top4'] for v in sp.values()):.2f} / "
              f"{statistics.median(v['top16'] for v in sp.values()):.2f}")
    print(f"{len(runs)} seeds; at step 100 per seed: actual " + " ".join(f"{v:.3f}" for v in M['actual/A_noncopy'][:, st.index(100)])
          + " | patch r=4 " + " ".join(f"{v:.3f}" for v in M['patch_r4/A_noncopy'][:, st.index(100)])
          + " | control " + " ".join(f"{v:.3f}" for v in M['control_r4/A_noncopy'][:, st.index(100)]))
    print(f"wrote {a.out}.png/.pdf")


if __name__ == "__main__":
    main()
