"""Paper figures for the K=1 toy: two standalone accuracy curves, one per readout.

    plots/toy_k1_linear_acc.{png,pdf}   z = h U        (the store-and-softmax toy as is)
    plots/toy_k1_norm_acc.{png,pdf}     z = rms(h) U   (one normalizer added)

Same store, same data, same optimizer, same seeds; the readout is the only difference.
Data: toy_final/k1/paper_base.json (paper_base_runs.py) at beta = 0.5, gain 1, readout rate
0.01, TEN seeds, injection rate gate-matched per seed and arm so B reaches 0.99 at step 200
(the earlier beta_sweep.json was three seeds at a fixed rate, whose 2.4x spread in B's clock
widened the band in time; LOG.md, 'where the toy's noise comes from'). Style follows the
transformer figures: navy A, crimson B, mean and one standard deviation, log injection
step, no grid, dpi 200.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM = "#1B2A4E", "#C4245F"
SEEDS = tuple(range(10))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 11, "axes.grid": False})


def curves(norm, fname="paper_base.json"):
    d = json.load(open(os.path.join(HERE, fname)))
    runs = [d[f"b0.5-n{norm}-g1.0-u0.01-s{s}"]["rows"] for s in SEEDS]
    st = np.array([x["step"] for x in runs[0]], float)
    A = np.array([[x["A"] for x in r] for r in runs])
    B = np.array([[x["B"] for x in r] for r in runs])
    return st, A, B


def one(norm, out, fname="paper_base.json"):
    st, A, B = curves(norm, fname)
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=200)
    for Y, col, lab in ((A, NAVY, "A"), (B, CRIM, "B")):
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        ax.plot(st, m, color=col, lw=2.0, label=lab)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("injection step")
    ax.set_ylabel("first-token accuracy")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "plots", f"{out}.{e}"), bbox_inches="tight")
    m = A.mean(0)
    tr = int(m.argmin())
    print(f"{out}: A trough {m[tr]:.3f} at step {int(st[tr])}, max after {m[tr:].max():.3f}, "
          f"end {m[-1]:.3f}; B reaches 0.99 at step "
          f"{int(st[np.argmax(B.mean(0) >= 0.99)]) if (B.mean(0) >= 0.99).any() else -1}")


if __name__ == "__main__":
    one(0, "toy_k1_linear_acc")
    one(1, "toy_k1_norm_acc")
    # (ii): the store's input normalizer rms(k) replaced by the constant sqrt(d) -- same
    # protocol, ten seeds, gate-matched (paper_base_runs.py --norms 1 --store_scale sqrt_d).
    one(1, "toy_k1_norm_acc_ii", "paper_base_ii.json")
