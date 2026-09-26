"""Does the norm of the transformer's hidden state grow throughout, like the toy's ||W||?
A follow-up, run on the actual model.

    plots/toy_plots_working/hidden_state_timecourse.{png,pdf}

Reuses the pulled cluster data (delta_structure/*.npz, weight_patch/*-head-acc-*.npz; no new
runs). At each residual-stream depth j, A's displacement splits EXACTLY as
    Delta h_a = delta + eps_a,   delta = mean_a(Delta h_a),   sum_a eps_a = 0
so mean_a||Delta h_a||^2 = ||delta||^2 + mean_a||eps_a||^2 with NO cross term -- same identity
as the toy's ||dW||^2 = ||s||^2 + ||dW_perp||^2 (FINDINGS 3.21.5 established this on the final
readout state; this script uses the per-layer version, delta_layer/eps_layer_rms, already
saved by analyze_delta_structure.py, at the LAST FEW residual-stream layers).

CAVEAT, stated once and left in the figure's reading: `eps_layer_rms` as saved is
mean_a(||eps_a||), not sqrt(mean_a||eps_a||^2) -- see analyze_delta_structure.py's
`e_lay_rms[k, j] = np.sqrt((e**2).sum(-1)).mean()`, sqrt before the mean, not after. The two
coincide only if ||eps_a|| does not vary much across individuals. So the two curves are shown
SEPARATELY, never combined into a single "total" norm the way the toy's exact split allowed.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, OLIVE = "#1B2A4E", "#2F8C7D", "#8A8C30"
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ARMS = [("demo4", "4 layers, with ballast"), ("demo4_noballast", "4 layers, no ballast"),
        ("standard", "8 layers, with ballast"), ("standard_noballast", "8 layers, no ballast")]
N_LAST = 2  # "the last few hidden states": average the last 2 residual-stream depths
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 9.5, "axes.grid": False})


def load(arm):
    acc = [np.load(f) for f in sorted(glob.glob(os.path.join(
        ROOT, "weight_patch", f"{arm}-head-acc-p*-disjoint-seed*.npz")))]
    ds = [np.load(f) for f in sorted(glob.glob(os.path.join(
        ROOT, "delta_structure", f"{arm}-p*-disjoint-t1200-seed*.npz")))]
    return acc, ds


def band(ax, x, Y, color, ls="-", lw=1.8, label=None):
    Y = np.asarray(Y); m = Y.mean(0); sd = Y.std(0, ddof=1) if len(Y) > 1 else 0 * m
    ax.plot(x, m, color=color, ls=ls, lw=lw, marker="o", ms=2.6, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6), dpi=200, sharex="col")
    for c, (arm, title) in enumerate(ARMS):
        acc, ds = load(arm)
        st_acc = acc[0]["steps"].astype(float)
        ax = axes[0, c]
        band(ax, st_acc, [a["acc_t"].mean(-1) for a in acc], NAVY)
        ax.set_title(title, fontsize=10.5); ax.set_ylim(0, 1.04)
        if c == 0: ax.set_ylabel("A, full-vocabulary accuracy")

        st = ds[0]["steps"].astype(float)
        delta_n, eps_n = [], []
        for d in ds:
            dl = d["delta_layer"][:, :, -N_LAST:, :].astype(np.float64)   # (S, attr, N_LAST, E)
            el = d["eps_layer_rms"][:, :, -N_LAST:].astype(np.float64)    # (S, attr, N_LAST)
            delta_n.append(np.linalg.norm(dl, axis=-1).mean(axis=(1, 2)))  # -> (S,)
            eps_n.append(el.mean(axis=(1, 2)))
        ax = axes[1, c]
        band(ax, st, delta_n, TEAL, label="$\\|\\delta\\|$ (coherent, last %d layers)" % N_LAST)
        band(ax, st, eps_n, OLIVE, label="mean$\\|\\varepsilon_a\\|$ (individual)")
        ax.set_ylim(0, None); ax.set_xlabel("injection step")
        if c == 0:
            ax.set_ylabel("hidden-state displacement")
            ax.legend(frameon=False, fontsize=7.5, loc="center right", handlelength=1.6)
        for r in range(2):
            axes[r, c].set_xscale("symlog", linthresh=10); axes[r, c].set_xlim(0, 1300)
    fig.tight_layout()
    out = os.path.join(ROOT, "plots", "toy_plots_working", "hidden_state_timecourse")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print(f"wrote {out}.png")
    for arm, _ in ARMS:
        acc, ds = load(arm)
        st = ds[0]["steps"].astype(float)
        dl = np.mean([d["delta_layer"][:, :, -N_LAST:, :].astype(np.float64) for d in ds], 0)
        el = np.mean([d["eps_layer_rms"][:, :, -N_LAST:].astype(np.float64) for d in ds], 0)
        dn = np.linalg.norm(dl, axis=-1).mean(axis=(1, 2)); en = el.mean(axis=(1, 2))
        pk = int(dn.argmax())
        print(f"{arm:>20}: ||delta|| peak {dn[pk]:.2f}@step{int(st[pk])}, end {dn[-1]:.2f} "
              f"({100*(1-dn[-1]/dn[pk]):.0f}% down from peak) | "
              f"mean||eps|| start {en[1]:.2f} end {en[-1]:.2f} (monotone: "
              f"{'yes' if (np.diff(en) >= -1e-9).all() else 'no'})")


if __name__ == "__main__":
    main()
