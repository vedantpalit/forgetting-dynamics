"""Exact vs knockout: is the transformer's 'de-coherence, not shrinkage' a property of the
model or of the decomposition?

Two per-block decompositions of A's mean shift exist for the MLP-free transformer:
  knockout  P_j = readout(theta_t) - readout(theta_t with block j's OV reverted), post-final-LN
            (dose_coherence/*.npz; the crossover figure and the rotation analysis use this)
  exact     inc_j = delta_layer[j+1] - delta_layer[j], pre-LN residual increments, which sum
            to the layer-8 displacement with no remainder (delta_structure/*.npz)
The knockout pieces sum to only ~half of delta. The exact pieces sum to all of it. This figure
puts the two side by side over the recovery window and reports the de-coherence share of the
fall under each, next to the toy's exact-decomposition share at K = 4 and 8.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE, GREY = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30", "#8A8F98"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 10, "axes.grid": False})
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def recon(n, g):
    return float(np.sqrt((n[:, None] * n[None, :] * g).sum()))


def gram(v):
    n = np.linalg.norm(v, axis=-1)
    u = v / np.clip(n[:, None], 1e-12, None)
    return n, u @ u.T


def series(P):
    """P: (T, 8, A, E) -> per-(T,A): sum||P||, ||sum P||, mean pairwise cos."""
    N = np.linalg.norm(P, axis=-1)                                   # (T,8,A)
    U = P / np.clip(N[..., None], 1e-12, None)
    G = np.einsum("tiae,tjae->taij", U, U)
    iu = np.triu_indices(P.shape[1], 1)
    return N.sum(1), np.linalg.norm(P.sum(1), axis=-1), G[..., iu[0], iu[1]].mean(-1)


def shares(P, i0, i1):
    """de-coherence and norm share of the fall in ||sum P|| from index i0 to i1, per attribute."""
    out = []
    for a in range(P.shape[2]):
        n0, g0 = gram(P[i0, :, a]); n1, g1 = gram(P[i1, :, a])
        A0, A1 = recon(n0, g0), recon(n1, g1); fall = A0 - A1
        out.append(((A0 - recon(n0, g1)) / fall, (A0 - recon(n1, g0)) / fall))
    return np.array(out)


def load_knockout():
    fs = sorted(glob.glob(os.path.join(ROOT, "dose_coherence", "mlp_free-baseline-p16000-disjoint-seed*.npz")))
    out = []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        out.append((d["steps"], d["P"].astype(np.float64)))
    return out


def load_exact():
    fs = sorted(glob.glob(os.path.join(ROOT, "delta_structure", "mlp_free-p16000-disjoint-t1200-seed*.npz")))
    out = []
    for f in fs:
        d = np.load(f)
        inc = np.diff(d["delta_layer"], axis=2)                       # (T, 6, 8, 512)
        out.append((d["steps"], np.transpose(inc, (0, 2, 1, 3))))     # -> (T, 8, 6, 512)
    return out


def main():
    ko, ex = load_knockout(), load_exact()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), dpi=200)
    for (name, runs, col) in (("knockout (OV reverted)", ko, PURP), ("exact (residual increments)", ex, NAVY)):
        st = runs[0][0]
        keep = (st >= 10) & (st <= 400)
        S = np.array([series(P)[0].mean(-1) for _, P in runs])[:, keep]
        Q = np.array([series(P)[1].mean(-1) for _, P in runs])[:, keep]
        C = np.array([series(P)[2].mean(-1) for _, P in runs])[:, keep]
        x = st[keep]
        for ax, Y, lab in ((axes[0], S, "sum_j ||P_j||"), (axes[1], Q, "||sum_j P_j||"), (axes[2], C, "mean pairwise cos")):
            m, sd = Y.mean(0), Y.std(0, ddof=1)
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.15, lw=0)
            ax.plot(x, m, color=col, lw=2, marker="o", ms=3.5, label=name)
            ax.set_xscale("log"); ax.set_xlabel("injection step"); ax.set_ylabel(lab)
            ax.axvspan(50, 200, color="0.5", alpha=0.08, lw=0)
    axes[0].set_title("magnitudes", fontsize=10); axes[1].set_title("the sum", fontsize=10)
    axes[2].set_title("alignment", fontsize=10); axes[2].axhline(0, color="0.6", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    # shares over 50 -> 200
    def sh(runs):
        st = runs[0][0]; i0, i1 = int(np.where(st == 50)[0][0]), int(np.where(st == 200)[0][0])
        v = np.concatenate([shares(P, i0, i1) for _, P in runs])
        return v.mean(0), v.std(0)
    (kd, kn), (ksd, knsd) = sh(ko)
    (ed, en), (esd, ensd) = sh(ex)
    # toy exact-decomposition shares from wp_depth_report (K=4,8: recovery peak ~0.0, long window 0.30 +/- 0.34)
    labels = ["transformer\nknockout", "transformer\nexact", "toy K=8 exact\nat recovery peak", "toy K=4,8 exact\nby step 5000"]
    vals = [kd, ed, 0.00, 0.30]; errs = [ksd, esd, 0.34, 0.34]
    cols = [PURP, NAVY, OLIVE, OLIVE]
    ax = axes[3]
    ax.bar(range(4), vals, yerr=errs, color=cols, capsize=3, width=0.62)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.axhline(0, color="0.6", lw=0.8); ax.axhline(1, color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("de-coherence share of the fall"); ax.set_ylim(-0.4, 1.15)
    ax.set_title("50 -> 200, same identity, two decompositions", fontsize=10)
    fig.tight_layout()
    out = os.path.join(ROOT, "plots", "toy_plots_working", "tf_exact_vs_knockout")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print(f"knockout 50->200: de-coherence {kd:.3f}+/-{ksd:.3f}, norm {kn:.3f}+/-{knsd:.3f}")
    print(f"exact    50->200: de-coherence {ed:.3f}+/-{esd:.3f}, norm {en:.3f}+/-{ensd:.3f}")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
