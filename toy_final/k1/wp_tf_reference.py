"""The transformer's per-block recovery signature, as the toy must reproduce it.

THE TARGET, IN ONE SENTENCE. Between the accuracy trough (step 50) and the recovery peak
(step 200) every one of the eight blocks KEEPS the size of its OV-attributable contribution
P_j -- the largest change is 0.20 on norms of 1.1-2.6, and the summed norm moves 12.24 ->
12.08 -- while the mean pairwise cosine among the eight halves, 0.485 -> 0.228, and their
vector sum falls 9.01 -> 6.87. Holding the cosines at their step-50 values and letting only
the norms move reproduces 0% of that fall; holding the norms and letting only the cosines
move reproduces 99% of it. The recovery signature is DE-COHERENCE AT CONSTANT MAGNITUDE.

WHAT P_j IS. P_j = delta_none - delta_(block j's OV reverted to step 0), the A-mean
post-final-LayerNorm readout displacement, per attribute, from `analyze_dose_coherence`
(identical by construction and numerically to `weight_patch`'s `removed`; verified cos 1.000
and equal norms at steps 50 and 200). It is a knockout attribution, NOT the residual-stream
increment block j adds -- that is a different quantity with a different time course
(FINDINGS 3.21.8; every increment SHRINKS to ~0.6x over this window). Panel 3's dotted
curves are the increments' input side, shown so the two are never conflated again.

PANEL 3 REFUTES THE OBVIOUS EXPLANATION. "P_j rotates because its input rotates" predicts the
dotted curve for layer j to track the solid curve for block j and to order the same way with
depth. It does neither: P_j's turn from step 50 to 200 is 59-72 deg and gets LARGER with
depth, while the input displacement's turn is 43-59 deg and gets SMALLER with depth
(r = -0.92 between the two across blocks 1-7; with depth, +0.77 for P_j and -0.88 for the
input). Block 0 is the cleanest case: its input is the embedding
displacement, whose norm is 0.01 -- numerically static, so its plotted angle is noise -- and
P_0 still turns 62 deg. CAVEAT, load-bearing: the dotted curve is the rotation of the input's
DISPLACEMENT, not of the input. Step-0 per-layer states are not saved anywhere on disk, so
the angle between block j's actual input at step 50 and at step 200 CANNOT be computed from
what exists, and is smaller than the dotted curve by an unknown amount.

||delta|| is drawn in panel 2 because it is what actually recovers (18.81 -> 11.81, -37%),
and because it runs ~2x ||sum_j P_j||: the cumulative patch is super-additive, so the
knockout pieces reproduce delta's SHAPE and about half its size. A toy matched on
||sum P_j|| is matched on the right object; a toy matched on ||delta|| is matched on a
number the decomposition does not claim to reconstruct.

Mean +/- sd over the three baseline seeds throughout.

Run: python toy_final/k1/wp_tf_reference.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "plots", "toy_plots_working")
STEM = "tf_reference_decoherence"

C_A = "#1B2A4E"        # navy -- the sum, the output quantity
C_B = "#C4245F"        # crimson -- the mean pairwise cosine
C_D = "#5B3A7A"        # purple -- ||delta||
C_I = "#8A8C30"        # olive -- input-side (residual increment) rotation
XTICKS = [10, 25, 50, 100, 200, 400]
TROUGH, PEAK = 50, 200


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })


def load_P():
    fs = sorted(glob.glob(os.path.join(ROOT, "dose_coherence",
                                       "mlp_free-baseline-p16000-disjoint-seed*.npz")))
    if not fs:
        raise SystemExit("no dose_coherence baseline npz")
    steps = np.load(fs[0], allow_pickle=True)["steps"]
    P = np.stack([np.load(f, allow_pickle=True)["P"].astype(np.float64) for f in fs])
    base = np.stack([np.load(f, allow_pickle=True)["base_norm"].astype(np.float64) for f in fs])
    return np.asarray(steps, float), P, base            # (S,T,8,A,E), (S,T,A)


def load_layer_delta():
    """Per-layer pre-LN A-mean displacement vectors -- the INPUT to each block, as a drift."""
    fs = sorted(glob.glob(os.path.join(ROOT, "delta_structure",
                                       "mlp_free-p16000-disjoint-t1200-seed*.npz")))
    if not fs:
        return None, None
    steps = np.load(fs[0], allow_pickle=True)["steps"]
    DL = np.stack([np.load(f, allow_pickle=True)["delta_layer"].astype(np.float64) for f in fs])
    return np.asarray(steps, float), DL                 # (S,T,A,9,E)


def ang(a, b):
    ua = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    ub = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip((ua * ub).sum(-1), -1.0, 1.0)))


def main():
    steps, P, base = load_P()
    S, T, B, A, E = P.shape
    sd = lambda x: x.std(0, ddof=1) if S > 1 else np.zeros_like(x.mean(0))

    N = np.linalg.norm(P, axis=-1)                      # (S,T,8,A)
    NB = N.mean(-1)                                     # (S,T,8) attribute-averaged
    U = P / np.clip(N[..., None], 1e-12, None)
    G = np.einsum("stiae,stjae->staij", U, U)
    iu = np.triu_indices(B, 1)
    cos = G[..., iu[0], iu[1]].mean(-1).mean(-1)        # (S,T)
    nsum = np.linalg.norm(P.sum(2), axis=-1).mean(-1)   # (S,T)
    dn = base.mean(-1)                                  # (S,T)
    i50 = int(np.where(steps == TROUGH)[0][0])
    rot = np.stack([[ang(P[:, i, j], P[:, i50, j]).mean(-1) for j in range(B)]
                    for i in range(T)], axis=0).transpose(2, 0, 1)     # (S,T,8)

    lsteps, DL = load_layer_delta()
    if DL is not None:
        keep = [int(np.where(lsteps == s)[0][0]) for s in steps if s in lsteps]
        ls = lsteps[keep]
        j50 = int(np.where(lsteps == TROUGH)[0][0])
        irot = np.stack([[ang(DL[:, i, :, l], DL[:, j50, :, l]).mean(-1) for l in range(8)]
                         for i in keep], axis=0).transpose(2, 0, 1)    # (S,len,8)

    style()
    cmap = plt.get_cmap("plasma")
    col = [cmap(0.08 + 0.80 * j / (B - 1)) for j in range(B)]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), dpi=200)

    # ---- panel 1: per-block magnitudes, and their sum
    ax = axes[0]
    for j in range(B):
        m, e = NB[:, :, j].mean(0), sd(NB[:, :, j])
        ax.fill_between(steps, m - e, m + e, color=col[j], alpha=0.18, lw=0)
        ax.plot(steps, m, color=col[j], lw=1.6, marker="o", ms=3.2, label=f"block {j}")
    m, e = NB.sum(-1).mean(0), sd(NB.sum(-1))
    ax.fill_between(steps, m - e, m + e, color="black", alpha=0.15, lw=0)
    ax.plot(steps, m, color="black", lw=2.4, marker="s", ms=4.0, label=r"$\sum_j\|P_j\|$")
    ax.set_ylabel(r"$\|P_j\|$   (black: $\sum_j\|P_j\|$)")
    ax.set_ylim(0, max(m + e) * 1.06)
    ax.set_title("magnitudes: nothing backs off", fontsize=11.5)
    ax.legend(fontsize=7.2, ncol=2, frameon=False, loc="center right",
              bbox_to_anchor=(1.0, 0.42))

    # ---- panel 2: coherence, the sum, and delta
    ax = axes[1]
    axr = ax.twinx()
    for arr, c, lab, mk in ((nsum, C_A, r"$\|\sum_j P_j\|$", "o"),
                            (dn, C_D, r"$\|\delta\|$", "^")):
        m, e = arr.mean(0), sd(arr)
        ax.fill_between(steps, m - e, m + e, color=c, alpha=0.20, lw=0)
        ax.plot(steps, m, color=c, lw=2.2, marker=mk, ms=4.5, label=lab)
    m, e = cos.mean(0), sd(cos)
    axr.fill_between(steps, m - e, m + e, color=C_B, alpha=0.20, lw=0)
    axr.plot(steps, m, color=C_B, lw=2.2, ls="--", marker="s", ms=4.5,
             label="mean pairwise $\\cos(P_i,P_j)$")
    ax.set_ylabel("norm", color=C_A)
    ax.tick_params(axis="y", colors=C_A)
    axr.set_ylabel("mean pairwise cosine", color=C_B)
    axr.tick_params(axis="y", colors=C_B)
    axr.set_ylim(0, 0.8)
    ax.set_ylim(0, 21)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="lower left")
    ax.set_title("the sum falls because the cosines do", fontsize=11.5)

    # ---- panel 3: rotation from the trough
    ax = axes[2]
    for j in range(B):
        m, e = rot[:, :, j].mean(0), sd(rot[:, :, j])
        ax.fill_between(steps, m - e, m + e, color=col[j], alpha=0.18, lw=0)
        ax.plot(steps, m, color=col[j], lw=1.6, marker="o", ms=3.2)
    if DL is not None:
        for j in range(B):
            ax.plot(ls, irot[:, :, j].mean(0), color=col[j], lw=1.1, ls=":", alpha=0.85)
        ax.plot([], [], color=C_I, lw=1.1, ls=":", label="input drift  $\\angle(\\Delta h_j)$")
    ax.plot([], [], color=C_I, lw=1.6, ls="-", label="$\\angle(P_j(t),P_j(50))$")
    ax.set_ylabel("angle from step 50 (deg)")
    ax.set_ylim(0, 95)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("rotation, solid $P_j$ / dotted its input", fontsize=11.5)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(steps.min(), steps.max())
        ax.set_xlabel("injection step")
        for x in (TROUGH, PEAK):
            ax.axvline(x, color="#9A9A9A", ls=":", lw=1.1, zorder=0)

    fig.suptitle("Transformer reference: what the toy's per-block signature must match "
                 "(MLP-free, p16000, disjoint, 3 seeds)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{STEM}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)

    print("\nTARGET NUMBERS (mean +/- sd over 3 seeds)")
    print(f"  {'step':>5} {'sum||P_j||':>16} {'||sum P_j||':>16} {'||delta||':>16} {'mean cos':>16}")
    for i, s in enumerate(steps):
        print(f"  {s:>5.0f} {NB.sum(-1)[:, i].mean():>9.3f}+/-{sd(NB.sum(-1))[i]:<6.3f} "
              f"{nsum[:, i].mean():>9.3f}+/-{sd(nsum)[i]:<6.3f} "
              f"{dn[:, i].mean():>9.3f}+/-{sd(dn)[i]:<6.3f} "
              f"{cos[:, i].mean():>9.4f}+/-{sd(cos)[i]:<6.4f}")


if __name__ == "__main__":
    main()
