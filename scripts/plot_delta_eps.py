"""The common and individual parts of A's residual displacement (paper Section 5, move two).

Delta_h_a splits, arithmetically, into a common part delta (the mean over A) and an individual
remainder eps_a. Substituting into the logit shift gives

    Dz_a[v] = <delta, u_v> + <eps_a, u_v>

whose first term is identical for every individual and so cannot reorder values within a region,
while the second is individual-specific and can. This figure shows the two components then also
BEHAVE differently, which the arithmetic does not guarantee.

LEFT -- time course. ||delta|| rises to 18.81 at the accuracy trough and falls 37% by step 200;
rms||eps|| climbs monotonically to 17.6 and never turns over. They cross inside the recovery
window, so the panel carries the ratio as well as the two magnitudes.

RIGHT -- subspace. The share of each component's squared norm lying in the top 20 principal
directions of A's step-0 readout representations, against TWO reference lines without which the
numbers mean nothing:

    0.292   a direction drawn from the representations' own covariance
            (those 20 PCs explain 29.2% of A's step-0 readout variance)
    0.039   a uniformly random direction in 512 dimensions (= 20/512)

eps sits at or above the covariance line throughout: it lives inside the structure A's
representations already had. delta sits below it at every step and moves further below.

PHASE MARKERS ARE THIS ARM'S OWN. The MLP-free trough is step 50 and its recovery peak is step
200 (accuracy 1.000 -> 0.119 -> 0.539), not the standard arm's 30 and 110. Using the other arm's
phases here would misplace both rules by a factor of two.

BOTH SHARES ARE EXACT. delta_structure now saves delta_share and eps_share -- mean over
individuals of the squared projections, over the mean squared norm -- so neither curve rests on
a distributional assumption. Older npz lacking those fields fall back to reconstructing the eps
fraction from eps_pc, which stores the mean ABSOLUTE projection and so needs a 2/pi Gaussian
factor; that path was ~1.6% high at the trough (0.344 against 0.338). The script prints which
source it used, so an approximate figure cannot be mistaken for an exact one.

Step 0 is excluded: both components are identically zero there and the subspace fraction is 0/0.

Run: uv run python -m scripts.plot_delta_eps
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots"
# Deliberately NOT the navy/crimson of the A-vs-B figures. Those two colours mean "population
# A" and "population B" everywhere else in the paper, and delta/eps are components of A's
# displacement, not populations -- reusing the pair would invite exactly the wrong reading.
# Deep purple / olive, from the palette sheet, is also clear of the crossover panel's
# navy / grey-blue / green, so no colour in the paper stands for two different objects.
C_DELTA = "#5B3A7A"      # deep purple -- the common component
C_EPS = "#8A8C30"        # olive -- the individual component
MARK = "#9A9A9A"
BENCH = "#6E7A88"

TROUGH_STEP, RECOVERY_END = 50, 200
# 800 is dropped: on a log axis its label collides with 1200.
XTICKS = [10, 25, 50, 100, 200, 400, 1200]
GAUSS = 2.0 / np.pi          # see the caveat in the module docstring
# Stopgap source, deleted once the npz carry the shares themselves.
LOG_SHARES = os.path.join("delta_structure", "exact_shares_from_logs.json")


def load():
    rs = sorted(glob.glob("residual_split/mlp_free-p16000-disjoint-t1200-seed*.npz"))
    ds = sorted(glob.glob("delta_structure/mlp_free-p16000-disjoint-t1200-seed*.npz"))
    if not rs or not ds:
        raise SystemExit("need residual_split/ and delta_structure/ npz")
    st = np.load(rs[0], allow_pickle=True)["steps"]
    D = np.stack([np.linalg.norm(np.load(f, allow_pickle=True)["delta"], axis=-1) for f in rs])
    E = np.stack([np.load(f, allow_pickle=True)["eps_rms"] for f in rs])
    dpc = np.stack([np.load(f, allow_pickle=True)["delta_pc"] for f in ds])
    epc = np.stack([np.load(f, allow_pickle=True)["eps_pc"] for f in ds])
    ev = np.stack([np.load(f, allow_pickle=True)["explained_var"] for f in ds])
    d0 = np.load(ds[0], allow_pickle=True)
    if "eps_share" in d0.files:
        # Exact: mean over individuals of the squared projections, over the mean squared norm.
        # No distributional assumption.
        fd = np.stack([np.load(f, allow_pickle=True)["delta_share"] for f in ds])
        fe = np.stack([np.load(f, allow_pickle=True)["eps_share"] for f in ds])
        exact = True
    elif os.path.exists(LOG_SHARES):
        # Stopgap: the exact shares transcribed from the run logs, where the script printed
        # them without saving them. Fewer seeds than the npz, and it disappears the moment the
        # re-run lands, but it is the exact quantity rather than a modelled one.
        import json
        j = json.load(open(LOG_SHARES, encoding="utf-8"))
        if [int(x) for x in j["steps"]] != [int(x) for x in st]:
            raise SystemExit(f"{LOG_SHARES} step grid does not match the npz")
        ks = sorted(j["delta_share"])
        fd = np.array([j["delta_share"][k] for k in ks])[:, :, None]
        fe = np.array([j["eps_share"][k] for k in ks])[:, :, None]
        exact = "logs"
    else:
        # Fallback for npz written before delta_share/eps_share were saved: eps_pc holds the
        # mean ABSOLUTE projection, so an energy fraction needs the 2/pi Gaussian factor.
        fd = (dpc ** 2).sum(-1) / np.clip(D ** 2, 1e-12, None)          # (S,T,A)
        fe = (epc ** 2).sum(-1) / np.clip(E ** 2, 1e-12, None) / GAUSS
        exact = False
    return (np.array(st, float), D.mean(-1), E.mean(-1), fd.mean(-1), fe.mean(-1),
            float(ev.mean(0).sum(1).mean()), ev.shape[-1], exact)


def crossing(x, a, b):
    """Where a - b changes sign, interpolated in log x."""
    d = a - b
    k = next((i for i in range(1, len(d)) if d[i - 1] > 0 >= d[i]), None)
    if k is None:
        return None
    t = d[k - 1] / (d[k - 1] - d[k])
    lx = np.log(x[k - 1]) + t * (np.log(x[k]) - np.log(x[k - 1]))
    return float(np.exp(lx)), float(a[k - 1] + t * (a[k] - a[k - 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out", default="delta_eps")
    ap.add_argument("--left_only", action="store_true",
                    help="the time course alone, with the phase designations (i)/(ii)/(iii) at "
                         "the trough and recovery end measured on the accuracy curve, as in "
                         "the mlpfree_p16000_a_vs_b figure")
    ap.add_argument("--trough_step", type=int, default=60)
    ap.add_argument("--peak_step", type=int, default=190)
    a = ap.parse_args()
    if a.left_only:
        return left_only(a)

    st, D, E, FD, FE, bench, npc, exact = load()
    keep = st > 0
    st, D, E, FD, FE = st[keep], D[:, keep], E[:, keep], FD[:, keep], FE[:, keep]
    ns = D.shape[0]
    sem = lambda x: x.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(x.mean(0))
    rnd = npc / 512.0

    print(f"{ns} seeds; top-{npc} PCs; covariance benchmark {bench:.4f}, random floor {rnd:.4f}")
    print("  subspace shares: " + {True: "EXACT, from the npz",
          "logs": f"EXACT, transcribed from the run logs ({LOG_SHARES})",
          False: "APPROXIMATE -- eps uses the 2/pi Gaussian factor"}[exact])
    print(f"  {'step':>6} {'||delta||':>10} {'rms||eps||':>11} {'ratio':>7} "
          f"{'delta subsp':>12} {'eps subsp':>10}")
    for i, s in enumerate(st):
        print(f"  {s:>6.0f} {D[:, i].mean():>10.3f} {E[:, i].mean():>11.3f} "
              f"{D[:, i].mean()/E[:, i].mean():>7.2f} {FD[:, i].mean():>12.4f} "
              f"{FE[:, i].mean():>10.4f}")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.5), dpi=200)
    fig.subplots_adjust(wspace=0.28)

    for ax in (axL, axR):
        for x in (TROUGH_STEP, RECOVERY_END):
            ax.axvline(x, color=MARK, ls=":", lw=1.3, zorder=0)
        if not a.linear:
            ax.set_xscale("log")
            ax.set_xticks(XTICKS)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(st.min(), st.max())
        ax.set_xlabel("Injection step")

    # --- left: time course -------------------------------------------------------------
    for Y, c, lab in ((D, C_DELTA, r"$\|\delta\|$  (common)"),
                      (E, C_EPS, r"rms$\,\|\varepsilon_a\|$  (individual)")):
        m, se = Y.mean(0), sem(Y)
        axL.fill_between(st, m - se, m + se, color=c, alpha=0.20, lw=0)
        axL.plot(st, m, color=c, lw=2.6, label=lab, zorder=3)
    cr = crossing(st, D.mean(0), E.mean(0))
    if cr:
        axL.plot([cr[0]], [cr[1]], marker="o", ms=8, mfc="white", mec=MARK, mew=1.8, zorder=4)
        axL.annotate(f"cross at step {cr[0]:.0f}", xy=cr, xytext=(16, 20),
                     textcoords="offset points", fontsize=10.5, color=MARK, style="italic")
    axL.set_ylabel("Displacement magnitude")
    axL.set_ylim(0, max((D.mean(0) + sem(D)).max(), (E.mean(0) + sem(E)).max()) * 1.22)
    axL.legend(frameon=False, fontsize=11.5, loc="lower right", handlelength=1.8)
    axL.text(TROUGH_STEP, axL.get_ylim()[1] * 0.985, "trough", color=MARK, style="italic",
             fontsize=10, ha="center", va="top")
    axL.text(RECOVERY_END, axL.get_ylim()[1] * 0.985, "recovery end", color=MARK,
             style="italic", fontsize=10, ha="center", va="top")

    # --- right: subspace ---------------------------------------------------------------
    # The reference lines are what make the numbers mean anything, so they are drawn first
    # and labelled on the axis rather than left to the caption.
    for y, lab in ((bench, "own covariance"), (rnd, "random direction")):
        axR.axhline(y, color=BENCH, ls="--", lw=1.4, zorder=1)
        axR.text(st.max() * 0.97, y + 0.008, lab, color=BENCH, fontsize=10,
                 ha="right", va="bottom")
    for Y, c, lab in ((FD, C_DELTA, r"$\delta$"), (FE, C_EPS, r"$\varepsilon_a$")):
        m, se = Y.mean(0), sem(Y)
        axR.fill_between(st, m - se, m + se, color=c, alpha=0.20, lw=0)
        axR.plot(st, m, color=c, lw=2.6, label=lab, zorder=3)
    axR.set_ylabel(f"share of squared norm in A's top {npc} PCs")
    axR.set_ylim(0, 0.42)
    axR.legend(frameon=False, fontsize=11.5, loc="lower left", handlelength=1.8)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{a.out}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")


def left_only(a):
    """One panel: ||delta|| and rms||eps_a||, phase boundaries and numerals synced with the
    accuracy figure of the same arm (mlpfree_p16000_a_vs_b: trough 60, recovery end 190)."""
    st, D, E, *_ = load()
    keep = st > 0
    st, D, E = st[keep], D[:, keep], E[:, keep]
    ns = D.shape[0]
    sem = lambda x: x.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(x.mean(0))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(5.6, 4.5), dpi=200)
    for Y, c, lab in ((D, C_DELTA, r"$\|\delta\|$  (common)"),
                      (E, C_EPS, r"rms$\,\|\varepsilon_a\|$  (individual)")):
        m, se = Y.mean(0), sem(Y)
        ax.fill_between(st, m - se, m + se, color=c, alpha=0.20, lw=0)
        ax.plot(st, m, color=c, lw=2.6, label=lab, zorder=3)
    top = max((D.mean(0) + sem(D)).max(), (E.mean(0) + sem(E)).max())
    ax.set_ylim(0, top * 1.14)
    if not a.linear:
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(st.min(), st.max())
    # phase boundaries and numerals, the same designations as the accuracy figure
    bounds = [st.min(), a.trough_step, a.peak_step, st.max()]
    for x in bounds[1:-1]:
        ax.axvline(x, color=MARK, ls=":", lw=1.4, zorder=1)
    mid = (lambda x0, x1: (x0 + x1) / 2) if a.linear else (lambda x0, x1: np.sqrt(x0 * x1))
    for (x0, x1), lab in zip(zip(bounds, bounds[1:]), ("(i)", "(ii)", "(iii)")):
        ax.text(mid(x0, x1), top * 1.14 * 1.01, lab, color=MARK, style="italic",
                fontsize=12, va="bottom", ha="center", clip_on=False)
    ax.set_xlabel("Injection step")
    ax.set_ylabel("Displacement magnitude")
    ax.legend(frameon=False, fontsize=11.5, loc="lower right", handlelength=1.8)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{a.out}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
