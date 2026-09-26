"""Figures for the gate-matched sweeps.

S1  one row per axis, three columns: crash depth, recovery, and how far the collective write
    falls back. Every cell is read with B's acquisition matched to the same step, so columns
    are comparable -- which the earlier unmatched sweep was not.

S2  the scaling-law test. If the permanent damage depends only on how loaded the store is,
    then points from the d sweep (which shrinks the store) and the load sweep (which grows
    the population) must fall on ONE curve when plotted against (n_A + n_D + n_B) / d.
    They are different experiments; agreeing is a real claim, not a re-plot.

S3  the dose sweep, which is deliberately NOT gate matched. Plotted against B's accuracy
    rather than the step, the curves should collapse: that is what "the crash runs on B's
    clock" means, and it is the toy version of the transformer's dose result.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 10, "axes.grid": False})
LABEL = {"d": "key dimension $d$", "nb": "number injected  $n_B$",
         "load": "pretrained population  $n_A = n_D$", "vocab": "vocabulary  $|V|$",
         "beta": "shared key fraction  $\\beta$", "dose": "injection-rate multiplier"}
LOGX = {"d", "nb", "load", "vocab"}


def summarise(r):
    rows = r["rows"]
    st = np.array([x["step"] for x in rows])
    A = np.array([x["A"] for x in rows]); B = np.array([x["B"] for x in rows])
    sw = np.array([x["s_on_w"] for x in rows])
    ep = np.array([x["eps"] for x in rows]); pn = np.array([x["post_norm"] for x in rows])
    tr = int(A.argmin()); pw = int(sw.argmax())
    return dict(crash=1 - A[tr], recovery=A[tr:].max() - A[tr], end=A[-1],
                drop=(1 - sw[-1] / sw[pw]) if sw[pw] > 0.05 else np.nan,
                eps=ep[-1] / pn[-1], trough_step=st[tr],
                bgate=st[np.argmax(B >= 0.99)] if (B >= 0.99).any() else np.nan,
                load=(r["na"] + r["nd"] + r["nb"]) / r["d"])


def cells(dat):
    """value -> norm -> list of per-seed summaries."""
    out = {}
    for r in dat.values():
        out.setdefault(r["value"], {}).setdefault(r["norm"], []).append(summarise(r))
    return out


def agg(c, val, norm, key):
    v = [x[key] for x in c.get(val, {}).get(norm, [])]
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return np.nan, np.nan
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def fig_grid(files):
    axes_names = [s for s in ("d", "nb", "load", "vocab", "beta") if s in files]
    if not axes_names:
        return
    fig, axes = plt.subplots(len(axes_names), 3, squeeze=False,
                             figsize=(11.4, 2.7 * len(axes_names)), dpi=200)
    for r, s in enumerate(axes_names):
        c = cells(files[s])
        vals = sorted(c)
        for col, (key, lab, color) in enumerate(
                ((("crash"), "crash depth", NAVY),
                 (("recovery"), "recovery", OLIVE),
                 (("drop"), "fall in the write", PURP))):
            ax = axes[r, col]
            for norm, ls, mk, nm in ((1, "-", "o", "normalized"), (0, "--", "s", "linear")):
                M = np.array([agg(c, v, norm, key)[0] for v in vals])
                S = np.array([agg(c, v, norm, key)[1] for v in vals])
                ok = np.isfinite(M)
                if ok.any():
                    ax.errorbar(np.array(vals)[ok], M[ok], yerr=S[ok], color=color, ls=ls,
                                marker=mk, ms=4, lw=1.7, capsize=2.5, label=nm)
            if s in LOGX:
                ax.set_xscale("log"); ax.minorticks_off()
                ax.set_xticks(vals); ax.set_xticklabels([f"{v:g}" for v in vals])
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(LABEL[s])
            if col == 0:
                ax.set_ylabel(lab)
            else:
                ax.set_ylabel(lab)
            if r == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_sweeps.{e}", bbox_inches="tight")
    print("wrote plots/toy_k1_sweeps.png")


def fig_collapse(files):
    """Does the permanent damage depend only on the load n/d?"""
    have = [s for s in ("d", "load", "nb") if s in files]
    if len(have) < 2:
        print("skipping collapse figure: need at least two of d / load / nb")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=200)
    marks = {"d": ("o", "vary $d$ (shrink the store)"),
             "load": ("s", "vary $n_A$ (grow the population)"),
             "nb": ("^", "vary $n_B$ (grow the injection)")}
    for ax, key, lab in ((axes[0], "eps", "individual component at the end\n"
                                          r"$\|\epsilon\| / \|$state$\|$"),
                         (axes[1], "end", "A's accuracy at the end")):
        for s in have:
            c = cells(files[s])
            mk, nm = marks[s]
            for norm, col, fill in ((1, PURP, "full"), (0, NAVY, "none")):
                xs, ys, es = [], [], []
                for v in sorted(c):
                    ld = np.mean([x["load"] for x in c[v].get(norm, [])]) if c[v].get(norm) else np.nan
                    m, sd = agg(c, v, norm, key)
                    if np.isfinite(m) and np.isfinite(ld):
                        xs.append(ld); ys.append(m); es.append(sd)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, ls="none", marker=mk, ms=6, color=col,
                                mfc=("none" if fill == "none" else col), capsize=2.5,
                                label=f"{nm}, {'normalized' if norm else 'linear'}")
        ax.set_xscale("log")
        ax.set_xlabel(r"load   $(n_A + n_D + n_B) / d$")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_scaling.{e}", bbox_inches="tight")
    print("wrote plots/toy_k1_scaling.png")


def fig_dose(files):
    if "dose" not in files:
        return
    dat = files["dose"]
    doses = sorted({r["value"] for r in dat.values()})
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(doses)))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), dpi=200, sharey=True)
    for i, dv in enumerate(doses):
        for ax, xkey, xlab in ((axes[0], "step", "injection step"),
                               (axes[1], "B", "B's accuracy  (B's clock)")):
            rs = [r for r in dat.values() if r["value"] == dv and r["norm"] == 1]
            if not rs:
                continue
            n = min(len(r["rows"]) for r in rs)
            A = np.array([[x["A"] for x in r["rows"][:n]] for r in rs]).mean(0)
            if xkey == "step":
                x = np.array([x["step"] for x in rs[0]["rows"][:n]], float)
                ax.set_xscale("symlog", linthresh=10)
            else:
                x = np.array([[x["B"] for x in r["rows"][:n]] for r in rs]).mean(0)
            ax.plot(x, A, color=cmap[i], lw=1.9, label=f"{dv:g}x")
            ax.set_xlabel(xlab)
    axes[0].set_ylabel("A's accuracy")
    axes[0].legend(frameon=False, fontsize=8, title="injection rate", title_fontsize=8)
    axes[0].set_title("against the step: the curves separate", fontsize=10)
    axes[1].set_title("against B's accuracy: they collapse", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_dose.{e}", bbox_inches="tight")
    print("wrote plots/toy_k1_dose.png")


def main():
    files = {}
    for f in sorted(glob.glob("sw_*.json")):
        s = os.path.basename(f)[3:-5]
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            print(f"  {f} not readable yet, skipping"); continue
        if d:
            files[s] = d
    print("loaded sweeps:", sorted(files))
    os.makedirs("../../plots", exist_ok=True)
    fig_grid(files); fig_collapse(files); fig_dose(files)

    for s in sorted(files):
        c = cells(files[s])
        print(f"\n=== sweep: {s} ({LABEL[s]}) ===")
        print(f"{'value':>8} {'load':>6} {'arm':>11} {'Bgate':>6} | {'crash':>12} "
              f"| {'recovery':>12} | {'write falls':>12} | {'eps/state':>10} | {'A end':>10}")
        for v in sorted(c):
            for norm, nm in ((1, "normalized"), (0, "linear")):
                if norm not in c[v]:
                    continue
                ld = np.mean([x["load"] for x in c[v][norm]])
                bg = np.nanmean([x["bgate"] for x in c[v][norm]])
                out = [agg(c, v, norm, k) for k in ("crash", "recovery", "drop", "eps", "end")]
                w = "        n/a" if not np.isfinite(out[2][0]) else f"{100*out[2][0]:>7.0f}+/-{100*out[2][1]:<4.0f}"
                print(f"{v:>8g} {ld:>6.2f} {nm:>11} {bg:>6.0f} | "
                      f"{out[0][0]:>6.2f}+/-{out[0][1]:<5.2f} | {out[1][0]:>6.2f}+/-{out[1][1]:<5.2f} "
                      f"| {w:>12} | {out[3][0]:>5.3f}+/-{out[3][1]:<4.3f} | {out[4][0]:>5.2f}+/-{out[4][1]:<4.2f}")


if __name__ == "__main__":
    main()
