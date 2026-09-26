"""The dimension sweep: it separates the two damage channels.

The collective channel (one shared write, reversible) lives along a single direction that
exists at every d, so nothing about it should depend on d. The individual channel (crosstalk
between keys that cannot all be orthogonal, not reversible) is a finite-capacity effect and
should fall off as the load n/d does.

beta = 0.5 isolates "both channels"; beta = 0 isolates the individual channel alone, since
with no shared key component there is no collective write to make.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, CRIM, PURP, OLIVE = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30"
SEEDS = (0, 1, 2)
NTOT = 150          # n_A + n_D + n_B, the number of keys the store must hold
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 10, "axes.grid": False})


def load():
    out = {}
    for f in sorted(glob.glob("d_sweep_*.json")):
        d = int(re.search(r"d_sweep_(\d+)\.json", f).group(1))
        out[d] = json.load(open(f))
    return out


def band(ax, x, Y, color, ls="-", lw=2.0, label=None):
    m, sd = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
    return m


def rows(dat, d, beta, norm, s):
    return dat[d][f"b{beta}-n{norm}-g{norm and 1.0 or 1.0}-u0.01-s{s}"]["rows"]


def stat(dat, d, beta, norm):
    """Index 5 is A read at the step B first reaches 0.99 -- A's damage ON B'S CLOCK. The
    learning-rate probe picks a different rate per d and per arm, so the step axis is not
    comparable across the sweep; this readout is."""
    vals = []
    for s in SEEDS:
        r = rows(dat, d, beta, norm, s)
        A = np.array([x["A"] for x in r])
        B = np.array([x["B"] for x in r])
        sw = np.array([x["s_on_w"] for x in r])
        ep = np.array([x["eps"] for x in r])
        pn = np.array([x["post_norm"] for x in r])
        tr = int(A.argmin()); pw = int(sw.argmax())
        drop = (1 - sw[-1] / sw[pw]) if sw[pw] > 0.05 else np.nan
        at_gate = float(A[int(np.argmax(B >= 0.99))]) if (B >= 0.99).any() else np.nan
        vals.append((1 - A[tr], A[tr:].max() - A[tr], drop, 1 - A[-1],
                     ep[-1] / pn[-1], at_gate))
    v = np.array(vals)
    return np.nanmean(v, 0), np.nanstd(v, 0, ddof=1)


def main():
    dat = load()
    ds = sorted(dat)
    print("loaded d =", ds)
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), dpi=200)

    shown = [d for d in (32, 128, 512) if d in dat] or ds[:3]
    for c, d in enumerate(shown):
        ax = axes[0, c]
        for norm, ls, lab in ((1, "-", "normalized"), (0, "--", "linear")):
            r0 = rows(dat, d, 0.5, norm, 0)
            st = np.array([x["step"] for x in r0], float)
            A = np.array([[x["A"] for x in rows(dat, d, 0.5, norm, s)] for s in SEEDS])
            B = np.array([[x["B"] for x in rows(dat, d, 0.5, norm, s)] for s in SEEDS])
            band(ax, st, A, NAVY, ls, label=f"A, {lab}")
            band(ax, st, B, CRIM, ls, lw=1.3, label=f"B, {lab}")
        ax.set_xscale("symlog", linthresh=10); ax.set_ylim(0, 1.03)
        ax.set_title(f"d = {d}   (load n/d = {NTOT / d:.1f})", fontsize=10.5)
        ax.set_xlabel("injection step")
        if c == 0:
            ax.set_ylabel("accuracy   ($\\beta$ = 0.5)")
            ax.legend(frameon=False, fontsize=7.5, loc="center left")

    def summary(ax, idx, beta, label, color, ylim=None):
        for norm, ls, mk, lab in ((1, "-", "o", "normalized readout"),
                                  (0, "--", "s", "linear (no norm)")):
            M = np.array([stat(dat, d, beta, norm)[0][idx] for d in ds])
            S = np.array([stat(dat, d, beta, norm)[1][idx] for d in ds])
            ok = ~np.isnan(M)
            ax.errorbar(np.array(ds)[ok], M[ok], yerr=S[ok], color=color, ls=ls, marker=mk,
                        ms=4, lw=1.8, capsize=2.5, label=lab)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(ds); ax.set_xticklabels(ds)
        ax.set_xlabel("key dimension $d$")
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)

    summary(axes[1, 0], 0, 0.5, "crash depth  $1-A_{\\mathrm{trough}}$", NAVY, (-0.05, 1.05))
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1, 0].set_title("crash: shrinks with $d$ (dose not matched)", fontsize=10)

    summary(axes[1, 1], 1, 0.5, "recovery  $A_{\\mathrm{after}}-A_{\\mathrm{trough}}$",
            OLIVE, (-0.05, 1.05))
    axes[1, 1].set_title("recovery: normalizer only, and needs a crash", fontsize=10)

    summary(axes[1, 2], 3, 0.0, "damage at $\\beta$ = 0\n$1-A_{\\mathrm{end}}$", PURP,
            (-0.05, 1.05))
    # accuracy is measured on 50 individuals, so anything under 0.02 is one item.
    axes[1, 2].axhline(1 / 50, color="0.6", lw=0.9, ls=":")
    axes[1, 2].annotate("one individual", (0.52, 0.045), xycoords="axes fraction",
                        fontsize=7.5, color="0.45")
    axes[1, 2].set_title("individual channel: gone by $d$ = 128", fontsize=10)

    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"../../plots/toy_k1_d_sweep.{e}", bbox_inches="tight")
    print("wrote plots/toy_k1_d_sweep.png")

    hdr = (f"{'d':>5} {'n/d':>5} {'B gate':>7} | {'crash (b=.5)':>18} | {'recovery':>18} | "
           f"{'A at B gate':>18} | {'damage b=0':>18}")
    for norm, nm in ((1, "normalized"), (0, "linear")):
        print(f"\n  --- {nm} readout ---\n{hdr}")
        for d in ds:
            m, s = stat(dat, d, 0.5, norm)
            m0, s0 = stat(dat, d, 0.0, norm)
            bg = np.mean([next((x["step"] for x in rows(dat, d, 0.5, norm, sd)
                                if x["B"] >= 0.99), np.nan) for sd in SEEDS])
            print(f"{d:>5} {NTOT/d:>5.1f} {bg:>7.0f} | {m[0]:>7.2f} +/- {s[0]:<7.2f} | "
                  f"{m[1]:>7.2f} +/- {s[1]:<7.2f} | {m[5]:>7.2f} +/- {s[5]:<7.2f} | "
                  f"{m0[3]:>7.3f} +/- {s0[3]:<7.3f}")


if __name__ == "__main__":
    main()
