"""A vs B first-token accuracy during injection, 8-layer standard arm, in the Zucchet style.

Same figure style as `llm/plot_curve.py` and `a_vs_b_zucchet_style.png`: navy A, crimson B,
serif type, boxed spines, no grid, a shaded band on each curve.

THE THREE PHASES ARE DIVIDED BY VERTICAL DOTTED RULES at the measured trough and peak --
(i) the crash, (ii) the recovery, (iii) the erosion -- with the numerals in the headroom above
y=1 rather than inside the plot, so they label the regions without sitting on the curves. The
y-axis is ticked only to 1.0 so that headroom does not read as accuracy above 100%.

WHERE THE RECOVERY ENDS. Not at the argmax. A is flat from step 110 -- 0.922 / 0.925 / 0.926 at
110 / 120 / 130 -- so the argmax at 130 is where the plateau's noise happens to peak, not where
the recovery finishes. The divider is the first step reaching 99% of A's post-trough maximum
(--peak_frac), which lands at 110 and does not chase a seed. --peak_step overrides outright.

Parsed from the injection logs in logs_scale8/ -- the eval line is printed every 10 steps, so
the curve is dense without a checkpoint pass and without new runs.

NOTE ON THE STEP AXIS. The logs print the GLOBAL step counter, carried over from pretraining,
so injection step 0 appears as `[step 16000]`. The offset is taken from the filename's own p<N>
field rather than assumed, because getting it wrong shifts the whole curve instead of erroring.

THE BAND IS THE STANDARD ERROR ACROSS SEEDS, not across items. With five seeds it is narrow
almost everywhere, which is the honest picture rather than a presentational choice -- the
alternative (min-max) would look wider without meaning more.

Run: uv run python -m scripts.plot_scale8_a_vs_b --xmax 800
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = "logs_scale8"
OUT_DIR = "plots"
A_COLOR = "#1B2A4E"
B_COLOR = "#C4245F"
MARK = "#9A9A9A"
# Own-half rank, in a_decomp mode: distinct from both A navy and B crimson.
RANK_COLOR = "#2F7A5A"
# Phase spans sit above the data; the y-axis is ticked only to 1.0.
SPAN_Y = 1.045

STEP = re.compile(r"\[step (\d+)\]")
FIELD = re.compile(r"(\w+)=(-?[\d.]+)")


def parse(path, pretrain_step):
    """{injection step: {"A": {field: value}, "B": {...}}} from the per-eval log line."""
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = STEP.search(line)
        if not m:
            continue
        rec = {}
        for pop, tag in (("A", "dataA:"), ("B", "dataB:"), ("Bal", "ballast:")):
            i = line.find(tag)
            if i < 0:
                continue
            seg = line[i + len(tag):].split(",")[0]
            rec[pop] = {k: float(v) for k, v in FIELD.findall(seg)}
        if "A" in rec:
            rows[int(m.group(1)) - pretrain_step] = rec
    return rows


def phases(acc):
    """(trough index, peak index) -- the crashed point with the largest subsequent rebound.

    Not the global minimum: the curve crashes, recovers, then erodes, so on a long enough
    window the global minimum is the last step. Candidates must be a real crash, 0.05 below
    the running maximum.
    """
    cand = [k for k in range(len(acc)) if acc[k] <= max(acc[:k + 1]) - 0.05]
    if not cand:
        return None, None
    i = max(cand, key=lambda k: max(acc[k:]) - acc[k])
    return i, i + int(np.argmax(acc[i:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=IN_DIR, help="logs_scale8 (standard) or logs_mlpfree (attention-only)")
    ap.add_argument("--prefix", default="scale8", help="log/figure stem: scale8 or mlpfree")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--xmax", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="a_vs_b", choices=["a_vs_b", "a_decomp", "combined", "roles"],
                    help="a_vs_b: A and B first-token accuracy. a_decomp: A's raw accuracy "
                         "against A's own-half top-1 rate -- the suppression signature, where "
                         "accuracy collapses while the correct value keeps winning inside its "
                         "own half")
    ap.add_argument("--no_markers", action="store_true")
    ap.add_argument("--trough_step", type=int, default=None,
                    help="override the (i)|(ii) divider; default is the measured trough")
    ap.add_argument("--peak_step", type=int, default=None,
                    help="override the (ii)|(iii) divider outright")
    ap.add_argument("--peak_frac", type=float, default=0.99,
                    help="the (ii)|(iii) divider is the first step reaching this fraction of "
                         "A's post-trough maximum")
    a = ap.parse_args()

    pat = os.path.join(a.in_dir, f"{a.prefix}_injection.p{a.pretrain_step}.{a.condition}.seed*.out")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no logs matching {pat}")
    runs = []
    for f in files:
        r = parse(f, a.pretrain_step)
        if r:
            runs.append(r)
        else:
            print(f"  WARNING: no eval lines parsed from {os.path.basename(f)}")
    if not runs:
        raise SystemExit("nothing parsed; the eval-line format may have changed")

    steps = [s for s in sorted(set.intersection(*(set(r) for r in runs)))
             if 0 <= s <= a.xmax]

    def series(pop, field):
        try:
            return np.array([[r[s][pop][field] for s in steps] for r in runs])
        except KeyError:
            raise SystemExit(f"{pop}/{field} is not in these logs; available for A: "
                             f"{sorted(runs[0][steps[0]]['A'])}")

    A = series("A", "first_acc")
    C = None
    if a.mode == "a_vs_b":
        B, lab_a, lab_b, col_b = series("B", "first_acc"), "A", "B", B_COLOR
    elif a.mode == "roles":
        # the two old populations by the structural role of their answers: A's values sit in
        # the half the new facts do NOT use, the ballast's in the half they do
        B, lab_a, lab_b, col_b = series("Bal", "first_acc"), \
            "old facts, answers the new facts do not use", \
            "old facts, answers the new facts use", RANK_COLOR
    elif a.mode == "combined":
        # one panel: A's accuracy, A's own-half top-1 and B's accuracy
        B, lab_a, lab_b, col_b = series("B", "first_acc"), "old facts, accuracy", "new facts", B_COLOR
        C = series("A", "rank_own_top1")
    else:
        B = series("A", "rank_own_top1")
        lab_a, lab_b, col_b = "A, accuracy", "A, own-half top-1", RANK_COLOR
    ns = A.shape[0]
    Am = A.mean(0)
    Bm = B.mean(0)
    Ase = A.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(Am)
    Bse = B.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(Bm)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 13,
        "axes.linewidth": 1.0, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)

    curves = [(Am, Ase, A_COLOR, "-", lab_a), (Bm, Bse, col_b, ":", lab_b)]
    if C is not None:
        Cm = C.mean(0); Cse = C.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros_like(Cm)
        curves.insert(1, (Cm, Cse, RANK_COLOR, "--", "old facts, own-half top-1"))
    for m, se, c, ls, lab in curves:
        ax.fill_between(steps, m - se, m + se, color=c, alpha=0.18, linewidth=0)
        ax.plot(steps, m, color=c, linewidth=2.4, linestyle=ls, label=lab, zorder=3)

    tr, pk = phases(list(Am))
    if tr is not None:
        # First step at or after the trough that is within one SE of the maximum -- see the
        # module docstring on why this and not the argmax.
        # The maximum must be taken AFTER the trough: A starts at 1.0, so a whole-window
        # max is the pre-crash baseline and nothing in the recovery ever reaches it.
        post_max = float(Am[tr:].max())
        pk = next(k for k in range(tr, len(steps)) if Am[k] >= a.peak_frac * post_max)
    if not a.no_markers and tr is not None:
        x_tr = a.trough_step if a.trough_step is not None else steps[tr]
        x_pk = a.peak_step if a.peak_step is not None else steps[pk]
        bounds = [0, x_tr, x_pk, a.xmax]
        for x in bounds[1:-1]:
            ax.axvline(x, color=MARK, linestyle=":", linewidth=1.4, zorder=1)
        for (x0, x1), lab in zip(zip(bounds, bounds[1:]), ("(i)", "(ii)", "(iii)")):
            ax.text((x0 + x1) / 2, SPAN_Y, lab, color=MARK, style="italic",
                    fontsize=12, va="bottom", ha="center", clip_on=False)

    ax.set_xlabel("Injection step")
    ax.set_ylabel("First-token accuracy")
    ax.set_xlim(0, a.xmax)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if a.mode == "combined":
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.04),
                  fontsize=12, handlelength=1.6)
    else:
        ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62),
                  fontsize=13, handlelength=1.6)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = a.out or (f"{a.prefix}_p{a.pretrain_step}_"
                     f"{a.mode}_to{a.xmax}")
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"  wrote {p}")

    tr, pk = phases(list(Am))
    print(f"\n  {ns} seeds, {len(steps)} eval points in [0, {a.xmax}]")
    if tr is not None:
        print(f"  A trough {Am[tr]:.3f} @{steps[tr]}  (B {Bm[tr]:.3f})   "
              f"A peak {Am[pk]:.3f} @{steps[pk]}  (B {Bm[pk]:.3f})")
    print(f"  {lab_a} at {steps[-1]}: {Am[-1]:.4f}   {lab_b}: {Bm[-1]:.4f}")
    if tr is not None:
        print(f"  at the trough (step {steps[tr]}): {lab_a} {Am[tr]:.4f}  {lab_b} {Bm[tr]:.4f}")


if __name__ == "__main__":
    main()
