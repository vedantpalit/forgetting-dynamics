"""Tables and figures for the drift-vs-write decomposition of the K-stack toy.

Reads wp_drift_K{2,4,8}.npz (written by wp_drift_stack.py) and answers, per block and per
depth: is the change in a block's A-mean contribution P_j driven by its INPUT moving (upstream
blocks having rewritten the residual stream) or by its OWN weights moving?

    python wp_drift_report.py
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "axes.grid": False, "figure.dpi": 200, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, CRIM, PURP, OLIV = "#1B2A4E", "#C4245F", "#5B3A7A", "#8A8C30"
OUT = "plots/toy_plots_working"
KS = [2, 4, 8]
SEEDS = [0, 1, 2]


def ang(a, b):
    c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


CRASH_WINDOW = 300          # the dense checkpoint window; every delta peak falls inside it


def load(K):
    """`pk` (the delta peak) is taken as stored -- it is unambiguous and is the checkpoint the
    counterfactuals are frozen at. `tr`/`rp` are RECOMPUTED here: the stored trough was a plain
    argmin over the whole run, which on a seed that never recovers picks up the slow late decay
    instead of the crash. The trough is the minimum inside the crash window; the recovery peak is
    A's maximum at or after it. A seed with no recovery then shows rp ~ tr, which is honest."""
    z = np.load(f"wp_drift_K{K}.npz")
    out = {}
    for s in SEEDS:
        r = {k.split("/")[1]: z[k] for k in z.files if k.startswith(f"s{s}/")}
        if not r:
            continue
        st, A = r["steps"], r["A"]
        w = st <= CRASH_WINDOW
        tr = int(np.argmin(np.where(w, A, np.inf)))
        rp = tr + int(np.argmax(A[tr:]))
        r["tr"], r["rp"] = np.int64(tr), np.int64(rp)
        out[s] = r
    return out


DATA = {K: load(K) for K in KS}
DATA = {K: v for K, v in DATA.items() if v}

# ----------------------------------------------------------------- task 2
print("=" * 100)
print("TASK 2  drift vs write, per block, at the delta-peak and at the recovery peak")
print("  share = projected share of ||dP_j||^2 : (||X||^2 + <DRIFT,WRITE>) / ||dP_j||^2, sums to 1")
print("=" * 100)
T2 = {}
for K in DATA:
    print(f"\n--- K = {K} ---")
    print(f"{'where':>10} {'blk':>4} {'|DRIFT|':>9} {'|WRITE|':>9} {'cos(D,W)':>9} "
          f"{'shD':>7} {'shW':>7} {'|dP_j|':>9}")
    rows = {}
    for label, key in [("delta-peak", "pk"), ("recov-peak", "rp"), ("rec-end", "END")]:
        for j in range(K):
            acc = []
            for s, r in DATA[K].items():
                t = len(r["steps"]) - 1 if key == "END" else int(r[key])
                d, w = r["DRIFT"][t, j], r["WRITE"][t, j]
                dp = d + w
                n2 = max(float(dp @ dp), 1e-30)
                acc.append([np.linalg.norm(d), np.linalg.norm(w),
                            float(d @ w / max(np.linalg.norm(d) * np.linalg.norm(w), 1e-30)),
                            (float(d @ d) + float(d @ w)) / n2,
                            (float(w @ w) + float(d @ w)) / n2, np.sqrt(n2)])
            m = np.mean(acc, 0); sd = np.std(acc, 0)
            rows[(label, j)] = (m, sd)
            print(f"{label:>10} {j:>4} {m[0]:9.2f} {m[1]:9.2f} {m[2]:9.2f} "
                  f"{m[3]:7.2f} {m[4]:7.2f} {m[5]:9.2f}")
    T2[K] = rows

# share at the last checkpoint too (recovery end), for the figure
T2E = {}
for K in DATA:
    for j in range(K):
        acc = []
        for s, r in DATA[K].items():
            d, w = r["DRIFT"][-1, j], r["WRITE"][-1, j]
            dp = d + w; n2 = max(float(dp @ dp), 1e-30)
            acc.append([(float(d @ d) + float(d @ w)) / n2, (float(w @ w) + float(d @ w)) / n2])
        T2E[(K, j)] = np.mean(acc, 0)

# ----------------------------------------------------------------- task 3
print("\n" + "=" * 100)
print("TASK 3  rotation of P_j from the delta-peak to the recovery peak, and the two "
      "counterfactuals")
print("  drift-only: W_j frozen at its peak value, input free.  write-only: input frozen, W_j free.")
print("=" * 100)
T3 = {}
for K in DATA:
    print(f"\n--- K = {K} ---")
    print(f"{'blk':>4} | {'actual':>7} {'driftonly':>9} {'writeonly':>9} (peak -> recovery peak)"
          f" | {'actual':>7} {'driftonly':>9} {'writeonly':>9} (peak -> end)"
          f" | {'|P| pk':>7} {'|P| rp':>7} {'|P| end':>7}")
    for j in range(K):
        acc = []
        for s, r in DATA[K].items():
            pk, rp, en = int(r["pk"]), int(r["rp"]), len(r["steps"]) - 1
            row = []
            for t in (rp, en):
                row += [ang(r["P"][pk, j], r["P"][t, j]),
                        ang(r["P"][pk, j], r["P_drift_only"][t, j]),
                        ang(r["P"][pk, j], r["P_write_only"][t, j])]
            row += [np.linalg.norm(r["P"][pk, j]), np.linalg.norm(r["P"][rp, j]),
                    np.linalg.norm(r["P"][en, j])]
            acc.append(row)
        m = np.mean(acc, 0); T3[(K, j)] = m
        print(f"{j:>4} | {m[0]:7.1f} {m[1]:9.1f} {m[2]:9.1f} {'':23}"
              f" | {m[3]:7.1f} {m[4]:9.1f} {m[5]:9.1f} {'':14}"
              f" | {m[6]:7.2f} {m[7]:7.2f} {m[8]:7.2f}")

# ----------------------------------------------------------------- task 4
print("\n" + "=" * 100)
print("TASK 4  de-coherence: mean pairwise cos(dP_i, dP_j) at the recovery peak")
print("  drift-only: every WRITE_j frozen at peak.  write-only: every DRIFT_j frozen at peak.")
print("=" * 100)
print(f"{'K':>3} {'at peak':>9} {'actual rp':>10} {'drift rp':>11} {'write rp':>11} "
      f"{'actual end':>11} {'drift end':>11} {'write end':>11}")
COH = {}
for K in DATA:
    if K < 2: continue
    acc = []
    series = []
    for s, r in DATA[K].items():
        pk, rp = int(r["pk"]), int(r["rp"])
        DR, WR = r["DRIFT"], r["WRITE"]
        def mc(dp):
            u = dp / np.clip(np.linalg.norm(dp, axis=-1, keepdims=True), 1e-30, None)
            return float(np.mean([u[i] @ u[j] for i in range(K) for j in range(i + 1, K)]))
        cur = [mc(DR[pk] + WR[pk]), mc(DR[rp] + WR[rp]),
               mc(DR[rp] + WR[pk]), mc(DR[pk] + WR[rp]), mc(DR[-1] + WR[-1]),
               mc(DR[-1] + WR[pk]), mc(DR[pk] + WR[-1])]
        acc.append(cur)
        T = len(r["steps"])
        series.append(np.array([[mc(DR[t] + WR[t]), mc(DR[t] + WR[pk]), mc(DR[pk] + WR[t])]
                                for t in range(T)]))
    m = np.mean(acc, 0)
    COH[K] = (m, np.mean(series, 0), list(DATA[K].values())[0]["steps"],
              int(list(DATA[K].values())[0]["pk"]))
    print(f"{K:>3} {m[0]:9.3f} {m[1]:10.3f} {m[2]:11.3f} {m[3]:11.3f} {m[4]:11.3f} "
          f"{m[5]:11.3f} {m[6]:11.3f}")

# ----------------------------------------------------------------- task 5
print("\n" + "=" * 100)
print("TASK 5  input rotation vs contribution rotation, delta-peak -> recovery peak")
print("=" * 100)
print(f"{'K':>3} {'blk':>4} {'rot m_j':>9} {'rot P_j':>9} | {'rot m_j end':>11} {'rot P_j end':>11}")
xs, ys, xe, ye = [], [], [], []
for K in DATA:
    for j in range(K):
        acc = []
        for s, r in DATA[K].items():
            pk, rp, en = int(r["pk"]), int(r["rp"]), len(r["steps"]) - 1
            acc.append([ang(r["M"][pk, j], r["M"][rp, j]), ang(r["P"][pk, j], r["P"][rp, j]),
                        ang(r["M"][pk, j], r["M"][en, j]), ang(r["P"][pk, j], r["P"][en, j])])
        m = np.mean(acc, 0)
        xs.append(m[0]); ys.append(m[1]); xe.append(m[2]); ye.append(m[3])
        print(f"{K:>3} {j:>4} {m[0]:9.1f} {m[1]:9.1f} | {m[2]:11.1f} {m[3]:11.1f}")
xs, ys, xe, ye = map(np.array, (xs, ys, xe, ye))
for nm, a, b in [("peak->recovery peak", xs, ys), ("peak->end", xe, ye)]:
    k = a > 1e-6
    print(f"\n{nm}: Pearson r all blocks {np.corrcoef(a, b)[0,1]:+.3f} (n={len(a)}); "
          f"blocks j>0 only {np.corrcoef(a[k], b[k])[0,1]:+.3f} (n={int(k.sum())})")

# ----------------------------------------------------------------- run summary
print("\n" + "=" * 100)
print("RUN SUMMARY")
print(f"{'K':>3} {'s':>2} {'ilr':>10} {'Bgate':>6} {'ident':>9} {'A0':>5} {'trough':>13} "
      f"{'recpeak':>13} {'Aend':>5} {'dpeak':>8}")
for K in DATA:
    for s, r in DATA[K].items():
        st = r["steps"]
        print(f"{K:>3} {s:>2} {float(r['inject_lr']):10.3e} {int(r['B_gate']):6d} "
              f"{float(r['identity_max_abs']):9.1e} {r['A'][0]:5.2f} "
              f"{r['A'][int(r['tr'])]:6.2f}@{st[int(r['tr'])]:<6d} "
              f"{r['A'][int(r['rp'])]:6.2f}@{st[int(r['rp'])]:<6d} {r['A'][-1]:5.2f} "
              f"{r['delta'][int(r['pk'])]:8.1f}")

# ----------------------------------------------------------------- figures
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png/.pdf")


# (a) stacked drift/write share per block, at peak and at recovery end
fig, axes = plt.subplots(1, len(DATA), figsize=(3.2 * len(DATA), 3.0), squeeze=False)
for ax, K in zip(axes[0], DATA):
    xb = np.arange(K); w = 0.38
    dpk = [T2[K][("delta-peak", j)][0][3] for j in range(K)]
    dend = [T2E[(K, j)][0] for j in range(K)]
    ax.bar(xb - w / 2, dpk, w, color=NAVY, label="drift")
    ax.bar(xb - w / 2, 1 - np.array(dpk), w, bottom=dpk, color=CRIM, label="write")
    ax.bar(xb + w / 2, dend, w, color=NAVY, alpha=0.45)
    ax.bar(xb + w / 2, 1 - np.array(dend), w, bottom=dend, color=CRIM, alpha=0.45)
    ax.axhline(0.5, color="0.4", lw=0.6, ls=":")
    ax.set_xticks(xb); ax.set_xlabel("block $j$"); ax.set_title(f"$K={K}$", fontsize=10)
    ax.set_ylim(0, 1)
    if K == list(DATA)[0]:
        ax.set_ylabel("share of $\\|\\Delta P_j\\|^2$"); ax.legend(frameon=False, fontsize=7)
fig.suptitle("solid = delta peak,  faded = recovery end", fontsize=8, y=1.02)
save(fig, "drift_share_by_block")

# (b) P rotation: actual vs the two counterfactuals
fig, axes = plt.subplots(1, len(DATA), figsize=(3.2 * len(DATA), 3.0), squeeze=False, sharey=True)
for ax, K in zip(axes[0], DATA):
    xb = np.arange(K)
    ax.plot(xb, [T3[(K, j)][3] for j in range(K)], "o-", color=NAVY, label="actual")
    ax.plot(xb, [T3[(K, j)][4] for j in range(K)], "s--", color=CRIM, label="drift only")
    ax.plot(xb, [T3[(K, j)][5] for j in range(K)], "^:", color=OLIV, label="write only")
    ax.axhspan(60, 72, color=PURP, alpha=0.12, lw=0)
    ax.set_xticks(xb); ax.set_xlabel("block $j$"); ax.set_title(f"$K={K}$", fontsize=10)
    if K == list(DATA)[0]:
        ax.set_ylabel("rotation of $P_j$, peak $\\to$ recovery (deg)")
        ax.legend(frameon=False, fontsize=7)
fig.suptitle("shaded band: the transformer's measured 60-72 deg", fontsize=8, y=1.02)
save(fig, "drift_rotation_counterfactual")

# (c) pairwise cos over time at the deepest K
if COH:
    Kc = max(COH)
    m, series, st, pk = COH[Kc]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    m1 = st > 0
    ax.plot(st[m1], series[m1, 0], color=NAVY, lw=1.6, label="actual")
    # the counterfactuals are frozen AT the peak, so they are only defined from the peak on
    m2 = st >= st[pk]
    ax.plot(st[m2], series[m2, 1], color=CRIM, lw=1.3, ls="--", label="drift only (WRITE frozen)")
    ax.plot(st[m2], series[m2, 2], color=OLIV, lw=1.3, ls=":", label="write only (DRIFT frozen)")
    ax.axvline(st[pk], color="0.5", lw=0.7)
    ax.annotate("$\\delta$ peak", (st[pk], ax.get_ylim()[1]), fontsize=7, color="0.4",
                ha="left", va="top", xytext=(3, -2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("injection step"); ax.set_ylabel("mean pairwise $\\cos(\\Delta P_i,\\Delta P_j)$")
    ax.set_title(f"de-coherence attribution, $K={Kc}$", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "drift_decoherence_K%d" % Kc)
