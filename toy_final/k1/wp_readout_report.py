"""Tasks 3 and 4.  The cell table, the H1 test, and the cold-row measurement.

H1 (to test, not assume): recovery fraction equals the STORE's share of B's acquisition,
because a class bias written into U has no anti-state force that can remove it.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json

import numpy as np

SWEEP = "wp_readout_sweep.json"
KEYS = ["depth", "recovery", "rec_frac", "trough", "end", "store_share", "read_share",
        "A_storeonly_at_trough", "A_readonly_at_trough", "A_storeonly_min", "A_readonly_min",
        "G_store_trough", "G_read_trough", "G_store_end", "G_read_end",
        "dU_rel", "dW_rel", "uX0", "uY0", "uX_end", "uY_end", "B_gate", "B_end",
        "s_on_w_peak", "s_on_w_end", "s_abs_on_w_0", "s_abs_on_w_end", "A0", "M_B_gain"]


def load():
    raw = json.load(open(SWEEP))
    cells = {}
    for k, r in raw.items():
        c = (r["nd"], r["u_scale"], r["w_scale"])
        cells.setdefault(c, []).append(r)
    return raw, cells


def agg(runs, key):
    return float(np.mean([r["summary"][key] for r in runs]))


def main():
    raw, cells = load()
    print(f"{len(raw)} runs, {len(cells)} cells, "
          f"{sum(1 for r in raw.values() if not r['matched'])} not gate-matched")

    print()
    print("=" * 142)
    print("CELL TABLE  (mean over seeds; normalized arm, beta=0.5, d=128, |V|=32, "
          "n_A=n_B=50, B gate matched to step 200)")
    print("=" * 142)
    hdr = (f"{'nD':>3} {'u_sc':>5} {'w_sc':>5} | {'A0':>4} {'trgh':>5} {'dpth':>5} "
           f"{'rec':>5} {'rec/d':>6} {'end':>5} | {'stSh':>6} {'rdSh':>6} | "
           f"{'cfW':>5} {'cfU':>5} | {'Gst':>6} {'Grd':>6} | {'dU/U':>6} {'dW/W':>5} "
           f"| {'Bgt':>4} {'Bend':>4} {'m':>1}")
    print(hdr); print("-" * 142)
    rowsout = []
    for c in sorted(cells):
        nd, us, ws = c
        R = cells[c]
        m = "" if all(r["matched"] for r in R) else "*"
        d = {k: agg(R, k) for k in KEYS}
        d.update(nd=nd, u_scale=us, w_scale=ws, matched=(m == ""))
        rowsout.append(d)
        print(f"{nd:>3} {us:>5g} {ws:>5g} | {d['A0']:>4.2f} {d['trough']:>5.2f} "
              f"{d['depth']:>5.2f} {d['recovery']:>+5.2f} {d['rec_frac']:>6.2f} "
              f"{d['end']:>5.2f} | {d['store_share']:>+6.2f} {d['read_share']:>+6.2f} | "
              f"{d['A_storeonly_at_trough']:>5.2f} {d['A_readonly_at_trough']:>5.2f} | "
              f"{d['G_store_trough']:>+6.2f} {d['G_read_trough']:>+6.2f} | "
              f"{d['dU_rel']:>6.3f} {d['dW_rel']:>5.3f} | {d['B_gate']:>4.0f} "
              f"{d['B_end']:>4.2f} {m:>1}")
    print("-" * 142)
    print("stSh/rdSh: store and readout share of B's mean correct-class margin gain at B's gate,")
    print("           from M(W_t,U_0) and M(W_0,U_t) against M(W_0,U_0); they need not sum to 1")
    print("           (the remainder is the W-U interaction).")
    print("cfW = A's accuracy AT THE TROUGH STEP with the readout reverted to pretrained (store")
    print("      carries the crash if this is low);  cfU = with the store reverted (readout carries).")
    print("Gst/Grd = the store and readout parts of A's mean Y-vs-X class-bias gap at the trough,")
    print("          in logits, from the exact split in wp_readout_lib.  * = gate not matched.")

    # ---------------------------------------------------------------- H1 -------------
    print()
    print("=" * 90)
    print("H1: does the recovery fraction track the STORE's share of B's acquisition?")
    print("=" * 90)
    good = [d for d in rowsout if d["depth"] > 0.05 and d["B_end"] >= 0.99]
    x = np.array([d["store_share"] for d in good])
    y = np.array([d["rec_frac"] for d in good])
    print(f"  cells with a real crash (depth > 0.05) and B learned: {len(good)}")
    print(f"  Pearson r(store share, recovery fraction) = {np.corrcoef(x, y)[0,1]:+.3f}")
    sp = lambda a: np.argsort(np.argsort(a))
    print(f"  Spearman                                  = "
          f"{np.corrcoef(sp(x), sp(y))[0,1]:+.3f}")
    print(f"  slope of a 1-parameter fit rec_frac = a * store_share: "
          f"a = {float((x@y)/(x@x)):+.3f}   (H1 predicts a = 1)")
    resid = y - x
    print(f"  rec_frac - store_share: mean {resid.mean():+.3f}, sd {resid.std():.3f}, "
          f"range [{resid.min():+.3f}, {resid.max():+.3f}]")

    print()
    print("  cells sorted by store share:")
    print(f"  {'nD':>3} {'u':>5} {'w':>5} | {'depth':>5} {'stSh':>6} {'rdSh':>6} "
          f"{'rec_frac':>8} | carrier")
    for d in sorted(good, key=lambda z: z["store_share"]):
        car = ("store" if d["A_storeonly_at_trough"] < d["A_readonly_at_trough"] - 0.05 else
               "readout" if d["A_readonly_at_trough"] < d["A_storeonly_at_trough"] - 0.05
               else "both/neither")
        print(f"  {d['nd']:>3} {d['u_scale']:>5g} {d['w_scale']:>5g} | {d['depth']:>5.2f} "
              f"{d['store_share']:>+6.2f} {d['read_share']:>+6.2f} {d['rec_frac']:>8.2f} | {car}")

    print()
    print("  CRASH WITHOUT RECOVERY  (depth > 0.30 and recovery < 10% of it):")
    hits = [d for d in rowsout if d["depth"] > 0.30 and d["rec_frac"] < 0.10]
    if not hits:
        print("    none.")
    for d in hits:
        print(f"    nD={d['nd']} u={d['u_scale']:g} w={d['w_scale']:g}: "
              f"A {d['A0']:.2f} -> {d['trough']:.2f} (depth {d['depth']:.2f}), "
              f"recovery {d['recovery']:+.2f} = {100*d['rec_frac']:.0f}% of it, end {d['end']:.2f}; "
              f"store share {d['store_share']:+.2f}, readout share {d['read_share']:+.2f}; "
              f"cf W-only {d['A_storeonly_at_trough']:.2f}, U-only {d['A_readonly_at_trough']:.2f}; "
              f"G_read {d['G_read_trough']:+.1f} -> {d['G_read_end']:+.1f} logits, "
              f"G_store {d['G_store_trough']:+.1f} -> {d['G_store_end']:+.1f}; "
              f"B_end {d['B_end']:.2f}, matched={d['matched']}")

    # ------------------------------------------------------------- cold rows ---------
    print()
    print("=" * 90)
    print("TASK 4.  Cold readout rows: ||U_v|| for v in Y (B's half) vs v in X (A's half)")
    print("=" * 90)
    print(f"  {'nD':>3} {'u':>5} {'w':>5} | {'uX0':>6} {'uY0':>6} {'Y/X0':>6} | "
          f"{'uXend':>6} {'uYend':>6} {'Y/Xend':>7} | {'Ygrow':>6} {'Xgrow':>6} | "
          f"{'Grd_end':>8} {'Gst_end':>8} | {'depth':>5} {'rec':>5}")
    for d in rowsout:
        print(f"  {d['nd']:>3} {d['u_scale']:>5g} {d['w_scale']:>5g} | {d['uX0']:>6.2f} "
              f"{d['uY0']:>6.2f} {d['uY0']/d['uX0']:>6.2f} | {d['uX_end']:>6.2f} "
              f"{d['uY_end']:>6.2f} {d['uY_end']/d['uX_end']:>7.2f} | "
              f"{d['uY_end']/d['uY0']:>6.2f} {d['uX_end']/d['uX0']:>6.2f} | "
              f"{d['G_read_end']:>+8.2f} {d['G_store_end']:>+8.2f} | "
              f"{d['depth']:>5.2f} {d['recovery']:>+5.2f}")

    json.dump(rowsout, open("wp_readout_cells.json", "w"))
    print("\nwrote wp_readout_cells.json")


if __name__ == "__main__":
    main()
