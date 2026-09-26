"""Task 2.  The two-channel sweep: n_D x u_scale x w_scale, normalized arm, gate-matched.

n_D in {0, 50}                 ballast off / on
u_scale in {0, 0.01, 0.1, 0.3, 1.0}    the readout channel's injection rate multiplier
w_scale in {1.0, 0.1, 0.01}            the store channel's

Every cell is gate-matched to B reaching 0.99 at step 200, by bisecting the injection rate
with BOTH multipliers held fixed -- so a cell is a fixed CHANNEL MIX, and the overall rate is
whatever that mix needs to learn B on the common clock.  Cells where no rate in the bracket
gets B to the gate are kept and flagged; they are exactly the cells where the channel left
open cannot learn B at all, which is itself the answer for that mix.

Pretraining does not depend on either multiplier, so it is done once per (n_D, seed) and
reused across the 15 knob cells.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import sys
import time

import numpy as np

import k1
import ballast_k1 as bk
import wp_readout_lib as wl

BETA, GAIN, NORM, STEPS, TARGET = 0.5, 1.0, 1, 5000, 200
NDS = [0, 50]
USCALES = [0.0, 0.01, 0.1, 0.3, 1.0]
WSCALES = [1.0, 0.1, 0.01]
SEEDS = [0, 1, 2]
OUT = "wp_readout_sweep.json"


def main():
    k1.configure(d=128, v=32, na=50, nb=50)
    out = {}
    t0 = time.time()
    for nd in NDS:
        for seed in SEEDS:
            bk.configure(nd=nd)
            p0, lr, pg, mu, A, Dd, B = bk.pretrain(seed, BETA, NORM, GAIN, nd)
            base = lr / k1.INJECT_RATIO
            for us in USCALES:
                for ws in WSCALES:
                    bk.configure(nd=nd)
                    ilr, g, ok = wl.match_gate2(p0, B, NORM, GAIN, us, ws, base, seed, TARGET)
                    rows, _ = wl.inject2(p0, mu, A, Dd, B, NORM, GAIN, us, ws,
                                         STEPS, seed, ilr)
                    q = wl.summarise2(rows)
                    key = f"nd{nd}-u{us}-w{ws}-s{seed}"
                    out[key] = dict(nd=nd, u_scale=us, w_scale=ws, seed=seed, beta=BETA,
                                    d=k1.D, v=k1.V, na=k1.NA, nb=k1.NB, pretrain_lr=lr,
                                    inject_lr=ilr, matched=bool(ok), probe_gate=g,
                                    rows=rows, summary=q)
                    flag = "" if ok else "  [GATE NOT MATCHED]"
                    print(f"nd={nd:2d} u={us:<5g} w={ws:<5g} s{seed} | ilr {ilr:.2e} "
                          f"Bgate {q['B_gate']:4d} Bend {q['B_end']:.2f} | "
                          f"A {q['A0']:.2f}->{q['trough']:.2f}@{q['trough_step']:<4d} "
                          f"rec {q['recovery']:+.2f} end {q['end']:.2f} | "
                          f"cf W-only {q['A_storeonly_at_trough']:.2f} "
                          f"U-only {q['A_readonly_at_trough']:.2f} | "
                          f"share store {q['store_share']:+.2f} read {q['read_share']:+.2f} | "
                          f"Gst {q['G_store_trough']:+.2f} Grd {q['G_read_trough']:+.2f} | "
                          f"dU {q['dU_rel']:.3f} dW {q['dW_rel']:.3f}{flag}", flush=True)
                    sys.stdout.flush()
            json.dump(out, open(OUT, "w"))
    print(f"wrote {OUT} ({len(out)} runs) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
