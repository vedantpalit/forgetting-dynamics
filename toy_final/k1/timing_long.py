"""TASK 4: one config, 20000 steps, dense logging, 2 seeds -- to separate the three time
scales (crash rise, recovery fall, erosion growth), which the 5000-step log-spaced grid
cannot resolve past the recovery."""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import sys

import numpy as np
import k1

STEPS = 20000


def dense(steps):
    g = set(range(0, 1001, 5)) | set(range(1000, 5001, 25)) | set(range(5000, steps + 1, 100))
    return sorted(s for s in g if s <= steps)


k1.grid = dense

if __name__ == "__main__":
    norm = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = {}
    for seed in (0, 1):
        p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, 0.5, norm, 1.0)
        ilr = lr / k1.INJECT_RATIO
        rows, _ = k1.inject(p0, lr, mu, A, Dd, B, norm, 1.0, 0.01, STEPS, seed, ilr=ilr)
        r = dict(sweep="long", value=0.0, norm=norm, seed=seed, beta=0.5, d=k1.D, v=k1.V,
                 na=k1.NA, nd=k1.ND, nb=k1.NB, gain=1.0, u_scale=0.01, pretrain_lr=lr,
                 pretrain_gate=pg, inject_lr=ilr, matched=True, probe_gate=-1, dose=1.0,
                 rows=rows)
        q = k1.summarise(r)
        out[f"long-n{norm}-s{seed}"] = r
        print(f"norm={norm} seed={seed} ilr={ilr:.4e} | trough {q['trough']:.2f}"
              f"@{q['trough_step']} -> {q['after']:.2f} end {q['end']:.2f} | "
              f"s.w {q['sw_peak']:.3f}@{q['sw_peak_step']} -> {q['sw_end']:.3f} | "
              f"eps {q['eps_end']:.3f} | Bgate {q['B_gate']}", flush=True)
        json.dump(out, open(f"timing_long_n{norm}.json", "w"))
    print("done")
