"""Data for the paper's first toy figure (plot_paper_k1.py): the base configuration, both
readouts, gate-matched, ten seeds.

    python paper_base_runs.py --seeds 0-9 --out paper_base.json

Why this exists next to beta_sweep.json: that file was three seeds at a fixed injection
rate, and at a fixed rate B's clock varies 2.4x across seeds, so the averaged curve's band
was widened by timing alone (LOG.md, 'where the toy's noise comes from'). Here the rate is
bisected per seed and per arm so B reaches 0.99 at step 200, as in every sweep figure; the
depth spread that remains is genuine instance spread, and ten seeds put the standard error
of the mean at ~0.05 instead of ~0.10. Keys match beta_sweep.json's so the plot script only
changes its file name.
"""
import argparse
import json

import k1
import sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--u_scale", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--match", type=int, default=200)
    ap.add_argument("--store_scale", default=None,
                    help="'sqrt_d' replaces the store's rms(k) with sqrt(d) k (k1.configure)")
    ap.add_argument("--norms", default="0,1")
    ap.add_argument("--out", default="paper_base.json")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    k1.configure(d=128, v=32, na=50, nd=50, nb=50, store_scale=a.store_scale)

    out = {}
    for seed in range(lo, hi + 1):
        for norm in [int(x) for x in a.norms.split(",")]:
            p0, lr, gate, mu, A, Dd, B = k1.pretrain(seed, a.beta, norm, a.gain)
            base = lr / k1.INJECT_RATIO
            ilr, g, ok = sweeps.match_gate(p0, A, Dd, B, norm, a.gain, a.u_scale, base, seed, a.match)
            rows, _ = k1.inject(p0, lr, mu, A, Dd, B, norm, a.gain, a.u_scale, a.steps, seed, ilr=ilr)
            key = f"b{a.beta}-n{norm}-g{a.gain}-u{a.u_scale}-s{seed}"
            out[key] = dict(seed=seed, beta=a.beta, norm=norm, u_scale=a.u_scale, gain=a.gain,
                            pretrain_lr=lr, pretrain_gate=gate, inject_lr=ilr, B_gate=g,
                            matched=bool(ok), rows=rows)
            s = k1.summarise(out[key])
            print(f"seed {seed} norm {norm}: B gate {g} (matched={ok}) | A trough {s['trough']:.2f}"
                  f"@{s['trough_step']} end {s['end']:.2f}", flush=True)
        json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
