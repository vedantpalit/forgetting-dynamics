"""K=1 toy: does recovery need the write to occupy a large SHARE of the state, or literally
a large ||W||? Sweep `gain` (the multiplier on rms(k)@W, see k1.fwd's
docstring), gate-matched so B always reaches 0.99 at step 200 -- so a_frac is the only thing
that changes across columns, not dose.

For each gain: pretrain, bisect the injection rate to the gate, inject, record ||dW|| (the
literal magnitude of the store's weight change), a_frac at the trough (the write's share of
the state, ||shift||/||h||), A's trough, and A's recovered fraction
f = (A_end - A_trough) / (1 - A_trough).

    python wp_gain_sweep.py --gains 0.05,0.1,0.25,0.5,0.7071,1,1.4142,2,4 --seeds 0,1,2 --out wp_gain_sweep.json
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp

import k1
import sweeps


def run_cell(gain, seed, d=128, vocab=32, na=50, nb=50, beta=0.5, u_scale=0.01,
             steps=5000, match_gate=200):
    k1.configure(d=d, v=vocab, na=na, nd=na, nb=nb)
    p0, lr, pg, mu, A, Dd, B = k1.pretrain(seed, beta, 1, gain)
    base_ilr = lr / k1.INJECT_RATIO
    ilr, gate, ok = sweeps.match_gate(p0, A, Dd, B, 1, gain, u_scale, base_ilr, seed, match_gate)
    rows, _ = k1.inject(p0, lr, mu, A, Dd, B, 1, gain, u_scale, steps, seed, ilr=ilr)
    A_arr = np.array([r["A"] for r in rows]); af = np.array([r["a_frac"] for r in rows])
    dW = np.array([r["dW"] for r in rows])
    tr = int(A_arr.argmin())
    f = float((A_arr[-1] - A_arr[tr]) / max(1.0 - A_arr[tr], 1e-9))
    return dict(gain=gain, seed=seed, matched=bool(ok), gate=gate, ilr=ilr,
                A_trough=float(A_arr[tr]), trough_step=int(rows[tr]["step"]),
                A_end=float(A_arr[-1]), recovered_frac=f,
                a_frac_trough=float(af[tr]), a_frac_end=float(af[-1]),
                dW_trough=float(dW[tr]), dW_end=float(dW[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for gain in [float(x) for x in a.gains.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            r = run_cell(gain, seed)
            out.append(r)
            print(f"gain={gain:<8g} seed={seed} matched={r['matched']} | "
                  f"a_frac(trough)={r['a_frac_trough']:.3f} dW(trough)={r['dW_trough']:.2f} | "
                  f"A trough={r['A_trough']:.3f}@{r['trough_step']} A end={r['A_end']:.3f} "
                  f"recovered={r['recovered_frac']:.3f}", flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
