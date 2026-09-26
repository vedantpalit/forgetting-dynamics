"""Does ||W|| grow throughout the phenomenon, or turn over like accuracy?

The question, precisely. Uses k1.inject's existing per-checkpoint `dW` (the full
||Delta W||_F) and `s_norm` (||s||, s = mu^T Delta W, the coherent/shared row) to derive
||Delta W_perp|| = sqrt(||Delta W||^2 - ||s||^2) exactly, since s and Delta W_perp are
Frobenius-orthogonal by construction (mu mu^T Delta W and (I - mu mu^T) Delta W split any
matrix into orthogonal pieces). Writes wp_dW_timecourse.json; see wp_dW_timecourse_fig.py
for the figure and the LOG.md entry for the reading.

    python wp_dW_timecourse.py --seeds 0,1,2 --out wp_dW_timecourse.json
"""
import argparse
import json

import numpy as np

import k1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="wp_dW_timecourse.json")
    a = ap.parse_args()
    k1.configure(d=128, v=32, na=50, nd=50, nb=50)

    out = {}
    for seed in [int(x) for x in a.seeds.split(",")]:
        p0, lr, gate, mu, A, Dd, B = k1.pretrain(seed, a.beta, 1, a.gain)
        rows, _ = k1.inject(p0, lr, mu, A, Dd, B, 1, a.gain, 0.01, a.steps, seed)
        st = np.array([r["step"] for r in rows])
        dW = np.array([r["dW"] for r in rows])
        s = np.array([r["s_norm"] for r in rows])
        Acc = np.array([r["A"] for r in rows])
        dWperp = np.sqrt(np.maximum(dW ** 2 - s ** 2, 0))
        out[str(seed)] = dict(step=st.tolist(), dW=dW.tolist(), s=s.tolist(),
                              dWperp=dWperp.tolist(), A=Acc.tolist())

        d_diff = np.diff(dW)
        tr = int(Acc.argmin())
        print(f"seed {seed}: A trough {Acc[tr]:.3f}@step{st[tr]}  "
              f"||dW|| trough={dW[tr]:.2f} end={dW[-1]:.2f}  "
              f"||s|| trough={s[tr]:.2f} peak={s.max():.2f}@step{st[int(s.argmax())]} end={s[-1]:.2f}  "
              f"||dW_perp|| trough={dWperp[tr]:.2f} end={dWperp[-1]:.2f}  "
              f"||dW|| decreasing at {100 * (d_diff < 0).mean():.1f}% of steps, "
              f"min step diff {d_diff.min():+.3e}", flush=True)

    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
