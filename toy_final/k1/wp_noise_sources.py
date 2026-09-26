"""Where does the toy's seed-to-seed noise come from?

Three candidate sources, separated by construction:
  (a) the pretrained MODEL: keys, values, the LR the probe picked, and the margins A has when
      pretraining stops (it stops at the first 100-step check where A and D are both >= 0.99,
      so margins sit wherever they happen to be at that moment);
  (b) the MINIBATCH stream during injection (k1.inject seeds its rng from `seed + 5`, and
      takes the model as an argument, so the same p0 can be injected under several streams);
  (c) accuracy granularity: n_A = 50, so A moves in steps of 0.02.

For each model seed: pretrain once, record lr / steps / A's pretrained margins / n_distinct_B,
then inject the SAME model under R different minibatch streams. The between-model variance of
the trough against the within-model (between-stream) variance says which of (a) or (b) owns
the noise; the correlation of the trough with the pretrained margin says what inside (a).

    python wp_noise_sources.py --seeds 0-7 --streams 3 --out wp_noise_sources.json
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

import k1


def margins(p, k, v, norm, gain):
    z = np.asarray(k1.fwd(p, jnp.asarray(k), norm, gain)[0])
    zy = z[np.arange(len(v)), v]
    z2 = z.copy(); z2[np.arange(len(v)), v] = -np.inf
    return zy - z2.max(-1)                       # margin to the best competitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--out", default="wp_noise_sources.json")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    k1.configure(d=128, v=32, na=50, nd=50, nb=50)

    out = []
    for seed in range(lo, hi + 1):
        p0, lr, t_pre, mu, A, Dd, B = k1.pretrain(seed, a.beta, 1, a.gain)
        (kA, vA), _, (kB, vB) = A, Dd, B
        m = margins(p0, kA, vA, 1, a.gain)
        rec = dict(seed=seed, lr=lr, t_pre=int(t_pre), n_distinct_B=int(np.unique(vB).size),
                   margin_mean=float(m.mean()), margin_std=float(m.std()),
                   margin_min=float(m.min()), margin_q25=float(np.quantile(m, 0.25)),
                   streams=[])
        for r in range(a.streams):
            rows, _ = k1.inject(p0, lr, mu, A, Dd, B, 1, a.gain, 0.01, a.steps, seed + 1000 * r)
            acc = np.array([x["A"] for x in rows]); st = np.array([x["step"] for x in rows])
            tr = int(acc.argmin())
            rec["streams"].append(dict(stream=r, trough=float(acc[tr]), trough_step=int(st[tr]),
                                       end=float(acc[-1]),
                                       recovered=float((acc[-1] - acc[tr]) / max(1 - acc[tr], 1e-9))))
        tro = [s["trough"] for s in rec["streams"]]
        print(f"seed {seed}: lr={lr} t_pre={t_pre} margin mean={m.mean():.2f} sd={m.std():.2f} "
              f"min={m.min():.2f} distinctB={rec['n_distinct_B']} | troughs over streams "
              f"{np.round(tro, 2).tolist()} (sd {np.std(tro):.3f})", flush=True)
        out.append(rec)
    json.dump(out, open(a.out, "w"))

    # variance decomposition of the trough
    per_model = np.array([[s["trough"] for s in r["streams"]] for r in out])   # (S, R)
    between = per_model.mean(1).var(ddof=1)
    within = per_model.var(1, ddof=1).mean()
    print(f"\ntrough: between-model sd {np.sqrt(between):.3f}, within-model (minibatch) sd "
          f"{np.sqrt(within):.3f}")
    mm = np.array([r["margin_mean"] for r in out]); tm = per_model.mean(1)
    lrs = np.array([r["lr"] for r in out]); tp = np.array([r["t_pre"] for r in out])
    nd = np.array([r["n_distinct_B"] for r in out])
    for name, x in (("margin mean", mm), ("margin min", np.array([r["margin_min"] for r in out])),
                    ("pretrain lr", lrs), ("pretrain steps", tp), ("n_distinct_B", nd)):
        if x.std() > 0:
            print(f"  corr(trough, {name:>14}) = {np.corrcoef(x, tm)[0, 1]:+.2f}")
    print(f"  lr values picked: {sorted(set(lrs.tolist()))}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
