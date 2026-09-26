"""Pretrain on A + D, inject B, log the curves (PLAN.md §5-6). One run per call.

Pretraining stops when A and D both reach the gate (0.99), capped. Injection runs until B
reaches the gate and then for as long again, so the recovery window is inside the run --
`toy/` §0.18c: a truncated arm cannot show a turnover it has not reached.

Plain SGD, minibatches sampled uniformly from the population being trained. The injection LR
is the pretrain LR divided by the transformer's ratio (13.33) unless overridden.

Every eval logs A / D / B full-vocabulary accuracy, A's own-half top-1 rate, and the
mechanistic quantities on A's final residual: ||delta||, rms||eps||, sum_j ||P_j||,
||sum_j P_j||, mean pairwise cosine among the P_j, and ||U(t) - U(0)||, ||W_j(t) - W_j(0)||.
"""
import argparse
import json
import os

import jax.numpy as jnp
import numpy as np

import populations as pop
import model as M

INJECT_RATIO = 400.0 / 30.0        # the transformer's pretrain/inject LR ratio
GATE = 0.99


def mech(params, params0, keys):
    """The Section-5 quantities on A's inputs, exact because the stack is a sum."""
    _, P, h = M.forward(params, jnp.asarray(keys))
    _, P0, h0 = M.forward(params0, jnp.asarray(keys))
    dh = np.asarray(h - h0)                              # (n, d)
    delta = dh.mean(0)
    eps = dh - delta[None]
    Pj = np.asarray(P - P0).mean(1)                      # (K, d): per-block contribution, A-mean
    N = np.linalg.norm(Pj, axis=-1)
    U_ = Pj / np.clip(N[:, None], 1e-12, None)
    G = U_ @ U_.T
    K = Pj.shape[0]
    iu = np.triu_indices(K, 1)
    return dict(
        delta=float(np.linalg.norm(delta)),
        eps=float(np.sqrt((eps ** 2).sum(-1)).mean()),
        sum_norms=float(N.sum()),
        norm_sum=float(np.linalg.norm(Pj.sum(0))),
        cos=float(G[iu].mean()) if K > 1 else float("nan"),
        dU=float(np.linalg.norm(np.asarray(params["U"] - params0["U"]))),
        dW=[float(np.linalg.norm(np.asarray(params["W"][j] - params0["W"][j])))
            for j in range(K)],
    )


def run_phase(params, keys, vals, lr, batch, rng, evals, gate_fn, cap, extra_after_gate,
              min_steps=0, every=10, fix_U=False):
    """Train until gate_fn(params) >= GATE, then for `extra_after_gate` x the steps it took
    (0 to stop at the gate), but never fewer than `min_steps` in total. The first grid stopped
    at 2x B's gate and every K=2 trough sat on the last step -- the window ended while A was
    still falling. Returns params and the step at which the gate was reached."""
    n = keys.shape[0]
    kj, vj = jnp.asarray(keys), jnp.asarray(vals)
    gate_step, stop = None, cap
    step = 0
    while step < stop:
        if step % every == 0:
            evals(step, params)
            if gate_step is None and gate_fn(params) >= GATE:
                gate_step = step
                stop = min(cap, max(min_steps, step + int(extra_after_gate * max(step, every))))
        idx = rng.integers(0, n, size=batch)
        params = M.sgd_step(params, kj[idx], vj[idx], lr, fix_U=fix_U)
        step += 1
    evals(step, params)
    return params, gate_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n_vals", type=int, default=128)
    ap.add_argument("--n_a", type=int, default=200)
    ap.add_argument("--n_d", type=int, default=200)
    ap.add_argument("--n_b", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", default="auto",
                    help="pretrain SGD lr, or 'auto': probe the grid below on a short budget "
                         "and take the LARGEST lr that reaches the gate. A deep stack "
                         "compounds the step through K matrices, so one lr does not serve "
                         "every K -- lr 0.5 pretrains K=1 in 690 steps and diverges at K=4.")
    ap.add_argument("--lr_grid", default="0.5,0.3,0.2,0.1,0.05,0.03,0.02,0.01,0.005")
    ap.add_argument("--probe_cap", type=int, default=2500)
    ap.add_argument("--inject_lr", type=float, default=None,
                    help="default: --lr / 13.33, the transformer's ratio")
    ap.add_argument("--pretrain_cap", type=int, default=20000)
    ap.add_argument("--inject_cap", type=int, default=20000)
    ap.add_argument("--extra_after_gate", type=float, default=8.0,
                    help="inject for this many x B's gate step after the gate")
    ap.add_argument("--min_inject", type=int, default=3000,
                    help="floor on injection length regardless of the gate")
    ap.add_argument("--fix_U", action="store_true",
                    help="freeze the readout during injection (pretraining still trains it)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    P = pop.build(a.n_a, a.n_d, a.n_b, a.d, a.n_vals, a.beta, seed=1000 + a.seed)
    k_pre = np.concatenate([P.keys_a, P.keys_d]); v_pre = np.concatenate([P.vals_a, P.vals_d])
    gate = lambda p: min(M.accuracy(p, P.keys_a, P.vals_a), M.accuracy(p, P.keys_d, P.vals_d))

    # ---- LR probe (PLAN.md 5: from a probe, not assumed) --------------------------------
    if a.lr == "auto":
        lr = None
        for cand in [float(x) for x in a.lr_grid.split(",")]:
            p_try = M.init(a.K, a.d, a.n_vals, seed=a.seed)
            r_try = np.random.default_rng(a.seed)
            _, g = run_phase(p_try, k_pre, v_pre, cand, a.batch, r_try, lambda s_, p_: None,
                             gate, a.probe_cap, extra_after_gate=0.0)
            if g is not None:
                lr = cand
                print(f"K={a.K} beta={a.beta} seed={a.seed}: lr probe -> {lr} "
                      f"(gate at {g} within {a.probe_cap})")
                break
        if lr is None:
            raise SystemExit(f"no lr in {a.lr_grid} reaches the gate within {a.probe_cap} steps")
    else:
        lr = float(a.lr)
    a.lr = lr
    inject_lr = a.inject_lr if a.inject_lr is not None else lr / INJECT_RATIO

    params = M.init(a.K, a.d, a.n_vals, seed=a.seed)
    rng = np.random.default_rng(a.seed)

    # ---- pretrain on A + D --------------------------------------------------------------
    pre_log = []
    def pre_eval(step, p):
        pre_log.append(dict(step=step, A=M.accuracy(p, P.keys_a, P.vals_a),
                            D=M.accuracy(p, P.keys_d, P.vals_d)))
    params, pre_gate = run_phase(params, k_pre, v_pre, lr, a.batch, rng, pre_eval, gate,
                                 a.pretrain_cap, extra_after_gate=0.0)
    if pre_gate is None:
        raise SystemExit(f"pretraining did not reach {GATE} within {a.pretrain_cap} steps "
                         f"(A {pre_log[-1]['A']:.3f}, D {pre_log[-1]['D']:.3f}); raise --lr "
                         f"or the cap")
    params0 = {k: v.copy() for k, v in params.items()}
    print(f"K={a.K} beta={a.beta} seed={a.seed}: pretrain gate at step {pre_gate}  "
          f"A {pre_log[-1]['A']:.3f} D {pre_log[-1]['D']:.3f}")

    # ---- inject B ---------------------------------------------------------------------
    inj_log = []
    def inj_eval(step, p):
        rec = dict(step=step,
                   A=M.accuracy(p, P.keys_a, P.vals_a),
                   A_own=M.accuracy(p, P.keys_a, P.vals_a, restrict=P.x_ids),
                   D=M.accuracy(p, P.keys_d, P.vals_d),
                   B=M.accuracy(p, P.keys_b, P.vals_b))
        rec.update(mech(p, params0, P.keys_a))
        inj_log.append(rec)
    params, b_gate = run_phase(params, P.keys_b, P.vals_b, inject_lr, a.batch, rng, inj_eval,
                               lambda p: M.accuracy(p, P.keys_b, P.vals_b),
                               a.inject_cap, extra_after_gate=a.extra_after_gate,
                               min_steps=a.min_inject, fix_U=a.fix_U)
    A = np.array([r["A"] for r in inj_log]); st = np.array([r["step"] for r in inj_log])
    tr = int(np.argmin(A))
    print(f"   inject: B gate at {b_gate}   A {A[0]:.3f} -> trough {A[tr]:.3f}@{st[tr]} -> "
          f"peak-after {A[tr:].max():.3f}@{st[tr + int(np.argmax(A[tr:]))]} -> end {A[-1]:.3f}")

    os.makedirs(a.out, exist_ok=True)
    stem = f"K{a.K}-beta{a.beta}-seed{a.seed}" + ("-fixU" if a.fix_U else "")
    json.dump(dict(args=vars(a), inject_lr=inject_lr, pretrain_gate=pre_gate, b_gate=b_gate,
                   pretrain=pre_log, inject=inj_log),
              open(os.path.join(a.out, f"{stem}.json"), "w"), indent=1)
    print(f"   wrote {os.path.join(a.out, stem)}.json")


if __name__ == "__main__":
    main()
