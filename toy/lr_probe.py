"""Quick LR probe for pretraining (PLAN.md §7.3). Picks a single default LR,
held fixed across all conditions per the user's explicit instruction — the
three-point grid around it is logged as robustness only, never a
per-condition tuning knob (a per-condition LR would make trough depth a
function of tuning rather than of the concentration/overlap knobs)."""

import numpy as np

import config as C
import populations as P
import train as T

LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
N_SEEDS = 5
PROBE_BUDGET = 2000
STABILITY_CHECK_STEPS = 200


def probe_one(lr, seed):
    cfg = C.ToyConfig(
        population=C.PopulationConfig(overlap=0.5, concentration=8),
        protocol=C.ProtocolConfig(pretrain_max_steps=PROBE_BUDGET,
                                   pretrain_ceiling_window=20,
                                   pretrain_extra_fraction=0.0),
        optimizer=C.OptimizerConfig(lr=lr),
        seed=seed,
    )
    ps = P.build_populations(cfg)
    try:
        store, history, ceiling_step, target_step = T.pretrain(cfg, ps, verbose=False)
    except RuntimeError:
        return {"lr": lr, "seed": seed, "reached_ceiling": False, "diverged": None,
                "stable": False, "ceiling_step": None, "final_acc": None}

    accs = np.array([h["acc_a"] for h in history])
    diverged = bool(np.any(np.isnan(accs)) or np.any(accs < -1))  # sanity guard
    stable = True
    if ceiling_step is not None:
        tail_start = min(ceiling_step + STABILITY_CHECK_STEPS, len(accs) - 1)
        stable = bool(accs[ceiling_step:tail_start + 1].min() >= 0.99 - 1e-9)

    return {
        "lr": lr, "seed": seed, "reached_ceiling": ceiling_step is not None,
        "ceiling_step": ceiling_step, "diverged": diverged, "stable": stable,
        "final_acc": float(accs[-1]),
    }


def run_probe():
    results = []
    for lr in LR_GRID:
        for seed in range(N_SEEDS):
            r = probe_one(lr, seed)
            results.append(r)
            print(f"lr={lr:<6g} seed={seed}  reached={r.get('reached_ceiling')}  "
                  f"ceiling_step={r.get('ceiling_step')}  stable={r.get('stable')}  "
                  f"final_acc={r.get('final_acc')}")

    print()
    print(f"{'lr':>8} {'n_reached':>10} {'n_stable':>9} {'mean_ceiling_step':>18}")
    summary = {}
    for lr in LR_GRID:
        rows = [r for r in results if r["lr"] == lr]
        n_reached = sum(1 for r in rows if r["reached_ceiling"])
        n_stable = sum(1 for r in rows if r["reached_ceiling"] and r.get("stable"))
        ceiling_steps = [r["ceiling_step"] for r in rows if r["reached_ceiling"]]
        mean_ceiling = float(np.mean(ceiling_steps)) if ceiling_steps else None
        summary[lr] = {"n_reached": n_reached, "n_stable": n_stable, "mean_ceiling_step": mean_ceiling}
        print(f"{lr:>8g} {n_reached:>10} {n_stable:>9} {str(mean_ceiling):>18}")

    candidates = [lr for lr in LR_GRID
                  if summary[lr]["n_reached"] == N_SEEDS and summary[lr]["n_stable"] == N_SEEDS]
    chosen = min(candidates) if candidates else None
    print()
    if chosen is not None:
        print(f"CHOSEN DEFAULT LR: {chosen} "
              f"(smallest LR with {N_SEEDS}/{N_SEEDS} seeds reaching ceiling and staying stable)")
    else:
        print("NO LR reached ceiling+stability in all 5 seeds within the probed grid — "
              "widen the grid before proceeding.")
    return results, summary, chosen


if __name__ == "__main__":
    run_probe()
