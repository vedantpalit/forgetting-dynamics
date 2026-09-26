"""Short pretrain-LR probe for the 8-layer/512-dim scale check, run before committing to
the full pretrain. The pilot's own 5-point LR sweep (2e-4 to 5e-3, a 25x range) didn't
discriminate at 4 layers -- everything in the grid worked about equally well, so 5e-4
was never actually validated as *the* right choice there, only *a* working one. At 8x
the parameters that may not hold, so this checks it directly rather than assuming the
pilot's choice carries over.

Not a full pretrain: each grid point gets a short, identical step budget, just enough
to see which LRs make healthy progress vs. diverge or stall. Same population/partition/
batch size as everything else in this project -- only the LR varies across points, and
model shape is fixed at the exact 8-layer/512-dim spec (scale8_common), never left to a
CLI flag.

Run:
  uv run python -m src.experiments.scale8_lr_probe
"""
import sys

import wandb

from src.config import parse_config
from src.experiments.knowledge_injection import (
    KIConfig, PRETRAIN_LR_GRID, build, init_state, make_optimizer,
    population_baseline_nats, reset_stream, train_loop,
)
from src.experiments.scale8_common import (
    MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, verify_architecture,
)
from dataclasses import replace

PROBE_STEPS = 1000
PROBE_EVAL_INTERVAL = 100


def main():
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.batch_size", "256",
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ] + sys.argv[1:]
    cfg = parse_config(KIConfig, description="8-layer scale check: pretrain LR probe")
    pop, model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c = build(
        cfg, max_eval_people=500)

    tx0 = make_optimizer(replace(cfg.pretrain.opt, peak_lr=PRETRAIN_LR_GRID[0]), PROBE_STEPS)
    state0 = init_state(model, cfg, data_cfg, tx0, cfg.seed + 10)
    verify_architecture(state0.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)

    baseline = population_baseline_nats(pop)
    eval_sets = [d for d in (data_a, data_ballast) if d]

    results = []
    for lr in PRETRAIN_LR_GRID:
        print(f"\n{'=' * 60}\nLR probe: peak_lr = {lr:g}  ({PROBE_STEPS} steps)\n{'=' * 60}")
        reset_stream(data_pre, cfg.seed + 1)
        tx = make_optimizer(replace(cfg.pretrain.opt, peak_lr=lr), PROBE_STEPS)
        state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
        with wandb.init(project=cfg.wandb_project, mode="disabled"):
            state, final_loss, metrics = train_loop(
                state, data_pre, PROBE_STEPS, cfg.pretrain.batch_size, eval_sets, cfg,
                eval_interval=PROBE_EVAL_INTERVAL, baseline=baseline)
        acc_a = metrics["dataA/first_token_accuracy"]
        acc_ballast = metrics["ballast/first_token_accuracy"]
        results.append((lr, final_loss, acc_a, acc_ballast))
        print(f"  -> final_loss={final_loss:.4f}  dataA_acc={acc_a:.4f}  ballast_acc={acc_ballast:.4f}")

    print(f"\n{'=' * 60}\nProbe summary ({PROBE_STEPS} steps each)\n{'=' * 60}")
    print(f"{'peak_lr':>10}  {'final_loss':>11}  {'dataA_acc':>10}  {'ballast_acc':>12}")
    for lr, loss, acc_a, acc_ballast in results:
        flag = ""
        if loss != loss:  # NaN check
            flag = "  <-- DIVERGED (NaN)"
        print(f"{lr:>10.1e}  {loss:>11.4f}  {acc_a:>10.4f}  {acc_ballast:>12.4f}{flag}")

    finite = [r for r in results if r[1] == r[1]]
    if finite:
        best = min(finite, key=lambda r: r[1])
        print(f"\nLowest final_loss at {PROBE_STEPS} steps: peak_lr={best[0]:g} "
              f"(final_loss={best[1]:.4f})")
        if best[0] in (PRETRAIN_LR_GRID[0], PRETRAIN_LR_GRID[-1]):
            print("WARNING: best point is at a grid edge -- consider extending the grid "
                  "before committing to the full pretrain.")
    else:
        print("\nEvery grid point diverged (all NaN) -- something is wrong beyond just "
              "LR choice; do not proceed to the full pretrain.")


if __name__ == "__main__":
    main()
