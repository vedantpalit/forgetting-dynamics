"""Injection driver for the 8-layer/512-dim scale check. Continues from one of the
dense pretrain checkpoints saved by scale8_pretrain.py (pretrain-{key}-step{N}.msgpack),
selected via --pretrain_step -- both step 6000 and step 16000 are being run, since
dataA/ballast were already at ceiling by step 6000 with no further change through
16000, while dataC's hallucination_gap kept drifting upward the whole way, a genuine
ambiguity about which checkpoint to treat as "the" pretrained model, not resolved by
just picking whichever step is last.

Purpose is narrow, per the rebuild's own scope: does the crash-recovery-decline shape
documented at 4 layers appear at this scale. high_overlap and disjoint, three seeds
each, 1200-step injection, dense checkpoint schedule matching the crossover
(0,10,25,50,100,200,400,600,800,1000,1200).

Run:
  uv run python -m src.experiments.scale8_injection --pretrain_step 16000 --inject_condition disjoint --inject_seed 0
"""
import argparse
import os
import sys
from dataclasses import replace

import wandb

from src.config import parse_config
from src.experiments.ckpt import experiment_key, load_state, save_params
from src.experiments.knowledge_injection import (
    B_HALF, INJECT_LR_RATIO, KIConfig, build, get_partition, inherit_moments, init_state,
    make_optimizer, own_half_baseline_nats, parse_steps, population_baseline_nats,
    pretrain_key, reset_stream, save_exposures, train_loop,
)
from src.experiments.scale8_common import MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, verify_architecture
from src.data.biography import NUM_ATTRIBUTES

DENSE_CHECKPOINT_STEPS = "0,10,25,50,100,200,400,600,800,1000,1200"
PRETRAIN_TOTAL_STEPS = 16000  # must match scale8_pretrain.py exactly -- part of the key
PRETRAIN_PEAK_LR = 5e-4


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pretrain_step", type=int, required=True,
                        choices=list(range(6000, 16001, 1000)))
    extra.add_argument("--inject_condition",
                       choices=["high_overlap", "disjoint", "all_values"], required=True)
    extra.add_argument("--inject_seed", type=int, required=True)
    known, remaining = extra.parse_known_args()

    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--model.model_dim", str(MODEL_DIM), "--model.num_heads", str(NUM_HEADS),
        "--model.num_layers", str(NUM_LAYERS), "--model.dropout_rate", "0",
        "--model.mlp_coefficient", str(MLP_COEFFICIENT),
        "--pretrain.total_steps", str(PRETRAIN_TOTAL_STEPS), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", known.inject_condition, "--inject.seed", str(known.inject_seed),
        "--inject.total_steps", "1200", "--inject.batch_size", "256", "--inject.eval_interval", "10",
        "--inject.checkpoint_steps", DENSE_CHECKPOINT_STEPS,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "offline",
    ] + remaining
    cfg = parse_config(KIConfig, description="8-layer scale check: injection")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)

    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = os.path.join(cfg.checkpoint_dir,
                             f"pretrain-{pre_key}-step{known.pretrain_step:05d}.msgpack")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(
            f"No pretrain checkpoint at {pre_path}. Check --pretrain_step matches one of "
            f"the steps scale8_pretrain.py actually saved (6000-16000, every 1000).")
    print(f"Continuing from pretrain step {known.pretrain_step}: {pre_path}")

    inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    n_layers, d_model, n_params = verify_architecture(state.params, MODEL_DIM, NUM_HEADS, NUM_LAYERS)

    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))
    print(f"Optimizer: {'inherited' if not inject_cfg.fresh_optimizer else 'fresh'}")
    print(f"pre_step (global step counter carried into injection) = {pre_step}")
    if pre_step != known.pretrain_step:
        raise ValueError(f"Loaded checkpoint's own step counter ({pre_step}) doesn't match "
                          f"the requested --pretrain_step ({known.pretrain_step}) -- "
                          f"the checkpoint filename and its contents disagree.")

    reset_stream(data_b, cfg.seed + 5 + inject_cfg.seed)
    # `if d` is not the right test for ballast -- see scale8_pretrain._nonempty. With
    # --num_ballast 0 an empty ballast dataset is truthy and crashes inside evaluate().
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c)
                 if d is not None and len(d.person_ids) > 0]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="scale8inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
        pretrain_step=known.pretrain_step, seed=cfg.seed)
    print(f"Injecting on dataB, pretrain_step={known.pretrain_step}, "
          f"condition={known.inject_condition}, seed={known.inject_seed}, "
          f"checkpoints at {sorted(ckpt_steps)}, key={key}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir,
                          f"scale8inject-p{known.pretrain_step}-{key}-step{step:04d}.msgpack")
        save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}")

    baseline = population_baseline_nats(pop)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    b_half = B_HALF[inject_cfg.condition]
    own_halves = {"dataA": "X", "ballast": "Y"}
    if b_half in ("X", "Y"):
        own_halves["dataB"] = b_half
    if cfg.c_half in ("X", "Y"):
        own_halves["dataC"] = cfg.c_half
    rank_ctx = {
        "x_ids": x_ids, "y_ids": y_ids, "own_halves": own_halves,
        "baseline_own": {"X": own_half_baseline_nats(halves, "X"),
                          "Y": own_half_baseline_nats(halves, "Y")},
    }

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode,
                    name=f"scale8-inject-p{known.pretrain_step}-{known.inject_condition}",
                    config={"pretrain_step": known.pretrain_step,
                            "condition": known.inject_condition, "seed": known.inject_seed}):
        state, final_loss, _ = train_loop(
            state, data_b, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)
    save_exposures(cfg, data_b, f"scale8inject-p{known.pretrain_step}-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")


if __name__ == "__main__":
    main()
