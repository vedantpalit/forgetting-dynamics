"""Pretrain driver for the without-ballast arm of the first-look demonstration figure.

Cannot reuse phase_pretrain unchanged: with num_ballast=0, build() itself is safe (verified
locally -- data_ballast comes back as a valid, structurally sound, 0-person dataset), but
phase_pretrain's eval_sets filter (`[d for d in (data_a, data_ballast, data_c) if d]`) still
includes it, since BiographyDataset has no __len__/__bool__ override and an empty dataset is
still truthy -- unlike data_c, which build() already guards with `if len(ids_c) else None`.
Calling .evaluate() on 0 rows crashes: the batch loop never runs, the accumulator stays
None, and _finalize_metrics(None) immediately subscripts it. Confirmed directly (dummy
forward_fn, real empty dataset) before writing this, not just traced through the code.

This driver sidesteps that by simply never putting data_ballast in eval_sets or the gate
check -- the same treatment data_c already gets when absent, just made explicit here since
ballast doesn't get that guard for free.

Uses the STANDARD (non-historical) pretrain_key() -- this is a brand-new run built with
today's code, nothing to reconstruct. num_ballast=0 and the 0.85 partition file's content
both already guarantee a different key from every existing checkpoint (pretrain_key hashes
population sizes and partition-file content), so there is no collision risk with the
existing crossover checkpoint reused by the with-ballast arm.
"""
from dataclasses import replace

import jax
import wandb

from src.config import parse_config
from src.experiments.ckpt import ckpt_path, save_state
from src.experiments.knowledge_injection import (
    KIConfig, build, init_state, make_optimizer, population_baseline_nats, pretrain_key,
    reset_stream, save_exposures, save_meta, train_loop,
)


def main():
    cfg = parse_config(KIConfig, description="Demo figure: pretrain A alone (no ballast)")
    if cfg.num_ballast != 0:
        raise ValueError(f"This driver is for the without-ballast arm specifically; "
                          f"got --num_ballast={cfg.num_ballast}, expected 0.")

    pop, model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.pretrain.max_eval_people)
    assert len(data_ballast.person_ids) == 0, "num_ballast=0 but data_ballast is non-empty"

    reset_stream(data_pre, cfg.seed + 1)
    tx = make_optimizer(cfg.pretrain.opt, cfg.pretrain.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
    print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(state.params)):,}")

    key = pretrain_key(cfg, data_cfg)
    path = ckpt_path(cfg.checkpoint_dir, "pretrain", key)
    baseline = population_baseline_nats(pop)

    # No ballast in eval_sets -- see module docstring for exactly why this can't be the
    # standard [d for d in (data_a, data_ballast, data_c) if d] filter.
    eval_sets = [d for d in (data_a, data_c) if d]

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="pretrain-no-ballast",
                    config={"num_ballast": 0, "arm": "without_ballast"}):
        state, final_loss, metrics = train_loop(
            state, data_pre, cfg.pretrain.total_steps, cfg.pretrain.batch_size,
            eval_sets, cfg, eval_interval=cfg.pretrain.eval_interval, baseline=baseline)
    acc = metrics["dataA/first_token_accuracy"]

    if path:
        save_state(path, state)
        save_meta(path, peak_lr=cfg.pretrain.opt.peak_lr, schedule=cfg.pretrain.opt.schedule,
                  total_steps=cfg.pretrain.total_steps, batch_size=cfg.pretrain.batch_size,
                  final_train_loss=final_loss, dataA_first_token_accuracy=acc,
                  baseline_nats=baseline, key=key, arm="without_ballast")
        print(f"Saved {path}")
    save_exposures(cfg, data_pre, f"pretrain-{key}")

    gate_passed = acc >= cfg.pretrain.target_acc
    print(f"\n=== GATE (no-ballast arm; only dataA is gated, there is no ballast) === "
          f"dataA/first_token_accuracy = {acc:.4f} (target {cfg.pretrain.target_acc}), "
          f"final train loss = {final_loss:.4f}")
    if not gate_passed:
        print("BELOW TARGET -- facts not learned. Injection results would be uninterpretable. "
              "Not proceeding.")
    return state, final_loss, acc, key


if __name__ == "__main__":
    main()
