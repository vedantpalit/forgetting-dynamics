"""Second normalization for the region-size-vs-concentration follow-up: ballast's MEAN
rank within its own half, at the trough (step 50, confirmed the same step across all 9
runs in the sweep), as a FRACTION of ballast's own pool size -- not the top-1 rate, which
isn't comparable across conditions whose own-half pool sizes differ (top-1-of-33 and
top-1-of-100 are different bars). rank_own_mean was computed by population_rank_metrics
during injection but never printed to the console/.out logs, only rank_own_top1 was -- this
reads it directly from the already-saved step-50 checkpoints instead of retraining anything.

No new phase, no new condition: same high_overlap / 400-step / partition_path pattern as
every other analyze_*.py script this project has, just looped over the three partition
files already in hand (ipv_control, conc_out66, f0.25).

Run with the SAME flags as the injection, once per partition -- used to reconstruct the checkpoint
filenames and the population/value-half assignment.
"""
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.config import parse_config
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key, first_token_positions,
    get_partition, init_state, load_meta, make_optimizer, meta_path, pretrain_key,
    rank_and_loss_within,
)
from src.train import eval_forward

TROUGH_STEP = 50  # confirmed identical across all 9 (condition, seed) runs in the sweep


def ballast_rank_mean(model, params, ds, own_ids_k, k, batch_size=256):
    """Mean rank of the correct value within ballast's own half, for attribute k, over
    the whole eval set -- rank_own_mean, the quantity population_rank_metrics computes
    internally but that never made it into the printed logs.
    """
    n = ds.eval_inputs.shape[0]
    ranks = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        cols, correct = first_token_positions(mask, targets, k)
        row_logits = logits[np.arange(sl.stop - sl.start), cols, :]
        rank, _ = rank_and_loss_within(row_logits, correct, own_ids_k)
        ranks.append(rank)
    return float(np.concatenate(ranks).mean())


def main():
    cfg = parse_config(KIConfig, description="Mean-rank/pool-fraction normalization")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(6)]

    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path}")

    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        meta = load_meta(pre_path)
        inject_opt = replace(inject_opt, peak_lr=float(meta["peak_lr"]) / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)
    print(f"Injection key: {key}  partition_path={cfg.partition_path}")

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{TROUGH_STEP:04d}.msgpack")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No step-{TROUGH_STEP} checkpoint at {p}")
    params = load_params(p, template_state.params)

    print(f"\nballast rank_own_mean at step {TROUGH_STEP}, per attribute and pooled:")
    per_attr_ranks = []
    for k in range(6):
        pool_k = len(y_ids[k])
        rmean = ballast_rank_mean(model, params, data_ballast, y_ids[k], k)
        frac = rmean / pool_k
        per_attr_ranks.append((rmean, pool_k, frac))
        print(f"  attr {k}: rank_own_mean={rmean:7.3f}  pool={pool_k:4d}  "
              f"rank/pool={frac:.4f}")

    pooled_frac = float(np.mean([f for _, _, f in per_attr_ranks]))
    pooled_rank = float(np.mean([r for r, _, _ in per_attr_ranks]))
    pooled_pool = float(np.mean([n for _, n, _ in per_attr_ranks]))
    print(f"\n  POOLED (mean over 6 attributes): rank_own_mean={pooled_rank:.3f}  "
          f"mean_pool_size={pooled_pool:.1f}  mean(rank/pool)={pooled_frac:.4f}")


if __name__ == "__main__":
    main()
