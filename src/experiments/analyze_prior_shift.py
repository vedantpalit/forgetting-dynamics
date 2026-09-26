"""Prior-shift diagnostic for Stage 1 injection.

Question: is the retention drop driven by an OUTPUT-MARGINAL shift — injection evidences
"the untrained half's values don't occur", suppressing that whole output region as a
blunt prior, independent of any individual's key — or by actual fact destruction?

Under the prior-shift account, ballast's argmax should die (its correct answers all live
in the suppressed half) while its facts stay intact: rank of the correct value should
stay ~1 when compared only against candidates from its OWN half, but be pushed down when
compared against the full pool (both halves) — because it is specifically OTHER-half
candidates outranking it, not other same-half individuals' facts crowding it out.

Three measurements, at each dense injection checkpoint (params-only, already on disk):
  1. Ballast: retention (from attribute_loss) vs. accuracy, every dense checkpoint.
  2. Same for A.
  3. THE decisive one: rank of the correct value-token within its own half vs. within
     the full pool, per population, per checkpoint.

Reuses `BiographyDataset.evaluate()` for (1)/(2) rather than reimplementing loss/accuracy.
Only the rank computation in (3) is new.

Run with the SAME flags used for the real injection job (config must match exactly --
it is used to reconstruct the checkpoint filenames and the population/value-half
assignment), e.g. with the injection's defaults:

  uv run python -m src.experiments.analyze_prior_shift \\
    --num_a 2000 --num_ballast 2000 --num_b 500 \\
    --pretrain.total_steps 8000 --pretrain.batch_size 256 --pretrain.opt.peak_lr 5e-4 \\
    --inject.condition high_overlap --inject.total_steps 400 --inject.batch_size 256 \\
    --inject.eval_interval 10 --seed 42
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from src.config import parse_config
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    B_HALF, NUM_ATTRIBUTES, INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key,
    init_state, load_meta, make_optimizer, meta_path, parse_steps,
    population_baseline_nats, pretrain_key,
)
from src.train import eval_forward


def first_token_positions(mask: np.ndarray, targets: np.ndarray, k: int):
    """Row-aligned (col, correct_token_id) for attribute k's first value-token position.

    Every biography contains exactly one first-token position per attribute (all 6
    attributes appear once per biography, just in random order), so this must find
    exactly one match per row.
    """
    rows, cols = np.where(mask == (k + 1))
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    assert len(rows) == mask.shape[0] and np.array_equal(rows, np.arange(mask.shape[0])), (
        f"attribute {k}: expected exactly one first-token position per row; "
        f"got {len(rows)} for {mask.shape[0]} rows — biography generation invariant broken?"
    )
    return cols, targets[rows, cols]


def rank_within(row_logits: np.ndarray, correct_token: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    """1-indexed rank of the correct token's logit among `candidate_ids` (1 = top choice)."""
    cand_logits = row_logits[:, candidate_ids]
    correct_logit = row_logits[np.arange(len(correct_token)), correct_token]
    return (cand_logits > correct_logit[:, None]).sum(axis=1) + 1


def value_ranks(model, params, ds, own_half: str, x_ids, y_ids, batch_size=256):
    """Per-person, per-attribute rank of the correct value among (a) its own half's
    candidates only, (b) the full pool (both halves). Returns two (N, 6) int arrays.
    """
    n = ds.eval_inputs.shape[0]
    own_rank = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int32)
    full_rank = np.zeros((n, NUM_ATTRIBUTES), dtype=np.int32)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(model.apply, params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        b = sl.stop - sl.start
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            row_logits = logits[np.arange(b), cols, :]
            own_ids = x_ids[k] if own_half == "X" else y_ids[k]
            assert np.isin(correct, own_ids).all(), (
                f"attribute {k}: some '{ds.name}' values are not in half {own_half} — "
                f"own_half assumption is wrong for this population/condition."
            )
            full_ids = np.concatenate([x_ids[k], y_ids[k]])
            own_rank[start:start + b, k] = rank_within(row_logits, correct, own_ids)
            full_rank[start:start + b, k] = rank_within(row_logits, correct, full_ids)
    return own_rank, full_rank


def summarize(name, step, acc, attr_loss, retention, own_rank, full_rank):
    frac_top_own = float((own_rank == 1).mean())
    frac_top_full = float((full_rank == 1).mean())
    print(f"  {name:8s} step={step:4d}  first_acc={acc:.3f}  attr_loss={attr_loss:.3f}  "
          f"retention={retention:6.3f}  "
          f"rank_own(mean={own_rank.mean():5.2f}, top1={frac_top_own:.3f})  "
          f"rank_full(mean={full_rank.mean():5.2f}, top1={frac_top_full:.3f})")


def main():
    cfg = parse_config(KIConfig, description="Prior-shift diagnostic for Stage 1 injection")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    baseline = population_baseline_nats(pop)

    halves = None
    from src.experiments.knowledge_injection import get_partition
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]

    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(f"No pretraining checkpoint at {pre_path}")

    # Replicate phase_inject's exact LR resolution -- the checkpoint key depends on it.
    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        meta = load_meta(pre_path)
        inject_opt = replace(inject_opt, peak_lr=float(meta["peak_lr"]) / INJECT_LR_RATIO)
    inject_cfg = replace(cfg.inject, opt=inject_opt)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key, seed=cfg.seed)
    print(f"Injection key: {key}  (peak_lr={inject_opt.peak_lr:g})")

    steps = sorted(parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps))
    b_half = B_HALF[inject_cfg.condition]
    populations = [
        ("dataA", data_a, "X"), ("ballast", data_ballast, "Y"),
        ("dataB", data_b, b_half), ("dataC", data_c, cfg.c_half),
    ]
    populations = [(n, d, h) for n, d, h in populations if d is not None and h != "mixed"]

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    l_pretrained = {}
    print(f"\n{'=' * 100}\nSteps: {steps}\n{'=' * 100}")
    for step in steps:
        p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
        if not os.path.exists(p):
            print(f"  [missing: {p} -- skipping step {step}]")
            continue
        params = load_params(p, template_state.params)
        for name, ds, own_half in populations:
            forward = lambda x: eval_forward(model.apply, params, x)
            m = ds.evaluate(forward, cfg.data.eval_batch_size)
            attr_loss = m[f"{name}/attribute_loss"]
            if step == steps[0]:
                l_pretrained[name] = attr_loss
            denom = baseline - l_pretrained[name]
            retention = (baseline - attr_loss) / denom if abs(denom) > 1e-6 else float("nan")
            own_rank, full_rank = value_ranks(model, params, ds, own_half, x_ids, y_ids)
            summarize(name, step, m[f"{name}/first_token_accuracy"], attr_loss, retention,
                      own_rank, full_rank)
        print()


if __name__ == "__main__":
    main()
