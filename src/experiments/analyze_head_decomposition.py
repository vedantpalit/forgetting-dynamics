"""Head decomposition: which term carries the behavioural recovery.

`analyze_prior_shift.py` established, behaviourally, that ballast's rank/accuracy against
the FULL candidate pool crashes fast then partially recovers (e.g. 50->200), while rank
against its OWN half stays high throughout. `analyze_weight_localization.py` then showed
the head bias for half-Y keeps getting MORE negative every phase, including during that
same recovery window -- a bias-only account cannot explain the recovery, since the thing
recovering and the thing measured are both getting worse at the same time.

The reconciliation this script tests: the head is `logit = h @ kernel + bias`. Accuracy and
rank are decided by the GAP between the correct token's logit and the best-scoring wrong
candidate's, not by either logit in isolation. Bias is shared between the correct answer and
every other candidate IN THE SAME HALF, so it mostly cancels in `gap_own` -- a bias that
deepens uniformly across half-Y suppresses the whole region against half-X (tanking
`gap_full`, hence rank_full/accuracy) without touching the ordering within half-Y at all
(leaving `gap_own`, hence rank_own, untouched by the bias term). If that is right, the
BIAS component should dominate `gap_full`'s trajectory and be roughly flat in `gap_own`,
while the KERNEL (context-dependent, h-mediated) component should be what actually moves
during recovery in both, and specifically be responsible for whatever recovery `gap_own`
shows in isolation.

Two gaps, both decomposed into bias_gap + kernel_gap:
  gap_full = logit[correct] - logit[best wrong candidate in the ATTRIBUTE'S FULL pool, X union Y]
  gap_own  = logit[correct] - logit[best wrong candidate in the population's OWN half only]
These map exactly onto the existing rank_full / rank_own metrics.

Populations: dataA (own_half X) is included alongside ballast (own_half Y) as a contrast --
A shares B's half in `high_overlap`, so if A shows the same suppress-then-recover shape in
this decomposition, the "suppression is targeted at half-Y specifically" reading is wrong.

Non-value check: the sharp test in analyze_weight_localization.py found non-value tokens'
bias ALSO drifts negative (-0.00015 in phase 0->10), smaller than half-Y's -0.00022 but not
zero -- flagged there as a softmax-competition effect, not explained. If that competition is
real, it should show up as a KERNEL-side effect at the value-prediction position (the model
learning, in context, to push non-value tokens' logits down when a value is expected) rather
than only riding on the same context-free bias drift. This script checks both: bias[non-value
tokens] (context-free, one number per checkpoint) against the mean kernel contribution
h @ kernel[:, non-value tokens] evaluated AT the value-prediction position (context-dependent,
per population per attribute).

Every decomposed gap is checked against the directly-read logit gap every batch
(bias_gap + kernel_gap == logit[correct] - logit[wrong], atol 1e-3) -- this is the
falsifier for having grabbed the wrong hidden state, not a formality.

Run with the SAME flags as the injection / analyze_prior_shift.py -- used to reconstruct the
checkpoint filenames and the population/value-half assignment.
"""
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from src.config import parse_config
from src.data.biography_corpus import ATTRIBUTE_NAMES
from src.experiments.ckpt import load_params
from src.experiments.knowledge_injection import (
    NUM_ATTRIBUTES, INJECT_LR_RATIO, KIConfig, build, ckpt_path, experiment_key,
    first_token_positions, get_partition, init_state, load_meta, make_optimizer,
    meta_path, parse_steps, pretrain_key, token_half_groups,
)


def forward_with_hidden(model, params, inputs):
    """Forward pass that also returns `h`, the backbone's output -- the head's input.
    `capture_intermediates=True` sows every submodule's __call__ output; the backbone is
    a plain constructor-supplied submodule of SequenceModel, so it is keyed by its
    attribute name, "backbone". If that naming is wrong for some flax version, this
    raises immediately rather than silently reading the wrong tensor -- the decomposition
    check below would catch it anyway, but this gives a clearer error first.
    """
    logits, mutated = model.apply(
        {"params": params}, inputs, deterministic=True,
        capture_intermediates=True, mutable=["intermediates"])
    try:
        h = mutated["intermediates"]["backbone"]["__call__"][0]
    except KeyError as exc:
        raise KeyError(
            f"Expected intermediates['backbone']['__call__']; got top-level intermediate "
            f"keys {list(mutated.get('intermediates', {}).keys())}. capture_intermediates "
            f"naming differs from what this script assumes -- inspect `mutated` and fix "
            f"the path before trusting any number below."
        ) from exc
    return np.asarray(logits), np.asarray(h)


def best_wrong(row_logits: np.ndarray, correct: np.ndarray, candidate_ids: np.ndarray):
    """Per row: (token id, RAW LOGIT) of the best-scoring candidate in candidate_ids that
    is NOT the correct token -- not the gap. candidate_ids is the same fixed array for
    every row in the batch (one attribute's value pool); correct varies per row, so the
    correct column is masked out per row, not once for the whole batch.
    """
    cand_logits = row_logits[:, candidate_ids]
    is_correct_col = candidate_ids[None, :] == correct[:, None]
    masked = np.where(is_correct_col, -np.inf, cand_logits)
    idx = masked.argmax(axis=1)
    return candidate_ids[idx], masked[np.arange(len(correct)), idx]


def decompose_gap(h_row: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                  correct: np.ndarray, wrong: np.ndarray):
    """gap = logit[correct] - logit[wrong] = bias_gap + kernel_gap, both (n,)."""
    bias_gap = bias[correct] - bias[wrong]
    kernel_gap = np.einsum("nd,nd->n", h_row, (kernel[:, correct] - kernel[:, wrong]).T)
    return bias_gap, kernel_gap


def population_decomposition(model, params, ds, own_half: str, x_ids, y_ids,
                             nonvalue_ids: np.ndarray, batch_size=256) -> dict:
    """Per-attribute arrays of bias_gap/kernel_gap (own and full), the fraction of
    best-wrong candidates that come from the OPPOSITE half (direct evidence of cross-half
    competition), and the non-value kernel-contribution check, at ONE checkpoint.
    """
    kernel = params["head"]["Dense_0"]["kernel"]  # (model_dim, vocab)
    bias = np.asarray(params["head"]["Dense_0"]["bias"])  # (vocab,)
    n = ds.eval_inputs.shape[0]
    per_attr = {k: {name: [] for name in (
        "bias_gap_full", "kernel_gap_full", "bias_gap_own", "kernel_gap_own",
        "wrong_full_opposite_half", "nonvalue_kernel")} for k in range(NUM_ATTRIBUTES)}
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits, h = forward_with_hidden(model, params, jnp.array(ds.eval_inputs[sl]))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        b = sl.stop - sl.start
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            rows = np.arange(b)
            row_logits = logits[rows, cols, :]
            h_row = h[rows, cols, :]

            own_ids = x_ids[k] if own_half == "X" else y_ids[k]
            opposite_ids = y_ids[k] if own_half == "X" else x_ids[k]
            full_ids = np.concatenate([x_ids[k], y_ids[k]])

            correct_logit = row_logits[rows, correct]
            wrong_full, wrong_full_logit = best_wrong(row_logits, correct, full_ids)
            wrong_own, wrong_own_logit = best_wrong(row_logits, correct, own_ids)
            direct_full = correct_logit - wrong_full_logit
            direct_own = correct_logit - wrong_own_logit

            bg_full, kg_full = decompose_gap(h_row, kernel, bias, correct, wrong_full)
            bg_own, kg_own = decompose_gap(h_row, kernel, bias, correct, wrong_own)
            assert np.allclose(bg_full + kg_full, direct_full, atol=1e-3), (
                f"attribute {k}: gap_full decomposition does not match the directly-read "
                f"logit gap -- wrong hidden state captured (see forward_with_hidden)."
            )
            assert np.allclose(bg_own + kg_own, direct_own, atol=1e-3), (
                f"attribute {k}: gap_own decomposition does not match the directly-read "
                f"logit gap -- wrong hidden state captured (see forward_with_hidden)."
            )

            nv_kernel = h_row @ kernel[:, nonvalue_ids]  # (b, n_nonvalue)

            d = per_attr[k]
            d["bias_gap_full"].append(bg_full)
            d["kernel_gap_full"].append(kg_full)
            d["bias_gap_own"].append(bg_own)
            d["kernel_gap_own"].append(kg_own)
            d["wrong_full_opposite_half"].append(np.isin(wrong_full, opposite_ids))
            d["nonvalue_kernel"].append(nv_kernel.mean(axis=1))
    return {k: {name: np.concatenate(v) for name, v in d.items()} for k, d in per_attr.items()}


def fmt_row(label, step, own_half, per_attr, nonvalue_bias):
    pooled = {name: np.concatenate([per_attr[k][name] for k in range(NUM_ATTRIBUTES)])
             for name in per_attr[0]}
    gap_full = pooled["bias_gap_full"] + pooled["kernel_gap_full"]
    gap_own = pooled["bias_gap_own"] + pooled["kernel_gap_own"]
    print(f"  {label:8s} step={step:4d} (own={own_half})  "
          f"gap_full={gap_full.mean():+8.4f} (bias={pooled['bias_gap_full'].mean():+8.4f} "
          f"kernel={pooled['kernel_gap_full'].mean():+8.4f})  "
          f"gap_own={gap_own.mean():+8.4f} (bias={pooled['bias_gap_own'].mean():+8.4f} "
          f"kernel={pooled['kernel_gap_own'].mean():+8.4f})  "
          f"wrong_full_opp_half={pooled['wrong_full_opposite_half'].mean():.3f}  "
          f"nonvalue_kernel={pooled['nonvalue_kernel'].mean():+8.5f} "
          f"(nonvalue_bias={nonvalue_bias:+8.5f})")


def fmt_per_attr(label, step, per_attr):
    for k in range(NUM_ATTRIBUTES):
        d = per_attr[k]
        gap_full = d["bias_gap_full"] + d["kernel_gap_full"]
        gap_own = d["bias_gap_own"] + d["kernel_gap_own"]
        print(f"    {label:8s} step={step:4d} {ATTRIBUTE_NAMES[k]:14s}  "
              f"gap_full={gap_full.mean():+8.4f} (bias={d['bias_gap_full'].mean():+8.4f} "
              f"kernel={d['kernel_gap_full'].mean():+8.4f})  "
              f"gap_own={gap_own.mean():+8.4f} (bias={d['bias_gap_own'].mean():+8.4f} "
              f"kernel={d['kernel_gap_own'].mean():+8.4f})  "
              f"wrong_full_opp_half={d['wrong_full_opposite_half'].mean():.3f}  "
              f"nonvalue_kernel={d['nonvalue_kernel'].mean():+8.5f}")


def main():
    cfg = parse_config(KIConfig, description="Head logit decomposition for Stage 1 injection")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    _, _, _, nonvalue_ids = token_half_groups(pop, halves)
    nonvalue_ids = np.array(sorted(nonvalue_ids))
    print(f"Non-value tokens: {len(nonvalue_ids)} of {pop.vocab_size} vocab tokens.")

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
    print(f"Injection key: {key}  (peak_lr={inject_opt.peak_lr:g})")

    steps = sorted(parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps))
    populations = [("dataA", data_a, "X"), ("ballast", data_ballast, "Y")]

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    template_state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)

    print(f"\n{'=' * 120}\nPooled gap decomposition (mean over all 6 attributes), "
          f"gap = bias_gap + kernel_gap = logit[correct] - logit[best wrong candidate]\n"
          f"gap_full uses the attribute's full pool (X union Y) -- what decides rank_full "
          f"/ accuracy. gap_own uses only the population's own half -- what decides "
          f"rank_own. wrong_full_opp_half = fraction of rows where the best FULL-pool "
          f"competitor comes from the OTHER half (direct evidence of cross-half "
          f"competition, not same-half crowding).\n{'=' * 120}")
    results_by_step = {}
    for step in steps:
        p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
        if not os.path.exists(p):
            print(f"  [missing: {p} -- skipping step {step}]")
            continue
        params = load_params(p, template_state.params)
        nonvalue_bias = float(np.asarray(params["head"]["Dense_0"]["bias"])[nonvalue_ids].mean())
        step_results = {}
        for label, ds, own_half in populations:
            per_attr = population_decomposition(model, params, ds, own_half, x_ids, y_ids, nonvalue_ids)
            fmt_row(label, step, own_half, per_attr, nonvalue_bias)
            step_results[label] = per_attr
        results_by_step[step] = step_results
        print()

    print(f"\n{'=' * 120}\nPer-attribute breakdown\n{'=' * 120}")
    for step, step_results in results_by_step.items():
        for label, per_attr in step_results.items():
            fmt_per_attr(label, step, per_attr)
        print()


if __name__ == "__main__":
    main()
