"""Behavioral and diagnostic metrics for the embed-to-unembed toy model.

Operates on plain numpy arrays (logits, masks, keys) with no dependency on
populations.py / model.py / train.py, so it can be validated standalone
against hand-constructed W/b instances (see metrics_sanity.py) before any
training code exists, per PLAN.md #8.
"""

import numpy as np


def compute_logits(W, b, keys):
    """keys: (n, d). W: (d, |V|). b: (|V|,). Returns (n, |V|)."""
    return keys @ W + b


def _masked(logits_arr, candidate_mask):
    if candidate_mask is None:
        return logits_arr
    out = np.where(candidate_mask[None, :], logits_arr, -np.inf)
    return out


def argmax_correct(logits_arr, correct_idx, candidate_mask=None):
    """Per-individual boolean: does argmax (optionally restricted to
    candidate_mask) land on the correct value."""
    scored = _masked(logits_arr, candidate_mask)
    pred = np.argmax(scored, axis=1)
    return pred == np.asarray(correct_idx)


def rank_of_correct(logits_arr, correct_idx, candidate_mask=None):
    """Per-individual rank (1 = best) of the correct value's logit among the
    candidate set. Ties broken by average rank (competition rank with a
    +0.5 nudge per tied competitor), so a plateau of equal logits doesn't
    produce a misleadingly sharp rank jump."""
    scored = _masked(logits_arr, candidate_mask)
    n = scored.shape[0]
    correct_idx = np.asarray(correct_idx)
    correct_logit = scored[np.arange(n), correct_idx]
    greater = np.sum(scored > correct_logit[:, None], axis=1)
    equal = np.sum(scored == correct_logit[:, None], axis=1) - 1  # exclude self
    return 1.0 + greater + 0.5 * equal


def margin(logits_arr, correct_idx, region_mask):
    """logit(correct) - max(logits over region_mask), per individual."""
    n = logits_arr.shape[0]
    correct_idx = np.asarray(correct_idx)
    correct_logit = logits_arr[np.arange(n), correct_idx]
    region_max = np.max(_masked(logits_arr, region_mask), axis=1)
    return correct_logit - region_max


def mean_max_softmax_confidence(logits_arr):
    """Per-individual max softmax probability (numerically stable)."""
    shifted = logits_arr - logits_arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def softmax_prob_correct(logits_arr, correct_idx):
    """Per-individual softmax probability on the correct value — the raw
    vector for the fact-level distribution dump (PLAN.md #6 item 11)."""
    shifted = logits_arr - logits_arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits_arr.shape[0]
    return probs[np.arange(n), np.asarray(correct_idx)]


def aggregate(per_individual_arr, mask=None):
    """mean/std/n over an optional boolean mask."""
    arr = np.asarray(per_individual_arr)
    if mask is not None:
        arr = arr[np.asarray(mask)]
    n = arr.shape[0]
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(n)}


def touched_untouched_split(per_individual_arr, touched_mask):
    touched_mask = np.asarray(touched_mask)
    return {
        "touched": aggregate(per_individual_arr, touched_mask),
        "untouched": aggregate(per_individual_arr, ~touched_mask),
    }


def _unit(v):
    return v / np.linalg.norm(v)


def decompose_update(delta_W, delta_b, direction_u, shared_basis_Q):
    """Decompose an applied (delta_W, delta_b) into three pieces, in this
    fixed order (PLAN.md #6 item 13's sign convention target):

      1. the rank-1 component of delta_W along `direction_u` (the mean-B-key
         direction) — this is what should carry a planted suppression-style
         uplift;
      2. the component of the *remainder* in the span of `shared_basis_Q`
         (name-part features shared between A and B) — this is what should
         carry a planted erosion-style corruption;
      3. whatever's left (orthogonal remainder).

    Order matters: Q is first re-orthogonalized against u (Gram-Schmidt) so
    the two components don't double-count any overlap between the two
    subspaces. direction_u must be a single (d,) vector; shared_basis_Q a
    (d, k) matrix whose columns need not already be orthonormal.
    """
    u = _unit(np.asarray(direction_u, dtype=float))
    rank1_row = u @ delta_W  # (|V|,) — logit-space effect of the u-direction component
    rank1_component = np.outer(u, rank1_row)  # (d, |V|)
    remainder = delta_W - rank1_component

    Q = np.asarray(shared_basis_Q, dtype=float)
    if Q.ndim == 1:
        Q = Q[:, None]
    # Gram-Schmidt: remove u from each basis column, then orthonormalize Q itself.
    Q = Q - np.outer(u, u @ Q)
    Q_orth, R = np.linalg.qr(Q)
    # QR's sign is not unique (Q_orth could come back as -Q); pin each column
    # to the same orientation as the pre-QR (post-deflation) input, via R's
    # diagonal sign, so the projection's sign is a property of the planted
    # direction, not of the QR implementation.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q_orth = Q_orth * signs[None, :]
    # np.linalg.qr can return more columns than rank; keep only non-degenerate ones.
    keep = np.linalg.norm(Q_orth, axis=0) > 1e-8
    Q_orth = Q_orth[:, keep]

    if Q_orth.shape[1] > 0:
        shared_rows = Q_orth.T @ remainder  # (k, |V|)
        shared_component = Q_orth @ shared_rows
    else:
        shared_rows = np.zeros((0, delta_W.shape[1]))
        shared_component = np.zeros_like(delta_W)

    orthogonal_remainder = remainder - shared_component

    return {
        "bias_norm": float(np.linalg.norm(delta_b)),
        "rank1_component_norm": float(np.linalg.norm(rank1_component)),
        "rank1_row": rank1_row,  # signed, per-value-column effect along u
        "shared_component_norm": float(np.linalg.norm(shared_component)),
        "shared_rows": shared_rows,  # signed, per-value-column effect in shared subspace
        "shared_component": shared_component,  # (d, |V|), for cross-checking against decompose_update_direct
        "orthogonal_remainder_norm": float(np.linalg.norm(orthogonal_remainder)),
    }


def decompose_update_direct(delta_W, delta_b, direction_u, shared_basis_Q):
    """Same three-way split as decompose_update, but never orthonormalizes
    shared_basis_Q. Projects directly onto its (possibly non-orthonormal)
    span by solving the normal equations against its Gram matrix, so the
    sign and scale of the shared-subspace term are inherited from the
    caller's own basis vectors rather than pinned by an orthonormalization
    convention (there is none here — no QR, no sign to fix). Used only as
    an independent cross-check of decompose_update (PLAN.md Addition 1);
    not the primary implementation."""
    u = _unit(np.asarray(direction_u, dtype=float))
    rank1_row = u @ delta_W
    rank1_component = np.outer(u, rank1_row)
    remainder = delta_W - rank1_component

    Q = np.asarray(shared_basis_Q, dtype=float)
    if Q.ndim == 1:
        Q = Q[:, None]
    Q = Q - np.outer(u, u @ Q)  # deflate against u, same as the QR path

    gram = Q.T @ Q
    coeffs, *_ = np.linalg.lstsq(gram, Q.T @ remainder, rcond=None)
    shared_component = Q @ coeffs

    orthogonal_remainder = remainder - shared_component

    return {
        "bias_norm": float(np.linalg.norm(delta_b)),
        "rank1_component_norm": float(np.linalg.norm(rank1_component)),
        "rank1_row": rank1_row,
        "shared_component_norm": float(np.linalg.norm(shared_component)),
        "shared_component": shared_component,
        "orthogonal_remainder_norm": float(np.linalg.norm(orthogonal_remainder)),
    }


def kernel_bias_attribution(W_current, b_current, b_pretrain, direction_u,
                             keys, correct_idx, candidate_mask=None):
    """Two counterfactual accuracies (PLAN.md spec metric #6):
      - bias frozen at its pre-injection value, W unchanged;
      - W with its component along `direction_u` projected out, b unchanged.
    """
    u = _unit(np.asarray(direction_u, dtype=float))

    logits_bias_frozen = compute_logits(W_current, b_pretrain, keys)
    acc_bias_frozen = float(
        argmax_correct(logits_bias_frozen, correct_idx, candidate_mask).mean()
    )

    W_deflated = W_current - np.outer(u, u @ W_current)
    logits_w_deflated = compute_logits(W_deflated, b_current, keys)
    acc_w_deflated = float(
        argmax_correct(logits_w_deflated, correct_idx, candidate_mask).mean()
    )

    return {
        "acc_bias_frozen": acc_bias_frozen,
        "acc_w_direction_deflated": acc_w_deflated,
    }
