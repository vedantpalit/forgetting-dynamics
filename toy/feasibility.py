"""Closed-form feasibility pre-check and critical-point calibration (PLAN.md §0.4, §0.5).

Two things live here, both cheap and both run BEFORE any training:

  * `ridge_precheck` -- solve the store in closed form and report whether the assignment is
    linearly realizable at all. If ridge argmax accuracy is at ceiling, gradient descent
    with a sane LR will get there; if it is far below, no learning rate will fix it. Paired
    with the key Gram matrix's rank and condition number and the load ratio, these three
    numbers diagnose a structural infeasibility in seconds -- which is what the first
    attempt at this toy needed and did not have.

  * `critical_push_per_individual` -- the exact magnitude of probability-weighted
    suppression at which an individual's correct value loses its own-half top rank. See
    the derivation in PLAN.md §0.5a; this is its general form, valid for any store rather
    than only the idealized fixture.
"""
import numpy as np


def ridge_precheck(keys, correct_idx, n_values, candidate_mask=None, lam_rel=1e-6):
    """W = (K^T K + lam I)^{-1} K^T Y in closed form, then argmax accuracy of W^T k.

    `lam_rel` is relative to mean(diag(K^T K)) so the regularization does not depend on the
    scale of the keys or on d. Returns a dict; `ridge_acc_full` is the number that decides
    feasibility.
    """
    n, d = keys.shape
    gram_feat = keys.T @ keys                       # (d, d)
    lam = lam_rel * float(np.trace(gram_feat)) / d
    Y = np.zeros((n, n_values))
    Y[np.arange(n), correct_idx] = 1.0
    W = np.linalg.solve(gram_feat + lam * np.eye(d), keys.T @ Y)   # (d, |V|)
    logits = keys @ W

    pred_full = logits.argmax(axis=1)
    out = {"ridge_acc_full": float((pred_full == correct_idx).mean())}
    if candidate_mask is not None:
        masked = np.where(candidate_mask[None, :], logits, -np.inf)
        out["ridge_acc_restricted"] = float((masked.argmax(axis=1) == correct_idx).mean())

    # Conditioning of the KEY Gram (n x n over individuals is the object that matters for
    # whether individuals are separable; report the feature-space rank too since that is
    # what caps it -- for the additive alpha=1 keys it is bounded by |F| + |L|).
    sv = np.linalg.svd(keys, compute_uv=False)
    tol = sv.max() * max(keys.shape) * np.finfo(float).eps
    out["key_rank"] = int((sv > tol).sum())
    out["key_dim"] = int(d)
    out["n_individuals"] = int(n)
    out["key_cond"] = float(sv.max() / sv[sv > tol].min())
    out["lam"] = float(lam)
    return out, W, logits


def critical_push_per_individual(logits, correct_idx, candidate_mask):
    """Exact per-individual critical magnitude of a probability-weighted suppression.

    A push of magnitude `s` shifts every logit by -s * p_i[j], with p_i the individual's
    CURRENT full-vocabulary softmax. The correct value c loses its top rank within the
    candidate set to competitor j once

        z[c] - s p[c]  <  z[j] - s p[j]      =>      s  >  (z[c] - z[j]) / (p[c] - p[j])

    so the critical point is the minimum of that ratio over candidates. p[c] is the largest
    entry for a store that is correct on this individual, so every denominator is positive
    and the ratio is well defined. Returns (n,) of critical magnitudes; inf where no
    candidate can ever overtake, nan where the store is already wrong.
    """
    n = logits.shape[0]
    z = logits
    m = z.max(axis=1, keepdims=True)
    p = np.exp(z - m)
    p /= p.sum(axis=1, keepdims=True)

    rows = np.arange(n)
    zc = z[rows, correct_idx][:, None]
    pc = p[rows, correct_idx][:, None]

    cand = np.where(candidate_mask)[0]
    zj, pj = z[:, cand], p[:, cand]
    dz, dp = zc - zj, pc - pj

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dp > 0, dz / dp, np.inf)
    self_col = cand[None, :] == np.asarray(correct_idx)[:, None]
    ratio = np.where(self_col, np.inf, ratio)
    crit = ratio.min(axis=1)

    already_wrong = (np.where(candidate_mask[None, :], z, -np.inf).argmax(axis=1)
                     != np.asarray(correct_idx))
    crit = np.where(already_wrong, np.nan, crit)
    return crit


def apply_probability_weighted_push(logits, push):
    """The suppression shape case (e) plants: subtract push * p_i[j] from every logit,
    with p_i the individual's own pre-push softmax."""
    m = logits.max(axis=1, keepdims=True)
    p = np.exp(logits - m)
    p /= p.sum(axis=1, keepdims=True)
    return logits - push * p
