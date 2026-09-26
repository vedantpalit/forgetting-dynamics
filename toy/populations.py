"""Population and key construction (PLAN.md §4.4, §4.4a).

Name-token pools are partitioned into reserved, disjoint sub-pools up front
(A's own, D's own, B's own-for-non-overlap), sized from `name_multiplicity`
so every population's marginal token multiplicity is exact by construction
(round-robin assignment), not left to random coverage.

Overlap controls TOKEN COVERAGE directly (PLAN.md §0.3b): a target subset of
A's last-name tokens is sampled explicitly, of size `overlap * a_pool_size`,
and overlapping B individuals are round-robined within it. WHICH of A's
tokens land in that subset is still uniform — collision concentration
remains deliberately not a knob — but HOW MANY is now the knob itself.
The earlier construction drew uniformly from all of A's tokens, letting the
coupon-collector effect cover 46 of 50 tokens at overlap 0.5 and 49.7 at
overlap 1.0, which left 80 and 6 untouched A individuals respectively and
destroyed the touched/untouched split's power at the top of the grid.

B's first names always come from B's own reserved pool, regardless of the
overlap condition — overlap is a last-name-only channel (PLAN.md §4.6). At
overlap=0 this makes CHECK 1's fixture (PLAN.md §5.1) structurally free of
any B/A key overlap, first name or last name alike; without this, overlap=0
would still share first-name tokens with A and CHECK 1 would not actually
be measuring the erosion-off case it's meant to isolate.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config as cfg_mod


def build_name_features(n, d, rho, rank_shared, rng):
    """PLAN.md §4.3: idealized (rho=0) i.i.d. Gaussian unit vectors, or
    correlated within a shared low-rank subspace (rho>0)."""
    base = rng.normal(size=(n, d))
    shared_basis = rng.normal(size=(d, rank_shared))
    coeffs = rng.normal(size=(n, rank_shared))
    shared_part = coeffs @ shared_basis.T
    combined = np.sqrt(1 - rho) * base + np.sqrt(rho) * shared_part
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return combined / norms


def build_private_features(n, d, rng):
    """One frozen random unit vector per individual -- the `u_i` of PLAN.md §0.1."""
    v = rng.normal(size=(n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def compose_keys(first_feats, last_feats, private_feats, first_idx, last_idx, alpha,
                 beta=0.0, mu=None):
    """k_i = alpha * (e_f + e_l)/sqrt(2) + sqrt(1 - alpha^2) * u_i   (PLAN.md §0.1).

    alpha=1 recovers the purely additive composed key, whose logit is exactly
    a_v(f) + b_v(l) and whose key span is at most |F| + |L| REGARDLESS of d -- the
    structural limit that stalled the first attempt. alpha<1 adds a private direction per
    individual, restoring expressivity while keeping the shared name-part substrate that
    the erosion channel needs. Two individuals sharing one name part have key cosine
    ~alpha^2/2, which is what the Gram check measures, so alpha is calibratable later.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    composed = (first_feats[first_idx] + last_feats[last_idx]) / np.sqrt(2)
    base = alpha * composed + np.sqrt(1.0 - alpha ** 2) * private_feats
    if beta == 0.0:
        return base
    if mu is None:
        raise ValueError("beta > 0 requires the shared mean direction mu")
    return np.sqrt(beta) * mu[None, :] + np.sqrt(1.0 - beta) * base


def _round_robin_pairs(n, pool_size):
    """n individuals, pool_size local slots, exact multiplicity n/pool_size
    on both first- and last-local-slot marginals, all n pairs unique.
    Requires n % pool_size == 0 (asserted by the caller / config)."""
    assert n % pool_size == 0
    idx = np.arange(n)
    first_local = idx % pool_size
    block = idx // pool_size
    last_local = (first_local + block) % pool_size
    return first_local, last_local


@dataclass
class PopulationSet:
    keys_a: np.ndarray
    values_a: np.ndarray
    first_a: np.ndarray
    last_a: np.ndarray
    touched_a: np.ndarray

    keys_d: Optional[np.ndarray]
    values_d: Optional[np.ndarray]

    keys_b: np.ndarray
    values_b: np.ndarray
    last_b: np.ndarray
    overlap_mask_b: np.ndarray

    keys_c: np.ndarray

    v_a_mask: np.ndarray
    v_b_mask: np.ndarray

    # Name-part feature tables, exposed so diagnostics can build the SHARED SUBSPACE
    # (the span of the last-name features B actually touches) without re-deriving the
    # rng stream. Read-only for everything downstream; nothing in the battery uses them.
    first_feats: Optional[np.ndarray] = None
    last_feats: Optional[np.ndarray] = None
    # The shared mean direction. Exposed so the bilinear model can recover each key's
    # FROZEN part exactly, base_i = (k_i - sqrt(beta)*mu) / sqrt(1-beta), without
    # rebuilding the population or re-deriving the rng stream.
    mu: Optional[np.ndarray] = None

    diagnostics: dict = field(default_factory=dict)


def build_populations(cfg: "cfg_mod.ToyConfig"):
    pop_seed, _train_seed = cfg.split_seeds()
    rng = np.random.default_rng(pop_seed)

    ecfg, pcfg, vcfg = cfg.embedding, cfg.population, cfg.vocab
    d = ecfg.d

    first_feats = build_name_features(ecfg.n_first, d, ecfg.feature_shared_strength,
                                       ecfg.rank_shared, rng)
    last_feats = build_name_features(ecfg.n_last, d, ecfg.feature_shared_strength,
                                      ecfg.rank_shared, rng)
    alpha = ecfg.alpha
    # Drawn from a SEPARATE generator, not `rng`: taking it from the main stream would
    # shift every subsequent draw and silently change the populations at beta=0 too.
    mu = np.random.default_rng(pop_seed + 7_000_000).normal(size=d)
    mu = mu / np.linalg.norm(mu)
    beta = ecfg.beta

    # --- Reserved, disjoint sub-pools (PLAN.md §4.4a) ---
    perm_first = rng.permutation(ecfg.n_first)
    perm_last = rng.permutation(ecfg.n_last)

    a_pool = pcfg.a_pool_size
    d_pool = pcfg.d_pool_size
    b_pool = pcfg.b_own_pool_size
    mult_a, mult_aux = pcfg.name_multiplicity, pcfg.aux_multiplicity

    a_first_pool = perm_first[:a_pool]
    d_first_pool = perm_first[a_pool:a_pool + d_pool]
    b_first_pool = perm_first[a_pool + d_pool:a_pool + d_pool + b_pool]

    a_last_pool = perm_last[:a_pool]
    d_last_pool = perm_last[a_pool:a_pool + d_pool]
    b_last_pool = perm_last[a_pool + d_pool:a_pool + d_pool + b_pool]

    # Assert pairwise disjointness at config-build time (PLAN.md requirement 4).
    assert len(set(a_first_pool.tolist()) & set(d_first_pool.tolist())) == 0
    assert len(set(a_first_pool.tolist()) & set(b_first_pool.tolist())) == 0
    assert len(set(d_first_pool.tolist()) & set(b_first_pool.tolist())) == 0
    assert len(set(a_last_pool.tolist()) & set(d_last_pool.tolist())) == 0
    assert len(set(a_last_pool.tolist()) & set(b_last_pool.tolist())) == 0
    assert len(set(d_last_pool.tolist()) & set(b_last_pool.tolist())) == 0

    # --- A: round-robin over its own reserved pools, exact multiplicity ---
    a_first_local, a_last_local = _round_robin_pairs(pcfg.n_a, a_pool)
    first_a = a_first_pool[a_first_local]
    last_a = a_last_pool[a_last_local]
    assert len(set(zip(first_a.tolist(), last_a.tolist()))) == pcfg.n_a, "A pairs not unique"
    a_first_mult = np.bincount(first_a, minlength=ecfg.n_first)
    a_last_mult = np.bincount(last_a, minlength=ecfg.n_last)
    assert np.all(a_first_mult[a_first_pool] == pcfg.name_multiplicity)
    assert np.all(a_last_mult[a_last_pool] == pcfg.name_multiplicity)

    priv_a = build_private_features(pcfg.n_a, d, rng)
    keys_a = compose_keys(first_feats, last_feats, priv_a, first_a, last_a, alpha, beta, mu)
    values_a = rng.integers(0, vcfg.v_a_size, size=pcfg.n_a)  # local index into V_A

    # --- D (ballast): round-robin over its own reserved pools ---
    keys_d = None
    values_d = None
    if pcfg.ballast_present:
        d_first_local, d_last_local = _round_robin_pairs(pcfg.n_ballast, d_pool)
        first_d = d_first_pool[d_first_local]
        last_d = d_last_pool[d_last_local]
        assert len(set(zip(first_d.tolist(), last_d.tolist()))) == pcfg.n_ballast
        priv_d = build_private_features(pcfg.n_ballast, d, rng)
        keys_d = compose_keys(first_feats, last_feats, priv_d, first_d, last_d, alpha, beta, mu)
        values_d = rng.integers(0, vcfg.v_b_size, size=pcfg.n_ballast) + vcfg.v_a_size

    # --- B: first names always from B's own reserved pool (round-robin);
    # last names split by the overlap knob ---
    b_first_local, b_last_local_own = _round_robin_pairs(pcfg.n_b, b_pool)
    first_b = b_first_pool[b_first_local]

    # Overlap controls TOKEN COVERAGE directly (PLAN.md §0.3b). The earlier construction
    # drew B's overlapping individuals uniformly from all of A's tokens, which let the
    # coupon-collector effect decouple the knob from what the touched/untouched split
    # needs: at overlap 0.5 it already covered 46 of 50 tokens, leaving 80 untouched A
    # individuals, and at overlap 1.0 only 6. Now a target subset of A's tokens is sampled
    # explicitly and overlapping B individuals are assigned only within it, so untouched
    # count is exact and predictable across the grid (960/720/480/240/0).
    n_touched_tokens = pcfg.n_touched_tokens
    n_overlap = pcfg.n_overlapping_b
    assert np.isclose(n_overlap / pcfg.n_b, pcfg.overlap), "overlap fraction not exactly representable"
    assert np.isclose(n_touched_tokens / a_pool, pcfg.overlap), \
        "overlap does not land on an integer number of touched A tokens"

    overlap_mask_b = np.zeros(pcfg.n_b, dtype=bool)
    overlap_idx = rng.choice(pcfg.n_b, size=n_overlap, replace=False)
    overlap_mask_b[overlap_idx] = True

    # WHICH of A's tokens get touched stays uniform — that is the collision-concentration
    # axis, still deliberately not a knob. What changed is only HOW MANY.
    touched_tokens = rng.choice(a_last_pool, size=n_touched_tokens, replace=False)

    last_b = np.empty(pcfg.n_b, dtype=int)
    if n_overlap:
        # Round-robin the overlapping individuals across the touched tokens, so
        # collisions per touched token is exactly n_overlap / n_touched_tokens with no
        # nuisance variance. That ratio is constant across the overlap grid by
        # construction (both scale with overlap), which is what stops token coverage from
        # being confounded with per-token injection pressure.
        last_b[overlap_mask_b] = touched_tokens[np.arange(n_overlap) % n_touched_tokens]
    # Non-overlapping individuals: round-robin over B's own reserved pool.
    nonoverlap_idx = np.where(~overlap_mask_b)[0]
    if len(nonoverlap_idx) > 0:
        _lf, local_last = _round_robin_pairs_flexible(len(nonoverlap_idx), b_pool)
        last_b[nonoverlap_idx] = b_last_pool[local_last]

    priv_b = build_private_features(pcfg.n_b, d, rng)
    keys_b = compose_keys(first_feats, last_feats, priv_b, first_b, last_b, alpha, beta, mu)

    concentration_values = rng.choice(vcfg.v_b_size, size=pcfg.concentration, replace=False)
    values_b = rng.choice(concentration_values, size=pcfg.n_b, replace=True) + vcfg.v_a_size

    # --- touched mask for A: last-name token touched by >=1 B individual ---
    touched_last_tokens = set(last_b[overlap_mask_b].tolist())
    touched_a = np.isin(last_a, list(touched_last_tokens)) if touched_last_tokens else \
        np.zeros(pcfg.n_a, dtype=bool)

    # --- C: held-out, name parts reused, exact pairs unseen ---
    used_first = np.concatenate([a_first_pool, d_first_pool, b_first_pool])
    used_last = np.concatenate([a_last_pool, d_last_pool, b_last_pool])
    real_pairs = set(zip(first_a.tolist(), last_a.tolist()))
    if pcfg.ballast_present:
        real_pairs |= set(zip(first_d.tolist(), last_d.tolist()))
    real_pairs |= set(zip(first_b.tolist(), last_b.tolist()))

    c_pairs = []
    attempts = 0
    max_attempts = pcfg.n_c * 50
    while len(c_pairs) < pcfg.n_c and attempts < max_attempts:
        attempts += 1
        f = rng.choice(used_first)
        l = rng.choice(used_last)
        if (f, l) not in real_pairs and (f, l) not in set(c_pairs):
            c_pairs.append((f, l))
    assert len(c_pairs) == pcfg.n_c, "could not find enough unseen (first,last) pairs for C"
    first_c = np.array([p[0] for p in c_pairs])
    last_c = np.array([p[1] for p in c_pairs])
    priv_c = build_private_features(pcfg.n_c, d, rng)
    keys_c = compose_keys(first_feats, last_feats, priv_c, first_c, last_c, alpha, beta, mu)

    v_a_mask = np.zeros(vcfg.v_total, dtype=bool)
    v_a_mask[:vcfg.v_a_size] = True
    v_b_mask = ~v_a_mask

    key_norms_a = np.linalg.norm(keys_a, axis=1)
    diagnostics = {
        "a_first_pool_size": int(a_pool),
        "a_last_pool_size": int(a_pool),
        "d_pool_size": int(d_pool),
        "b_own_pool_size": int(b_pool),
        "n_overlap_realized": int(n_overlap),
        "overlap_fraction_realized": float(n_overlap / pcfg.n_b),
        "n_a_touched": int(touched_a.sum()),
        "n_a_untouched": int((~touched_a).sum()),
        "n_distinct_a_last_tokens_touched": len(touched_last_tokens),
        "n_touched_tokens_target": int(n_touched_tokens),
        "collisions_per_touched_token": float(pcfg.collisions_per_touched_token),
        "alpha": float(alpha),
        "beta": float(beta),
        "mean_key_norm_b": float(np.linalg.norm(keys_b.mean(axis=0))),
        "mean_key_norm_a": float(np.linalg.norm(keys_a.mean(axis=0))),
        # Key norm varies with the random inner product between the two name-part
        # features; a systematic touched/untouched difference would confound the split.
        "key_norm_touched_mean": float(key_norms_a[touched_a].mean()) if touched_a.any() else float("nan"),
        "key_norm_untouched_mean": float(key_norms_a[~touched_a].mean()) if (~touched_a).any() else float("nan"),
        "key_norm_all_mean": float(key_norms_a.mean()),
        "key_norm_all_std": float(key_norms_a.std()),
        "a_last_multiplicity_min": int(a_last_mult[a_last_pool].min()) if pcfg.n_a else None,
        "a_last_multiplicity_max": int(a_last_mult[a_last_pool].max()) if pcfg.n_a else None,
    }

    return PopulationSet(
        keys_a=keys_a, values_a=values_a, first_a=first_a, last_a=last_a, touched_a=touched_a,
        keys_d=keys_d, values_d=values_d,
        keys_b=keys_b, values_b=values_b, last_b=last_b, overlap_mask_b=overlap_mask_b,
        keys_c=keys_c,
        v_a_mask=v_a_mask, v_b_mask=v_b_mask,
        first_feats=first_feats, last_feats=last_feats,
        mu=mu,
        diagnostics=diagnostics,
    )


def _round_robin_pairs_flexible(n, pool_size):
    """Same as _round_robin_pairs but tolerates n not dividing pool_size
    evenly (used for B's non-overlap subset, whose size depends on the
    overlap fraction and is not guaranteed to divide b_own_pool_size)."""
    idx = np.arange(n)
    first_local = idx % pool_size
    block = idx // pool_size
    last_local = (first_local + block) % pool_size
    return first_local, last_local
