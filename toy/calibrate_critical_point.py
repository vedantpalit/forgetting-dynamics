"""Addition 3, re-measured at battery geometry (PLAN.md §0.5).

The orthogonal hand-built fixture gave a SHARP phase transition: restricted rank exactly 1.0
below a critical push and fully saturated above it, with no graded regime. That is not a
fixture artifact to smooth away -- it is a statement about what a probability-weighted
suppression does. This script asks whether the structure survives at the geometry the battery
actually runs in (alpha < 1, |V_A| = 256, real name-sharing multiplicity), and locates the
critical point rather than reporting a single leakage number.

Why the answer is not obvious in advance: each INDIVIDUAL has a sharp threshold (the
derivation is exact, see PLAN.md §0.5a), but under alpha < 1 the store is no longer a perfect
one-hot readout, so the per-individual thresholds are heterogeneous. A population of sharp
thresholds with a spread distribution produces a GRADED population curve. Whether the toy
looks sharp or graded is therefore a question about the spread of the threshold distribution,
and that spread is what this measures.

Run: python -m calibrate_critical_point   (from toy/)
"""
import numpy as np

from feasibility import (
    apply_probability_weighted_push, critical_push_per_individual, ridge_precheck,
)
from metrics import rank_of_correct
from populations import build_name_features, build_private_features, compose_keys

# Battery geometry (PLAN.md §0.1, §0.3).
N_A = 2000
N_FIRST = N_LAST = 100          # A's reserved sub-pools
MULTIPLICITY = 20               # A individuals per name token
D = 2048
V_A = V_B = 256
ALPHAS = (1.0, 0.85, 0.7, 0.5, 0.3)
PUSH_GRID = np.concatenate([np.linspace(0, 20, 41), np.linspace(21, 60, 40)])


def build_A(alpha, seed):
    rng = np.random.default_rng(seed)
    first = build_name_features(N_FIRST, D, 0.0, 8, rng)
    last = build_name_features(N_LAST, D, 0.0, 8, rng)
    priv = build_private_features(N_A, D, rng)
    idx = np.arange(N_A)
    first_idx = idx % N_FIRST
    last_idx = (first_idx + idx // N_FIRST) % N_LAST      # round-robin, exact multiplicity
    keys = compose_keys(first, last, priv, first_idx, last_idx, alpha)
    values = rng.integers(0, V_A, size=N_A)
    return keys, values, last_idx


def main():
    v_a_mask = np.zeros(V_A + V_B, dtype=bool)
    v_a_mask[:V_A] = True

    print(f"Battery geometry: n_A={N_A}, |F|=|L|={N_FIRST}, multiplicity={MULTIPLICITY}, "
          f"d={D}, |V_A|={V_A}, |V|={V_A + V_B}, load ratio n_A/|V_A|={N_A / V_A:.1f}\n")

    for alpha in ALPHAS:
        keys, values, _ = build_A(alpha, seed=0)
        pre, W, logits = ridge_precheck(keys, values, V_A + V_B, v_a_mask)

        crit = critical_push_per_individual(logits, values, v_a_mask)
        finite = crit[np.isfinite(crit)]

        print(f"=== alpha = {alpha} ===")
        print(f"  ridge argmax acc (full vocab) = {pre['ridge_acc_full']:.4f}   "
              f"(restricted = {pre['ridge_acc_restricted']:.4f})")
        print(f"  key rank = {pre['key_rank']} of d={pre['key_dim']} "
              f"(n={pre['n_individuals']}); cond = {pre['key_cond']:.3g}")
        if len(finite) == 0:
            print("  no finite critical points\n")
            continue
        qs = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
        print(f"  per-individual critical push: min={qs[0]:.3f} p05={qs[1]:.3f} "
              f"p25={qs[2]:.3f} p50={qs[3]:.3f} p75={qs[4]:.3f} p95={qs[5]:.3f} "
              f"max={qs[6]:.3f}")
        print(f"    spread p05->p95 = {qs[5] - qs[1]:.3f}  "
              f"(relative to median: {(qs[5] - qs[1]) / qs[3]:.2f}x)")

        # Empirical population curve, on a grid auto-scaled to the measured critical point.
        # A fixed grid is useless here: the critical push scales with the STORE's confidence
        # (s* = margin / (p_c - p_j)), and the ridge store's margins are ~1 with p_own ~ 1/|V|,
        # so its critical point is ~300 where the CE-trained fixture's was ~10.
        grid = np.linspace(0.9 * qs[0], 1.1 * qs[6], 60)
        ranks = []
        for s in grid:
            r = rank_of_correct(apply_probability_weighted_push(logits, s), values, v_a_mask)
            ranks.append(r.mean())
        ranks = np.array(ranks)
        worst = float(V_A)
        leaves = grid[np.argmax(ranks > 1.01)] if (ranks > 1.01).any() else np.nan
        half = grid[np.argmax(ranks > worst / 2)] if (ranks > worst / 2).any() else np.nan
        sat = grid[np.argmax(ranks > 0.95 * worst)] if (ranks > 0.95 * worst).any() else np.nan
        print(f"  population restricted mean rank: leaves 1.01 at push={leaves:.2f}, "
              f"half-saturated at {half:.2f}, 95%-saturated at {sat:.2f}")
        print(f"    transition width (1.01 -> 95% sat) = "
              f"{sat - leaves:.2f} push units = {(sat - leaves) / max(leaves, 1e-9):.2f}x "
              f"the onset")
        show = np.percentile(grid, [0, 20, 40, 45, 50, 55, 60, 80, 100])
        pairs = "  ".join(f"{v:.1f}:{ranks[np.argmin(np.abs(grid - v))]:.2f}" for v in show)
        print(f"    curve (push:mean_rank) {pairs}\n")


if __name__ == "__main__":
    main()
