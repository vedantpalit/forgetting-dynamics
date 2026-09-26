"""Direct verification that the injection batch sampler can only ever draw from B --
never A or D (ballast) -- across real get_batch() calls, not by re-deriving it from
_generate()'s code path in isolation.

This has never been directly checked before. What existed was verification of
population MEMBERSHIP under each condition (who draws which values) -- not whether the
batch sampler itself, at the mechanical level of cache refills and slicing, could ever
leak a draw from outside its own person_ids array. Raised when discussing the setup;
closing it here before the 8-layer rebuild, since it's cheap and
orthogonal to model scale (BiographyDataset.get_batch never touches the model).

Method: BiographyDataset.get_batch() already tracks exposure_counts internally
(np.bincount(persons[sl], ...), incremented every call) but does not return the drawn
person ids to the caller. Rather than modify that shared method to add a return value
just for this one-off check, this observes exposure_counts BEFORE and AFTER each
get_batch() call -- the delta is exactly the set of person ids drawn in that batch,
with zero changes to shared code. This validates the REAL path (cache refill, slicing,
CACHE_MULTIPLIER) end to end, not a reimplementation of it.

Run (no GPU needed, pure CPU, no model construction):
  uv run python -m src.experiments.verify_injection_sampler
"""
import sys

import numpy as np

from src.config import parse_config
from src.experiments.knowledge_injection import KIConfig, build, reset_stream

N_BATCHES = 50
BATCH_SIZE = 256


def main():
    sys.argv = [sys.argv[0],
        "--num_a", "2000", "--num_ballast", "2000", "--num_b", "500", "--num_c", "500",
        "--inject.condition", "disjoint", "--inject.seed", "0",
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--checkpoint_dir", "", "--wandb_mode", "disabled",
    ] + sys.argv[1:]
    cfg = parse_config(KIConfig, description="Verify the injection batch sampler never leaks A/D")
    pop, model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c = build(
        cfg, max_eval_people=5)

    ids_a = set(data_a.person_ids.tolist())
    ids_ballast = set(data_ballast.person_ids.tolist())
    ids_b = set(data_b.person_ids.tolist())
    assert ids_a.isdisjoint(ids_ballast) and ids_a.isdisjoint(ids_b) and ids_ballast.isdisjoint(ids_b), \
        "Population id ranges themselves overlap -- this is a deeper bug than the sampler."
    print(f"A ids: [{min(ids_a)}, {max(ids_a)}], n={len(ids_a)}")
    print(f"ballast ids: [{min(ids_ballast)}, {max(ids_ballast)}], n={len(ids_ballast)}")
    print(f"B ids: [{min(ids_b)}, {max(ids_b)}], n={len(ids_b)}")

    inject_set = data_b  # condition is disjoint/high_overlap, never "identical", so
                          # the real injection driver always trains on data_b -- verified
                          # directly rather than assumed (see knowledge_injection.py's
                          # phase_inject: inject_set = data_a if condition == "identical" else data_b).
    reset_stream(inject_set, cfg.seed + 5 + cfg.inject.seed)

    rng = __import__("jax").random.PRNGKey(0)
    all_drawn = set()
    leaked_from_a, leaked_from_ballast, leaked_from_elsewhere = set(), set(), set()

    for i in range(N_BATCHES):
        before = inject_set.exposure_counts.copy()
        rng, batch_rng = __import__("jax").random.split(rng)
        inject_set.get_batch(batch_rng, BATCH_SIZE)
        after = inject_set.exposure_counts
        delta = after - before
        drawn_this_batch = set(np.nonzero(delta)[0].tolist())
        all_drawn |= drawn_this_batch
        leaked_from_a |= (drawn_this_batch & ids_a)
        leaked_from_ballast |= (drawn_this_batch & ids_ballast)
        leaked_from_elsewhere |= (drawn_this_batch - ids_a - ids_ballast - ids_b)
        if i < 3 or i == N_BATCHES - 1:
            print(f"  batch {i}: {len(drawn_this_batch)} unique ids drawn "
                  f"(cache refill: {inject_set._cache is not None and inject_set._cache_idx <= BATCH_SIZE})")

    total_draws = N_BATCHES * BATCH_SIZE
    print(f"\n=== Result: {N_BATCHES} batches x {BATCH_SIZE} = {total_draws} total draws ===")
    print(f"Unique individuals drawn: {len(all_drawn)} (of B's {len(ids_b)})")
    print(f"Drawn from A: {len(leaked_from_a)}  {'<<< LEAK' if leaked_from_a else '(zero, clean)'}")
    print(f"Drawn from ballast: {len(leaked_from_ballast)}  "
          f"{'<<< LEAK' if leaked_from_ballast else '(zero, clean)'}")
    print(f"Drawn from anywhere else (not A/ballast/B): {len(leaked_from_elsewhere)}  "
          f"{'<<< LEAK' if leaked_from_elsewhere else '(zero, clean)'}")
    all_from_b = all_drawn <= ids_b
    print(f"\nAll {total_draws} draws confirmed to come exclusively from B: {all_from_b}")
    if not all_from_b:
        print("FAIL -- the sampler leaked outside B. Do not proceed to the LR probe until "
              "this is understood and fixed.")
        sys.exit(1)
    print("PASS -- zero leakage across every batch checked.")


if __name__ == "__main__":
    main()
