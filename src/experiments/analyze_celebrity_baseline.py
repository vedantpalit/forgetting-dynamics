"""Pre-registration check for the celebrities experiment: before running anything, does
`margin_own` have usable room to move UPWARD at the pretrained checkpoint (accuracy/rank_own
are already confirmed degenerate there -- rank_own_top1=1.000 exactly, forcing mean rank_own
to exactly 1.0, no spread at all), and does that baseline margin already differ by shared_count
against a candidate celebrity subset before any celebrity fine-tuning has happened?

Two checks, same already-existing pretrained checkpoint, no new training:

1. Full distribution (min/max/deciles) of A's own-half margin at step 0 of injection (=
   the pretrained state). Mean/std alone (9.36/0.57, already visible in every existing
   analyze_name_overlap.py run) can't say whether the upper tail is open or already
   compressed toward a ceiling -- which is what actually matters for a reinforcement
   account, since reinforcement should push margins UP.
2. The SAME distribution split by shared_count (0-3) against a candidate celebrity subset
   (n=100, random, fixed seed CELEB_SEED so it's reproducible and the same split can be
   reused for the real experiment later) drawn from A itself. If the four groups already
   differ in baseline margin before any training, that's a confound to control for, not
   an artifact of nothing having happened yet -- important to know now, not after the runs.
"""
import numpy as np

from src.config import parse_config
from src.experiments.analyze_name_overlap import (
    crossover_pretrain_key, load_inject_checkpoint, per_person_margin_own,
)
from src.experiments.knowledge_injection import KIConfig, build, get_partition

CELEB_SEED = 71
N_CELEB = 100


def decile_report(values):
    n = len(values)
    order = np.argsort(values, kind="stable")
    bin_idx = np.empty(n, dtype=int)
    bin_idx[order] = (np.arange(n) * 10) // n
    return [float(values[bin_idx == d].mean()) for d in range(10)]


def main():
    cfg = parse_config(KIConfig, description="Celebrity-experiment pre-check: margin_own headroom")
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(cfg, cfg.inject.max_eval_people)

    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(6)]

    ids_a = data_a.person_ids
    print(f"Loading pretrained checkpoint (crossover key {crossover_pretrain_key(cfg, data_cfg)})...")
    pretrained_params, key = load_inject_checkpoint(cfg, data_cfg, model, 0)
    print(f"Loaded inject-{key}-step0000.msgpack\n")

    margin = per_person_margin_own(model, pretrained_params, data_a, x_ids).mean(axis=1)

    print("=== 1. Full distribution of margin_own, all of A (n=2000), pretrained checkpoint ===")
    print(f"mean={margin.mean():.3f}  std={margin.std():.3f}  min={margin.min():.3f}  max={margin.max():.3f}")
    deciles = decile_report(margin)
    print("deciles (1=lowest margin .. 10=highest margin):")
    for i, d in enumerate(deciles, 1):
        print(f"  decile {i:2d}: mean={d:.3f}")
    top_gap = deciles[9] - deciles[8]
    mid_gap = deciles[5] - deciles[4]
    print(f"gap between decile 10 and decile 9: {top_gap:.3f}  (vs. decile 6-5 gap: {mid_gap:.3f}) "
          f"-- a shrinking top gap relative to mid-distribution gaps is the signature of a "
          f"compressing ceiling; a comparable or larger gap says the upper end is still open.")

    print("\n=== 2. margin_own split by shared_count against a candidate celebrity subset ===")
    print(f"celebrity subset: n={N_CELEB}, random, seed={CELEB_SEED} (fixed, reusable for the real run)")
    rng = np.random.default_rng(CELEB_SEED)
    celeb_local = rng.choice(len(ids_a), size=N_CELEB, replace=False)
    is_celeb = np.zeros(len(ids_a), dtype=bool)
    is_celeb[celeb_local] = True
    remainder_local = np.where(~is_celeb)[0]

    names_a = pop.person_names[ids_a]
    names_celeb = names_a[celeb_local]
    names_remainder = names_a[remainder_local]

    mult = np.zeros((len(remainder_local), 3))
    for j in range(3):
        vals, counts = np.unique(names_celeb[:, j], return_counts=True)
        m = dict(zip(vals.tolist(), counts.tolist()))
        mult[:, j] = [m.get(v, 0) for v in names_remainder[:, j]]
    shared_count = (mult > 0).sum(axis=1)
    margin_remainder = margin[remainder_local]

    counts_hist = {c: int((shared_count == c).sum()) for c in range(4)}
    print(f"remainder shared_count distribution: {counts_hist}")
    print("baseline (pre-training) margin_own by shared_count -- should NOT differ if groups start equal:")
    for c in range(4):
        m = margin_remainder[shared_count == c]
        print(f"  shared_count={c}  n={len(m):4d}  mean={m.mean():.3f}  std={m.std():.3f}")
    overall_gap = margin_remainder[shared_count == 3].mean() - margin_remainder[shared_count == 0].mean()
    print(f"gap (shared_count=3 minus shared_count=0), baseline: {overall_gap:+.3f} "
          f"-- compare this to whatever post-fine-tuning gap the real experiment finds; "
          f"a baseline gap this size or larger would need controlling for.")


if __name__ == "__main__":
    main()
