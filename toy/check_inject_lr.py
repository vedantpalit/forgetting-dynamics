"""Characterize the injection learning rate against the critical push (PLAN.md §0.4b).

Addition 3 established that restricted rank is untouched below the critical push s* and
destroyed above it. s_eff scales with injection LR, so injection LR decides which side of
that transition the battery runs on. This measures s_eff / s* rather than assuming it, at
the matched ratio and at 3x either side.

The ratio is NOT tuned to make CHECK 1 pass: it is set to the real experiments'
INJECT_LR_RATIO (4e-4 / 3e-5 = 40/3), an external protocol choice made independently of
anything this toy shows. If the margin lands subcritical that agrees with the MLP-free
ablation, whose 0.33-unit rank transient says the real model operates below critical. If it
lands supercritical, that is a finding about the toy's dynamics, not a threshold to adjust.

Also reports the lagging individual: pretraining converges to acc_A = 0.9990, i.e. one
individual of 960 still wrong. Cross-entropy on separable data approaches the max-margin
solution only logarithmically, so a lagging individual after finite steps is expected
optimizer behaviour rather than a defect, and ridge reaching 1.0 in closed form is
consistent with that. Checked anyway, because a permanently-wrong individual sitting in the
touched group would add noise exactly where CHECK 2 reads.

Run: python -m check_inject_lr   (from toy/)
"""
import numpy as np

import config as C
import populations as P
from feasibility import critical_push_per_individual
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
RATIOS = (MATCHED_RATIO * 3, MATCHED_RATIO, MATCHED_RATIO / 3)
SEEDS = (0, 1, 2)


def probe_cfg(ratio, seed, overlap=0.5):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7),
        population=C.PopulationConfig(overlap=overlap),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=ratio),
        protocol=C.ProtocolConfig(pretrain_max_steps=4000, pretrain_extra_fraction=0.25,
                                   finetune_max_steps=400, eval_dense_until=60,
                                   eval_every_sparse=20, finetune_flat_window=1000),
        seed=seed)


def main():
    print("=== Lagging individual at the end of pretraining ===")
    lagging = {}
    for seed in SEEDS:
        cfg = probe_cfg(MATCHED_RATIO, seed)
        ps = P.build_populations(cfg)
        store, hist, ceil, tgt = pretrain(cfg, ps)
        import metrics as m
        ok = m.argmax_correct(store.logits(ps.keys_a), ps.values_a)
        wrong = np.where(~ok)[0]
        lagging[seed] = (store, ps, cfg, wrong)
        in_touched = ps.touched_a[wrong].sum() if len(wrong) else 0
        print(f"  seed {seed}: acc_A={ok.mean():.4f}  wrong={list(wrong)}  "
              f"of which touched={int(in_touched)}  "
              f"(touched fraction of A = {ps.touched_a.mean():.2f})")
    all_wrong = [set(w.tolist()) for (_s, _p, _c, w) in lagging.values()]
    common = set.intersection(*all_wrong) if all_wrong else set()
    print(f"  same individual across all {len(SEEDS)} seeds? "
          f"{'YES: ' + str(sorted(common)) if common else 'NO — differs by seed'}")

    print("\n=== s_eff vs the critical push s* (seed 0, overlap 0.5) ===")
    store0, ps0, cfg0, _w = lagging[SEEDS[0]]
    logits_a0 = store0.logits(ps0.keys_a)
    s_star = critical_push_per_individual(logits_a0, ps0.values_a, ps0.v_a_mask)
    finite = s_star[np.isfinite(s_star)]
    print(f"  s* over A: p05={np.percentile(finite, 5):.3f} "
          f"median={np.median(finite):.3f} p95={np.percentile(finite, 95):.3f} "
          f"({len(finite)}/{len(s_star)} finite)")

    # PER-INDIVIDUAL margin. s* is per-individual, so the distribution and especially the
    # supercritical FRACTION are the quantities that matter; a mean-vs-median comparison
    # would hide a supercritical tail (PLAN.md §0.4c).
    print(f"  {'ratio':>8} {'inj_lr':>8} {'margin p05':>11} {'p50':>8} {'p95':>8} "
          f"{'%supercrit':>11} {'A_acc_end':>10} {'A_rank_end':>11} {'trough':>8}")
    for ratio in RATIOS:
        cfg = probe_cfg(ratio, SEEDS[0])
        ps = P.build_populations(cfg)
        curve, cap, _store = finetune(cfg, ps, store0)
        end = curve[-1]
        trough = min(c["a_full_acc_mean"] for c in curve)
        se = end["s_eff_per_individual"]
        ok = np.isfinite(s_star)
        margin = se[ok] / s_star[ok]                     # >1 means supercritical
        frac_super = float((margin > 1.0).mean())
        tag = "" if frac_super == 0 else ("  <- STRADDLING" if frac_super < 0.99
                                           else "  <- SUPERCRITICAL")
        print(f"  {ratio:>8.2f} {cfg.optimizer.inject_lr:>8.4f} "
              f"{np.percentile(margin, 5):>11.4f} {np.percentile(margin, 50):>8.4f} "
              f"{np.percentile(margin, 95):>8.4f} {frac_super:>11.1%} "
              f"{end['a_full_acc_mean']:>10.4f} {end['a_restr_rank_mean']:>11.3f} "
              f"{trough:>8.4f}{tag}")

    print("\n  (matched ratio is the middle row; the other two are 3x either side, run as a"
          "\n   reported robustness axis rather than a tuned point)")


if __name__ == "__main__":
    main()
