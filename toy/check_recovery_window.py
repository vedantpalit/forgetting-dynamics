"""Does recovery appear, and does the trough stay off the floor? (PLAN.md §0.4c)

The 400-step probe left A floored at the matched ratio with no recovery, and rank at 1.001
even at overlap 0.5 where erosion should exist -- both signs that the window was too short
rather than that the phenomena are absent. This runs the matched ratio out far enough to
tell "never" from "not yet", and reports:

  * the per-individual margin distribution s_eff/s* and the SUPERCRITICAL FRACTION, which
    is the quantity that matters since s* is per-individual (a mean-vs-median comparison
    would hide a straddling tail);
  * B's own acquisition curve, since recovery is hypothesised to wait on B individuals
    becoming mutually resolvable -- so B's curve says whether a missing recovery is
    "never" or "not yet";
  * whether A's trough sits ON the floor, which would make CHECK 3 and CHECK 4 vacuous
    (a floor cannot be distinguished from an invariance).

Run: python -m check_recovery_window   (from toy/)
"""
import numpy as np

import config as C
import populations as P
from feasibility import critical_push_per_individual
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
STEPS = 2000


def cfg_for(overlap, ratio=MATCHED_RATIO, seed=0):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7),
        population=C.PopulationConfig(overlap=overlap),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=ratio),
        protocol=C.ProtocolConfig(pretrain_max_steps=4000, pretrain_extra_fraction=0.25,
                                   finetune_max_steps=STEPS, eval_dense_until=100,
                                   eval_every_sparse=25, finetune_flat_window=10 ** 6),
        seed=seed)


def main():
    base = cfg_for(0.5)
    ps0 = P.build_populations(base)
    store0, _hist, ceil, tgt = pretrain(base, ps0)
    print(f"pretrained: ceiling@{ceil}, trained to {tgt}\n")

    logits_a0 = store0.logits(ps0.keys_a)
    s_star = critical_push_per_individual(logits_a0, ps0.values_a, ps0.v_a_mask)
    ok = np.isfinite(s_star)

    for overlap in (0.0, 0.5):
        cfg = cfg_for(overlap)
        ps = P.build_populations(cfg)
        curve, cap, _st = finetune(cfg, ps, store0)

        steps = np.array([c["step"] for c in curve])
        acc = np.array([c["a_full_acc_mean"] for c in curve])
        rank = np.array([c["a_restr_rank_mean"] for c in curve])
        bacc = np.array([c["b_full_acc_mean"] for c in curve])

        i_tr = int(acc.argmin())
        peak_after = acc[i_tr:].max()
        recov = (peak_after - acc[i_tr]) / max(acc[0] - acc[i_tr], 1e-12)

        margin = curve[-1]["s_eff_per_individual"][ok] / s_star[ok]
        frac_super = float((margin > 1.0).mean())

        print(f"=== overlap {overlap} (matched ratio, {STEPS} steps, "
              f"{'cap-truncated' if cap else 'stopped on flat window'}) ===")
        print(f"  margin s_eff/s* per individual: p05={np.percentile(margin, 5):.4f} "
              f"p50={np.percentile(margin, 50):.4f} p95={np.percentile(margin, 95):.4f}  "
              f"supercritical fraction = {frac_super:.2%}")
        print(f"  A acc: baseline={acc[0]:.4f} trough={acc[i_tr]:.4f}@{steps[i_tr]} "
              f"peak_after={peak_after:.4f} recovery_fraction={recov:.3f} end={acc[-1]:.4f}")
        print(f"  A rank_own: start={rank[0]:.3f} at_trough={rank[i_tr]:.3f} "
              f"max={rank.max():.3f} end={rank[-1]:.3f}")
        print(f"  B acc: @10={bacc[min(2, len(bacc)-1)]:.3f} "
              f"@100={bacc[np.argmin(np.abs(steps - 100))]:.3f} "
              f"@500={bacc[np.argmin(np.abs(steps - 500))]:.3f} end={bacc[-1]:.3f}")
        floored = acc[i_tr] < 0.01
        print(f"  trough on the floor? {'YES -- CHECK 3/4 would be vacuous here' if floored else 'no'}")
        show = [0, 25, 50, 100, 200, 400, 800, 1200, 1600, 2000]
        pairs = "  ".join(f"{v}:{acc[np.argmin(np.abs(steps - v))]:.3f}" for v in show)
        print(f"  A curve {pairs}\n")


if __name__ == "__main__":
    main()
