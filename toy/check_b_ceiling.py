"""Diagnostic 3: run to B's ceiling -- is "no recovery" a fact or a truncation?

Diagnostics 1/1b/2 closed two questions: erosion is structurally unavailable in stage 1
(the V_A columns receive only the probability-weighted mean key, common across V_A), and
the bias carries 100% of the suppression (freezing b leaves A at baseline while ||dW||
reaches 36). Neither closes the RECOVERY question, because the recovery hypothesis is that
the coherent write along B's mean key stops paying once B's individuals have to be told
apart from one another -- and B never resolves in any arm run so far: 0.175 (overlap 0.5,
b frozen), 0.554 (overlap 0, b frozen), 0.912/0.975 (b free). Recovery is hypothesised to
track B's resolution, not step count, so "no recovery" is not established while B is short
of ceiling.

Both arms are run because they ask different halves of the question:
  * b free    -- does the region-level bias shift reverse once B is fully resolved, in the
                 configuration the battery actually uses;
  * b frozen  -- does suppression appear AT ALL through W once B is genuinely learned, or
                 does the value-space closure hold all the way to B's ceiling.

Stopping is on the protocol's own condition (B >= finetune_min_b_acc AND A flat), not a
fixed step count, with a hard cap that is FLAGGED rather than treated as convergence.

Run: python -m check_b_ceiling   (from toy/)
"""
import sys

import numpy as np

import config as C
import populations as P
from feasibility import critical_push_per_individual
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
MAX_STEPS = 12000
# argv: [overlap] [arms]  -- arms in {"both", "free"}. Defaults reproduce the original run.
OVERLAP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
ARMS = sys.argv[2] if len(sys.argv) > 2 else "both"


def cfg_for(seed=0):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7),
        population=C.PopulationConfig(overlap=OVERLAP),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=MATCHED_RATIO),
        protocol=C.ProtocolConfig(pretrain_max_steps=4000, pretrain_extra_fraction=0.25,
                                   finetune_max_steps=MAX_STEPS, eval_dense_until=200,
                                   eval_every_sparse=50, finetune_flat_window=50,
                                   finetune_min_b_acc=0.99),
        seed=seed)


def report(tag, curve, cap, s_star):
    steps = np.array([c["step"] for c in curve])
    acc = np.array([c["a_full_acc_mean"] for c in curve])
    rank = np.array([c["a_restr_rank_mean"] for c in curve])
    bacc = np.array([c["b_full_acc_mean"] for c in curve])

    i = int(acc.argmin())
    peak = acc[i:].max()
    j = i + int(acc[i:].argmax())
    recov = (peak - acc[i]) / max(acc[0] - acc[i], 1e-12)
    ok = np.isfinite(s_star)
    margins = np.array([c["s_eff_per_individual"][ok] / s_star[ok] for c in curve])
    peak_margin = margins.max(axis=0)

    print(f"--- {tag} ---")
    print(f"  stopped at step {steps[-1]} "
          f"({'CAP-TRUNCATED -- B did NOT reach ceiling' if cap else 'B ceiling + A flat'})")
    print(f"  B acc: end={bacc[-1]:.4f}  reached_ceiling={bacc[-1] >= 0.99}")
    print(f"  A acc: baseline={acc[0]:.4f} trough={acc[i]:.4f}@{steps[i]} "
          f"peak_after={peak:.4f}@{steps[j]} recovery_fraction={recov:.3f} end={acc[-1]:.4f}")
    print(f"  A rank_own: start={rank[0]:.3f} at_trough={rank[i]:.3f} "
          f"max={rank.max():.3f}@{steps[int(rank.argmax())]} end={rank[-1]:.3f}")
    print(f"  peak margin s_eff/s*: p05={np.percentile(peak_margin,5):.4f} "
          f"p50={np.percentile(peak_margin,50):.4f} p95={np.percentile(peak_margin,95):.4f}  "
          f"ever-supercritical={float((margins>1.0).any(axis=0).mean()):.2%}")
    show = [0, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000]
    show = [v for v in show if v <= steps[-1]]
    for name, arr in (("A acc ", acc), ("A rank", rank), ("B acc ", bacc)):
        print(f"  {name} " + "  ".join(
            f"{v}:{arr[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
    print()


def main():
    cfg = cfg_for()
    ps = P.build_populations(cfg)
    store0, _h, ceil, tgt = pretrain(cfg, ps)
    s_star = critical_push_per_individual(store0.logits(ps.keys_a), ps.values_a, ps.v_a_mask)
    print(f"=== overlap {OVERLAP}, matched ratio, run to B ceiling (cap {MAX_STEPS}), "
          f"pretrained ceiling@{ceil} ===\n")
    arms = [("b free (battery config)", False)]
    if ARMS == "both":
        arms.append(("b FROZEN (diagnostic ablation)", True))
    for tag, frozen in arms:
        curve, cap, _st = finetune(cfg, ps, store0, freeze_bias=frozen)
        report(tag, curve, cap, s_star)


if __name__ == "__main__":
    main()
