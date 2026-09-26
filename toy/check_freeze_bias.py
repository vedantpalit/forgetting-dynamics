"""Diagnostic 2: freeze b during injection -- a direct test of bias dominance.

THIS IS A DIAGNOSTIC ABLATION, NOT A BATTERY CONFIGURATION. Freezing b removes a degree of
freedom the transformer does not have in the same form: there, a uniform per-value shift has
to be built out of the same parameters that store the facts, so it is competed away as B
resolves, whereas this toy's free-standing b is a shortcut that never stops paying. The
hypothesis under test is that this difference is what explains the absent recovery -- so
closing the shortcut and seeing whether the expected dynamics appear is a diagnosis, not a
fix, and the battery still runs with b free.

Mechanism, stated so the prediction is falsifiable: with b frozen, B's loss can only be
reduced by writing to W, forcing the update into the key-dependent channel. That is where
erosion lives, and it is also where the suppression push becomes something that must
eventually reverse -- a coherent write along B's mean key direction stops paying once B's
individuals have to be told apart from each other, whereas a bias shift never stops paying.

Reports the same quantities as the 2000-step run, for frozen and free b side by side off the
SAME pretrained store, so the comparison is exact rather than seed-matched: A's accuracy
curve, A's own-half rank at start/trough/max/end, B's acquisition, the per-individual margin
distribution with the supercritical fraction, and the decomposition norms.

Run: python -m check_freeze_bias   (from toy/)
"""
import numpy as np

import config as C
import populations as P
from check_update_decomposition import cfg_for, make_probe
from feasibility import critical_push_per_individual
from train import finetune, pretrain

STEPS = 2000
OVERLAPS = (0.0, 0.5)
PROBE_STEPS = (0, 25, 50, 100, 200, 400, 800, 1200, 2000)


def report(tag, curve, cap, s_star):
    steps = np.array([c["step"] for c in curve])
    acc = np.array([c["a_full_acc_mean"] for c in curve])
    rank = np.array([c["a_restr_rank_mean"] for c in curve])
    bacc = np.array([c["b_full_acc_mean"] for c in curve])

    i = int(acc.argmin())
    peak = acc[i:].max()
    recov = (peak - acc[i]) / max(acc[0] - acc[i], 1e-12)

    ok = np.isfinite(s_star)
    # Peak over time of the per-individual margin, not the endpoint: with b frozen the push
    # is not monotone by hypothesis, so the endpoint would understate how far it got.
    margins = np.array([c["s_eff_per_individual"][ok] / s_star[ok] for c in curve])
    peak_margin = margins.max(axis=0)
    frac_super = float((margins > 1.0).any(axis=0).mean())

    print(f"--- {tag} ({'CAP-TRUNCATED' if cap else 'stopped on flat window'}) ---")
    print(f"  margin s_eff/s* per individual, PEAK over time: "
          f"p05={np.percentile(peak_margin, 5):.4f} p50={np.percentile(peak_margin, 50):.4f} "
          f"p95={np.percentile(peak_margin, 95):.4f}  "
          f"ever-supercritical fraction = {frac_super:.2%}")
    print(f"  A acc: baseline={acc[0]:.4f} trough={acc[i]:.4f}@{steps[i]} "
          f"peak_after={peak:.4f} recovery_fraction={recov:.3f} end={acc[-1]:.4f}")
    print(f"  A rank_own: start={rank[0]:.3f} at_trough={rank[i]:.3f} "
          f"max={rank.max():.3f}@{steps[int(rank.argmax())]} end={rank[-1]:.3f}")
    show = [0, 50, 100, 200, 400, 800, 1200, 2000]
    print("  A acc curve   " + "  ".join(
        f"{v}:{acc[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
    print("  A rank curve  " + "  ".join(
        f"{v}:{rank[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
    print("  B acc curve   " + "  ".join(
        f"{v}:{bacc[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
    print(f"  {'step':>6} {'||db||':>9} {'||dW||':>9} {'rank1':>9} {'shared':>9} {'orth':>9}"
          f" {'shared/dW':>10}")
    for v in PROBE_STEPS:
        r = curve[int(np.argmin(np.abs(steps - v)))]
        dwn = max(r["dec_dW_norm"], 1e-12)
        print(f"  {r['step']:>6} {r['dec_bias_norm']:>9.4f} {r['dec_dW_norm']:>9.4f} "
              f"{r['dec_rank1_norm']:>9.4f} {r['dec_shared_norm']:>9.4f} "
              f"{r['dec_orth_norm']:>9.4f} {r['dec_shared_norm']/dwn:>10.4f}")
    print()


def main():
    for overlap in OVERLAPS:
        cfg = cfg_for(overlap=overlap)
        ps = P.build_populations(cfg)
        store0, _h, ceil, tgt = pretrain(cfg, ps)
        s_star = critical_push_per_individual(
            store0.logits(ps.keys_a), ps.values_a, ps.v_a_mask)
        probe = make_probe(ps, set(PROBE_STEPS))
        print(f"=== overlap {overlap} (matched ratio, {STEPS} steps, pretrained ceiling@{ceil}) ===")
        print(f"  touched A={ps.diagnostics['n_a_touched']} "
              f"tokens={ps.diagnostics['n_distinct_a_last_tokens_touched']} "
              f"collisions/token={ps.diagnostics['collisions_per_touched_token']:.2f}\n")
        for tag, frozen in (("b FROZEN (diagnostic ablation)", True), ("b free (battery config)", False)):
            curve, cap, _st = finetune(cfg, ps, store0, probe_fn=probe, freeze_bias=frozen)
            report(tag, curve, cap, s_star)


if __name__ == "__main__":
    main()
