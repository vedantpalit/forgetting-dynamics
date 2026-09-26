"""Rung 2.5: does a W-carried constant shift reverse when the key can move? (PLAN.md 0.11)

GATED AND PASSED. The representation-vs-readout split on the real checkpoints
(FINDINGS 3.16) put 100% of the constant component's fall in A's REPRESENTATION: across five
runs `c_readout` peaked at the FINAL injection step (0.0% drop) with const-only accuracy
pinned at baseline throughout, while `c_rep` tracked `c_full` in both norm and accuracy. Over
the recovery window the two are OPPOSED -- `c_rep` falls 38.7% while `c_readout` rises 145%.
So recovery is A's key changing, and a trainable shared key direction is the right model.

MODEL. k_i(mu) = sqrt(beta)*mu + sqrt(1-beta)*base_i, with `W` and `mu` trainable and `b`
and every base_i frozen. `mu` is trained in BOTH phases (see bilinear.py), so it arrives at
injection already shaped by A and D -- which is why every rotation diagnostic here is
measured from the START OF INJECTION, not from the start of pretraining.

OBSERVABLE IS ||c||, NOT ACCURACY. Rung 1 showed accuracy floors hard in this toy, and the
transformer's signature is ||c|| turning over between trough and peak. A turnover with
accuracy still pinned counts as the mechanism appearing; accuracy alone would miss it. The
toy's ||c|| is decomposed the SAME WAY as the transformer's (rep / readout / cross), so the
correspondence is checkable in matching units rather than eyeballed. A proper vary-only
counterfactual is reported too, per rung 1's caveat that a full-vocabulary ||v|| dominated by
V_B columns is not the analog of the transformer's.

THREE PRE-REGISTERED BRANCHES.
  (1) ||c|| turns over AND A recovers -- mechanism confirmed, the toy has both channels.
  (2) ||c|| turns over but A does not recover -- partial; the mechanism is present but too
      weak. Report the magnitude and what would scale it.
  (3) ||c|| saturates as in rung 1 -- trainable mu is insufficient, and the next candidate
      is trainable name-part embeddings (rung 3).

EXPECTATION REGISTERED IN ADVANCE. A's keys depend on mu too, so a rotation large enough to
relieve A should also disturb A's own retrieval. Expect the constant component to fall
ALONGSIDE genuine degradation elsewhere. That is not a confound -- it is the erosion channel
arriving through the key rather than through the columns, which is what the transformer does.
A's own-half rank is logged throughout so it is measured, not inferred.

Run: python -m check_trainable_mu   (from toy/)
"""
import sys

import numpy as np

import bilinear as BL
import config as C
import populations as P

MATCHED_RATIO = 40.0 / 3.0
BETA = 0.5
MAX_STEPS = 12000
OVERLAPS = (0.0, 0.5)
PRETRAIN_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000


def cfg_for(overlap, seed=0):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7, beta=BETA),
        population=C.PopulationConfig(overlap=overlap),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=MATCHED_RATIO),
        protocol=C.ProtocolConfig(pretrain_max_steps=PRETRAIN_MAX,
                                  pretrain_extra_fraction=0.25,
                                  finetune_max_steps=MAX_STEPS, eval_dense_until=200,
                                  eval_every_sparse=50, finetune_flat_window=50,
                                  finetune_min_b_acc=0.99),
        seed=seed)


def main():
    for overlap in OVERLAPS:
        cfg = cfg_for(overlap)
        ps = P.build_populations(cfg)
        store0, ceil, tgt = BL.pretrain(cfg, ps)
        curve, cap, _st = BL.finetune(cfg, ps, store0)

        g = lambda k: np.array([c[k] for c in curve])
        steps = g("step")
        acc, rank, bacc = g("a_full_acc_mean"), g("a_restr_rank_mean"), g("b_full_acc_mean")
        cf, cr, cro, cc = (g("c_full_norm"), g("c_rep_norm"),
                           g("c_readout_norm"), g("c_cross_norm"))
        ang, dmu = g("mu_angle_deg"), g("d_mu_norm")

        i_tr = int(acc.argmin())
        peak = acc[i_tr:].max()
        j = i_tr + int(acc[i_tr:].argmax())
        recov = (peak - acc[i_tr]) / max(acc[0] - acc[i_tr], 1e-12)

        print(f"########## overlap={overlap}  beta={BETA}  (mu TRAINABLE, b frozen) ##########")
        print(f"  pretrained ceiling@{ceil} -> {tgt};  stopped at {steps[-1]} "
              f"({'CAP-TRUNCATED, B short of ceiling' if cap else 'B ceiling + flat'})")
        print(f"  B acc end={bacc[-1]:.4f}")
        print(f"  A acc: baseline={acc[0]:.4f} trough={acc[i_tr]:.4f}@{steps[i_tr]} "
              f"peak_after={peak:.4f}@{steps[j]} recovery_fraction={recov:.3f} end={acc[-1]:.4f}")
        print(f"  A rank_own: start={rank[0]:.3f} max={rank.max():.3f}@"
              f"{steps[int(rank.argmax())]} end={rank[-1]:.3f}   "
              f"(degradation through the KEY is expected, not a confound)")
        print(f"  {'component':>12} {'max':>10} {'@step':>7} {'end':>10} {'drop':>8}  "
              f"{'const-only acc @max':>20}")
        for name, arr in (("c_full", cf), ("c_rep", cr), ("c_readout", cro), ("c_cross", cc)):
            i = int(arr.argmax())
            key = name.replace("c_", "c_") + "_acc"
            a_at = curve[i][f"{name}_acc"]
            print(f"  {name:>12} {arr[i]:>10.4f} {steps[i]:>7} {arr[-1]:>10.4f} "
                  f"{1 - arr[-1] / max(arr[i], 1e-12):>7.1%}  {a_at:>20.4f}")
        print(f"  mu rotation from injection start: angle max={ang.max():.3f} deg "
              f"end={ang[-1]:.3f} deg;  ||d_mu|| end={dmu[-1]:.4f}")
        print(f"  mean shift on V_A: end={curve[-1]['c_mean_va']:+.4f}")
        print(f"  vary-only: acc end={curve[-1]['vary_acc']:.4f} "
              f"rank end={curve[-1]['vary_rank']:.3f}")
        show = [v for v in (0, 50, 100, 200, 400, 800, 1600, 3000, 6000, 9000, 12000)
                if v <= steps[-1]]
        for name, arr in (("A acc  ", acc), ("A rank ", rank), ("B acc  ", bacc),
                          ("c_full ", cf), ("c_rep  ", cr), ("c_read ", cro),
                          ("mu ang ", ang)):
            print(f"  {name} " + "  ".join(
                f"{v}:{arr[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
        print()

    print("Branches: (1) ||c|| turns over AND A recovers = confirmed; (2) turns over, no")
    print("accuracy recovery = partial; (3) saturates as in rung 1 = insufficient, rung 3 next.")


if __name__ == "__main__":
    main()
