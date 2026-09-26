"""Does concentration rescue recovery, and lift the trough off the floor?

The 2000-step run at concentration 8 showed suppression but NO recovery and NO erosion
(rank pinned at 1.001 throughout, both overlaps). Before reading that as "the toy lacks
both channels", concentration has to be varied: it is the axis CHECK 3 says governs the
trough, and PLAN.md §0.7 already pre-registers that recovery may be weak or absent at LOW
concentration, because with few distinct B values B's individuals never need to be
distinguished and the per-example negative phase that is supposed to reverse the coherent
uplift never has to fire.

Concentration 8 of |V_B|=256 is very low -- 240 B individuals over 8 values, 30 each. This
probes {1, 8, 64} at the matched ratio to see whether trough depth and recovery move with
it, which is simultaneously CHECK 3's question and the floor-vacuity question.

Run: python -m check_concentration_probe   (from toy/)
"""
import numpy as np

import config as C
import populations as P
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
STEPS = 2000
CONCENTRATIONS = (1, 8, 64)


def cfg_for(conc, overlap=0.5, seed=0):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7),
        population=C.PopulationConfig(overlap=overlap, concentration=conc),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=MATCHED_RATIO),
        protocol=C.ProtocolConfig(pretrain_max_steps=4000, pretrain_extra_fraction=0.25,
                                   finetune_max_steps=STEPS, eval_dense_until=100,
                                   eval_every_sparse=25, finetune_flat_window=10 ** 6),
        seed=seed)


def main():
    base = cfg_for(8)
    ps0 = P.build_populations(base)
    store0, _h, ceil, tgt = pretrain(base, ps0)
    print(f"pretrained once (shared across conditions): ceiling@{ceil}, to {tgt}\n")
    print(f"  {'conc':>5} {'trough':>8} {'@step':>6} {'peak_after':>11} {'recov':>7} "
          f"{'end':>8} {'rank_max':>9} {'rank_end':>9} {'B@end':>7} {'floored':>8}")
    for conc in CONCENTRATIONS:
        cfg = cfg_for(conc)
        ps = P.build_populations(cfg)
        curve, cap, _st = finetune(cfg, ps, store0)
        steps = np.array([c["step"] for c in curve])
        acc = np.array([c["a_full_acc_mean"] for c in curve])
        rank = np.array([c["a_restr_rank_mean"] for c in curve])
        bacc = np.array([c["b_full_acc_mean"] for c in curve])
        i = int(acc.argmin())
        peak = acc[i:].max()
        recov = (peak - acc[i]) / max(acc[0] - acc[i], 1e-12)
        print(f"  {conc:>5} {acc[i]:>8.4f} {steps[i]:>6} {peak:>11.4f} {recov:>7.3f} "
              f"{acc[-1]:>8.4f} {rank.max():>9.3f} {rank[-1]:>9.3f} {bacc[-1]:>7.3f} "
              f"{'YES' if acc[i] < 0.01 else 'no':>8}")
        show = [0, 50, 100, 200, 400, 800, 1200, 2000]
        print("        A curve  " + "  ".join(
            f"{v}:{acc[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
        print("        B curve  " + "  ".join(
            f"{v}:{bacc[np.argmin(np.abs(steps - v))]:.3f}" for v in show))


if __name__ == "__main__":
    main()
