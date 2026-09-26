"""Is the erosion channel closed by DISJOINT VALUE HALVES specifically? (PLAN.md §0.10d)

Diagnostics 1/1b/2/3 showed that on B-only data, with frozen features, a linear store and
value halves disjoint after pretraining, the V_A columns receive a near-common write and A's
ordering within V_A cannot change. That is a conclusion about this TRAINING SIGNAL, not
about linear stores generally, and the natural suspect is disjointness itself.

Erosion needs p_i[j] to vary across j in V_A for B's individuals. That requires the model to
place differing probability on specific A values for specific B individuals -- and the
disjoint-halves construction suppresses exactly that, since after pretraining every B
individual's probability over V_A is close to flat. Disjointness was inherited from Q1,
where it exists to control the SUPPRESSION channel; in the toy it has the side effect of
closing the EROSION channel.

Here B draws its values from V_A instead of V_B, at overlap 0.5, matched ratio, everything
else unchanged. B's targets are then values A already uses, the one-hot term lands in V_A
columns, and the column writes are differentiated by construction rather than near-common.

  * rank MOVES        -> erosion is available, and disjointness was what closed it.
  * rank does NOT move -> the unavailability is deeper, and the multi-attribute shared-store
                          variant becomes the real candidate.

Implementation note: the remap is done on the built PopulationSet by subtracting the V_A
offset from `values_b`, NOT by adding a config field. Same `concentration` distinct values,
same individuals, same keys, same pretrained store -- only the region the targets live in
changes. Keeping it out of the config also keeps a diagnostic variant from leaking into a
battery condition_id.

Run: python -m check_shared_values   (from toy/)
"""
import sys

import numpy as np

import config as C
import populations as P
from check_update_column_split import make_probe
from feasibility import critical_push_per_individual
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
MAX_STEPS = 12000
OVERLAP = 0.5
PROBE_STEPS = (25, 100, 200, 400, 800, 1600, 2800)
# argv[1] in {"free", "frozen", "both"}; default "both".
ARMS = sys.argv[1] if len(sys.argv) > 1 else "both"


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


def main():
    cfg = cfg_for()
    ps = P.build_populations(cfg)
    store0, _h, ceil, tgt = pretrain(cfg, ps)

    # Remap B's targets from V_B into V_A, preserving the identity of the concentration
    # values (same local indices, different region).
    ps.values_b = ps.values_b - cfg.vocab.v_a_size
    assert ps.values_b.min() >= 0 and ps.values_b.max() < cfg.vocab.v_a_size

    shared_vals = np.unique(ps.values_b)
    # How many A individuals hold one of the values B is now being taught -- these are the
    # REINFORCED ones; the rest face a one-hot push on columns that are not theirs.
    n_a_on_shared = int(np.isin(ps.values_a, shared_vals).sum())
    print(f"=== B draws values from V_A, overlap {OVERLAP}, matched ratio, "
          f"pretrained ceiling@{ceil} ===")
    print(f"  distinct shared values: {len(shared_vals)} of |V_A|={cfg.vocab.v_a_size}   "
          f"A individuals holding one of them: {n_a_on_shared} of {cfg.population.n_a}")
    print(f"  touched A={ps.diagnostics['n_a_touched']} "
          f"tokens={ps.diagnostics['n_distinct_a_last_tokens_touched']} "
          f"collisions/token={ps.diagnostics['collisions_per_touched_token']:.2f}\n")

    s_star = critical_push_per_individual(store0.logits(ps.keys_a), ps.values_a, ps.v_a_mask)
    arms = [("b free", False), ("b FROZEN", True)]
    if ARMS == "free":
        arms = arms[:1]
    elif ARMS == "frozen":
        arms = arms[1:]
    for tag, frozen in arms:
        print()
        print("########## " + tag + " ##########")
        run_arm(cfg, ps, store0, s_star, frozen)


def run_arm(cfg, ps, store0, s_star, frozen):
    curve, cap, _st = finetune(cfg, ps, store0, freeze_bias=frozen,
                               probe_fn=make_probe(ps, set(PROBE_STEPS), ps.v_a_mask))

    steps = np.array([c["step"] for c in curve])
    acc = np.array([c["a_full_acc_mean"] for c in curve])
    rank = np.array([c["a_restr_rank_mean"] for c in curve])
    bacc = np.array([c["b_full_acc_mean"] for c in curve])
    rt = np.array([c["a_restr_rank_touched_mean"] for c in curve])
    ru = np.array([c["a_restr_rank_untouched_mean"] for c in curve])

    i = int(acc.argmin())
    peak = acc[i:].max()
    recov = (peak - acc[i]) / max(acc[0] - acc[i], 1e-12)
    ok = np.isfinite(s_star)
    margins = np.array([c["s_eff_per_individual"][ok] / s_star[ok] for c in curve])
    pm = margins.max(axis=0)

    print(f"  stopped at step {steps[-1]} "
          f"({'CAP-TRUNCATED -- B did NOT reach ceiling' if cap else 'B ceiling + A flat'})")
    print(f"  B acc end={bacc[-1]:.4f}  reached_ceiling={bacc[-1] >= 0.99}")
    print(f"  A acc: baseline={acc[0]:.4f} trough={acc[i]:.4f}@{steps[i]} "
          f"peak_after={peak:.4f} recovery_fraction={recov:.3f} end={acc[-1]:.4f}")
    print(f"  A rank_own: start={rank[0]:.3f} at_trough={rank[i]:.3f} "
          f"max={rank.max():.3f}@{steps[int(rank.argmax())]} end={rank[-1]:.3f}   "
          f"<< THE READOUT")
    print(f"  A rank_own touched:   start={rt[0]:.3f} max={rt.max():.3f} end={rt[-1]:.3f}")
    print(f"  A rank_own untouched: start={ru[0]:.3f} max={ru.max():.3f} end={ru[-1]:.3f}")
    print(f"  peak margin s_eff/s*: p05={np.percentile(pm,5):.4f} p50={np.percentile(pm,50):.4f} "
          f"p95={np.percentile(pm,95):.4f}  ever-supercritical={float((margins>1.0).any(axis=0).mean()):.2%}")
    show = [v for v in (0, 100, 200, 400, 800, 1600, 2800, 5000, 8000, 12000) if v <= steps[-1]]
    for name, arr in (("A acc  ", acc), ("A rank ", rank), ("A rankT", rt),
                      ("A rankU", ru), ("B acc  ", bacc)):
        print(f"  {name} " + "  ".join(f"{v}:{arr[np.argmin(np.abs(steps - v))]:.3f}" for v in show))

    print(f"\n  === V_A vs V_B column norms (the mechanism readout) ===")
    print(f"  {'step':>6} " + " ".join(f"{n:>21}" for n in ("rank1", "shared", "orth", "dW_total")))
    print(f"  {'':>6} " + " ".join(f"{'V_A':>10}{'V_B':>11}" for _ in range(4)))
    for v in PROBE_STEPS:
        r = curve[int(np.argmin(np.abs(steps - v)))]
        if f"col_rank1_normA" not in r:
            continue
        row = " ".join(f"{r[f'col_{n}_normA']:>10.4f}{r[f'col_{n}_normB']:>11.4f}"
                       for n in ("rank1", "shared", "orth", "dW_total"))
        print(f"  {r['step']:>6} {row}")
    print(f"\n  === induced within-V_A SPREAD of dW, and A with the bias deleted ===")
    print(f"  {'step':>6} {'meanA':>12} {'spreadA':>12} {'nobias acc':>12} {'nobias rank':>12}")
    for v in PROBE_STEPS:
        r = curve[int(np.argmin(np.abs(steps - v)))]
        if "col_dW_total_shiftA_spread" not in r:
            continue
        print(f"  {r['step']:>6} {r['col_dW_total_shiftA_mean']:>12.5f} "
              f"{r['col_dW_total_shiftA_spread']:>12.5f} {r['cf_nobias_acc']:>12.4f} "
              f"{r['cf_nobias_rank']:>12.3f}")


if __name__ == "__main__":
    main()
