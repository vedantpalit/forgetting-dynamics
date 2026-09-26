"""Diagnostic 1: where does B's update actually land? (bias dominance vs cancellation)

The 2000-step run left A's own-half rank at 1.001 at start, trough, max AND end at
overlap 0.5, where 480 A individuals share last-name tokens with B at 5 collisions per
token. Erosion did not accumulate too little to notice; it did not happen. The shared-bias
account explains suppression but says nothing about why the shared-feature channel is
silent: B's updates land in span{k_i : i in B}, which overlaps A's key space substantially
at overlap 0.5, so writes into those shared directions should move A's within-region
ranking somewhat regardless of what b does.

Two hypotheses, distinguished here:

  BIAS DOMINANCE  -- the update is almost entirely in b plus the rank-one mean-key
      component, with the shared-subspace component near zero. Erosion is then not absent
      in principle but outcompeted: the bias is such an efficient way to reduce B's loss
      that gradient descent barely writes to W at all.

  CANCELLATION    -- the shared-subspace component is substantial, but its effect on A's
      within-half ranking cancels. Large shared norms, no rank movement.

Norms alone cannot separate these, because "substantial norm" and "no effect" is exactly
what cancellation looks like. So this reports BOTH the three-way decomposition norms over
time AND, at selected steps, counterfactual re-evaluations of A with one component of the
realized update removed. If deleting the shared component leaves A's rank unchanged while
the component's norm is large, that is cancellation; if the norm itself is negligible,
that is dominance.

Subspace choices, stated because they are judgement calls:
  * direction_u   = the normalized mean of B's keys (the coherent direction along which a
    suppression-style uplift is written), matching decompose_update's contract.
  * shared_basis_Q = the last-name features B actually touches, last_feats[touched_tokens]
    (24 columns at overlap 0.5). These are precisely the directions A and B keys have in
    common at this overlap -- B's first names always come from B's own reserved pool, so
    the last-name features are the entire shared channel.

Run: python -m check_update_decomposition   (from toy/)
"""
import numpy as np

import config as C
import metrics as m
import populations as P
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
STEPS = 2000
OVERLAP = 0.5
CF_STEPS = (0, 25, 50, 100, 200, 400, 800, 1200, 2000)


def cfg_for(overlap=OVERLAP, seed=0):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7),
        population=C.PopulationConfig(overlap=overlap),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=MATCHED_RATIO),
        protocol=C.ProtocolConfig(pretrain_max_steps=4000, pretrain_extra_fraction=0.25,
                                   finetune_max_steps=STEPS, eval_dense_until=100,
                                   eval_every_sparse=25, finetune_flat_window=10 ** 6),
        seed=seed)


def make_probe(ps, cf_steps):
    u = ps.keys_b.mean(axis=0)
    u = u / np.linalg.norm(u)
    touched_tokens = np.unique(ps.last_b[ps.overlap_mask_b])
    Q = ps.last_feats[touched_tokens].T if len(touched_tokens) else np.zeros((ps.keys_a.shape[1], 0))

    def probe(step, store, W0, b0):
        dW = store.W - W0
        db = store.b - b0
        dec = m.decompose_update(dW, db, u, Q) if Q.shape[1] else None
        out = {
            "dec_bias_norm": float(np.linalg.norm(db)),
            "dec_dW_norm": float(np.linalg.norm(dW)),
            "dec_rank1_norm": dec["rank1_component_norm"] if dec else float("nan"),
            "dec_shared_norm": dec["shared_component_norm"] if dec else float("nan"),
            "dec_orth_norm": dec["orthogonal_remainder_norm"] if dec else float("nan"),
        }
        if step in cf_steps and dec is not None:
            rank1 = np.outer(u, u @ dW)
            shared = dec["shared_component"]
            variants = {
                "full": (store.W, store.b),
                "nobias": (store.W, b0),                    # delta_b removed
                "norank1": (store.W - rank1, store.b),      # coherent mean-key write removed
                "noshared": (store.W - shared, store.b),    # shared-subspace write removed
            }
            for name, (Wc, bc) in variants.items():
                lg = ps.keys_a @ Wc + bc
                out[f"cf_{name}_acc"] = float(m.argmax_correct(lg, ps.values_a).mean())
                out[f"cf_{name}_rank"] = float(
                    m.rank_of_correct(lg, ps.values_a, ps.v_a_mask).mean())
        return out

    return probe


def main():
    cfg = cfg_for()
    ps = P.build_populations(cfg)
    print(f"overlap={OVERLAP}  touched A individuals={ps.diagnostics['n_a_touched']}  "
          f"touched tokens={ps.diagnostics['n_distinct_a_last_tokens_touched']}  "
          f"collisions/token={ps.diagnostics['collisions_per_touched_token']:.2f}")
    store0, _h, ceil, tgt = pretrain(cfg, ps)
    print(f"pretrained: ceiling@{ceil}, to {tgt}\n")

    curve, cap, _st = finetune(cfg, ps, store0, probe_fn=make_probe(ps, set(CF_STEPS)))
    steps = np.array([c["step"] for c in curve])

    def at(v, key):
        return curve[int(np.argmin(np.abs(steps - v)))][key]

    print("=== Decomposition norms over time (Frobenius) ===")
    print(f"  {'step':>6} {'||db||':>10} {'||dW||':>10} {'rank1':>10} {'shared':>10} "
          f"{'orth':>10} {'shared/dW':>10} {'rank1/dW':>10}")
    for v in CF_STEPS:
        i = int(np.argmin(np.abs(steps - v)))
        r = curve[i]
        dwn = max(r["dec_dW_norm"], 1e-12)
        print(f"  {r['step']:>6} {r['dec_bias_norm']:>10.4f} {r['dec_dW_norm']:>10.4f} "
              f"{r['dec_rank1_norm']:>10.4f} {r['dec_shared_norm']:>10.4f} "
              f"{r['dec_orth_norm']:>10.4f} {r['dec_shared_norm']/dwn:>10.4f} "
              f"{r['dec_rank1_norm']/dwn:>10.4f}")

    print("\n=== Counterfactual A metrics with one component of the update removed ===")
    print("    (acc = full-vocab argmax; rank = mean rank of the correct value within V_A, of 256)")
    print(f"  {'step':>6} | {'full acc':>9} {'rank':>7} | {'nobias acc':>11} {'rank':>7} | "
          f"{'norank1 acc':>12} {'rank':>7} | {'noshared acc':>13} {'rank':>7}")
    for v in CF_STEPS:
        i = int(np.argmin(np.abs(steps - v)))
        r = curve[i]
        if "cf_full_acc" not in r:
            continue
        print(f"  {r['step']:>6} | {r['cf_full_acc']:>9.4f} {r['cf_full_rank']:>7.3f} | "
              f"{r['cf_nobias_acc']:>11.4f} {r['cf_nobias_rank']:>7.3f} | "
              f"{r['cf_norank1_acc']:>12.4f} {r['cf_norank1_rank']:>7.3f} | "
              f"{r['cf_noshared_acc']:>13.4f} {r['cf_noshared_rank']:>7.3f}")

    print(f"\n  A acc: baseline={curve[0]['a_full_acc_mean']:.4f} end={curve[-1]['a_full_acc_mean']:.4f}"
          f"   A rank_own end={curve[-1]['a_restr_rank_mean']:.3f}"
          f"   B acc end={curve[-1]['b_full_acc_mean']:.3f}"
          f"   {'CAP-TRUNCATED' if cap else 'stopped on flat window'}")


if __name__ == "__main__":
    main()
