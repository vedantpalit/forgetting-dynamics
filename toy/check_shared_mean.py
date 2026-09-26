"""Does a W-CARRIED constant shift reverse? (PLAN.md 0.11)

THIS IS A TEST OF A MECHANISM CLAIM, NOT A FIX FOR THE TOY. The transformer measurement
(docs/shift_decomposition.md) produced a claim: a constant logit shift carried by W gets
bought back because it shares parameters with B's own fact storage, while one carried by a
free bias does not, because it shares parameters with nothing. The shared mean gives the toy
a W-carried constant shift. Whether it reverses is a prediction the toy either confirms or
refutes.

WHY THE TOY COULD NOT PRODUCE ONE BEFORE. Measured here: at beta=0 the mean key norm is
0.104 (A) and 0.135 (B), i.e. the key set has essentially no common direction, so a W update
along k_bar_B produces almost no logit shift -- the induced constant shift was 0.00059
against the transformer's ||c|| of ~300, and freezing b gave literally zero suppression over
8100 steps. With k_i = sqrt(beta)*mu + sqrt(1-beta)*(previous), k_bar_B acquires norm
~sqrt(beta) (verified: 0.514 / 0.714 / 0.950 at beta = 0.25 / 0.5 / 0.9), so the constant
shift becomes W-carried by construction.

`b` is FROZEN in every arm. That is the whole point: with b free the toy's suppression is
100% bias-carried and the question cannot be asked.

THREE PRE-REGISTERED BRANCHES, written before the run.

  (1) The shift appears AND reverses -- ||c|| rises then falls while B resolves. The
      mechanism claim is confirmed and the toy is complete: suppression, recovery and their
      carrier, all in a system with a closed-form solution.

  (2) The shift does not appear -- ||c|| stays negligible even at large beta. A common key
      direction is not sufficient to make the shift W-carried, and the reason needs finding
      before anything else is built.

  (3) The shift appears and still does NOT reverse. This is the live one, and it is a
      RESULT, not a failed test. In the toy, W's V_A and V_B columns are DISJOINT
      parameters: B's facts are stored in V_B columns, the constant shift on A lives in V_A
      columns, and writing mu-direction content into V_A does not obviously compete with
      writing B's individual facts into V_B. "Shares parameters" may be true at the level of
      the matrix W and false at the level of the columns that matter. The transformer has no
      such separation, because B's learning changes what A's KEY is, not merely what is read
      off it. If this is the outcome it sharpens the next rung rather than closing the line
      -- see PLAN.md 0.11's ladder.

INSTRUMENTATION. ||c|| and ||v|| are logged at every eval (train.py), defined exactly as the
transformer measurement defines them. Accuracy is floored in this toy and these are
continuous, so a slight turnover in ||c|| while accuracy stays pinned at zero is the
mechanism appearing in weak form -- reading accuracy alone would miss it. Both are logged so
the race between c shrinking and v growing is visible on the same terms as the real model's
308.6 -> 191.7 from trough to recovery peak.

Run: python -m check_shared_mean   (from toy/)
"""
import sys

import numpy as np

import config as C
import populations as P
from feasibility import critical_push_per_individual, ridge_precheck
from train import finetune, pretrain

MATCHED_RATIO = 40.0 / 3.0
MAX_STEPS = 12000
OVERLAP = 0.5
# argv: [comma-separated betas] [pretrain_max_steps]
BETAS = (tuple(float(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1
         else (0.0, 0.5, 0.9))
# High beta crowds the keys (cond 586 at 0.9) so gradient descent needs a longer
# budget to reach the ceiling the protocol requires before injection. The closed-form
# pre-check confirms the solution EXISTS at every beta (rank 1200 = n, ridge 1.0000),
# so this is a budget, not a feasibility knob.
PRETRAIN_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 6000


def cfg_for(beta, seed=0):
    return C.ToyConfig(
        embedding=C.EmbeddingConfig(alpha=0.7, beta=beta),
        population=C.PopulationConfig(overlap=OVERLAP),
        optimizer=C.OptimizerConfig(lr=3.0, inject_lr_ratio=MATCHED_RATIO),
        protocol=C.ProtocolConfig(pretrain_max_steps=PRETRAIN_MAX, pretrain_extra_fraction=0.25,
                                  finetune_max_steps=MAX_STEPS, eval_dense_until=200,
                                  eval_every_sparse=50, finetune_flat_window=50,
                                  finetune_min_b_acc=0.99),
        seed=seed)


def main():
    print("=== Feasibility pre-check per beta (closed form, before any training) ===")
    print(f"  {'beta':>6} {'mean|k_bar_B|':>14} {'key_rank':>9} {'n':>6} {'ridge_acc':>10} "
          f"{'key_cond':>12}")
    usable = []
    for beta in BETAS:
        cfg = cfg_for(beta)
        ps = P.build_populations(cfg)
        keys = np.concatenate([ps.keys_a, ps.keys_d], axis=0)
        vals = np.concatenate([ps.values_a, ps.values_d], axis=0)
        info, _W, _lg = ridge_precheck(keys, vals, cfg.vocab.v_total)
        ok = info["key_rank"] >= keys.shape[0] and info["ridge_acc_full"] > 0.99
        print(f"  {beta:>6.2f} {ps.diagnostics['mean_key_norm_b']:>14.4f} "
              f"{info['key_rank']:>9} {keys.shape[0]:>6} {info['ridge_acc_full']:>10.4f} "
              f"{info['key_cond']:>12.1f}{'' if ok else '   <- INFEASIBLE, skipping'}")
        if ok:
            usable.append(beta)
    print()

    for beta in usable:
        cfg = cfg_for(beta)
        ps = P.build_populations(cfg)
        store0, _h, ceil, tgt = pretrain(cfg, ps)
        s_star = critical_push_per_individual(
            store0.logits(ps.keys_a), ps.values_a, ps.v_a_mask)
        curve, cap, _st = finetune(cfg, ps, store0, freeze_bias=True)

        steps = np.array([c["step"] for c in curve])
        acc = np.array([c["a_full_acc_mean"] for c in curve])
        rank = np.array([c["a_restr_rank_mean"] for c in curve])
        bacc = np.array([c["b_full_acc_mean"] for c in curve])
        cn = np.array([c["c_norm"] for c in curve])
        vn = np.array([c["v_norm"] for c in curve])
        cva = np.array([c["c_mean_va"] for c in curve])

        i_tr = int(acc.argmin())
        peak = acc[i_tr:].max()
        recov = (peak - acc[i_tr]) / max(acc[0] - acc[i_tr], 1e-12)
        i_cmax = int(cn.argmax())
        c_drop = 1.0 - cn[-1] / max(cn[i_cmax], 1e-12)

        print(f"########## beta={beta}  (b FROZEN, overlap {OVERLAP}, matched ratio) ##########")
        print(f"  pretrained ceiling@{ceil} -> {tgt};  stopped at step {steps[-1]} "
              f"({'CAP-TRUNCATED, B short of ceiling' if cap else 'B ceiling + A flat'})")
        print(f"  B acc end={bacc[-1]:.4f}")
        print(f"  A acc: baseline={acc[0]:.4f} trough={acc[i_tr]:.4f}@{steps[i_tr]} "
              f"peak_after={peak:.4f} recovery_fraction={recov:.3f} end={acc[-1]:.4f}")
        print(f"  A rank_own: start={rank[0]:.3f} max={rank.max():.3f}@{steps[int(rank.argmax())]} "
              f"end={rank[-1]:.3f}")
        print(f"  ||c||: max={cn[i_cmax]:.4f}@{steps[i_cmax]}  end={cn[-1]:.4f}  "
              f"drop from max={c_drop:>6.1%}   <-- DOES IT TURN OVER?")
        print(f"  ||v||: at c-max={vn[i_cmax]:.4f}  end={vn[-1]:.4f}  "
              f"growth={vn[-1] / max(vn[i_cmax], 1e-12):.2f}x")
        print(f"  mean shift on V_A: at c-max={cva[i_cmax]:+.4f}  end={cva[-1]:+.4f}")
        show = [v for v in (0, 50, 100, 200, 400, 800, 1600, 3000, 6000, 9000, 12000)
                if v <= steps[-1]]
        for name, arr in (("A acc ", acc), ("A rank", rank), ("B acc ", bacc),
                          ("||c|| ", cn), ("||v|| ", vn)):
            print(f"  {name} " + "  ".join(
                f"{v}:{arr[np.argmin(np.abs(steps - v))]:.3f}" for v in show))
        print()

    print("Branches: (1) c rises then falls = mechanism confirmed; (2) c stays ~0 = shift")
    print("never appears; (3) c rises and stays = W-carried is not enough, and the V_A/V_B")
    print("column disjointness is the suspect (PLAN.md 0.11).")


if __name__ == "__main__":
    main()
