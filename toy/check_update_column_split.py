"""Diagnostic 1b: split the update by VALUE COLUMN, not just by row space.

Diagnostic 1 produced two readings that disagree. The norms say the shared-subspace
component is substantial (34% of ||dW||, growing); the counterfactuals say deleting it
changes nothing about A, and that removing delta_b alone restores A to exactly its 0.9990
baseline at every step.

The likely reason they disagree is a defect in the norm measurement as specified:
`decompose_update` splits by ROW SPACE (directions in R^d) across all 512 value columns at
once. But B's own learning writes large updates into the V_B columns, and those columns
cannot affect A's ordering WITHIN V_A. So a large "shared subspace" norm may be entirely
about how B's facts get stored, and say nothing about the erosion channel.

The mechanism this tests, stated as a falsifiable prediction rather than an explanation:
for a value column j in V_A the gradient is (1/n) sum_i p_i[j] k_i -- there is no one-hot
term, because no B individual has a V_A target. Those weights are near-uniform across V_A
and decay toward zero, so every V_A column should receive nearly the SAME vector (the
probability-weighted mean key). A write common to all of V_A shifts A's logits uniformly and
cannot reorder within V_A. If that is right, erosion is not outcompeted and not cancelled;
it is structurally unavailable through W under cross-entropy on B-only data, and the shared
FEATURE channel being open in key space is irrelevant because the channel that is closed is
in VALUE space.

Reported per component (bias, rank-1 mean-key, shared subspace, orthogonal remainder, and
the total dW), at selected steps:

  * norm restricted to V_A columns vs V_B columns -- separates "writes about B" from
    "writes that reach A's value region";
  * the induced logit shift on A, S = keys_a @ component, summarized as the mean over V_A
    columns (a uniform region shift, which moves full-vocab accuracy but NOT rank) and the
    within-row SPREAD across V_A columns (the only part that can reorder within V_A);
  * that spread against the baseline within-V_A margin (correct minus runner-up), which is
    the scale it has to beat to change rank at all.

Run: python -m check_update_column_split   (from toy/)
"""
import numpy as np

import config as C
import metrics as m
import populations as P
from check_update_decomposition import cfg_for
from train import finetune, pretrain

CF_STEPS = (25, 100, 200, 400, 800, 1200, 2000)


def make_probe(ps, cf_steps, v_a_mask):
    u = ps.keys_b.mean(axis=0)
    u = u / np.linalg.norm(u)
    touched = np.unique(ps.last_b[ps.overlap_mask_b])
    Q = ps.last_feats[touched].T
    va, vb = v_a_mask, ~v_a_mask

    def probe(step, store, W0, b0):
        if step not in cf_steps:
            return {}
        dW = store.W - W0
        db = store.b - b0
        dec = m.decompose_update(dW, db, u, Q)
        rank1 = np.outer(u, u @ dW)
        shared = dec["shared_component"]
        orth = dW - rank1 - shared

        out = {}
        comps = [("dW_total", dW), ("rank1", rank1), ("shared", shared), ("orth", orth)]
        for name, X in comps:
            S = ps.keys_a @ X                      # (n_a, |V|) induced logit shift on A
            out[f"col_{name}_normA"] = float(np.linalg.norm(X[:, va]))
            out[f"col_{name}_normB"] = float(np.linalg.norm(X[:, vb]))
            out[f"col_{name}_shiftA_mean"] = float(S[:, va].mean())
            out[f"col_{name}_shiftA_spread"] = float(S[:, va].std(axis=1).mean())
            out[f"col_{name}_shiftB_mean"] = float(S[:, vb].mean())
        # Counterfactual: A re-evaluated with the bias held at its pre-injection value and
        # W as trained. Separates "the bias did it" from "W did it" on the realized state,
        # which norms alone cannot do (diagnostic 1's two halves disagreed for exactly this
        # reason).
        lg_nobias = ps.keys_a @ store.W + b0
        out["cf_nobias_acc"] = float(m.argmax_correct(lg_nobias, ps.values_a).mean())
        out["cf_nobias_rank"] = float(
            m.rank_of_correct(lg_nobias, ps.values_a, ps.v_a_mask).mean())

        # bias: the induced shift is db itself, identical for every individual
        out["col_bias_normA"] = float(np.linalg.norm(db[va]))
        out["col_bias_normB"] = float(np.linalg.norm(db[vb]))
        out["col_bias_shiftA_mean"] = float(db[va].mean())
        out["col_bias_shiftA_spread"] = float(db[va].std())
        out["col_bias_shiftB_mean"] = float(db[vb].mean())
        return out

    return probe


def main():
    cfg = cfg_for()
    ps = P.build_populations(cfg)
    store0, _h, ceil, tgt = pretrain(cfg, ps)
    print(f"overlap=0.5  pretrained ceiling@{ceil}, to {tgt}")

    # The scale the within-V_A spread has to beat to reorder anything.
    lg0 = store0.logits(ps.keys_a)
    va = ps.v_a_mask
    masked = np.where(va[None, :], lg0, -np.inf)
    correct = masked[np.arange(len(ps.values_a)), ps.values_a]
    tmp = masked.copy()
    tmp[np.arange(len(ps.values_a)), ps.values_a] = -np.inf
    runner_up = tmp.max(axis=1)
    gap = correct - runner_up
    print(f"baseline within-V_A margin (correct - runner-up): "
          f"p05={np.percentile(gap,5):.4f} p50={np.median(gap):.4f} p95={np.percentile(gap,95):.4f}\n")

    curve, cap, _st = finetune(cfg, ps, store0,
                               probe_fn=make_probe(ps, set(CF_STEPS), va))
    steps = np.array([c["step"] for c in curve])

    print("=== Norm split by value column (Frobenius) ===")
    print(f"  {'step':>6} " + " ".join(f"{n:>21}" for n in
                                        ("bias", "rank1", "shared", "orth", "dW_total")))
    print(f"  {'':>6} " + " ".join(f"{'V_A':>10}{'V_B':>11}" for _ in range(5)))
    for v in CF_STEPS:
        r = curve[int(np.argmin(np.abs(steps - v)))]
        row = " ".join(f"{r[f'col_{n}_normA']:>10.4f}{r[f'col_{n}_normB']:>11.4f}"
                       for n in ("bias", "rank1", "shared", "orth", "dW_total"))
        print(f"  {r['step']:>6} {row}")

    print("\n=== Induced logit shift on A: mean over V_A / SPREAD across V_A / mean over V_B ===")
    print("    (only the SPREAD can reorder within V_A; the mean is a uniform region shift)")
    for n in ("bias", "rank1", "shared", "orth", "dW_total"):
        print(f"  -- {n} --")
        print(f"     {'step':>6} {'meanA':>12} {'spreadA':>12} {'meanB':>12} {'spread/margin':>14}")
        for v in CF_STEPS:
            r = curve[int(np.argmin(np.abs(steps - v)))]
            sp = r[f"col_{n}_shiftA_spread"]
            print(f"     {r['step']:>6} {r[f'col_{n}_shiftA_mean']:>12.5f} {sp:>12.5f} "
                  f"{r[f'col_{n}_shiftB_mean']:>12.5f} {sp/np.median(gap):>14.5f}")

    print(f"\n  A acc end={curve[-1]['a_full_acc_mean']:.4f} rank_own end={curve[-1]['a_restr_rank_mean']:.3f} "
          f"B acc end={curve[-1]['b_full_acc_mean']:.3f}  {'CAP-TRUNCATED' if cap else 'flat-window stop'}")


if __name__ == "__main__":
    main()
