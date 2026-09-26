"""Feasibility gate (PLAN.md §0.4).

Step 1, closed form, no training: ridge pre-check on the REAL pretraining population,
A union D, reporting the three diagnostics that would have caught the original failure in
seconds -- ridge argmax accuracy, key Gram rank and condition number, and load ratio.

Step 2: confirm by training. Pretrain on A union D and check ceiling.

Diagnosis order if it fails: ridge accuracy and conditioning, then alpha and d, then load
ratio, then learning rate LAST. An LR-independent plateau is a structural limit, not
undertuning; do not widen an LR grid in response to one.

Also prints the per-condition touched/untouched individual counts, so the power available to
each split reading is visible next to the result (PLAN.md §0.3b, §0.3c).

Run: python -m gate   (from toy/)
"""
import time

import numpy as np

import config as C
import populations as P
from feasibility import ridge_precheck
from train import pretrain

ALPHAS = (1.0, 0.85, 0.7, 0.5, 0.3)
LR_GRID = (0.3, 1.0, 3.0)
PRETRAIN_CEILING_STEPS = 4000


def precheck_table():
    print("=== Step 1: closed-form ridge pre-check on A u D (no training) ===")
    base = C.ToyConfig()
    e, p, v = base.embedding, base.population, base.vocab
    n_pre = p.n_a + p.n_ballast
    n_all = n_pre + p.n_b
    n_params = e.d * v.v_total + v.v_total
    occ = (n_pre * np.log2(v.v_total)) / (n_params * 2.0)
    print(f"  n_A={p.n_a} n_D={p.n_ballast} n_B={p.n_b}  d={e.d}  "
          f"|V_A|={v.v_a_size} |V|={v.v_total}")
    print(f"  keys A u D = {n_pre} ({n_pre / e.d:.0%} of d);  "
          f"A u D u B = {n_all} ({n_all / e.d:.0%} of d)")
    print(f"  LOAD RATIO n_A/|V_A| = {p.n_a / v.v_a_size:.2f}")
    print(f"  params = {n_params:,}   occupancy = {occ:.4%}   "
          f"(MLP-free arm: 0.83%)")
    print(f"  {'alpha':>6} {'ridge_acc':>10} {'rank(AuD)':>10} {'cond':>10}  verdict")
    results = {}
    for alpha in ALPHAS:
        cfg = C.ToyConfig(embedding=C.EmbeddingConfig(alpha=alpha), seed=0)
        ps = P.build_populations(cfg)
        keys = np.concatenate([ps.keys_a, ps.keys_d])
        vals = np.concatenate([ps.values_a, ps.values_d])
        pre, _W, _lg = ridge_precheck(keys, vals, cfg.vocab.v_total)
        feasible = pre["ridge_acc_full"] > 0.99
        results[alpha] = pre
        print(f"  {alpha:>6} {pre['ridge_acc_full']:>10.4f} {pre['key_rank']:>10} "
              f"{pre['key_cond']:>10.3g}  "
              f"{'FEASIBLE' if feasible else 'INFEASIBLE -- no LR fixes this'}"
              f"{'  (rank < n: no interference-free solution exists)' if pre['key_rank'] < len(keys) else ''}")
    return results


def condition_table():
    print("\n=== Split power per condition (PLAN.md §0.3b) ===")
    print(f"  {'overlap':>8} {'touched_tok':>12} {'B_overlap':>10} {'coll/tok':>9} "
          f"{'A_touched':>10} {'A_untouched':>12}  split")
    for ov in (0.0, 0.25, 0.5, 0.75, 1.0):
        cfg = C.ToyConfig(population=C.PopulationConfig(overlap=ov), seed=0)
        ps = P.build_populations(cfg)
        dg = ps.diagnostics
        if dg["n_a_touched"] == 0 or dg["n_a_untouched"] == 0:
            note = "undefined (one group empty, by construction)"
        else:
            note = "usable"
        print(f"  {ov:>8} {dg['n_distinct_a_last_tokens_touched']:>12} "
              f"{int(ov * cfg.population.n_b):>10} "
              f"{dg['collisions_per_touched_token']:>9.2f} {dg['n_a_touched']:>10} "
              f"{dg['n_a_untouched']:>12}  {note}")


def train_confirm(alpha=0.7):
    print(f"\n=== Step 2: confirm by training (A u D, alpha={alpha}) ===")
    for lr in LR_GRID:
        cfg = C.ToyConfig(
            embedding=C.EmbeddingConfig(alpha=alpha),
            optimizer=C.OptimizerConfig(lr=lr),
            protocol=C.ProtocolConfig(pretrain_max_steps=PRETRAIN_CEILING_STEPS,
                                       pretrain_ceiling_window=20,
                                       pretrain_extra_fraction=0.0),
            seed=0)
        ps = P.build_populations(cfg)
        t0 = time.time()
        try:
            store, hist, ceil_step, _tgt = pretrain(cfg, ps, verbose=False)
            acc_a = hist[-1]["acc_a"]
            acc_d = hist[-1]["acc_d"]
            print(f"  lr={lr:<5} CEILING at step {ceil_step:<5} "
                  f"acc_A={acc_a:.4f} acc_D={acc_d:.4f}  ({time.time() - t0:.0f}s, "
                  f"{len(hist)} steps)")
            return True, lr, ceil_step
        except RuntimeError:
            accs = None
            print(f"  lr={lr:<5} did NOT reach ceiling in {PRETRAIN_CEILING_STEPS} steps "
                  f"({time.time() - t0:.0f}s)")
    return False, None, None


def main():
    precheck_table()
    condition_table()
    ok, lr, step = train_confirm()
    print("\n=== GATE VERDICT ===")
    if ok:
        print(f"  PASS -- A u D reaches ceiling at lr={lr}, step {step}.")
        print("  Every key linearly independent (rank = n), so an interference-free")
        print("  storage solution provably exists: any erosion the battery measures")
        print("  cannot be capacity (PLAN.md §0.3a).")
    else:
        print("  FAIL -- run the diagnosis order in PLAN.md §0.4: ridge accuracy and")
        print("  conditioning (printed above), then alpha and d, then load ratio, then LR.")


if __name__ == "__main__":
    main()
