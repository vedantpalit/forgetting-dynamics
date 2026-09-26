"""Metric sanity harness (PLAN.md §8). Hand-constructed W/b instances where
the right answer is known by direct computation, not by running any training
code. This is a hard gate: run_experiment.py / run_battery.py must refuse to
run until every case here passes.

Fixture, shared by all four cases: d=8, four first-name slots (dims 0-3),
four last-name slots (dims 4-7), key(f,l) = (e_f + e_l)/sqrt(2). |V|=8,
V_A={0,1,2,3}, V_B={4,5,6,7}. Population A = {A0..A3}, individual A_i has
first-name slot i, last-name slot i, correct value i (all in V_A). Because
each A_i's key occupies a disjoint pair of one-hot dims, the four keys are
exactly mutually orthogonal — this is what makes the "perfect store" and the
planted-corruption cases exactly hand-computable rather than approximate.
"""

import numpy as np

import metrics as m

D = 8
V = 8
V_A = np.array([True, True, True, True, False, False, False, False])
V_B = ~V_A

SCALE = 10.0


def _eye(i):
    v = np.zeros(D)
    v[i] = 1.0
    return v


KEYS_A = np.stack([(_eye(i) + _eye(4 + i)) / np.sqrt(2) for i in range(4)])  # (4, 8)
CORRECT_A = np.array([0, 1, 2, 3])


def perfect_store():
    """W0 = SCALE * sum_i outer(key_i, onehot(v_i)), b0 = 0."""
    W = np.zeros((D, V))
    for i in range(4):
        W[:, CORRECT_A[i]] = SCALE * KEYS_A[i]
    b = np.zeros(V)
    return W, b


def check_a():
    """Perfect store: all accuracies 1, mean ranks 1."""
    W, b = perfect_store()
    logits = m.compute_logits(W, b, KEYS_A)

    full_acc = m.argmax_correct(logits, CORRECT_A)
    restr_acc = m.argmax_correct(logits, CORRECT_A, candidate_mask=V_A)
    full_rank = m.rank_of_correct(logits, CORRECT_A)
    restr_rank = m.rank_of_correct(logits, CORRECT_A, candidate_mask=V_A)

    ok = (
        np.all(full_acc == True)
        and np.all(restr_acc == True)
        and np.allclose(full_rank, 1.0)
        and np.allclose(restr_rank, 1.0)
    )

    # Opportunistic smoke test of two more metric functions on this fixture
    # (not separately gated — just checking they return finite, sane values).
    marg = m.margin(logits, CORRECT_A, V_B)
    conf = m.mean_max_softmax_confidence(logits)
    prob_correct = m.softmax_prob_correct(logits, CORRECT_A)
    smoke_ok = (
        np.all(np.isfinite(marg))
        and np.all(marg > 0)  # correct value beats every V_B candidate
        and np.all(np.isfinite(conf))
        and np.all(np.isfinite(prob_correct))
    )

    detail = {
        "full_acc": full_acc.tolist(),
        "restr_acc": restr_acc.tolist(),
        "full_rank": full_rank.tolist(),
        "restr_rank": restr_rank.tolist(),
        "margin": marg.tolist(),
        "smoke_ok": bool(smoke_ok),
    }
    return ok and smoke_ok, detail


def check_b():
    """Perfect store + uniform bias uplift on V_B, large enough to flip
    argmax: full-vocab argmax crashes, restricted argmax/rank unchanged."""
    W, b = perfect_store()
    b = b.copy()
    UPLIFT = 20.0
    b[V_B] = UPLIFT  # > SCALE, so full-vocab argmax moves into V_B for everyone
    logits = m.compute_logits(W, b, KEYS_A)

    full_acc = m.argmax_correct(logits, CORRECT_A)
    restr_acc = m.argmax_correct(logits, CORRECT_A, candidate_mask=V_A)
    restr_rank = m.rank_of_correct(logits, CORRECT_A, candidate_mask=V_A)

    ok = (
        np.all(full_acc == False)  # crashed
        and np.all(restr_acc == True)  # unchanged
        and np.allclose(restr_rank, 1.0)  # unchanged
    )
    detail = {
        "full_acc": full_acc.tolist(),
        "restr_acc": restr_acc.tolist(),
        "restr_rank": restr_rank.tolist(),
    }
    return ok, detail


def check_c():
    """Perfect store with the value association corrupted along one shared
    (last-name) feature direction (dim 4, A0's last-name slot, l0): restricted
    mean rank degrades for A0 only (touched), A1-A3 (untouched) unaffected."""
    W, b = perfect_store()
    W = W.copy()
    CORRUPT = 20.0
    W[4, 0] -= CORRUPT  # corrupt the l0-direction contribution to A0's own value
    logits = m.compute_logits(W, b, KEYS_A)

    restr_acc = m.argmax_correct(logits, CORRECT_A, candidate_mask=V_A)
    restr_rank = m.rank_of_correct(logits, CORRECT_A, candidate_mask=V_A)

    touched = np.array([True, False, False, False])
    split_acc = m.touched_untouched_split(restr_acc.astype(float), touched)
    split_rank = m.touched_untouched_split(restr_rank, touched)

    ok = (
        restr_acc[0] == False
        and np.all(restr_acc[1:] == True)
        and restr_rank[0] > 1.0
        and np.allclose(restr_rank[1:], 1.0)
        and split_acc["touched"]["mean"] == 0.0
        and split_acc["untouched"]["mean"] == 1.0
    )
    detail = {
        "restr_acc": restr_acc.tolist(),
        "restr_rank": restr_rank.tolist(),
        "split_acc": split_acc,
        "split_rank": split_rank,
    }
    return ok, detail


def check_d():
    """Sign checks on the decomposition. Plant, on top of the perfect-store
    baseline: (1) a rank-1 uplift along direction u=e0 that raises every
    V_B column (mean-B-key-direction-style suppression signal), plus a
    matching bias uplift on V_B; (2) the same shared-feature corruption as
    check_c (dim 4, column 0). Decomposition must recover the uplift in the
    rank-1/bias terms (positive sign) and the corruption in the shared-
    subspace term (negative sign), with ~zero orthogonal remainder."""
    u = _eye(0)
    UPLIFT_W = 5.0
    BIAS_UPLIFT = 3.0
    CORRUPT = 20.0

    delta_W = np.zeros((D, V))
    delta_W[0, V_B] = UPLIFT_W  # rank-1 uplift along u, on V_B columns
    delta_W[4, 0] -= CORRUPT  # shared-subspace (l0-direction) corruption

    delta_b = np.zeros(V)
    delta_b[V_B] = BIAS_UPLIFT

    shared_basis_Q = _eye(4)[:, None]  # the l0 direction, as in check_c

    result = m.decompose_update(delta_W, delta_b, u, shared_basis_Q)

    rank1_row = result["rank1_row"]
    shared_rows = result["shared_rows"][0]

    ok = (
        np.allclose(rank1_row[V_A], 0.0)
        and np.allclose(rank1_row[V_B], UPLIFT_W)  # positive: uplift, mean-direction term
        and np.all(delta_b[V_B] > 0)  # positive: uplift, bias term
        and np.isclose(shared_rows[0], -CORRUPT)  # negative: corruption, shared-subspace term
        and np.allclose(np.delete(shared_rows, 0), 0.0)
        and np.isclose(result["orthogonal_remainder_norm"], 0.0, atol=1e-8)
    )
    detail = {
        "rank1_row": rank1_row.tolist(),
        "shared_rows": shared_rows.tolist(),
        "bias_norm": result["bias_norm"],
        "orthogonal_remainder_norm": result["orthogonal_remainder_norm"],
    }
    return ok, detail


def run_all():
    cases = {
        "(a) perfect store": check_a,
        "(b) planted suppression (bias uplift)": check_b,
        "(c) planted erosion (shared-feature corruption)": check_c,
        "(d) sign checks on the decomposition": check_d,
    }
    results = {}
    all_ok = True
    for name, fn in cases.items():
        ok, detail = fn()
        results[name] = {"pass": bool(ok), "detail": detail}
        all_ok = all_ok and ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      detail: {detail}")
    print()
    print("ALL PASS" if all_ok else "AT LEAST ONE FAILURE — do not proceed to experiments")
    return all_ok, results


# ---------------------------------------------------------------------------
# Additions (closed list of three, requested after the four cases above
# passed). These test specificity/robustness, not new planted signals:
#   1. the decomposition's answer doesn't depend on the QR-sign convention;
#   2. the touched/untouched split isn't, by itself, a confound generator;
#   3. a *realistic*-shaped suppression channel (probability-weighted, not
#      uniform) doesn't get silently mistaken for erosion by the metrics.
# ---------------------------------------------------------------------------


def check_addition1_convention_independence():
    """The QR-based decompose_update and the QR-free decompose_update_direct
    must agree on the shared-subspace component, on every fixture above that
    actually plants a shared-subspace effect (cases c and d)."""
    results = {}
    all_ok = True

    # From check_c: corruption only, no bias, u = e0 (unused direction here,
    # decompose still needs *some* u; use e1, orthogonal to the corrupted
    # dim 4, so this exercises a case where u carries no signal at all).
    W_a, b_a = perfect_store()
    W_c, b_c = W_a.copy(), b_a.copy()
    CORRUPT = 20.0
    W_c[4, 0] -= CORRUPT
    delta_W_c = W_c - W_a
    delta_b_c = b_c - b_a
    u_c = _eye(1)
    Q_c = _eye(4)[:, None]

    qr_c = m.decompose_update(delta_W_c, delta_b_c, u_c, Q_c)
    direct_c = m.decompose_update_direct(delta_W_c, delta_b_c, u_c, Q_c)
    agree_c = np.allclose(qr_c["shared_component"], direct_c["shared_component"], atol=1e-8)
    results["case_c_fixture"] = {
        "qr_shared_norm": qr_c["shared_component_norm"],
        "direct_shared_norm": direct_c["shared_component_norm"],
        "agree": bool(agree_c),
    }
    all_ok = all_ok and agree_c

    # From check_d: uplift (rank-1 along u=e0) + corruption (shared dim e4).
    u = _eye(0)
    UPLIFT_W = 5.0
    delta_W_d = np.zeros((D, V))
    delta_W_d[0, V_B] = UPLIFT_W
    delta_W_d[4, 0] -= CORRUPT
    delta_b_d = np.zeros(V)
    delta_b_d[V_B] = 3.0
    Q_d = _eye(4)[:, None]

    qr_d = m.decompose_update(delta_W_d, delta_b_d, u, Q_d)
    direct_d = m.decompose_update_direct(delta_W_d, delta_b_d, u, Q_d)
    agree_d = np.allclose(qr_d["shared_component"], direct_d["shared_component"], atol=1e-8)
    results["case_d_fixture"] = {
        "qr_shared_norm": qr_d["shared_component_norm"],
        "direct_shared_norm": direct_d["shared_component_norm"],
        "agree": bool(agree_d),
    }
    all_ok = all_ok and agree_d

    return all_ok, results


def _fit_linear_store(keys, correct, n_values, steps, lr, seed):
    """Minimal full-batch cross-entropy fit of a frozen-key linear store.
    Standalone here (not train.py, which doesn't exist yet) — used only to
    give Addition 2 a *trained*, not hand-summed, W/b, so its baseline has
    the graded, population-geometry-driven interference a hand-summed
    perfect store cannot produce (every hand-summed column there is written
    by exactly one individual, so argmax/rank metrics saturate at the
    perfect value regardless of key overlap; only gradient accumulation
    across individuals who share key directions produces graded leakage)."""
    rng = np.random.default_rng(seed)
    d = keys.shape[1]
    W = rng.normal(scale=0.01, size=(d, n_values))
    b = np.zeros(n_values)
    n = keys.shape[0]
    onehot_targets = np.zeros((n, n_values))
    onehot_targets[np.arange(n), correct] = 1.0
    for _ in range(steps):
        logits = keys @ W + b
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad_logits = (probs - onehot_targets) / n
        W -= lr * (keys.T @ grad_logits)
        b -= lr * grad_logits.sum(axis=0)
    return W, b


def check_addition2_placebo_split():
    """At overlap=0 (no B at all), assign a *fake* touched label to A using
    the same selection rule the real pipeline would use (a fraction of A's
    own last-name tokens), and check whether that selection alone produces
    a material difference in A's metrics — i.e. whether touched/untouched
    is confounded by A's own internal key-sharing structure, independent of
    any real erosion."""
    rng = np.random.default_rng(0)
    n_A = 40
    n_last = 8
    d = n_A + n_last  # unique first name per individual + a small last-name pool
    n_values = n_A

    first_dims = np.arange(n_A)
    last_idx = rng.integers(0, n_last, size=n_A)  # unbalanced on purpose

    def onehot(i, dim):
        v = np.zeros(dim)
        v[i] = 1.0
        return v

    keys = np.stack([
        (onehot(first_dims[i], d) + onehot(n_A + last_idx[i], d)) / np.sqrt(2)
        for i in range(n_A)
    ])
    correct = np.arange(n_A)

    # A short, partial fit (not trained to convergence) — deliberately, so
    # the store retains genuine graded interference tied to last-name
    # cluster size (a fully-converged store separates every individual
    # perfectly regardless of key overlap, which would make this placebo
    # test vacuous: rank/accuracy pinned at their ceiling values for
    # everyone, no variance left for a spurious split to show up in).
    # Confirmed empirically before adopting these numbers: at steps=5,
    # lr=0.2, mean full-vocab accuracy is ~0.375 (genuinely mid-training)
    # and rank correlates with last-name cluster size (r~0.23), i.e. the
    # confound this check exists to quantify is actually present in the
    # fixture, not assumed.
    W, b = _fit_linear_store(keys, correct, n_values, steps=5, lr=0.2, seed=1)
    logits = m.compute_logits(W, b, keys)
    rank = m.rank_of_correct(logits, correct)
    acc = m.argmax_correct(logits, correct).astype(float)

    n_trials = 200
    overlap_fraction = 0.5
    n_touch_tokens = int(np.ceil(overlap_fraction * n_last))

    rank_diffs, acc_diffs = [], []
    for _ in range(n_trials):
        touched_tokens = rng.choice(n_last, size=n_touch_tokens, replace=False)
        fake_touched = np.isin(last_idx, touched_tokens)
        split_rank = m.touched_untouched_split(rank, fake_touched)
        split_acc = m.touched_untouched_split(acc, fake_touched)
        rank_diffs.append(split_rank["touched"]["mean"] - split_rank["untouched"]["mean"])
        acc_diffs.append(split_acc["touched"]["mean"] - split_acc["untouched"]["mean"])
    rank_diffs = np.array(rank_diffs)
    acc_diffs = np.array(acc_diffs)

    floor = {
        "baseline_rank_mean": float(rank.mean()),
        "baseline_rank_std": float(rank.std()),
        "baseline_acc_mean": float(acc.mean()),
        "rank_diff_mean": float(rank_diffs.mean()),
        "rank_diff_abs_mean": float(np.abs(rank_diffs).mean()),
        "rank_diff_abs_max": float(np.abs(rank_diffs).max()),
        "acc_diff_mean": float(acc_diffs.mean()),
        "acc_diff_abs_mean": float(np.abs(acc_diffs).mean()),
        "n_trials": n_trials,
        "overlap_fraction": overlap_fraction,
    }
    # No comparison against the real overlap-driven split — it doesn't exist
    # yet. Only requirement here: the procedure produces finite numbers.
    ok = np.isfinite(rank_diffs).all() and np.isfinite(acc_diffs).all()
    return ok, floor


def check_addition3_case_e():
    """Superposition of a probability-weighted suppression (the CE
    negative-phase shape: push down each A individual's own columns in
    proportion to that individual's *current* softmax probability on them —
    which, for a confident store, is concentrated on the individual's own
    correct value) with a shared-feature corruption (as in case c), on a
    fixture sized to match the real config's vocab (|V_A|=64)."""
    rng = np.random.default_rng(2)
    n_A = 20
    n_first, n_last = n_A, n_A  # unique first+last per A individual (orthogonal keys)
    d = n_first + n_last
    V_A_size, V_B_size = 64, 64
    V_total = V_A_size + V_B_size
    v_a_mask = np.zeros(V_total, dtype=bool)
    v_a_mask[:V_A_size] = True
    v_b_mask = ~v_a_mask
    SCALE_E = 10.0

    def onehot(i, dim):
        v = np.zeros(dim)
        v[i] = 1.0
        return v

    keys = np.stack([
        (onehot(i, d) + onehot(n_first + i, d)) / np.sqrt(2) for i in range(n_A)
    ])
    correct = np.arange(n_A)  # first n_A slots of V_A, rest of V_A unused

    W0 = np.zeros((d, V_total))
    for i in range(n_A):
        W0[:, correct[i]] = SCALE_E * keys[i]
    b0 = np.zeros(V_total)

    baseline_logits = m.compute_logits(W0, b0, keys)
    p = np.exp(baseline_logits - baseline_logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)  # (n_A, V_total), each row peaked at its own value

    UPLIFT = 20.0  # same order as case (b)/(d)'s crash-inducing uplift

    def build(push_magnitude, with_corruption):
        delta_W = np.zeros((d, V_total))
        for i in range(n_A):
            delta_W += np.outer(keys[i], -push_magnitude * p[i])
        delta_b = np.zeros(V_total)
        delta_b[v_b_mask] = UPLIFT
        if with_corruption:
            # corrupt the last-name direction of A0 alone, as in case (c)
            delta_W[n_first + 0, correct[0]] -= 20.0
        W = W0 + delta_W
        b = b0 + delta_b
        return W, b, delta_W, delta_b

    out = {}
    for push in (1.0, 5.0, 9.0, 9.5, 10.0, 10.5, 11.0, 15.0, UPLIFT):
        W, b, delta_W, delta_b = build(push, with_corruption=False)
        logits = m.compute_logits(W, b, keys)
        restr_rank = m.rank_of_correct(logits, correct, candidate_mask=v_a_mask)
        full_acc = m.argmax_correct(logits, correct).astype(float)
        out[f"push={push}_suppression_only"] = {
            "restricted_rank_mean": float(restr_rank.mean()),
            "restricted_rank_max": float(restr_rank.max()),
            "full_vocab_acc_mean": float(full_acc.mean()),
        }

    # Combined: suppression at the crash-inducing magnitude + corruption on A0.
    W, b, delta_W, delta_b = build(UPLIFT, with_corruption=True)
    logits = m.compute_logits(W, b, keys)
    restr_rank = m.rank_of_correct(logits, correct, candidate_mask=v_a_mask)
    u = keys.mean(axis=0)  # mean-B-key-direction stand-in for this diagnostic
    Q = onehot(n_first + 0, d)[:, None]  # A0's last-name direction
    decomp = m.decompose_update(delta_W, delta_b, u, Q)
    out["combined_suppression_and_corruption"] = {
        "restricted_rank_A0": float(restr_rank[0]),
        "restricted_rank_others_mean": float(restr_rank[1:].mean()),
        "decomposition": {
            "bias_norm": decomp["bias_norm"],
            "rank1_component_norm": decomp["rank1_component_norm"],
            "shared_component_norm": decomp["shared_component_norm"],
            "orthogonal_remainder_norm": decomp["orthogonal_remainder_norm"],
        },
    }

    leakage_at_crash_magnitude = out[f"push={UPLIFT}_suppression_only"]["restricted_rank_mean"]
    ok = np.isfinite(leakage_at_crash_magnitude)
    out["leakage_number_for_CHECK1"] = leakage_at_crash_magnitude
    out["V_A_size"] = V_A_size
    return ok, out


def run_additions():
    print("=== Addition 1: convention independence ===")
    ok1, r1 = check_addition1_convention_independence()
    print(f"{'PASS' if ok1 else 'FAIL'}  QR-based and direct-projection decompositions agree")
    print(f"      {r1}")
    print()

    print("=== Addition 2: placebo split (overlap=0, fake touched labels) ===")
    ok2, r2 = check_addition2_placebo_split()
    print(f"{'PASS (ran cleanly)' if ok2 else 'FAIL'}  placebo floor computed")
    print(f"      {r2}")
    print()

    print("=== Addition 3: case (e), realistic (probability-weighted) suppression ===")
    ok3, r3 = check_addition3_case_e()
    print(f"{'PASS (ran cleanly)' if ok3 else 'FAIL'}  case (e) computed")
    print(f"      {r3}")
    print()

    return ok1 and ok2 and ok3, {"addition1": r1, "addition2": r2, "addition3": r3}


if __name__ == "__main__":
    ok, _ = run_all()
    print()
    ok_add, _ = run_additions()
    raise SystemExit(0 if (ok and ok_add) else 1)
