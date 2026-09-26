"""Geometry selection for the battery (PLAN.md §0.3).

Runs the closed-form ridge pre-check on the REAL pretraining population, A union D, with
reserved disjoint name sub-pools, for each candidate geometry and each alpha. Also reports
the rank of the full A union D union B key set, since after injection the store must hold
all three, and the occupancy of the resulting configuration.

Two constraints the chosen geometry has to satisfy:
  * total keys comfortably below d, so every key is linearly independent with margin and an
    interference-free storage solution provably exists (PLAN.md §0.3a);
  * fast enough for 20 seeds across the full condition grid -- per-step cost is O(n d |V|).

Run: python -m select_geometry   (from toy/)
"""
import numpy as np

from feasibility import ridge_precheck
from populations import build_name_features, build_private_features, compose_keys

BITS_PER_PARAM = 2.0  # Allen-Zhu convention, as used by FINDINGS.md §4.5 and scale8_common


def _round_robin(n, pool):
    """Exact multiplicity n/pool on both marginals, all pairs unique. Needs pool >= n/pool."""
    idx = np.arange(n)
    first = idx % pool
    last = (first + idx // pool) % pool
    return first, last


def build_union(n_a, n_d, n_b, mult_a, mult_db, d, alpha, v_a, v_b, n_first, n_last, seed):
    rng = np.random.default_rng(seed)
    first_feats = build_name_features(n_first, d, 0.0, 8, rng)
    last_feats = build_name_features(n_last, d, 0.0, 8, rng)

    pa, pd, pb = n_a // mult_a, n_d // mult_db, n_b // mult_db
    for nm, pool, mult in (("A", pa, mult_a), ("D", pd, mult_db), ("B", pb, mult_db)):
        if pool < mult:
            raise ValueError(f"{nm}: pool {pool} < multiplicity {mult}; round-robin would "
                              f"produce duplicate pairs")
    if pa + pd + pb > min(n_first, n_last):
        raise ValueError(f"pools {pa}+{pd}+{pb} exceed |F|={n_first} / |L|={n_last}")

    perm_f, perm_l = rng.permutation(n_first), rng.permutation(n_last)
    slices = {"A": (0, pa), "D": (pa, pa + pd), "B": (pa + pd, pa + pd + pb)}

    out = {}
    for nm, n, mult in (("A", n_a, mult_a), ("D", n_d, mult_db), ("B", n_b, mult_db)):
        lo, hi = slices[nm]
        fl, ll = _round_robin(n, hi - lo)
        f_idx, l_idx = perm_f[lo:hi][fl], perm_l[lo:hi][ll]
        priv = build_private_features(n, d, rng)
        out[nm] = compose_keys(first_feats, last_feats, priv, f_idx, l_idx, alpha)

    values_a = rng.integers(0, v_a, size=n_a)
    values_d = rng.integers(0, v_b, size=n_d) + v_a
    return out, values_a, values_d


def occupancy(n_stored, n_params, v_total, n_attrs=1):
    bits = n_stored * n_attrs * np.log2(v_total)
    return bits / (n_params * BITS_PER_PARAM), bits


GEOMETRIES = [
    dict(label="current (as pre-checked on A alone)",
         n_a=2000, n_d=500, n_b=500, mult_a=20, mult_db=20, d=2048,
         v_a=256, v_b=256, n_first=200, n_last=200),
    dict(label="proposed (smaller populations, same d)",
         n_a=1000, n_d=250, n_b=250, mult_a=20, mult_db=10, d=2048,
         v_a=256, v_b=256, n_first=128, n_last=128),
]


def main():
    for geo in GEOMETRIES:
        n_a, n_d, n_b, d = geo["n_a"], geo["n_d"], geo["n_b"], geo["d"]
        v_a, v_b = geo["v_a"], geo["v_b"]
        v_total = v_a + v_b
        n_params = d * v_total + v_total
        pre_keys, all_keys = n_a + n_d, n_a + n_d + n_b
        occ, bits = occupancy(pre_keys, n_params, v_total)

        print(f"########## {geo['label']} ##########")
        print(f"  n_A={n_a} n_D={n_d} n_B={n_b}  d={d}  |V_A|={v_a} |V|={v_total}")
        print(f"  keys pretrained (A u D) = {pre_keys}  ({pre_keys / d:.0%} of d)   "
              f"full run (A u D u B) = {all_keys}  ({all_keys / d:.0%} of d)")
        print(f"  load ratio n_A/|V_A| = {n_a / v_a:.2f}")
        print(f"  params = {n_params:,}   occupancy (N=A+D, 1 attr, {np.log2(v_total):.0f} "
              f"bits/fact, 2 bits/param) = {occ:.4%}")
        print(f"  {'alpha':>6} {'ridge_acc(AuD)':>15} {'rank(AuD)':>10} {'rank(AuDuB)':>12} "
              f"{'cond':>10}")
        for alpha in (1.0, 0.85, 0.7, 0.5, 0.3):
            try:
                pops, va, vd = build_union(alpha=alpha, seed=0, **{
                    k: geo[k] for k in ("n_a", "n_d", "n_b", "mult_a", "mult_db", "d",
                                         "v_a", "v_b", "n_first", "n_last")})
            except ValueError as e:
                print(f"  {alpha:>6} INVALID: {e}")
                continue
            keys_pre = np.concatenate([pops["A"], pops["D"]])
            vals_pre = np.concatenate([va, vd])
            pre, _W, _lg = ridge_precheck(keys_pre, vals_pre, v_total)

            keys_all = np.concatenate([pops["A"], pops["D"], pops["B"]])
            sv = np.linalg.svd(keys_all, compute_uv=False)
            tol = sv.max() * max(keys_all.shape) * np.finfo(float).eps
            rank_all = int((sv > tol).sum())

            print(f"  {alpha:>6} {pre['ridge_acc_full']:>15.4f} {pre['key_rank']:>10} "
                  f"{rank_all:>12} {pre['key_cond']:>10.3g}")
        print()


if __name__ == "__main__":
    main()
