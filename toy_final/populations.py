"""Populations for question 1 (PLAN.md §4): random keys with a shared direction, three
populations, one attribute split into value halves X and Y.

Keys are random PER INDIVIDUAL, not composed from name parts. `toy/` §0.1 showed additive
name-part keys cannot span pair->value maps regardless of d; composition is the overlap knob
for erosion and is deferred (§9.2).

    k_i = sqrt(beta) * mu + sqrt(1 - beta) * g_i / ||g_i||

`mu` is one fixed unit vector shared by EVERY individual in EVERY population. beta = 0 is the
uniform-sphere control, where `toy/` found no collective write is possible; beta = 0.5 is
where Rung 1 produced the rank-sparing crash.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pops:
    keys_a: np.ndarray      # (n_A, d)
    vals_a: np.ndarray      # (n_A,)  indices into V, all in half X
    keys_d: np.ndarray
    vals_d: np.ndarray      # all in half Y
    keys_b: np.ndarray
    vals_b: np.ndarray      # all in half Y, NEW individuals
    x_ids: np.ndarray       # value ids in half X
    y_ids: np.ndarray       # value ids in half Y
    mu: np.ndarray
    beta: float


def _keys(n, d, beta, mu, rng):
    g = rng.normal(size=(n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    if beta == 0.0:
        return g
    return np.sqrt(beta) * mu[None, :] + np.sqrt(1.0 - beta) * g


def build(n_a, n_d, n_b, d, n_vals, beta, seed):
    """Value pool split 50/50 into X and Y by a fixed permutation. A draws from X, D and B
    from Y. Values are assigned round-robin so every value in a half is used equally often,
    which keeps the no-knowledge baseline exactly 1/|half| for every population."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_vals)
    x_ids, y_ids = np.sort(perm[: n_vals // 2]), np.sort(perm[n_vals // 2:])
    mu = rng.normal(size=d)
    mu /= np.linalg.norm(mu)
    ka, kd, kb = (_keys(n, d, beta, mu, rng) for n in (n_a, n_d, n_b))
    va = x_ids[np.arange(n_a) % len(x_ids)]
    vd = y_ids[np.arange(n_d) % len(y_ids)]
    vb = y_ids[rng.permutation(n_b) % len(y_ids)]
    return Pops(ka, va, kd, vd, kb, vb, x_ids, y_ids, mu, beta)
