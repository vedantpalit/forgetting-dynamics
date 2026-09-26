"""Hutchinson estimator of the Hessian trace (loss-landscape sharpness).

tr(H) = E_v[ v^T H v ] with v Rademacher (+/-1). The Hessian-vector product
H v is computed matrix-free as jvp(grad(loss)), so no Hessian is materialized.
Used during finetuning eval to probe how sharp the minimum the student sits in
is, both for the retained task (CE on A) and for the arm's own finetuning
objective (CE / KL on B).

Note: jax-metal miscompiles the KL path the same way it does distill_step; run
the KL trace on CPU/CUDA only (the NTP CE path is fine on Metal).
"""
from functools import partial

import jax
import jax.numpy as jnp

from src.data.metrics import compute_cross_entropy
from src.divergence import divergence_loss, divergence_terms


def _tree_vdot(a, b) -> jax.Array:
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def _hutchinson(loss_fn, params, rng, num_probes: int) -> jax.Array:
    """Mean of v^T (H v) over `num_probes` Rademacher probes of `loss_fn` at `params`."""
    grad_fn = jax.grad(loss_fn)

    def one_probe(key):
        v = jax.tree.map(
            lambda p, k: jax.random.rademacher(k, p.shape, dtype=p.dtype),
            params, _tree_split_like(params, key),
        )
        hv = jax.jvp(grad_fn, (params,), (v,))[1]
        return _tree_vdot(v, hv)

    # Sequential probes (lax.map, batch 1): a vmap over probes multiplies the
    # full jvp-of-grad activation memory by num_probes and OOMs on 80GB H100s.
    return jnp.mean(jax.lax.map(one_probe, jax.random.split(rng, num_probes)))


def _tree_split_like(params, key):
    """A distinct PRNG key per leaf, laid out with the same treedef as `params`."""
    leaves, treedef = jax.tree.flatten(params)
    return jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))


@partial(jax.jit, static_argnums=(0, 6))
def ce_hessian_trace(apply_fn, params, inputs, targets, mask, rng, num_probes: int) -> jax.Array:
    """tr(H) of masked cross-entropy at `params` over a fixed batch."""
    def loss(p):
        logits = apply_fn({"params": p}, inputs, deterministic=True)
        return compute_cross_entropy(logits, targets, mask)
    return _hutchinson(loss, params, rng, num_probes)


def kl_logit_hessian_trace(apply_fn, params, teacher_params, sequences, spec,
                           position_mask=None) -> jax.Array:
    """Exact tr of the Hessian of the divergence w.r.t. the STUDENT LOGITS.

    spec is a parsed divergence tuple (src.divergence.parse_divergence).
    Plain KLs (tau = 1) use a closed form per position (p = student probs,
    q = teacher probs):
      forward  KL(q||p): H = diag(p) - p p^T          -> tr = 1 - sum p^2
      reverse  KL(p||q): with u = log p - log q, g = sum p*u:
                         tr = sum p(1-p) + sum (u - g) * p(1-2p)
    Other families (alpha, js, temperature-scaled KL) use an exact autodiff
    Hessian per position, chunked over rows (heavier, still just forward
    passes + a V x V Hessian per position — no parameter-space work).
    Averaged over `position_mask` positions (all if None), matching the
    divergence loss normalization.
    """
    if spec[0] == "kl" and spec[2] == 1.0:
        return _kl_logit_hessian_closed(apply_fn, params, teacher_params, sequences,
                                        spec[1], position_mask)
    return _generic_logit_hessian_trace(apply_fn, params, teacher_params, sequences,
                                        spec, position_mask)


@partial(jax.jit, static_argnums=(0, 4))
def _kl_logit_hessian_closed(apply_fn, params, teacher_params, sequences, kl_direction: str,
                             position_mask=None) -> jax.Array:
    inputs = sequences[:, :-1]
    student_logp = jax.nn.log_softmax(apply_fn({"params": params}, inputs, deterministic=True), axis=-1)
    p = jnp.exp(student_logp)
    if kl_direction == "forward":
        tr = 1.0 - jnp.sum(p * p, axis=-1)
    else:
        teacher_logp = jax.nn.log_softmax(
            apply_fn({"params": teacher_params}, inputs, deterministic=True), axis=-1)
        u = student_logp - teacher_logp
        g = jnp.sum(p * u, axis=-1, keepdims=True)
        tr = jnp.sum(p * (1 - p), axis=-1) + jnp.sum((u - g) * p * (1 - 2 * p), axis=-1)
    if position_mask is None:
        return jnp.mean(tr)
    return jnp.sum(tr * position_mask) / jnp.maximum(jnp.sum(position_mask), 1)


@partial(jax.jit, static_argnums=(0, 4, 6))
def _generic_logit_hessian_trace(apply_fn, params, teacher_params, sequences, spec: tuple,
                                 position_mask=None, chunk: int = 8) -> jax.Array:
    """Exact per-position logit-Hessian trace for any divergence spec via autodiff."""
    inputs = sequences[:, :-1]
    student_logits = apply_fn({"params": params}, inputs, deterministic=True)
    teacher_logp = jax.nn.log_softmax(
        apply_fn({"params": teacher_params}, inputs, deterministic=True), axis=-1)
    vocab = student_logits.shape[-1]
    z = student_logits.reshape(-1, vocab)
    t = teacher_logp.reshape(-1, vocab)
    w = (jnp.ones(z.shape[0]) if position_mask is None
         else position_mask.reshape(-1).astype(z.dtype))

    def row_trace(z_row, t_row):
        f = lambda zz: divergence_terms(jax.nn.log_softmax(zz), t_row, spec)
        return jnp.trace(jax.hessian(f)(z_row))

    pad = (-z.shape[0]) % chunk
    z, t = jnp.pad(z, ((0, pad), (0, 0))), jnp.pad(t, ((0, pad), (0, 0)))
    wp = jnp.pad(w, (0, pad))
    tr = jax.lax.map(lambda args: jax.vmap(row_trace)(*args),
                     (z.reshape(-1, chunk, vocab), t.reshape(-1, chunk, vocab))).reshape(-1)
    return jnp.sum(tr * wp) / jnp.maximum(jnp.sum(wp), 1)


@partial(jax.jit, static_argnums=(0, 6, 7))
def kl_hessian_trace(apply_fn, params, teacher_params, sequences, position_mask,
                     rng, spec: tuple, num_probes: int) -> jax.Array:
    """tr(H) of the token-level distillation divergence at `params` over fixed sequences.

    Mirrors distill._distill_step_jit for the given spec, averaged over
    `position_mask` positions.
    """
    inputs = sequences[:, :-1]
    teacher_logp = jax.nn.log_softmax(apply_fn({"params": teacher_params}, inputs, deterministic=True), axis=-1)

    def loss(p):
        student_logp = jax.nn.log_softmax(apply_fn({"params": p}, inputs, deterministic=True), axis=-1)
        return divergence_loss(student_logp, teacher_logp, spec, position_mask)

    return _hutchinson(loss, params, rng, num_probes)
