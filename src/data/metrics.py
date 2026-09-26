"""Evaluation metrics."""
import jax
import jax.numpy as jnp


@jax.jit
def compute_cross_entropy(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """Masked cross-entropy loss."""
    # logits: (B, T, V), targets: (B, T)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    per_token = -jnp.sum(log_probs * one_hot, axis=-1)  # (B, T)
    return jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)
