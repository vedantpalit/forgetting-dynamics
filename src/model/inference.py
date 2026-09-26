"""Autoregressive token generation with KV cache (prefill + decode)."""
from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(3, 4, 5))
def _generate_jit(params, prompt, rng, max_new_tokens, apply_fn, temperature):
    """Prefill + lax.scan decode loop, compiled as a single program."""
    _, L = prompt.shape
    total_len = L + max_new_tokens

    def sample(logits, rng):
        if temperature == 0.0:
            return jnp.argmax(logits, axis=-1), rng
        rng, key = jax.random.split(rng)
        return jax.random.categorical(key, logits / temperature, axis=-1), rng

    logits, cache = apply_fn(
        {"params": params}, prompt,
        deterministic=True, init_cache=True, max_seq_len=total_len,
    )
    token, rng = sample(logits[:, -1, :], rng)

    def body(carry, step):
        cache, token, rng = carry
        logits, cache = apply_fn(
            {"params": params}, token[:, None],
            deterministic=True, cache=cache, decode_step=step,
        )
        next_token, rng = sample(logits[:, -1, :], rng)
        return (cache, next_token, rng), token

    (_, last_token, _), tokens = jax.lax.scan(
        body, (cache, token, rng), jnp.arange(L, total_len - 1)
    )
    return jnp.concatenate([prompt, tokens.T, last_token[:, None]], axis=1)


def generate(state, prompt, max_new_tokens, temperature=1.0, rng_key=None):
    """Autoregressive generation with KV cache.

    Args:
        state: TrainState with params and apply_fn
        prompt: (B, L) token prompt
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        rng_key: JAX PRNGKey for sampling

    Returns:
        Full sequence including prompt: (B, L + max_new_tokens)
    """
    prompt = jnp.array(prompt)
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    if max_new_tokens == 0:
        return prompt
    return _generate_jit(
        state.params, prompt, rng_key, max_new_tokens, state.apply_fn, temperature,
    )
