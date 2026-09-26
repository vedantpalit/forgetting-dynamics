"""Input layers."""
from typing import Any
import jax.numpy as jnp
import flax.linen as nn


class EmbeddingInput(nn.Module):
    """Embedding layer for discrete token inputs."""
    vocab_size: int
    model_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.dtype,
        )(x)
