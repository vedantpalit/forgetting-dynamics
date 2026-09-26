"""Output heads."""
from typing import Any
import jax.numpy as jnp
import flax.linen as nn


class ClassificationHead(nn.Module):
    """Output head for classification (logits over vocab)."""
    vocab_size: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.vocab_size, dtype=self.dtype)(x)
