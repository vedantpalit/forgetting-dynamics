"""Base dataset class."""
from abc import ABC, abstractmethod
import jax


class BaseDataset(ABC):
    """Abstract base class for sequence datasets."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return a batch with keys: inputs, targets, mask."""

    @staticmethod
    @abstractmethod
    def loss_fn(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
        """Compute loss. Used as static arg in train_step JIT."""
