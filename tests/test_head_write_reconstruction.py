"""Does the per-head write reconstruction rebuild the block's own attention write?

`analyze_head_write_variance` reimplements LayerNorm and the head slicing of W_out in numpy in
order to split a block's attention write into 64 per-head pieces. That is exactly the kind of
reimplementation the pattern-patch work avoided on purpose, because a wrong epsilon, a wrong
reduction axis, or a transposed head slice all produce plausible numbers rather than an error.

With the MLP disabled the block is exactly `x_out = x_in + Attn(LN(x_in))`, so the block's total
attention write is `x_out - x_in` -- both sown, nothing reimplemented -- and the per-head pieces
must sum to it. This test pins that, and pins that the check can fail: perturbing the LayerNorm
epsilon by a plausible wrong value (1e-5, another common default) must break it.

Small on purpose (2 layers, d=64), same construction path as the real model.

Run: uv run python -m tests.test_head_write_reconstruction
"""
import jax
import jax.numpy as jnp
import numpy as np

from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.experiments.mlpfree_common import create_mlpfree_model
import src.experiments.analyze_head_write_variance as hw

SMALL = dict(model_dim=64, num_heads=4, num_layers=2, dropout_rate=0.0,
             mlp_coefficient=0, activation="gelu", positional_encoding="rope",
             max_seq_len=32, dtype="float32")
VOCAB, SEQ, N = 37, 16, 8


def _run():
    model = create_mlpfree_model(ModelConfig(**SMALL),
                                 DataConfig(task_type="biography", sequence_length=SEQ,
                                            vocab_size=VOCAB))
    x = jnp.asarray((np.arange(N * SEQ) % VOCAB).reshape(N, SEQ), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)["params"]
    _, aux = model.apply({"params": params}, x, deterministic=True,
                         capture_intermediates=True, mutable=["intermediates"])
    return params, aux


def test_reconstruction(params, aux):
    for j in range(SMALL["num_layers"]):
        per_head, total, bo = hw.head_writes(params, aux, j, SMALL["num_heads"])
        err = np.abs(per_head.sum(2) + bo - total).max()
        scale = np.abs(total).max()
        assert err < 1e-4 * max(scale, 1.0), (
            f"block {j}: per-head sum rebuilds the block write to only {err:.3e} "
            f"(write magnitude {scale:.3f})")
        print(f"  ok  block {j}: per-head sum == block write, max error {err:.2e} "
              f"(write magnitude {scale:.3f})")


def test_wrong_epsilon_is_caught(params, aux):
    """A plausible wrong LayerNorm epsilon must break the reconstruction, not pass quietly."""
    real = hw.LN_EPS
    hw.LN_EPS = 1e-5
    try:
        per_head, total, bo = hw.head_writes(params, aux, 0, SMALL["num_heads"])
        err = np.abs(per_head.sum(2) + bo - total).max()
        assert err > 1e-4, (
            f"epsilon 1e-5 instead of 1e-6 left the reconstruction error at {err:.3e}; "
            f"the check cannot distinguish a wrong LayerNorm")
        print(f"  ok  a wrong LayerNorm epsilon breaks it (error {err:.2e}), so the check bites")
    finally:
        hw.LN_EPS = real


def test_head_slice_orientation(params, aux):
    """Transposing the head slice of W_out must break it -- pins the reshape convention."""
    j, H = 0, SMALL["num_heads"]
    p = params["backbone"][f"block_{j}"]["CausalSelfAttention_0"]
    Wo = np.asarray(p["out"]["kernel"]).copy()
    E = Wo.shape[-1]
    D = E // H
    # Reverse the head order: a wrong-but-plausible slicing of the same weights.
    p["out"]["kernel"] = jnp.asarray(Wo.reshape(H, D, E)[::-1].reshape(E, E))
    try:
        per_head, total, bo = hw.head_writes(params, aux, j, H)
        err = np.abs(per_head.sum(2) + bo - total).max()
        assert err > 1e-4, (
            f"reversing the head order left the error at {err:.3e}; the reconstruction is "
            f"insensitive to which slice belongs to which head")
        print(f"  ok  a wrong head slice breaks it (error {err:.2e})")
    finally:
        p["out"]["kernel"] = jnp.asarray(Wo)


if __name__ == "__main__":
    params, aux = _run()
    test_reconstruction(params, aux)
    test_wrong_epsilon_is_caught(params, aux)
    test_head_slice_orientation(params, aux)
    print("\nall head-write reconstruction tests passed")
