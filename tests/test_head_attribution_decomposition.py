"""The per-head residual decomposition must reproduce the model's own logits.

`analyze_head_attribution` attributes the constant shift `c` head by head by splitting the
residual stream into per-head contributions and pushing each through the final LayerNorm and
readout. That split is exact in principle -- the MLP-free block is `x = x + Attn(LN(x))` and
LayerNorm is affine once its scale is fixed -- so summing the components must give back the
logits. If it does not, every per-head number is meaningless.

The script asserts this at runtime on real checkpoints. This test pins it on a small randomly
initialised model, so a break is caught without a cluster job and without checkpoints.

WHY THIS TEST EXISTS, and two wrong diagnoses worth not repeating. The first version
reconstructed real checkpoints only to 1.29e-02, identically across three seeds.

  1. Blamed flax's fast variance. Measured: it agrees with numpy's two-pass form to 2.4e-07
     and both reconstruct to 9.5e-07. Not the cause.
  2. Blamed float32 cancellation across 64 heads. Moving to float64 changed the error by
     nothing, and the measured cancellation ratio was 0.5x -- there was none to lose.

The actual cause was that the MODEL was the imprecise side. JAX defaults float32 matmuls to
TF32 on GPU (10 mantissa bits, ~3e-4 relative), so the cluster's logits carried ~0.03 of
rounding at max|logit| 27.8, while this test on CPU -- where float32 is real float32 -- passed
all along. That is why the failure was invisible locally and identical across seeds. The script
now sets jax_default_matmul_precision="highest".

This test cannot catch that class of bug on a CPU runner, which is exactly why the script keeps
its own runtime assert on the real device.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.experiments.mlpfree_common import create_mlpfree_model
from src.experiments.analyze_head_attribution import (
    component_contributions, centred, logit_map,
)

COLS = np.array([[3, 5, 7, 9, 11, 13]])   # stand-in value slots; any six positions will do


def test_components_reproduce_logits():
    model_cfg = ModelConfig(model_dim=32, num_heads=4, num_layers=3, mlp_coefficient=0,
                            dropout_rate=0.0)
    data_cfg = DataConfig(vocab_size=41, sequence_length=17)
    model = create_mlpfree_model(model_cfg, data_cfg)

    rng = jax.random.PRNGKey(0)
    x = jax.random.randint(rng, (4, data_cfg.sequence_length), 0, data_cfg.vocab_size)
    params = model.init(rng, x, deterministic=True)["params"]

    cols = np.repeat(COLS, x.shape[0], axis=0)
    heads, other, mul = component_contributions(
        model, params, x, cols, model_cfg.num_layers, model_cfg.num_heads, model_cfg.model_dim)
    wg, offset = logit_map(params, data_cfg.vocab_size)

    recon = centred(heads, mul[..., None]).sum(axis=2) + centred(other, mul)
    recon = recon @ wg + offset
    real = np.asarray(model.apply({"params": params}, x, deterministic=True))
    bi = np.arange(x.shape[0])[:, None]

    err = float(np.abs(recon - real[bi, cols]).max())
    assert err < 1e-3, f"decomposition does not reproduce the logits: max err {err:.3e}"


def test_head_split_covers_every_head():
    """Every head must appear exactly once: an off-by-one in the out-kernel slicing would
    still reconstruct if two heads' slices overlapped and a third were dropped, provided the
    dropped one happened to be small. Checking the shape is not enough."""
    model_cfg = ModelConfig(model_dim=32, num_heads=4, num_layers=3, mlp_coefficient=0,
                            dropout_rate=0.0)
    data_cfg = DataConfig(vocab_size=41, sequence_length=17)
    model = create_mlpfree_model(model_cfg, data_cfg)
    rng = jax.random.PRNGKey(1)
    x = jax.random.randint(rng, (2, data_cfg.sequence_length), 0, data_cfg.vocab_size)
    params = model.init(rng, x, deterministic=True)["params"]

    heads, _, _ = component_contributions(
        model, params, x, np.repeat(COLS, x.shape[0], axis=0),
        model_cfg.num_layers, model_cfg.num_heads, model_cfg.model_dim)
    assert heads.shape[2] == model_cfg.num_layers * model_cfg.num_heads
    # No head is identically zero, which is what a mis-sliced kernel row range would produce.
    per_head = np.abs(heads).sum(axis=(0, 1, 3))
    assert (per_head > 0).all(), f"some head contributes nothing: {per_head}"
