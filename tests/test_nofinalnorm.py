"""Does `final_norm=False` remove exactly the final LayerNorm, and nothing else?

Three things pinned before any cluster time is spent:

  1. The standard arm is untouched: with the default `final_norm=True`, the parameter tree
     is identical to the pre-edit one (the mlpfree golden fixture covers the byte-level
     forward output; here the tree structure and count).
  2. The ablated model differs from the standard one by exactly 2 * model_dim parameters
     (the removed LayerNorm's scale and bias), the backbone has no top-level LayerNorm, and
     every block still has both of its own.
  3. It trains: one gradient step at the small size changes the parameters and the loss is
     finite. The head reads an un-normalized residual, which is the point of the arm, but
     it must not be NaN at init.

Also checks `verify_no_final_norm` REJECTS the standard model (a check that cannot fail
proves nothing) and that the two arms' checkpoint keys differ.

Small (2 layers, d=64) for the structural checks; the parameter difference is also checked
at the real 8L/512 size via `jax.eval_shape`, which needs no forward pass.

Run: uv run python -m tests.test_nofinalnorm
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict

from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.model.factory import create_model
from src.experiments.nofinalnorm_common import (
    MLP_COEFFICIENT, MODEL_DIM, NUM_HEADS, NUM_LAYERS, create_nofinalnorm_model,
    verify_no_final_norm,
)

SMALL = dict(model_dim=64, num_heads=4, num_layers=2, dropout_rate=0.0,
             mlp_coefficient=4, activation="gelu", positional_encoding="rope",
             max_seq_len=32, dtype="float32")
VOCAB, SEQ, N = 37, 16, 8


def _count(p):
    return int(sum(x.size for x in jax.tree.leaves(p)))


def _build(cfg, ablated):
    dcfg = DataConfig(task_type="biography", sequence_length=SEQ, vocab_size=VOCAB)
    mc = ModelConfig(**cfg)
    return create_nofinalnorm_model(mc, dcfg) if ablated else create_model(mc, dcfg)


def test_structure():
    x = jnp.asarray((np.arange(N * SEQ) % VOCAB).reshape(N, SEQ), dtype=jnp.int32)
    std = _build(SMALL, False); abl = _build(SMALL, True)
    p_std = std.init(jax.random.PRNGKey(0), x, deterministic=True)["params"]
    p_abl = abl.init(jax.random.PRNGKey(0), x, deterministic=True)["params"]

    k_std = set(flatten_dict(p_std)); k_abl = set(flatten_dict(p_abl))
    removed = k_std - k_abl
    assert not (k_abl - k_std), f"ablated model has leaves the standard one lacks: {k_abl - k_std}"
    assert removed == {("backbone", "LayerNorm_0", "scale"), ("backbone", "LayerNorm_0", "bias")}, removed
    assert _count(p_std) - _count(p_abl) == 2 * SMALL["model_dim"]
    verify_no_final_norm(p_abl, verbose=False)
    try:
        verify_no_final_norm(p_std, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("verify_no_final_norm accepted a model WITH a final LayerNorm")
    print(f"  ok  removed exactly {sorted(removed)}; count differs by {2 * SMALL['model_dim']}; "
          f"verifier rejects the standard model")
    return abl, p_abl, x


def test_trains(model, params, x):
    targets = jnp.asarray((np.arange(N * SEQ) * 7 % VOCAB).reshape(N, SEQ), dtype=jnp.int32)

    def loss_fn(p):
        logits = model.apply({"params": p}, x, deterministic=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    l0, g = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(float(l0)), f"loss at init is {l0}"
    tx = optax.adam(1e-3); st = tx.init(params)
    upd, _ = tx.update(g, st, params)
    p1 = optax.apply_updates(params, upd)
    l1 = loss_fn(p1)
    moved = sum(float(jnp.abs(a - b).sum()) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(p1)))
    assert moved > 0 and np.isfinite(float(l1))
    print(f"  ok  one step: loss {float(l0):.4f} -> {float(l1):.4f}, params moved")


def test_full_size_count():
    dcfg = DataConfig(task_type="biography", sequence_length=64, vocab_size=1000)
    cfg = ModelConfig(model_dim=MODEL_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                      dropout_rate=0.0, mlp_coefficient=MLP_COEFFICIENT, max_seq_len=64)
    x = jnp.zeros((1, 64), jnp.int32)
    n_std = _count(jax.eval_shape(lambda: create_model(cfg, dcfg).init(
        jax.random.PRNGKey(0), x, deterministic=True))["params"])
    n_abl = _count(jax.eval_shape(lambda: create_nofinalnorm_model(cfg, dcfg).init(
        jax.random.PRNGKey(0), x, deterministic=True))["params"])
    assert n_std - n_abl == 2 * MODEL_DIM, (n_std, n_abl)
    print(f"  ok  8L/512: {n_std:,} -> {n_abl:,} params (difference {n_std - n_abl})")


def test_keys_differ():
    import sys
    from src.config import parse_config
    from src.experiments.knowledge_injection import KIConfig, pretrain_key
    from src.experiments.nofinalnorm_common import nofinalnorm_pretrain_key
    from src.data.config import DataConfig
    saved = sys.argv
    sys.argv = [sys.argv[0], "--model.model_dim", "512", "--model.num_heads", "8",
                "--model.num_layers", "8", "--model.mlp_coefficient", "4",
                "--partition_path", "data/biography/value_partition.npz", "--seed", "42"]
    try:
        cfg = parse_config(KIConfig, description="test")
    finally:
        sys.argv = saved
    dcfg = DataConfig(task_type="biography", sequence_length=64, vocab_size=1000)
    assert pretrain_key(cfg, dcfg) != nofinalnorm_pretrain_key(cfg, dcfg)
    print("  ok  the two arms' pretrain keys differ")


if __name__ == "__main__":
    print("test_nofinalnorm")
    model, params, x = test_structure()
    test_trains(model, params, x)
    test_full_size_count()
    test_keys_differ()
    print("all passed")
