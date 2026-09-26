"""Regression tests for the MLP-free ablation's two edits to `src/model/backbone.py`.

Both edits are claimed to be no-ops for the standard arm:

  1. `mlp_enabled: bool = True` on `TransformerBlock` / `TransformerBackbone`. A flax module
     attribute creates no parameters, and the default keeps the MLP branch, so the standard
     arm's parameter tree and forward output should be *bit-identical* to before the edit.
  2. One `self.sow("intermediates", "attn_weights", attn)` in `CausalSelfAttention`. `sow`
     into a non-mutable collection is a no-op that adds no parameters and does not touch the
     returned value.

"Should be" is not evidence, so this test pins it. The golden fixture in
`tests/fixtures/backbone_noop_golden.json` was generated from the **pre-edit** backbone
(`python -m tests.test_mlpfree_noop --regenerate`, run before the edit landed) and is
compared byte-for-byte afterwards. Regenerating it after an edit would defeat the entire
point, so `--regenerate` refuses to overwrite an existing fixture.

The no-op tests deliberately run at a *small* configuration (2 layers, d=64): they exercise
exactly the same construction path and the same `sow`, in seconds rather than the ~11 minutes
an 8L/512 forward pass costs to compile on CPU. Parameter counts at the real 8L/512 scale are
checked separately via `jax.eval_shape`, which needs no forward pass at all.

Run: uv run python -m tests.test_mlpfree_noop
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.model.factory import create_model

FIXTURE = Path(__file__).parent / "fixtures" / "backbone_noop_golden.json"

# Small on purpose -- see module docstring.
SMALL_MODEL = dict(model_dim=64, num_heads=4, num_layers=2, dropout_rate=0.0,
                    mlp_coefficient=4, activation="gelu", positional_encoding="rope",
                    max_seq_len=32, dtype="float32")
SMALL_VOCAB = 37
SMALL_SEQ = 16
INIT_SEED = 0


def _small_model():
    model_cfg = ModelConfig(**SMALL_MODEL)
    data_cfg = DataConfig(task_type="biography", sequence_length=SMALL_SEQ,
                           vocab_size=SMALL_VOCAB)
    return create_model(model_cfg, data_cfg)


def _fixed_input():
    # Deterministic, no RNG: token ids must be < vocab_size or nn.Embed's jnp.take
    # fills with NaN (see tests/test_ckpt_key.py's note on mode="fill").
    return jnp.asarray((np.arange(2 * SMALL_SEQ) % SMALL_VOCAB).reshape(2, SMALL_SEQ),
                        dtype=jnp.int32)


def _structure_digest(params):
    """SHA1 over sorted 'path:shape:dtype' triples -- catches any added, removed, resized,
    or retyped parameter, without depending on value bits."""
    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        entries.append(f"{name}:{jnp.shape(leaf)}:{jnp.asarray(leaf).dtype}")
    entries.sort()
    return hashlib.sha1("\n".join(entries).encode()).hexdigest()


def _forward_digest(logits):
    """SHA1 over the raw output bytes -- bit-identity, not approximate equality."""
    return hashlib.sha1(np.asarray(logits).tobytes()).hexdigest()


def _standard_arm_digests():
    model = _small_model()
    x = _fixed_input()
    params = model.init(jax.random.PRNGKey(INIT_SEED), x, deterministic=True)
    logits = model.apply(params, x, deterministic=True)
    n_params = int(sum(np.size(leaf) for leaf in jax.tree.leaves(params)))
    return {
        "structure_sha1": _structure_digest(params),
        "forward_sha1": _forward_digest(logits),
        "n_params": n_params,
        "config": {"model": SMALL_MODEL, "vocab_size": SMALL_VOCAB,
                    "seq_len": SMALL_SEQ, "init_seed": INIT_SEED},
    }


def regenerate():
    if FIXTURE.exists():
        raise SystemExit(
            f"{FIXTURE} already exists. Regenerating it after an edit would defeat the "
            f"purpose of a before/after golden test -- delete it deliberately if you really "
            f"intend to re-baseline, and say so in the commit message.")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    golden = _standard_arm_digests()
    FIXTURE.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"Wrote golden fixture -> {FIXTURE}")
    for k, v in golden.items():
        if k != "config":
            print(f"  {k}: {v}")


# --- tests ---

def test_1_standard_arm_structure_unchanged(golden):
    got = _standard_arm_digests()
    assert got["structure_sha1"] == golden["structure_sha1"], (
        f"standard arm's parameter tree changed: {got['structure_sha1']} != "
        f"{golden['structure_sha1']}")
    assert got["n_params"] == golden["n_params"], (
        f"standard arm's parameter count changed: {got['n_params']} != {golden['n_params']}")
    print(f"  [1] standard arm parameter tree bit-identical "
          f"({got['n_params']:,} params, sha1 {got['structure_sha1'][:12]})")


def test_2_standard_arm_forward_unchanged(golden):
    got = _standard_arm_digests()
    assert got["forward_sha1"] == golden["forward_sha1"], (
        f"standard arm's forward output changed: {got['forward_sha1']} != "
        f"{golden['forward_sha1']}. The edits are NOT no-ops.")
    print(f"  [2] standard arm forward output bit-identical "
          f"(sha1 {got['forward_sha1'][:12]})")


def test_3_mlpfree_has_no_mlp_tensors():
    from src.experiments.mlpfree_common import create_mlpfree_model, verify_no_mlp

    model_cfg = ModelConfig(**{**SMALL_MODEL, "mlp_coefficient": 0})
    data_cfg = DataConfig(task_type="biography", sequence_length=SMALL_SEQ,
                           vocab_size=SMALL_VOCAB)
    model = create_mlpfree_model(model_cfg, data_cfg)
    x = _fixed_input()
    params = model.init(jax.random.PRNGKey(INIT_SEED), x, deterministic=True)
    n_mlp = verify_no_mlp(params["params"] if "params" in params else params, verbose=False)
    assert n_mlp == 0
    logits = model.apply(params, x, deterministic=True)
    assert logits.shape == (2, SMALL_SEQ, SMALL_VOCAB)
    assert np.all(np.isfinite(np.asarray(logits)))
    print(f"  [3] MLP-free model has zero MLP tensors and produces finite logits")


def test_4_full_scale_param_counts():
    """8L/512 counts for both arms, via jax.eval_shape -- no forward pass, no JIT, instant."""
    from src.experiments.mlpfree_common import create_mlpfree_model

    vocab, seq = 1881, 128  # vocab from the real preprocessed corpus; seq irrelevant to counts
    std_cfg = ModelConfig(model_dim=512, num_heads=8, num_layers=8, dropout_rate=0.0,
                           mlp_coefficient=4, max_seq_len=512)
    free_cfg = ModelConfig(model_dim=512, num_heads=8, num_layers=8, dropout_rate=0.0,
                            mlp_coefficient=0, max_seq_len=512)
    data_cfg = DataConfig(task_type="biography", sequence_length=seq, vocab_size=vocab)
    x = jnp.zeros((1, seq), dtype=jnp.int32)

    def count(model):
        shapes = jax.eval_shape(
            lambda: model.init(jax.random.PRNGKey(0), x, deterministic=True))
        return int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(shapes)))

    n_std = count(create_model(std_cfg, data_cfg))
    n_free = count(create_mlpfree_model(free_cfg, data_cfg))
    removed = n_std - n_free
    print(f"  [4] 8L/512 params: standard={n_std:,}  mlp_free={n_free:,}  "
          f"removed={removed:,} ({removed / n_std:.1%})  "
          f"mlp_free is {n_free / n_std:.1%} of standard")
    assert n_free < n_std
    # Per block we remove: LayerNorm (2*512) + Dense(512->2048) + Dense(2048->512).
    expected_removed = 8 * (2 * 512 + (512 * 2048 + 2048) + (2048 * 512 + 512))
    assert removed == expected_removed, (
        f"removed {removed:,} params, expected {expected_removed:,} -- the MLP-free block "
        f"is not removing exactly the pre-MLP LayerNorm and the two Dense layers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                     help="write the golden fixture (only valid BEFORE the backbone edits)")
    args = ap.parse_args()
    if args.regenerate:
        regenerate()
        return

    if not FIXTURE.exists():
        raise SystemExit(f"No golden fixture at {FIXTURE}. Generate it from the PRE-EDIT "
                          f"backbone with: python -m tests.test_mlpfree_noop --regenerate")
    golden = json.loads(FIXTURE.read_text())

    print("MLP-free ablation: backbone edit no-op tests")
    test_1_standard_arm_structure_unchanged(golden)
    test_2_standard_arm_forward_unchanged(golden)
    test_3_mlpfree_has_no_mlp_tensors()
    test_4_full_scale_param_counts()
    print("ALL PASS")


if __name__ == "__main__":
    main()
