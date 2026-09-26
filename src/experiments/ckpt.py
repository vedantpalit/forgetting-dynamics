"""Checkpoint keying and shape-checked (de)serialization.

Why this module exists
----------------------
`onpolicy_cl_biography._fingerprint` hashes `vars_nested(cfg.data)` — the *pre-override*
`DataConfig`, whose `sequence_length` and `vocab_size` are still the placeholder 64/64. The
real values are set later, in `data_cfg = replace(cfg.data, ...)`, and never reach the hash.
So any pool or template change yields a byte-identical checkpoint key.

What is actually silent, all of it measured in `tests/test_ckpt_key.py` rather than assumed:

  * A **vocab** mismatch is *not* silent. Flax validates `self.param` shapes against the
    module's declared shape at apply time and raises `ScopeParamShapeError` in both
    directions (test 8), with no help from this module.
  * An out-of-range token id yields **NaN**, not a clamped row — `nn.Embed` calls
    `jnp.take`, whose default `mode="fill"` fills out-of-bounds gathers (test 7).
  * **`sequence_length` under RoPE is the silent one.** It alters no parameter shape at
    all, so flax's check and any shape assertion are both blind to it, and a stale
    checkpoint loads perfectly cleanly. Only the key can catch it. **Test 9** demonstrates
    the upstream collision and this module's fix end to end; read it first.

Hence:
  1. Key on the post-`replace` config, so sequence_length/vocab_size actually enter the hash.
     This is the primary defence and the only one that sees a sequence_length change.
  2. Fold in a content hash of the preprocessed .npz *and* the source text of the data
     modules — the .npz hash alone misses generation-code edits (a changed sampler in
     `biography.py` produces different data from an identical .npz).
  3. Assert restored leaf shapes and dtypes against a freshly initialised template on load.
     Retained as **diagnostics**: it fails at load time naming the checkpoint file, instead
     of failing mid-forward-pass with a scope error that never mentions checkpoints.
"""
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from src.config import vars_nested

# Resolved from this file, not by import, so fingerprinting has no import side effects.
_SRC_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE_FILES = (
    _SRC_ROOT / "data" / "biography.py",
    _SRC_ROOT / "data" / "biography_corpus.py",
    _SRC_ROOT / "data" / "preprocess_biography.py",
)


class CheckpointMismatch(RuntimeError):
    """A checkpoint's structure, shapes, or dtypes disagree with the model loading it."""


def file_hash(path) -> str:
    """SHA1 of a file's bytes. Public: anything that determines training data through a
    file's *content* rather than its path (the preprocessed corpus, a value partition)
    should be hashed this way into a checkpoint key -- a path alone says nothing about
    what changed if the file at that path was overwritten with different content.
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_fingerprint(data_cfg, npz_path, source_files=DATA_SOURCE_FILES) -> str:
    """Hash of everything that determines the training data.

    Args:
        data_cfg: the POST-`replace` DataConfig (real sequence_length / vocab_size).
        npz_path: preprocessed corpus; hashed by content.
        source_files: data-generation modules; hashed by content.
    """
    parts = {
        "data_cfg": vars_nested(data_cfg),
        "npz": file_hash(npz_path),
        "src": {Path(p).name: file_hash(p) for p in source_files},
    }
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def experiment_key(model_cfg, data_cfg, npz_path, **extra) -> str:
    """12-hex key over model config, data fingerprint, and any extra phase-specific fields."""
    parts = {
        "model": vars_nested(model_cfg),
        "data": data_fingerprint(data_cfg, npz_path),
        "extra": {k: vars_nested(v) for k, v in sorted(extra.items())},
    }
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def ckpt_path(checkpoint_dir: str, name: str, key: str) -> str:
    """Path for a named checkpoint, or "" when checkpointing is disabled."""
    if not checkpoint_dir:
        return ""
    return os.path.join(checkpoint_dir, f"{name}-{key}.msgpack")


# --- shape-checked (de)serialization ---

def _key_str(k) -> str:
    return str(getattr(k, "key", getattr(k, "idx", k)))


def _leaves_with_paths(tree):
    return [("/".join(_key_str(k) for k in path), leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _assert_compatible(restored, template, what: str):
    """Raise CheckpointMismatch unless every leaf matches in shape and dtype."""
    r_struct = jax.tree_util.tree_structure(restored)
    t_struct = jax.tree_util.tree_structure(template)
    if r_struct != t_struct:
        raise CheckpointMismatch(
            f"{what}: pytree structure differs.\n  checkpoint: {r_struct}\n  model:      {t_struct}"
        )
    for (path, r), (_, t) in zip(_leaves_with_paths(restored), _leaves_with_paths(template)):
        if jnp.shape(r) != jnp.shape(t):
            raise CheckpointMismatch(
                f"{what}: shape mismatch at '{path}' — checkpoint {jnp.shape(r)}, "
                f"model {jnp.shape(t)}. The checkpoint was trained on different data "
                f"(vocab or sequence length); regenerate it rather than loading."
            )
        if jnp.asarray(r).dtype != jnp.asarray(t).dtype:
            raise CheckpointMismatch(
                f"{what}: dtype mismatch at '{path}' — checkpoint "
                f"{jnp.asarray(r).dtype}, model {jnp.asarray(t).dtype}."
            )


def save_params(path: str, params):
    """Params only — used for the dense injection checkpoints, where optimizer state
    would triple the footprint for no analytic gain."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))


def load_params(path: str, template_params):
    """Restore params, asserting shapes against a freshly initialised template."""
    with open(path, "rb") as f:
        restored = serialization.from_bytes(template_params, f.read())
    _assert_compatible(restored, template_params, f"params from {path}")
    return restored


def save_state(path: str, state):
    """Params + optimizer state + step — used at phase boundaries, so the
    inherited-moments injection arm survives a process boundary."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"params": state.params, "opt_state": state.opt_state, "step": int(state.step)}
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(payload))


def load_state(path: str, template_state):
    """Restore {params, opt_state, step}, asserting params against `template_state`.

    Returns (params, opt_state, step). Only params are shape-checked: the optimizer state
    is deliberately allowed to differ in structure, because the injection phase builds a
    different optimizer (constant schedule) than the phase that wrote the checkpoint.
    """
    template = {"params": template_state.params,
                "opt_state": template_state.opt_state,
                "step": 0}
    with open(path, "rb") as f:
        restored = serialization.from_bytes(template, f.read())
    _assert_compatible(restored["params"], template_state.params, f"params from {path}")
    return restored["params"], restored["opt_state"], int(restored["step"])
