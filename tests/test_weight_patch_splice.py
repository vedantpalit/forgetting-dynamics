"""Does the weight splice patch exactly what it claims, and can it move the output at all?

`analyze_weight_patch` measured a SMALL effect for the QK group. A small effect and a splice
that landed nowhere produce the same number, so the difference has to be established by test
rather than by reading the code. This pins three things:

  1. `splice_keys` raises on a path that matches nothing, instead of silently patching zero
     leaves.
  2. `verify` accepts a correct splice: intended leaves bit-identical to the source tree,
     every other leaf bit-identical to the destination, and both positive controls pass.
  3. `verify` REJECTS a sabotaged splice. Without this the checks could be vacuous -- a
     verification that cannot fail proves nothing about the run it guards.

Two random inits are not enough to exercise (2): flax initialises LayerNorm scale/bias and
Dense bias DETERMINISTICALLY, so those leaves are equal between any two inits and a naive
difference test cannot see whether they were spliced. Every leaf is perturbed here instead.

Small on purpose (2 layers, d=64) -- same construction path, seconds instead of minutes.

Run: uv run python -m tests.test_weight_patch_splice
"""
import types

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from src.data.biography import NUM_ATTRIBUTES
from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.experiments.mlpfree_common import create_mlpfree_model
from src.model.factory import create_model
import src.experiments.analyze_weight_patch as wp

SMALL = dict(model_dim=64, num_heads=4, num_layers=2, dropout_rate=0.0,
             mlp_coefficient=0, activation="gelu", positional_encoding="rope",
             max_seq_len=32, dtype="float32")
VOCAB, SEQ, N = 37, 16, 8


def _setup(mlp_coefficient=0):
    cfg = dict(SMALL, mlp_coefficient=mlp_coefficient)
    dcfg = DataConfig(task_type="biography", sequence_length=SEQ, vocab_size=VOCAB)
    model = (create_mlpfree_model(ModelConfig(**cfg), dcfg) if mlp_coefficient == 0
             else create_model(ModelConfig(**cfg), dcfg))
    x = jnp.asarray((np.arange(N * SEQ) % VOCAB).reshape(N, SEQ), dtype=jnp.int32)
    p0 = model.init(jax.random.PRNGKey(0), x, deterministic=True)["params"]

    # EVERY leaf perturbed -- see module docstring on why two random inits will not do.
    flat = flatten_dict(p0)
    keys = jax.random.split(jax.random.PRNGKey(1), len(flat))
    pt = unflatten_dict({k: v + 0.05 * jax.random.normal(keys[i], v.shape)
                         for i, (k, v) in enumerate(sorted(flat.items()))})

    # Targets at the value slots, so the accuracy scoring has something to score.
    cols = np.tile(np.arange(2, 2 + NUM_ATTRIBUTES), (N, 1)).astype(np.int64)
    targets = np.zeros((N, SEQ), dtype=np.int64)
    rng = np.random.default_rng(3)
    for k in range(NUM_ATTRIBUTES):
        targets[np.arange(N), cols[:, k]] = rng.integers(0, VOCAB, N)
    ds = types.SimpleNamespace(eval_inputs=np.asarray(x), eval_targets=targets)
    return model, p0, pt, ds, cols


def test_missing_path_raises(model, p0, pt):
    wp.GROUPS["_bogus"] = [("CausalSelfAttention_0", "no_such_layer", "kernel")]
    try:
        wp.splice(pt, p0, "_bogus", {0})
    except KeyError as e:
        assert "silent no-op" in str(e), str(e)
        print("  ok  a path matching nothing raises instead of patching zero leaves")
    else:
        raise AssertionError("a nonexistent parameter path spliced without error")
    finally:
        del wp.GROUPS["_bogus"]


def test_verify_accepts(model, p0, pt, ds, cols):
    # Covers the global groups too (embed / finalln / head), which take a different key path
    # and, for `head`, an inverted positive control -- it sits downstream of the readout tap,
    # so zeroing it must NOT move the readout.
    for group in wp.all_groups():
        r0 = wp.readout(model, p0, ds, cols, N)
        wp.verify(model, p0, pt, group, SMALL["num_layers"], ds, cols, N, r0)
        print(f"  ok  verify passes for group {group!r}")


def test_verify_rejects_sabotage(model, p0, pt, ds, cols):
    """A splice that silently does nothing must be caught, not measured."""
    real = wp.splice_keys
    wp.splice_keys = lambda dst, src, keys: dst          # lands nowhere
    try:
        r0 = wp.readout(model, p0, ds, cols, N)
        try:
            wp.verify(model, p0, pt, "qk", SMALL["num_layers"], ds, cols, N, r0)
        except SystemExit as e:
            print(f"  ok  verify rejects a no-op splice: {str(e).splitlines()[0][:70]}")
        else:
            raise AssertionError("verify passed a splice that patched nothing")
    finally:
        wp.splice_keys = real


def test_accuracy_scoring(model, p0, pt, ds, cols):
    """The head-splice accuracy read used for the ballast contrast.

    (body_0, head_0) is the step-0 model and must reproduce its accuracy exactly; the two
    hybrid splices must each be a genuine intervention (differ from both endpoints somewhere),
    and the own-half read must be an argmax inside the own half: it can never credit a value
    outside it.
    """
    own_ids = [np.arange(k * 5, k * 5 + 5) for k in range(NUM_ATTRIBUTES)]   # 5-value halves
    acc0, own0 = wp.accuracy_stats(model, p0, ds, cols, N, own_ids)
    same = wp.splice(p0, p0, "head", {0})[0]
    a_same, o_same = wp.accuracy_stats(model, same, ds, cols, N, own_ids)
    assert np.array_equal(acc0, a_same) and np.array_equal(own0, o_same), "self-splice moved it"
    acc_t, _ = wp.accuracy_stats(model, pt, ds, cols, N, own_ids)
    hyb_gt = wp.splice(p0, pt, "head", {0})[0]      # body 0, head t
    hyb_g0 = wp.splice(pt, p0, "head", {0})[0]      # body t, head 0
    f0, ft = flatten_dict(p0), flatten_dict(pt)
    fgt, fg0 = flatten_dict(hyb_gt), flatten_dict(hyb_g0)
    head = set(wp.GLOBAL_GROUPS["head"])
    for k in f0:
        src_gt, src_g0 = (ft if k in head else f0), (f0 if k in head else ft)
        assert np.array_equal(fgt[k], src_gt[k]) and np.array_equal(fg0[k], src_g0[k]), k
    print(f"  ok  head splices land on the right side of every leaf; A(0)={acc0.mean():.3f} "
          f"A(t)={acc_t.mean():.3f}")
    # own-half read never credits a value outside the half
    t = ds.eval_targets
    for k in range(NUM_ATTRIBUTES):
        outside = ~np.isin(t[np.arange(N), cols[:, k]], own_ids[k])
        # rows whose target lies outside the own half cannot be scored correct there
        _, own_k = wp.accuracy_stats(model, p0, ds, cols, N, own_ids)
        assert own_k[k] <= 1.0 - outside.mean() + 1e-9, "own-half read credited an outside value"
    print("  ok  own-half accuracy is an argmax inside the half")


def test_standard_model_paths():
    """Every global group's path exists in a model WITH MLPs too, and the delta_structure taps
    it needs (input_layer, backbone/block_i, backbone/LayerNorm_0) are present. This is what
    lets the ballast arms, which have MLPs, run through the same code."""
    model, p0, pt, ds, cols = _setup(mlp_coefficient=4)
    flat = flatten_dict(p0)
    for g, keys in wp.GLOBAL_GROUPS.items():
        for k in keys:
            assert k in flat, f"{g}: {'/'.join(k)} missing in the standard model"
    for j in range(SMALL["num_layers"]):
        for leaf in wp.GROUPS["ln"]:
            assert ("backbone", f"block_{j}") + leaf in flat
    r0 = wp.readout(model, p0, ds, cols, N)
    wp.verify(model, p0, pt, "head", SMALL["num_layers"], ds, cols, N, r0)
    x = jnp.asarray(ds.eval_inputs)
    _, aux = model.apply({"params": p0}, x, deterministic=True, capture_intermediates=True,
                         mutable=["intermediates"])
    it = aux["intermediates"]
    assert "input_layer" in it and "backbone" in it and "LayerNorm_0" in it["backbone"]
    for j in range(SMALL["num_layers"]):
        assert f"block_{j}" in it["backbone"]
    print("  ok  standard (MLP) model: global-group paths, head verify, and residual taps")


if __name__ == "__main__":
    model, p0, pt, ds, cols = _setup()
    test_missing_path_raises(model, p0, pt)
    test_verify_accepts(model, p0, pt, ds, cols)
    test_verify_rejects_sabotage(model, p0, pt, ds, cols)
    test_accuracy_scoring(model, p0, pt, ds, cols)
    test_standard_model_paths()
    print("\nall weight-splice tests passed")
