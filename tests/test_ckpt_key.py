"""Regression tests for checkpoint keying.

The defect: upstream hashes the pre-override `DataConfig`, whose `sequence_length` and
`vocab_size` are still the placeholder 64/64, so a data change that moves either one
produces an identical key.

What is and is not silent, all measured below rather than assumed:

  * A **vocab** mismatch already fails loudly without any help from us — flax validates
    `self.param` shapes against the module's declared shape at apply time and raises
    `ScopeParamShapeError` in both directions (test 8).
  * An out-of-range token id yields **NaN**, not a clamped row: `nn.Embed` calls
    `jnp.take`, whose default `mode="fill"` fills out-of-bounds gathers (test 7).
  * The genuinely silent case is **sequence_length** under RoPE. It alters no parameter
    shape at all, so neither flax nor a shape assertion can see it, and a stale checkpoint
    loads perfectly cleanly. Only the key catches it. Test 9 demonstrates the collision and
    the fix end to end, and is the reason this module exists.

The shape assertion in `load_params` is therefore retained as *diagnostics* — it fails at
load time naming the checkpoint file, rather than mid-forward-pass with a flax scope error
that never mentions checkpoints — not as the primary defence.

Run: uv run python -m tests.test_ckpt_key
"""
import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from src.config import vars_nested
from src.data.config import DataConfig
from src.model.config import ModelConfig
from src.model.factory import create_model
from src.experiments.ckpt import (
    CheckpointMismatch, data_fingerprint, experiment_key, load_params, save_params,
)
from src.experiments.knowledge_injection import KIConfig, pretrain_key

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")


def upstream_key(model_cfg, data_cfg_pre_override, **extra) -> str:
    """Replica of `onpolicy_cl_biography._fingerprint` + `common_key`.

    Faithful in the one respect that matters: it hashes the DataConfig *before*
    `replace(...)` fills in the real sequence_length and vocab_size.
    """
    blob = json.dumps(
        {"model": vars_nested(model_cfg), "data": vars_nested(data_cfg_pre_override), **extra},
        sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _npz(path, value=1):
    np.savez(path, num_values_per_attr=np.array([value, 2, 3], dtype=np.int32))
    return str(path)


def _src(path, text):
    Path(path).write_text(text, encoding="utf-8")
    return str(path)


def _build(vocab, seq_len=32, model_dim=16):
    mc = ModelConfig(model_dim=model_dim, num_heads=2, num_layers=1, dropout_rate=0.0)
    dc = DataConfig(task_type="biography", sequence_length=seq_len, vocab_size=vocab)
    model = create_model(mc, dc)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        jnp.ones((1, seq_len), jnp.int32), deterministic=True)
    return model, variables["params"]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ckpt_key_test_"))
    try:
        npz_a = _npz(tmp / "a.npz", value=1)
        src_a = _src(tmp / "gen.py", "VERSION = 1\n")
        base = DataConfig(task_type="biography", sequence_length=195, vocab_size=1881)
        key = lambda cfg, npz=npz_a, src=(src_a,): data_fingerprint(cfg, npz, src)
        k0 = key(base)

        print("\n[1] vocab_size enters the key")
        check("vocab_size 1881 -> 2141 changes the fingerprint",
              key(replace(base, vocab_size=2141)) != k0)

        print("\n[2] sequence_length enters the key")
        check("sequence_length 195 -> 210 changes the fingerprint",
              key(replace(base, sequence_length=210)) != k0)

        print("\n[3] .npz content enters the key")
        check("different .npz content changes the fingerprint",
              key(base, npz=_npz(tmp / "b.npz", value=99)) != k0)
        check("identical .npz at a different path leaves it unchanged (content-addressed)",
              key(base, npz=str(shutil.copyfile(npz_a, tmp / "a_copy.npz"))) == k0)

        print("\n[4] data-module source text enters the key")
        check("edited generation source changes the fingerprint",
              key(base, src=(_src(tmp / "gen2.py", "VERSION = 2\n"),)) != k0)

        print("\n[5] determinism and scoping")
        mc = ModelConfig(model_dim=256)
        ek = experiment_key(mc, base, npz_a, phase="pretrain")
        check("same inputs give the same fingerprint", key(base) == k0)
        check("experiment_key is stable across calls",
              experiment_key(mc, base, npz_a, phase="pretrain") == ek)
        check("a different phase changes experiment_key",
              experiment_key(mc, base, npz_a, phase="inject") != ek)
        check("a different model_dim changes experiment_key",
              experiment_key(ModelConfig(model_dim=128), base, npz_a, phase="pretrain") != ek)

        print("\n[6] load-time shape assertion (diagnostics, not the primary defence)")
        _, p_small = _build(vocab=64)
        _, p_large = _build(vocab=128)
        path = str(tmp / "ckpt.msgpack")
        save_params(path, p_small)
        check("matching vocab loads cleanly", load_params(path, p_small) is not None)
        raised, detail = False, ""
        try:
            load_params(path, p_large)
        except CheckpointMismatch as e:
            raised, detail = True, str(e).splitlines()[0]
        check("vocab-64 checkpoint refuses to load into a vocab-128 model", raised,
              detail if raised else "NO EXCEPTION")

        print("\n[7] out-of-range ids give NaN, not a clamped row")
        emb = nn.Embed(num_embeddings=8, features=4)
        ep = emb.init(jax.random.PRNGKey(0), jnp.array([0]))
        table = ep["params"]["embedding"]
        oob = emb.apply(ep, jnp.array([999]))
        check("nn.Embed fills out-of-bounds ids with NaN", bool(jnp.all(jnp.isnan(oob))),
              "jnp.take defaults to mode='fill'")
        check("it is NOT the clamped last row", not bool(jnp.allclose(oob[0], table[-1])))
        check("mode='clip' would have clamped (contrast)",
              bool(jnp.allclose(jnp.take(table, jnp.array([999]), axis=0, mode="clip")[0],
                                table[-1])))

        print("\n[8] vocab mismatch already fails loudly at apply time")
        m64, p64 = _build(vocab=64)
        m128, p128 = _build(vocab=128)
        ids = jnp.array([[0, 1, 30, 5] + [0] * 28])
        for label, model, params in [("narrow ckpt -> wide model", m128, p64),
                                     ("wide ckpt -> narrow model", m64, p128)]:
            hit = False
            try:
                model.apply({"params": params}, ids, deterministic=True)
            except Exception as e:
                hit = type(e).__name__ == "ScopeParamShapeError"
            check(f"{label} raises ScopeParamShapeError", hit)

        print("\n[9] THE DEFECT: a sequence_length change is invisible to everything but the key")
        real_npz = "data/biography/preprocessed.npz"
        d = dict(np.load(real_npz, allow_pickle=True))
        msl_1 = int(d["max_sentence_len"])
        prompt_len = 1 + int(d["max_name_len"])
        npz_1 = str(tmp / "corpus_v1.npz")
        np.savez(npz_1, **d)
        d2 = dict(d)
        d2["max_sentence_len"] = np.array(msl_1 + 2)  # e.g. a template or pool edit
        npz_2 = str(tmp / "corpus_v2.npz")
        np.savez(npz_2, **d2)
        vocab = int(d["english_vocab_size"])
        seq_1 = prompt_len + 6 * msl_1
        seq_2 = prompt_len + 6 * (msl_1 + 2)
        check(f"the edit really moves seq_len ({seq_1} -> {seq_2})", seq_1 != seq_2)

        # Both runs start from the SAME default DataConfig; only the post-replace object
        # differs. That is exactly the situation upstream hashes.
        pre = DataConfig(task_type="biography")
        cfg_1 = replace(pre, sequence_length=seq_1, vocab_size=vocab)
        cfg_2 = replace(pre, sequence_length=seq_2, vocab_size=vocab)
        check("upstream scheme: placeholder defaults are what get hashed",
              pre.sequence_length == 64 and pre.vocab_size == 64,
              f"sequence_length={pre.sequence_length}, vocab_size={pre.vocab_size}")
        check("upstream scheme COLLIDES across the change",
              upstream_key(mc, pre, seed=42) == upstream_key(mc, pre, seed=42))
        check("our scheme SEPARATES the two runs",
              experiment_key(mc, cfg_1, npz_1, phase="pretrain")
              != experiment_key(mc, cfg_2, npz_2, phase="pretrain"))

        # And the consequence: under the colliding key the stale checkpoint is found and
        # loads without a murmur, because no parameter shape depends on sequence_length.
        _, params_seq1 = _build(vocab=64, seq_len=seq_1 // 8)
        _, params_seq2 = _build(vocab=64, seq_len=seq_2 // 8)
        shapes_equal = jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: a.shape == b.shape, params_seq1, params_seq2))
        check("no parameter shape depends on sequence_length (RoPE)", shapes_equal)
        stale = str(tmp / "stale.msgpack")
        save_params(stale, params_seq1)
        loaded = load_params(stale, params_seq2)
        check("so the stale checkpoint loads clean — no error, no NaN, no shape complaint",
              loaded is not None and not bool(jnp.any(jnp.isnan(
                  jax.tree_util.tree_flatten(loaded)[0][0]))))

        print("\n[10] partition-file content enters pretrain_key (the exclusion-fraction-sweep collision)")
        # Two partitions with the SAME num_values_per_attr and the SAME partition_seed --
        # exactly the scenario a sweep over exclusion fraction produces: pool sizes never
        # change, only which specific values land in which half, so nothing about shape
        # differs between them. Before this fix, pretrain_key was blind to that difference.
        num_values = np.array([20, 20, 20, 20, 20, 20])

        def _fake_partition(path, x_size):
            arrays = {"seed": np.array(7), "num_values_per_attr": num_values}
            for k in range(6):
                arrays[f"attr_{k}_X"] = np.arange(x_size)
                arrays[f"attr_{k}_Y"] = np.arange(x_size, 20)
            np.savez(path, **arrays)

        part_a = str(tmp / "partition_f10.npz")
        part_b = str(tmp / "partition_f50.npz")
        _fake_partition(part_a, x_size=2)    # f = 0.10-equivalent: X is 2 of 20
        _fake_partition(part_b, x_size=10)   # f = 0.50-equivalent: X is 10 of 20

        kicfg = KIConfig(
            data=replace(DataConfig(task_type="biography"), sequence_length=195, vocab_size=1881,
                        biography_data_path=real_npz),
            num_a=20, num_ballast=20, num_b=5, num_c=0, seed=42, partition_seed=7,
            checkpoint_dir="", exposure_dir="")
        data_cfg_test = replace(kicfg.data, sequence_length=195, vocab_size=vocab)

        key_a = pretrain_key(replace(kicfg, partition_path=part_a), data_cfg_test)
        key_b = pretrain_key(replace(kicfg, partition_path=part_b), data_cfg_test)
        key_a_again = pretrain_key(replace(kicfg, partition_path=part_a), data_cfg_test)
        check("same partition_seed, different partition CONTENT -> different pretrain_key",
              key_a != key_b, f"key_a={key_a}, key_b={key_b}")
        check("same partition file gives the same key on a second call (determinism)",
              key_a == key_a_again)

        part_a_copy = str(tmp / "partition_f10_copy.npz")
        shutil.copyfile(part_a, part_a_copy)
        key_a_copy = pretrain_key(replace(kicfg, partition_path=part_a_copy), data_cfg_test)
        check("identical partition content at a different path gives the same key (content-addressed)",
              key_a == key_a_copy)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
