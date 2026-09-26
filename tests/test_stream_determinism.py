"""Batch-stream determinism across processes.

`reset_stream` was written to keep the batch sequence identical across LR grid points
*within* one process. Submitting the sweep as five independent jobs moves that
guarantee across a process boundary instead, where it now depends on the population,
partition, and RNG all being reconstructed identically from the seed.

If that failed, the sweep would be unfair in a new way — grid points would differ in
learning rate *and* in the data they saw — so it is checked rather than assumed:

  1. Two independent builds in one process give byte-identical batches.
  2. Changing only `pretrain.opt.peak_lr` leaves the stream untouched (the property the
     sweep actually rests on).
  3. A different `seed` gives a different stream, so 1 and 2 are not vacuous.
  4. Two separate OS processes agree — catching anything that leaks through interpreter
     state, hash randomization, or dict ordering.

Run: uv run python -m tests.test_stream_determinism
"""
import hashlib
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from src.data.biography import NUM_ATTRIBUTES
from src.experiments.knowledge_injection import KIConfig, build, reset_stream

PASS, FAIL = [], []

# Small population: determinism does not depend on scale, and this keeps the test quick.
SMALL = dict(num_a=200, num_ballast=200, num_b=50, num_c=50)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")


def make_cfg(seed=42, peak_lr=None, partition_path=None):
    cfg = KIConfig(seed=seed, **SMALL)
    cfg.partition_path = partition_path
    if peak_lr is not None:
        cfg.pretrain = replace(cfg.pretrain, opt=replace(cfg.pretrain.opt, peak_lr=peak_lr))
    return cfg


def stream_hash(cfg, n_batches=3, batch_size=32):
    """SHA-1 over the first few training batches."""
    _, _, _, data_pre, *_ = build(cfg, max_eval_people=1)
    reset_stream(data_pre, cfg.seed + 1)
    h = hashlib.sha1()
    for _ in range(n_batches):
        batch = data_pre.get_batch(jax.random.PRNGKey(0), batch_size)
        for key in ("inputs", "targets", "mask"):
            h.update(np.asarray(batch[key]).tobytes())
    return h.hexdigest()


def group_fingerprint(pop, dataset):
    """Who these individuals are: their name triples and their actual assigned values."""
    ids = dataset.person_ids
    h = hashlib.sha1()
    h.update(np.asarray(pop.person_names[ids]).tobytes())
    for k in range(NUM_ATTRIBUTES):
        h.update(np.argmax(pop.value_probs[k][ids], axis=1).astype(np.int32).tobytes())
    return h.hexdigest()


def population_fingerprints(cfg, max_eval_people):
    pop, _, _, _, data_a, data_ballast, data_b, data_c = build(cfg, max_eval_people)
    groups = [("A", data_a), ("ballast", data_ballast), ("B", data_b), ("C", data_c)]
    return {name: group_fingerprint(pop, ds) for name, ds in groups if ds}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="stream_det_"))
    part = str(tmp / "partition.npz")

    print("\n[1] two builds in one process")
    h1 = stream_hash(make_cfg(partition_path=part))
    h2 = stream_hash(make_cfg(partition_path=part))
    check("identical config gives identical batches", h1 == h2, h1[:12])

    print("\n[2] the learning rate does not touch the data stream")
    lo = stream_hash(make_cfg(peak_lr=2e-4, partition_path=part))
    hi = stream_hash(make_cfg(peak_lr=5e-3, partition_path=part))
    check("peak_lr 2e-4 and 5e-3 give the same batches", lo == hi and lo == h1,
          "grid points differ only in LR, as the sweep requires")

    print("\n[3] the seed does change the stream (guards against a vacuous test)")
    other = stream_hash(make_cfg(seed=43, partition_path=part))
    check("seed 43 gives a different stream", other != h1, other[:12])

    print("\n[4] two separate OS processes agree")
    outs = []
    for _ in range(2):
        res = subprocess.run(
            [sys.executable, "-m", "tests.test_stream_determinism", "--emit", part],
            capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-800:])
            check("subprocess ran", False, f"exit {res.returncode}")
            break
        outs.append(res.stdout.strip().splitlines()[-1])
    else:
        check("two independent processes produce the same stream hash",
              outs[0] == outs[1], outs[0][:12])
        check("and it matches the in-process hash", outs[0] == h1)

    print("\n[5] the pretrain/inject boundary reconstructs the same individuals")
    # The two phases now build with different max_eval_people (500 vs 0, per phase), so
    # this also checks that the eval-subsample size cannot leak into who the people are.
    pre = population_fingerprints(make_cfg(partition_path=part), max_eval_people=500)
    inj = population_fingerprints(make_cfg(partition_path=part), max_eval_people=0)
    for group in ("A", "ballast", "B", "C"):
        check(f"{group} is identical across the phase boundary", pre[group] == inj[group],
              pre[group][:12])

    print("\n[6] the condition rewrites B only")
    cfg_hi = make_cfg(partition_path=part)
    cfg_dis = make_cfg(partition_path=part)
    cfg_dis.inject = replace(cfg_dis.inject, condition="disjoint")
    hi = population_fingerprints(cfg_hi, max_eval_people=0)
    dis = population_fingerprints(cfg_dis, max_eval_people=0)
    check("A is unaffected by the condition", hi["A"] == dis["A"])
    check("ballast is unaffected by the condition", hi["ballast"] == dis["ballast"])
    check("B does change with the condition", hi["B"] != dis["B"],
          "high_overlap draws X, disjoint draws Y")

    print("\n[7] population C enters the checkpoint key")
    # C is never trained on, but its size feeds BiographyPopulation, and changing
    # num_people shifts RNG consumption in _assign_values — which moves every
    # individual's train/eval template split, A's included. If the key missed that, a
    # run with C would silently reuse a checkpoint trained on different template splits.
    import contextlib
    import io

    from src.experiments.knowledge_injection import pretrain_key

    def pop_and_key(num_c):
        cfg = KIConfig(num_a=200, num_ballast=200, num_b=50, num_c=num_c)
        cfg.partition_path = part
        with contextlib.redirect_stdout(io.StringIO()):
            pop = build(cfg, max_eval_people=1)[0]
        data_cfg = replace(cfg.data, sequence_length=pop.seq_len, vocab_size=pop.vocab_size)
        return pop, pretrain_key(cfg, data_cfg)

    pop0, key0 = pop_and_key(0)
    pop50, key50 = pop_and_key(50)
    ids = np.arange(200)
    check("A's names are unchanged by adding C",
          np.array_equal(pop0.person_names[ids], pop50.person_names[ids]))
    check("but A's template split DOES move",
          not np.array_equal(pop0.train_templates[ids], pop50.train_templates[ids]),
          "larger num_people shifts RNG consumption upstream of _split_templates")
    check("so num_c must change the checkpoint key — and does", key0 != key50,
          f"{key0} vs {key50}")

    print("\n[8] eval-only pretrain fields do NOT enter the checkpoint key")
    # eval_interval and target_acc affect logging cadence and pass/fail reporting, not what
    # gets trained. If they leaked into the key (as eval_interval briefly did), two runs with
    # identical training config but different eval settings would hash to different
    # checkpoints, forcing an unnecessary retrain instead of reusing the existing one —
    # exactly the mismatch a real inject smoke test hit against a real pretrain checkpoint.
    base_cfg = KIConfig(num_a=200, num_ballast=200, num_b=50, num_c=50)
    base_cfg.partition_path = part
    with contextlib.redirect_stdout(io.StringIO()):
        pop8 = build(base_cfg, max_eval_people=1)[0]
    data_cfg8 = replace(base_cfg.data, sequence_length=pop8.seq_len, vocab_size=pop8.vocab_size)
    key_default = pretrain_key(base_cfg, data_cfg8)
    for field, value in [("eval_interval", 999), ("target_acc", 0.42), ("max_eval_people", 12345)]:
        varied = replace(base_cfg, pretrain=replace(base_cfg.pretrain, **{field: value}))
        check(f"{field} does not change the checkpoint key",
              pretrain_key(varied, data_cfg8) == key_default,
              f"changed {field} to {value}")
    varied_steps = replace(base_cfg, pretrain=replace(base_cfg.pretrain, total_steps=1))
    check("total_steps DOES change the checkpoint key (control — it's a training knob)",
          pretrain_key(varied_steps, data_cfg8) != key_default)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    if "--emit" in sys.argv:
        # Child mode: print only the hash, so the parent can compare across processes.
        print(stream_hash(make_cfg(partition_path=sys.argv[sys.argv.index("--emit") + 1])))
    else:
        main()
