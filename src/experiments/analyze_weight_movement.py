"""How far did each attention weight group move during injection? (paper Section 5, move four)

This is the control that licenses the QK-versus-OV comparison. Restoring OV weights removes 84%
of delta and restoring QK weights removes 11%, but that contrast only says something about ROLE
if the two groups moved comparably in the first place -- otherwise it says the larger revert did
more, which is not interesting.

PARAMETER READS ONLY. No forward passes, no model evaluation: load two checkpoints and compare
tensors. Seconds per seed, which is why running it across all three seeds rather than quoting
one is worth doing at all.

THE COMPARISON IS KERNEL TO KERNEL. `query`, `key` and `value` are use_bias=False, so `out.bias`
has no QK counterpart. Including it raises OV's mean from 1.93% to 2.32% against QK's 1.62% and
turns a matched comparison into an unmatched one dressed as a matched one. Both are reported;
the kernel figure is the one the argument rests on.

NORMALISED BY max(||W_0||, ||W_t||), not by ||W_0||. A bias that pretrained to near zero would
otherwise divide by ~0 and report a meaningless 1e+11, which is what the first version of this
did.

Run:
  uv run python -m src.experiments.analyze_weight_movement --patch_step 50
"""
import argparse
import os
from dataclasses import replace

import numpy as np
from flax.traverse_util import flatten_dict

from src.experiments.ckpt import experiment_key, load_params
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, build, init_state, make_optimizer,
)
from src.experiments.analyze_mlpfree import ARMS, PRETRAIN_PEAK_LR, _cfg_for
from src.experiments.analyze_weight_patch import GROUPS, group_keys, structural_check
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key

OUT_DIR = "weight_movement"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mlp_free")
    ap.add_argument("--pretrain_step", type=int, default=16000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--condition", default="disjoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--total_steps", type=int, default=1200)
    ap.add_argument("--patch_step", type=int, default=50)
    a = ap.parse_args()
    if a.arm != "mlp_free":
        raise SystemExit("mlp_free only")
    seeds = [int(s) for s in a.seeds.split(",")]

    spec = dict(ARMS[a.arm])
    per_seed = {}
    n_layers = None
    for seed in seeds:
        cfg = _cfg_for(a.arm, a.pretrain_step, seed, a.condition, a.total_steps)
        pop, _, data_cfg, _, _da, _b, _c, _d = build(cfg, cfg.inject.max_eval_people)
        model = create_mlpfree_model(cfg.model, data_cfg)
        n_layers = cfg.model.num_layers
        inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
        key = experiment_key(
            cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
            inject=replace(cfg.inject, opt=inject_opt, max_eval_people=0),
            pretrain_key=mlpfree_pretrain_key(cfg, data_cfg), pretrain_step=a.pretrain_step,
            seed=cfg.seed, **spec["key_extra"])
        template = init_state(model, cfg, data_cfg,
                              make_optimizer(inject_opt, cfg.inject.total_steps), cfg.seed + 20)

        def path_for(s):
            return os.path.join(a.checkpoint_dir,
                                f"{spec['prefix']}-p{a.pretrain_step}-{key}-step{s:04d}.msgpack")
        for s in (0, a.patch_step):
            if not os.path.exists(path_for(s)):
                raise SystemExit(f"checkpoint missing: {path_for(s)}")
        f0, ft = structural_check(load_params(path_for(0), template.params),
                                  load_params(path_for(a.patch_step), template.params))
        rows = {}
        for g in sorted(GROUPS):
            for k in group_keys(g, range(n_layers)):
                d = np.linalg.norm(ft[k] - f0[k])
                scale = max(np.linalg.norm(f0[k]), np.linalg.norm(ft[k]), 1e-12)
                rows[(g, k[1], "/".join(k[3:]))] = d / scale
        per_seed[seed] = rows
        print(f"  seed {seed}: {len(rows)} leaves compared, key={key}")

    leaves = sorted(per_seed[seeds[0]])
    M = np.array([[per_seed[s][k] for k in leaves] for s in seeds])       # (S, L)
    ns = len(seeds)
    sem = M.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.zeros(M.shape[1])

    def agg(pred):
        idx = [i for i, k in enumerate(leaves) if pred(k)]
        v = M[:, idx].mean(1)                                            # per seed
        return v.mean(), (v.std(ddof=1) / np.sqrt(ns) if ns > 1 else 0.0)

    print(f"\nweight movement ||W(t)-W(0)|| / max(||W(0)||,||W(t)||), step {a.patch_step}, "
          f"{ns} seeds")
    print(f"  {'group':>6} {'leaf':>14} {'mean':>10} {'SE':>9}")
    for g in sorted(GROUPS):
        for leaf in sorted({k[2] for k in leaves if k[0] == g}):
            m, e = agg(lambda k, g=g, leaf=leaf: k[0] == g and k[2] == leaf)
            print(f"  {g:>6} {leaf:>14} {m:>10.5f} {e:>9.5f}")
    print()
    qk, qk_e = agg(lambda k: k[0] == "qk")
    ovk, ovk_e = agg(lambda k: k[0] == "ov" and k[2].endswith("kernel"))
    ova, ova_e = agg(lambda k: k[0] == "ov")
    print(f"  QK kernels          {qk:.5f} +/- {qk_e:.5f}")
    print(f"  OV kernels          {ovk:.5f} +/- {ovk_e:.5f}   ratio to QK {ovk/qk:.3f}")
    print(f"  OV all leaves       {ova:.5f} +/- {ova_e:.5f}   ratio to QK {ova/qk:.3f}"
          f"   (NOT matched: out.bias has no QK counterpart)")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{a.arm}-p{a.pretrain_step}-{a.condition}-step{a.patch_step}"
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{stem}.npz"),
        seeds=np.array(seeds), leaves=np.array([f"{g}|{b}|{l}" for g, b, l in leaves]),
        movement=M.astype(np.float64), patch_step=a.patch_step,
        pretrain_step=a.pretrain_step, condition=a.condition)
    print(f"\n  wrote {os.path.join(OUT_DIR, stem)}.npz")


if __name__ == "__main__":
    main()
