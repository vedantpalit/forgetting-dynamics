"""One place that knows how to reload each injection arm's checkpoints.

The fixed-checkpoint analyses (`analyze_weight_patch`, `analyze_unembedding_geometry`,
`analyze_delta_structure`) were written against the MLP-free 8-layer arm only. The ballast
contrast needs the same measurements on four more arms whose checkpoints already exist:

    standard             8L, with ballast     scale8inject-p{step}-{key}-step{s:04d}.msgpack
    standard_noballast   8L, no ballast       same prefix, different key (num_ballast=0)
    demo4                4L, with ballast     demoinject-with_ballast-{key}-step{s:04d}.msgpack
    demo4_noballast      4L, no ballast       demoinject-without_ballast-{key}-step{s:04d}.msgpack
    mlp_free             8L, MLP-free         mlpfreeinject-p{step}-{key}-step{s:04d}.msgpack
    no_final_norm        8L, no final LN      nofinalnorminject-p{step}-{key}-step{s:04d}.msgpack

Each arm's key is rebuilt by replaying the EXACT argv its driver used (`scale8_injection.py`,
`demo_injection.sh`), because the key hashes the config: an extra or missing flag gives a
real-but-wrong key and a silent 404. The two families derive their injection learning rate
differently, and that difference is load-bearing for the demo key (see
`analyze_demo_injection.py`'s note on the -1.0 sentinel): scale8 sets it from the constant,
demo derives it from the pretrain checkpoint's meta file and falls back to the same constant.

Nothing here trains, and the returned context is only what a forward pass or a parameter read
needs. Every path returned is checked to exist by the caller, never assumed.
"""
import os
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, List

from src.config import parse_config
from src.experiments.analyze_name_overlap import crossover_pretrain_key
from src.experiments.ckpt import ckpt_path, experiment_key, load_params
from src.experiments.demo_injection import demo_inject_key
from src.experiments.knowledge_injection import (
    INJECT_LR_RATIO, KIConfig, build, init_state, load_meta, make_optimizer, meta_path,
    pretrain_key,
)
from src.experiments.mlpfree_common import create_mlpfree_model, mlpfree_pretrain_key
from src.experiments.nofinalnorm_common import create_nofinalnorm_model, nofinalnorm_pretrain_key

DENSE = "0,10,25,50,100,200,400,600,800,1000,1200"
PRETRAIN_PEAK_LR = 5e-4          # every driver here pretrained at this peak

ARM_SPECS = {
    "mlp_free": dict(family="scale8", num_ballast=2000, mlp_coefficient=0,
                     phase="mlpfreeinject", prefix="mlpfreeinject", key_extra={"arm": "mlp_free"},
                     model=(512, 8, 8), pretrain_steps=16000, default_pretrain_step=16000),
    "no_final_norm": dict(family="scale8", num_ballast=2000, mlp_coefficient=4,
                          phase="nofinalnorminject", prefix="nofinalnorminject",
                          key_extra={"arm": "no_final_norm"},
                          model=(512, 8, 8), pretrain_steps=16000, default_pretrain_step=16000),
    "standard": dict(family="scale8", num_ballast=2000, mlp_coefficient=4,
                     phase="scale8inject", prefix="scale8inject", key_extra={},
                     model=(512, 8, 8), pretrain_steps=16000, default_pretrain_step=16000),
    "standard_noballast": dict(family="scale8", num_ballast=0, mlp_coefficient=4,
                               phase="scale8inject", prefix="scale8inject", key_extra={},
                               model=(512, 8, 8), pretrain_steps=16000,
                               default_pretrain_step=16000),
    "demo4": dict(family="demo", num_ballast=2000, arm_name="with_ballast",
                  model=(256, 4, 4), pretrain_steps=8000, default_pretrain_step=8000),
    "demo4_noballast": dict(family="demo", num_ballast=0, arm_name="without_ballast",
                            model=(256, 4, 4), pretrain_steps=8000, default_pretrain_step=8000),
}


@dataclass
class ArmContext:
    arm: str
    spec: dict
    cfg: Any
    model: Any
    data_cfg: Any
    pop: Any
    data_a: Any
    data_b: Any
    data_ballast: Any
    data_c: Any
    template_params: Any
    key: str
    pretrain_step: int
    steps: List[int]
    n_layers: int
    path_for: Callable[[int], str]

    def load(self, step):
        p = self.path_for(step)
        if not os.path.exists(p):
            raise SystemExit(f"checkpoint missing for arm={self.arm} step={step}: {p}")
        return load_params(p, self.template_params)


def _argv_scale8(spec, seed, condition, total_steps, schedule):
    """`scale8_injection.py`'s argv, with the ballast count as the one variable."""
    d, h, L = spec["model"]
    return [
        "--num_a", "2000", "--num_ballast", str(spec["num_ballast"]), "--num_b", "500",
        "--num_c", "500",
        "--model.model_dim", str(d), "--model.num_heads", str(h), "--model.num_layers", str(L),
        "--model.dropout_rate", "0", "--model.mlp_coefficient", str(spec["mlp_coefficient"]),
        "--pretrain.total_steps", str(spec["pretrain_steps"]), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", condition, "--inject.seed", str(seed),
        "--inject.total_steps", str(total_steps), "--inject.batch_size", "256",
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", schedule,
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ]


def _argv_demo(spec, seed, condition, total_steps, schedule):
    """`demo_injection.sh`'s argv, verbatim. No dropout or mlp flags: the driver passed
    none, and adding them would change the hash."""
    d, h, L = spec["model"]
    return [
        "--num_a", "2000", "--num_ballast", str(spec["num_ballast"]), "--num_b", "500",
        "--num_c", "500",
        "--pretrain.total_steps", str(spec["pretrain_steps"]), "--pretrain.batch_size", "256",
        "--pretrain.opt.peak_lr", str(PRETRAIN_PEAK_LR),
        "--inject.condition", condition, "--inject.seed", str(seed),
        "--inject.total_steps", str(total_steps), "--inject.batch_size", "256",
        "--inject.eval_interval", "10", "--inject.checkpoint_steps", schedule,
        "--model.model_dim", str(d), "--model.num_heads", str(h), "--model.num_layers", str(L),
        "--partition_path", "data/biography/value_partition.npz", "--seed", "42",
        "--wandb_mode", "disabled",
    ]


def load_arm(arm, seed, condition="disjoint", total_steps=1200, checkpoint_dir="checkpoints",
             pretrain_step=None, schedule=None):
    if arm not in ARM_SPECS:
        raise SystemExit(f"unknown arm {arm!r}; known: {sorted(ARM_SPECS)}")
    spec = ARM_SPECS[arm]
    if pretrain_step is None:
        pretrain_step = spec["default_pretrain_step"]
    if schedule is None:
        schedule = DENSE
    argv = (_argv_scale8 if spec["family"] == "scale8" else _argv_demo)(
        spec, seed, condition, total_steps, schedule)
    saved = sys.argv
    sys.argv = [sys.argv[0]] + argv
    try:
        cfg = parse_config(KIConfig, description=f"fixed-checkpoint analysis ({arm})")
    finally:
        sys.argv = saved
    cfg = replace(cfg, checkpoint_dir=checkpoint_dir)

    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = build(
        cfg, cfg.inject.max_eval_people)
    if arm == "mlp_free":
        model = create_mlpfree_model(cfg.model, data_cfg)
    elif arm == "no_final_norm":
        model = create_nofinalnorm_model(cfg.model, data_cfg)

    if spec["family"] == "scale8":
        pre_key = {"mlp_free": mlpfree_pretrain_key,
                   "no_final_norm": nofinalnorm_pretrain_key}.get(arm, pretrain_key)(cfg, data_cfg)
        inject_opt = replace(cfg.inject.opt, peak_lr=PRETRAIN_PEAK_LR / INJECT_LR_RATIO)
        inject_cfg = replace(cfg.inject, opt=inject_opt)
        key = experiment_key(
            cfg.model, data_cfg, cfg.data.biography_data_path, phase=spec["phase"],
            inject=replace(inject_cfg, max_eval_people=0), pretrain_key=pre_key,
            pretrain_step=pretrain_step, seed=cfg.seed, **spec["key_extra"])
        pattern = f"{spec['prefix']}-p{pretrain_step}-{key}-step{{s:04d}}.msgpack"
    else:
        pre_key = (crossover_pretrain_key(cfg, data_cfg) if spec["num_ballast"] > 0
                   else pretrain_key(cfg, data_cfg))
        inject_opt = cfg.inject.opt
        if inject_opt.peak_lr <= 0:
            # The driver derived the rate from the pretrain checkpoint's meta and baked it
            # into the key. The meta lives next to the checkpoint on the cluster; the
            # fallback is the value every one of these runs recorded, so the key is the
            # same either way, and the caller's existence check catches any drift.
            pre_path = ckpt_path(checkpoint_dir, "pretrain", pre_key)
            meta = (load_meta(pre_path)
                    if pre_path and os.path.exists(meta_path(pre_path)) else None)
            peak = float(meta["peak_lr"]) if meta is not None else PRETRAIN_PEAK_LR
            if meta is None:
                print(f"  [arms] no pretrain meta at {meta_path(pre_path) if pre_path else '?'}; "
                      f"using peak_lr={PRETRAIN_PEAK_LR} to derive the injection rate")
            inject_opt = replace(inject_opt, peak_lr=peak / INJECT_LR_RATIO)
        inject_cfg = replace(cfg.inject, opt=inject_opt)
        key = demo_inject_key(cfg, data_cfg, pre_key, inject_cfg)
        pattern = f"demoinject-{spec['arm_name']}-{key}-step{{s:04d}}.msgpack"

    template = init_state(model, cfg, data_cfg,
                          make_optimizer(inject_cfg.opt, inject_cfg.total_steps), cfg.seed + 20)
    steps = sorted({int(s) for s in schedule.split(",") if s.strip()} | {total_steps})
    steps = [s for s in steps if s <= total_steps]

    def path_for(s):
        return os.path.join(checkpoint_dir, pattern.format(s=s))

    print(f"  [arms] arm={arm} family={spec['family']} num_ballast={spec['num_ballast']} "
          f"model={spec['model']} pretrain_step={pretrain_step} key={key}")
    return ArmContext(arm=arm, spec=spec, cfg=cfg, model=model, data_cfg=data_cfg, pop=pop,
                      data_a=data_a, data_b=data_b, data_ballast=data_ballast, data_c=data_c,
                      template_params=template.params, key=key, pretrain_step=pretrain_step,
                      steps=steps, n_layers=cfg.model.num_layers, path_for=path_for)
