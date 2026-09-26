"""On-policy distillation for continual learning on synthetic biographies.

Protocol, one WandB run per phase:
  1. pretraining_teacher: NTP on the new-people subset B; evaluated on B.
  2. pretraining_student: NTP on the old-people subset A, optionally mixed
     with a fraction of B (finetune_mix_fraction, default 0); evaluated on A.
  3. finetuning_student: the student acquires B via distillation on B name
     prompts — on-policy (default: sample from the student, reverse KL to the
     teacher), off-policy (--finetune.sample_from teacher --finetune.kl
     forward), or on real B sequences (--finetune.sample_from data) — or via
     plain NTP on B as a baseline; evaluated on A (forgetting) and B
     (acquisition) every finetune.eval_interval steps.

The WandB step axis is the number of training steps that model has taken:
0..pretrain for the pretraining runs, continuing at pretrain.total_steps for
the finetuning run. Teacher/student pretraining are checkpointed and reused
across runs with matching config, in which case their WandB runs are skipped.

A and B are disjoint subsets of one shared population, so their people,
names, and facts never overlap. By default |B| = |A| / 8.

--phase selects what to run: "teacher" / "student" pretrain one model and
save its checkpoint (one wandb run each, so both are sweepable), "finetune"
loads the checkpoints (training them only if missing) and runs one
finetuning arm, "all" (default) runs the full pipeline.

Run: uv run python -m src.experiments.onpolicy_cl_biography --phase all
"""
import hashlib
import json
import os
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import serialization

from src.config import Config, TrainConfig, parse_config, vars_nested
from src.data.biography import BiographyDataset, BiographyPopulation
from src.data.config import DataConfig
from src.distill import distill_step, sample_trajectories
from src.divergence import divergence_metric_name, parse_divergence
from src.sharpness import ce_hessian_trace, kl_hessian_trace, kl_logit_hessian_trace
from src.model.config import ModelConfig
from src.model.factory import create_model
from src.train import TrainState, create_optimizer, create_train_state, eval_forward, train_step


@dataclass
class FinetuneConfig:
    method: str = "distill"  # "distill" or "ntp"
    sample_from: str = "student"  # "student" (on-policy), "teacher" (off-policy), or "data" (real B sequences)
    kl: str = "auto"  # "reverse" = KL(student||teacher), "forward" = KL(teacher||student); "auto" = canonical pairing (student/data -> reverse, teacher -> forward)
    # Divergence spec overriding `kl` when non-empty (see src.divergence):
    # "kl:forward[:tT]" / "kl:reverse[:tT]" / "alpha:A" / "js:B"
    divergence: str = ""
    optimizer: str = "adamw"  # "adamw" or "sgd" (momentum 0.9, no weight decay)
    schedule: str = "cosine"  # "cosine" or "constant" (see TrainConfig)
    total_steps: int = 2000
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0
    temperature: float = 1.0  # sampling temperature
    eval_interval: int = 25
    # "linear" = every eval_interval steps; "log" = ~64 log-spaced eval points
    # over total_steps (dense early, sparse late)
    eval_schedule: str = "linear"
    # Sharpness probe (Hutchinson tr(Hessian)) at finetune eval steps; 0 disables.
    hessian_probes: int = 0
    hessian_batch_size: int = 256
    # Probe only every N steps (plus the step-0 baseline); 0 = every eval step.
    # Lets eval_interval stay small for smooth accuracy curves without paying
    # the Hessian cost at each eval.
    hessian_interval: int = 0
    # Average every trace over this many independent batches (fresh data rows
    # and fresh sampled trajectories per batch) for a lower-variance estimate.
    hessian_num_batches: int = 1


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(
        task_type="biography", batch_size=256, eval_batch_size=256))
    pretrain: TrainConfig = field(default_factory=lambda: TrainConfig(total_steps=8000))
    teacher: TrainConfig = field(default_factory=lambda: TrainConfig(total_steps=4000))
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    # Population split
    num_pretrain_people: int = 8000
    num_finetune_people: int = 1000
    finetune_mix_fraction: float = 0.0  # fraction of B examples in student pretraining batches (stochastically rounded per batch)
    # Stylistic gap: per-dataset probability that a biography is rendered with
    # French templates (values/names stay English). 0 = monolingual English.
    lang_mix_a: float = 0.0
    lang_mix_b: float = 0.0
    # Per-sentence language switching probability (Markov chain over the 6
    # sentences; start language drawn with lang_mix). 0 = per-biography language.
    lang_switch_a: float = 0.0
    lang_switch_b: float = 0.0
    # Convenience for paired sweeps: X = percent ENGLISH in A; if >= 0 sets
    # lang_mix_a = 1 - X/100 and lang_mix_b = X/100 (reversed teacher mix).
    lang_split_x: float = -1.0
    # Paired step scaling for population sweeps (wandb grids can't tie two
    # parameters): scale pretrain.total_steps = pretrain_steps_scale *
    # num_pretrain_people (capped at pretrain_steps_max if > 0) and/or
    # teacher.total_steps = 4 * num_finetune_people, leaving everything else
    # (in particular finetune.total_steps) untouched.
    scale_pretrain_steps: bool = False
    pretrain_steps_scale: float = 1.0
    pretrain_steps_max: int = 0
    scale_teacher_steps: bool = False
    # Which part of the protocol to run: "teacher" / "student" (pretraining,
    # one wandb run each — sweepable), "finetune", "joint" (joint-training
    # baseline: NTP on A ∪ B, evaluated on both), or "all" (full pipeline)
    phase: str = "all"
    # Student pretraining early stop: end once dataA first-token accuracy
    # reaches this at an eval step (0 disables). pretrain.total_steps becomes a
    # max budget; enters the student checkpoint key when set.
    pretrain_target_acc: float = 0.0
    # Checkpointing: teacher/student pretraining are reused across runs whose
    # relevant config matches (keyed by hash); "" disables
    checkpoint_dir: str = "checkpoints"
    # Cap each dataset's eval set to a fixed random subset of people (0 = all);
    # eval-only, does not affect training or checkpoint keys
    max_eval_people: int = 0
    # Auto-scale step budgets from population size (keeps exposures/person fixed
    # across scales, matching the hand-tuned Nx sweeps): pretrain = n_pretrain,
    # teacher = 4*n_finetune, finetune = n_pretrain/4. Lets a single grid sweep
    # population size without per-cell step tuning.
    auto_scale_steps: bool = False
    # Logging
    log_interval: int = 50
    eval_interval: int = 500  # pretraining phases; finetuning uses finetune.eval_interval
    wandb_project: str = "transformer-lab"
    wandb_mode: str = "online"
    seed: int = 42


def evaluate_model(state, datasets, batch_size, step, extra=None):
    """Evaluate on each dataset's full eval set, log to wandb, print a summary.

    extra: optional dict of extra scalars (e.g. sharpness) merged into the log.
    """
    forward = lambda x: eval_forward(state.apply_fn, state.params, x)
    metrics = {}
    for ds in datasets:
        metrics.update(ds.evaluate(forward, batch_size))
    metrics["total_knowledge"] = sum(
        metrics[f"{ds.name}/knowledge"] for ds in datasets if not ds.name.endswith("_p0"))
    if extra:
        metrics.update(extra)
    wandb.log(metrics, step=step)
    print(f"[step {step}] " + ", ".join(
        f"{ds.name}: attr_acc={metrics[f'{ds.name}/attribute_accuracy']:.3f} "
        f"first_acc={metrics[f'{ds.name}/first_token_accuracy']:.3f}" for ds in datasets
    ))
    return metrics


def log_eval_steps(total_steps: int, num_points: int = 64) -> set:
    """~num_points log-spaced eval steps in [1, total_steps] (dense early)."""
    pts = np.unique(np.round(np.logspace(0, np.log10(max(total_steps, 2)), num_points)))
    return set(int(p) for p in pts if 1 <= p <= total_steps)


def train_ntp(state, get_batch, steps, batch_size, eval_datasets, eval_interval, cfg, step0=0,
              sharpness_fn=None, eval_steps=None, stop_when=None):
    """Standard next-token-prediction training loop with logging.

    sharpness_fn: optional (state, gstep) -> dict merged into eval logs.
    eval_steps: optional set of (relative) steps to eval at, overriding eval_interval.
    stop_when: optional (metric_key, target) — stop once the eval metric
      reaches target (steps then acts as a max budget).
    """
    nan_check_interval = max(1, steps // 100)
    for i in range(steps):
        rng, batch_rng = jax.random.split(state.rng)
        state = state.replace(rng=rng)
        state, metrics = train_step(state, get_batch(batch_rng, batch_size), BiographyDataset.loss_fn)
        gstep = step0 + i + 1
        if (i + 1) % nan_check_interval == 0 and jnp.isnan(metrics["loss"]):
            print(f"NaN loss at step {gstep}, stopping.")
            break
        if gstep % cfg.log_interval == 0:
            wandb.log({"train_loss": float(metrics["loss"])}, step=gstep)
        do_eval = (i + 1 in eval_steps) if eval_steps is not None else gstep % eval_interval == 0
        if do_eval or i + 1 == steps:
            extra = sharpness_fn(state, gstep) if sharpness_fn else None
            eval_metrics = evaluate_model(state, eval_datasets, cfg.data.eval_batch_size, gstep, extra)
            if stop_when and eval_metrics.get(stop_when[0], 0.0) >= stop_when[1]:
                print(f"Target {stop_when[0]} >= {stop_when[1]} reached at step {gstep}, stopping.")
                break
    return state


def _fingerprint(**kwargs) -> str:
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_or_train(state, path, train_fn):
    """Load params from a checkpoint if present, else train (in its own wandb run) and save."""
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            state = state.replace(params=serialization.from_bytes(state.params, f.read()))
        print(f"Loaded checkpoint {path}")
        return state
    state = train_fn(state)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(state.params))
        print(f"Saved checkpoint {path}")
    return state


def main():
    cfg = parse_config(ExperimentConfig, description="On-policy distillation CL on biographies")
    print(cfg)
    if cfg.phase not in ("teacher", "student", "finetune", "joint", "all"):
        raise ValueError(f"Unknown phase: {cfg.phase}")
    if cfg.phase == "teacher" and cfg.finetune_mix_fraction > 0:
        # Register an (empty) run so sweep grids count this cell as complete
        print("Teacher is mix-independent; skipping duplicate sweep cell.")
        wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="skipped_duplicate").finish()
        return

    if cfg.auto_scale_steps:
        cfg.pretrain = replace(cfg.pretrain, total_steps=cfg.num_pretrain_people)
        cfg.teacher = replace(cfg.teacher, total_steps=4 * cfg.num_finetune_people)
        cfg.finetune = replace(cfg.finetune, total_steps=max(1, cfg.num_pretrain_people // 4))
        print(f"Auto-scaled steps: pretrain={cfg.pretrain.total_steps}, "
              f"teacher={cfg.teacher.total_steps}, finetune={cfg.finetune.total_steps}")
    if cfg.scale_pretrain_steps:
        steps = int(cfg.pretrain_steps_scale * cfg.num_pretrain_people)
        if cfg.pretrain_steps_max > 0:
            steps = min(steps, cfg.pretrain_steps_max)
        cfg.pretrain = replace(cfg.pretrain, total_steps=steps)
        print(f"Scaled pretrain steps: {cfg.pretrain.total_steps}")
    if cfg.scale_teacher_steps:
        cfg.teacher = replace(cfg.teacher, total_steps=4 * cfg.num_finetune_people)
        print(f"Scaled teacher steps: {cfg.teacher.total_steps}")
    if cfg.lang_split_x >= 0:
        cfg.lang_mix_a = 1.0 - cfg.lang_split_x / 100.0
        cfg.lang_mix_b = cfg.lang_split_x / 100.0
        print(f"Language split X={cfg.lang_split_x}: lang_mix_a={cfg.lang_mix_a}, lang_mix_b={cfg.lang_mix_b}")

    # --- Data: one population, disjoint A (old) / B (new) subsets ---
    num_people = cfg.num_pretrain_people + cfg.num_finetune_people
    population = BiographyPopulation(
        cfg.data.biography_data_path, num_people, seed=cfg.seed,
        support_size=cfg.data.support_size, num_train_templates=cfg.data.num_train_templates,
        use_french=any(x > 0 for x in (cfg.lang_mix_a, cfg.lang_mix_b,
                                       cfg.lang_switch_a, cfg.lang_switch_b)),
    )
    data_a = BiographyDataset(population, np.arange(cfg.num_pretrain_people), name="dataA",
                              seed=cfg.seed + 1, max_eval_people=cfg.max_eval_people,
                              lang_mix=cfg.lang_mix_a, lang_switch=cfg.lang_switch_a)
    data_b = BiographyDataset(population, np.arange(cfg.num_pretrain_people, num_people), name="dataB",
                              seed=cfg.seed + 2, max_eval_people=cfg.max_eval_people,
                              lang_mix=cfg.lang_mix_b, lang_switch=cfg.lang_switch_b)
    # When the teacher's B is rendered with sentence switching, also evaluate B
    # acquisition in the switch-free (p=0) format the student pretrained on.
    # Excluded from total_knowledge (it re-counts dataB's people).
    data_b_p0 = None
    if cfg.lang_switch_b > 0:
        data_b_p0 = BiographyDataset(population, np.arange(cfg.num_pretrain_people, num_people),
                                     name="dataB_p0", seed=cfg.seed + 2,
                                     max_eval_people=cfg.max_eval_people, lang_mix=cfg.lang_mix_b)
    print(f"Vocab: {population.vocab_size}, seq_len: {population.seq_len}, prompt_len: {population.prompt_len}")
    print(f"People: {len(data_a.person_ids)} old (A), {len(data_b.person_ids)} new (B)")

    data_cfg = replace(cfg.data, sequence_length=population.seq_len, vocab_size=population.vocab_size)
    model = create_model(cfg.model, data_cfg)

    def init_state(train_cfg, seed):
        full_cfg = Config(model=cfg.model, data=data_cfg, train=train_cfg, seed=seed)
        return create_train_state(model, full_cfg, jax.random.PRNGKey(seed), None, create_optimizer(train_cfg))

    def wandb_run(phase):
        return wandb.init(
            project=cfg.wandb_project, mode=cfg.wandb_mode, name=phase, config=vars_nested(cfg),
        )

    # Checkpoint keys: everything that affects the trained parameters of each phase
    common_key = dict(
        model=vars_nested(cfg.model), data=vars_nested(cfg.data), seed=cfg.seed,
        people=[cfg.num_pretrain_people, cfg.num_finetune_people],
    )
    # Language rendering enters each phase's key only for the dataset that
    # phase trains on (teacher: B, student: A), and only when nonzero — keeps
    # monolingual hashes stable and lets e.g. one p=0 student serve every
    # lang_switch_b teacher cell.
    lang_key_a = [cfg.lang_mix_a, cfg.lang_switch_a] if cfg.lang_mix_a > 0 or cfg.lang_switch_a > 0 else None
    lang_key_b = [cfg.lang_mix_b, cfg.lang_switch_b] if cfg.lang_mix_b > 0 or cfg.lang_switch_b > 0 else None
    if population.use_french:  # bilingual vocab changes model shapes
        common_key["vocab"] = population.vocab_size

    def ckpt_path(name, **key):
        if not cfg.checkpoint_dir:
            return ""
        return os.path.join(cfg.checkpoint_dir, f"{name}-{_fingerprint(**common_key, **key)}.msgpack")

    # --- joint-training baseline: NTP on A ∪ B (uniform sampling) ---
    if cfg.phase == "joint":
        print("\n=== joint-training baseline: NTP on A ∪ B ===")
        data_all = BiographyDataset(population, np.arange(num_people), name="dataJoint",
                                    seed=cfg.seed + 3, max_eval_people=cfg.max_eval_people)
        joint = init_state(cfg.pretrain, cfg.seed + 60)
        print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(joint.params)):,}")

        def train_joint(state):
            with wandb_run("pretraining_joint"):
                return train_ntp(state, data_all.get_batch, cfg.pretrain.total_steps,
                                 cfg.data.batch_size, [data_a, data_b], cfg.eval_interval, cfg)

        load_or_train(joint, ckpt_path("joint", train=vars_nested(cfg.pretrain)), train_joint)
        return

    # --- pretraining_teacher: NTP on new data (B) ---
    teacher = None
    if cfg.phase in ("teacher", "finetune", "all"):
        print("\n=== pretraining_teacher: NTP on new data (B) ===")
        teacher = init_state(cfg.teacher, cfg.seed + 10)
        print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(teacher.params)):,}")

        def train_teacher(state):
            with wandb_run("pretraining_teacher"):
                return train_ntp(state, data_b.get_batch, cfg.teacher.total_steps, cfg.data.batch_size,
                                 [data_b], cfg.eval_interval, cfg)

        teacher_key = dict(train=vars_nested(cfg.teacher))
        if lang_key_b is not None:
            teacher_key["lang"] = lang_key_b
        teacher = load_or_train(teacher, ckpt_path("teacher", **teacher_key), train_teacher)
        if cfg.phase == "teacher":
            return

    # --- pretraining_student: NTP on old data (A), optional B mixture ---
    print("\n=== pretraining_student: NTP on old data (A) ===")
    if cfg.finetune_mix_fraction > 0:
        exposures = (cfg.finetune_mix_fraction * cfg.data.batch_size * cfg.pretrain.total_steps
                     / cfg.num_finetune_people)
        print(f"B mix: ~{exposures:.2f} expected biographies per B person during pretraining")
    mix_rng = np.random.default_rng(cfg.seed + 40)

    def pretrain_batch(rng, batch_size):
        exact = cfg.finetune_mix_fraction * batch_size
        n_new = int(exact) + int(mix_rng.random() < exact - int(exact))  # stochastic rounding
        batch = data_a.get_batch(rng, batch_size - n_new)
        if n_new == 0:
            return batch
        new = data_b.get_batch(rng, n_new)
        return {k: jnp.concatenate([batch[k], new[k]], axis=0) for k in batch}

    student = init_state(cfg.pretrain, cfg.seed + 20)

    def train_student(state):
        with wandb_run("pretraining_student"):
            stop = (("dataA/first_token_accuracy", cfg.pretrain_target_acc)
                    if cfg.pretrain_target_acc > 0 else None)
            return train_ntp(state, pretrain_batch, cfg.pretrain.total_steps, cfg.data.batch_size,
                             [data_a], cfg.eval_interval, cfg, stop_when=stop)

    student_key = dict(train=vars_nested(cfg.pretrain), mix=cfg.finetune_mix_fraction)
    if cfg.pretrain_target_acc > 0:
        student_key["target_acc"] = cfg.pretrain_target_acc
    if lang_key_a is not None:
        student_key["lang"] = lang_key_a
    if cfg.finetune_mix_fraction > 0 and lang_key_b is not None:  # mixed-in B batches
        student_key["lang_b"] = lang_key_b
    student = load_or_train(student, ckpt_path("student", **student_key), train_student)
    if cfg.phase == "student":
        return

    # --- finetuning_student on new data (B) ---
    ft = cfg.finetune
    print(f"\n=== finetuning_student on B ({ft.method}) ===")
    ft_train_cfg = TrainConfig(
        learning_rate=ft.learning_rate, weight_decay=ft.weight_decay,
        warmup_steps=ft.warmup_steps, total_steps=ft.total_steps, grad_clip=ft.grad_clip,
        schedule=ft.schedule,
    )
    student = TrainState.create(
        apply_fn=model.apply, params=student.params,
        tx=create_optimizer(ft_train_cfg, ft.optimizer), rng=jax.random.PRNGKey(cfg.seed + 30),
    )
    step0 = cfg.pretrain.total_steps
    eval_sets = [data_a, data_b] + ([data_b_p0] if data_b_p0 is not None else [])

    def resolve_spec():
        """Divergence spec tuple: explicit ft.divergence wins over the kl knobs."""
        if ft.divergence:
            return parse_divergence(ft.divergence)
        kl = {"teacher": "forward"}.get(ft.sample_from, "reverse") if ft.kl == "auto" else ft.kl
        return ("kl", kl, 1.0)

    def sharpness_metrics(state, gstep):
        """Sharpness of the retained task (A) and of the arm's own objective (B).

        Parameter-space Hutchinson traces (dataA/hessian_trace,
        finetune/hessian_trace) are probed at step0 and every hessian_interval
        steps (every eval if 0). The exact LOGIT-Hessian trace of the distill
        objective (finetune/logit_hessian_trace) is cheap and logged at every
        eval step. Every trace is averaged over hessian_num_batches independent
        batches (fresh random eval rows + fresh trajectories per batch).
        """
        if ft.hessian_probes <= 0:
            return None
        do_param = gstep == step0 or ft.hessian_interval <= 0 or gstep % ft.hessian_interval == 0
        hb, probes, nb = ft.hessian_batch_size, ft.hessian_probes, max(1, ft.hessian_num_batches)
        base_rng = jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 50), gstep)

        def rows(ds, key):
            n = ds.eval_inputs.shape[0]
            return np.asarray(jax.random.choice(key, n, (min(hb, n),), replace=False))

        def ce_trace(ds, key):
            k_rows, k_probe = jax.random.split(key)
            idx = rows(ds, k_rows)
            mask = (jnp.array(ds.eval_mask[idx]) != 0).astype(jnp.float32)
            return float(ce_hessian_trace(
                model.apply, state.params, jnp.array(ds.eval_inputs[idx]),
                jnp.array(ds.eval_targets[idx]), mask, k_probe, probes))

        out = {}
        if ft.method == "ntp":
            if not do_param:
                return None
            out["finetune/hessian_trace"] = float(np.mean(
                [ce_trace(data_b, jax.random.fold_in(base_rng, 100 + j)) for j in range(nb)]))
        else:
            # The logit trace is logged at every eval step for all divergence
            # specs. The generic autodiff path (alpha/js/temperature) costs a
            # few seconds per batch on H100 — with log-spaced evals the
            # earlier hessian_interval gating left only 2 logged points per
            # run, so the extra cost buys the whole trace-evolution curve.
            logit_traces, param_traces = [], []
            for j in range(nb):
                k_rows, k_gen, k_probe = jax.random.split(jax.random.fold_in(base_rng, 100 + j), 3)
                idx = rows(data_b, k_rows)
                if ft.sample_from == "data":
                    sequences = jnp.concatenate(
                        [jnp.array(data_b.eval_inputs[idx]), jnp.array(data_b.eval_targets[idx][:, -1:])], axis=1)
                else:
                    prompts = jnp.array(data_b.eval_inputs[idx][:, :population.prompt_len])
                    sampler = state if ft.sample_from == "student" else teacher
                    sequences = sample_trajectories(sampler, prompts, population.seq_len, ft.temperature, k_gen)
                pos_mask = jnp.broadcast_to(
                    (jnp.arange(sequences.shape[1] - 1) >= population.prompt_len - 1)[None, :],
                    (sequences.shape[0], sequences.shape[1] - 1)).astype(jnp.float32)
                logit_traces.append(float(kl_logit_hessian_trace(
                    model.apply, state.params, teacher.params, sequences, resolve_spec(), pos_mask)))
                if do_param:
                    param_traces.append(float(kl_hessian_trace(
                        model.apply, state.params, teacher.params, sequences,
                        pos_mask, k_probe, resolve_spec(), probes)))
            out["finetune/logit_hessian_trace"] = float(np.mean(logit_traces))
            if do_param:
                out["finetune/hessian_trace"] = float(np.mean(param_traces))
        if do_param:
            out["dataA/hessian_trace"] = float(np.mean(
                [ce_trace(data_a, jax.random.fold_in(base_rng, 200 + j)) for j in range(nb)]))
        return out

    eval_steps = log_eval_steps(ft.total_steps) if ft.eval_schedule == "log" else None

    with wandb_run("finetuning_student"):
        # baseline before finetuning
        evaluate_model(student, eval_sets, cfg.data.eval_batch_size, step0, sharpness_metrics(student, step0))
        if ft.method == "ntp":
            student = train_ntp(student, data_b.get_batch, ft.total_steps, ft.batch_size,
                                eval_sets, ft.eval_interval, cfg, step0=step0,
                                sharpness_fn=sharpness_metrics, eval_steps=eval_steps)
        elif ft.method == "distill":
            if ft.sample_from not in ("student", "teacher", "data"):
                raise ValueError(f"Unknown sample_from: {ft.sample_from}")
            spec = resolve_spec()
            kl_key = divergence_metric_name(spec)
            for i in range(ft.total_steps):
                rng, gen_rng = jax.random.split(student.rng)
                student = student.replace(rng=rng)
                if ft.sample_from == "data":
                    batch = data_b.get_batch(gen_rng, ft.batch_size)
                    sequences = jnp.concatenate([batch["inputs"], batch["targets"][:, -1:]], axis=1)
                else:
                    prompts, _ = data_b.get_prompts(ft.batch_size)
                    sampler = student if ft.sample_from == "student" else teacher
                    sequences = sample_trajectories(sampler, prompts, population.seq_len, ft.temperature, gen_rng)
                student, metrics = distill_step(
                    student, teacher.params, sequences, population.prompt_len, model.apply, spec
                )
                gstep = step0 + i + 1
                if gstep % cfg.log_interval == 0:
                    wandb.log({kl_key: metrics["kl"]}, step=gstep)
                do_eval = (i + 1 in eval_steps) if eval_steps is not None \
                    else gstep % ft.eval_interval == 0
                if do_eval or i + 1 == ft.total_steps:
                    print(f"[distill step {gstep}] {kl_key}={metrics['kl']:.4f}")
                    evaluate_model(student, eval_sets, cfg.data.eval_batch_size, gstep,
                                   sharpness_metrics(student, gstep))
        else:
            raise ValueError(f"Unknown finetune method: {ft.method}")


if __name__ == "__main__":
    main()
