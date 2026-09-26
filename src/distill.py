"""On-policy distillation: sample from the student, token-level divergence to the teacher."""
from functools import partial

import jax
import jax.numpy as jnp

from src.divergence import divergence_loss, parse_divergence
from src.model.inference import generate
from src.train import TrainState


def sample_trajectories(
    student_state: TrainState,
    prompts: jax.Array,
    seq_len: int,
    temperature: float,
    rng: jax.Array,
) -> jax.Array:
    """Sample completions from the student given prompts. Returns (B, seq_len)."""
    return generate(
        student_state,
        prompts,
        max_new_tokens=seq_len - prompts.shape[1],
        temperature=temperature,
        rng_key=rng,
    )


def distill_step(
    student_state: TrainState,
    teacher_params: dict,
    sequences: jax.Array,
    prompt_len: int,
    teacher_apply_fn,
    divergence: str | tuple = "kl:reverse",
) -> tuple[TrainState, dict]:
    """One distillation step over the full vocabulary.

    divergence is a spec string (see src.divergence) or a pre-parsed spec
    tuple. "kl:reverse" minimizes KL(student || teacher) (mode-seeking, the
    on-policy choice); "kl:forward" minimizes KL(teacher || student)
    (mass-covering, the off-policy choice); alpha/js/temperature specs
    interpolate between the two. The loss is computed at every position of the
    sampled continuation (label positions >= prompt_len - 1); the prompt itself
    is excluded. The returned value is validated host-side: jax-metal is
    rejected outright, and a negative or NaN loss aborts (all supported
    divergences are >= 0).
    """
    spec = parse_divergence(divergence) if isinstance(divergence, str) else divergence
    if jax.default_backend().lower() == "metal":
        raise RuntimeError(
            "jax-metal miscompiles distill_step (flips the loss/gradient sign). "
            "Run locally with JAX_PLATFORMS=cpu; CUDA is unaffected."
        )
    gen_mask = jnp.broadcast_to(
        (jnp.arange(sequences.shape[1] - 1) >= prompt_len - 1)[None, :],
        (sequences.shape[0], sequences.shape[1] - 1),
    )
    new_state, metrics = _distill_step_jit(
        student_state, teacher_params, sequences, gen_mask, teacher_apply_fn, spec
    )
    kl = float(metrics["kl"])
    if not (kl > -1e-4):  # catches negative divergence and NaN
        raise RuntimeError(f"Divergence is {kl}: divergence or backend miscompilation.")
    return new_state, {"kl": kl}


@partial(jax.jit, static_argnums=(4, 5))
def _distill_step_jit(
    student_state: TrainState,
    teacher_params: dict,
    sequences: jax.Array,
    position_mask: jax.Array,
    teacher_apply_fn,
    spec: tuple,
) -> tuple[TrainState, dict]:
    dropout_rng = jax.random.fold_in(student_state.rng, student_state.step)
    inputs = sequences[:, :-1]

    teacher_logits = teacher_apply_fn({"params": teacher_params}, inputs, deterministic=True)
    teacher_logp = jax.nn.log_softmax(teacher_logits, axis=-1)

    def compute_loss(params):
        student_logits = student_state.apply_fn(
            {"params": params}, inputs,
            deterministic=False, rngs={"dropout": dropout_rng},
        )
        student_logp = jax.nn.log_softmax(student_logits, axis=-1)
        return divergence_loss(student_logp, teacher_logp, spec, position_mask)

    loss, grads = jax.value_and_grad(compute_loss)(student_state.params)
    new_state = student_state.apply_gradients(grads=grads)
    return new_state, {"kl": loss}
