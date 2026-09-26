"""Training utilities: train state, optimizer, JIT step and eval forward."""
from typing import Any, Callable
from functools import partial
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from src.config import Config, TrainConfig


class TrainState(train_state.TrainState):
    """Training state with additional RNG field."""
    rng: jax.Array


def create_optimizer(train_config: TrainConfig, optimizer: str = "adamw") -> optax.GradientTransformation:
    """Create optimizer with warmup + cosine decay and gradient clipping.

    optimizer: "adamw" (with weight decay) or "sgd" (momentum 0.9, no decay).
    """
    # Warmup, then either cosine decay or a flat constant learning rate.
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=train_config.learning_rate,
        transition_steps=train_config.warmup_steps,
    )
    sched = getattr(train_config, "schedule", "cosine")
    if sched == "constant":
        decay_fn = optax.constant_schedule(train_config.learning_rate)
    elif sched == "cosine":
        decay_fn = optax.cosine_decay_schedule(
            init_value=train_config.learning_rate,
            decay_steps=max(1, train_config.total_steps - train_config.warmup_steps),
        )
    else:
        raise ValueError(f"Unknown schedule: {sched}")
    schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[train_config.warmup_steps],
    )

    if optimizer == "sgd":
        opt = optax.sgd(learning_rate=schedule, momentum=0.9)
    elif optimizer == "adamw":
        opt = optax.adamw(learning_rate=schedule, weight_decay=train_config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    return optax.chain(optax.clip_by_global_norm(train_config.grad_clip), opt)


def create_train_state(
    model: nn.Module,
    config: Config,
    rng: jax.Array,
    dataset: Any,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Initialize parameters and create train state."""
    rng, init_rng = jax.random.split(rng)

    # Create dummy input based on task type
    if config.data.task_type == "biography":
        dummy_input = jnp.ones((1, config.data.sequence_length), dtype=jnp.int32)
    else:
        raise ValueError(f"Unknown task_type: {config.data.task_type}")

    init_rngs = {"params": init_rng, "dropout": init_rng}
    variables = model.init(init_rngs, dummy_input, deterministic=True)
    params = variables["params"]

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng,
    )


@partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    batch: dict,
    loss_fn: Callable,
) -> tuple[TrainState, dict]:
    """Single training step. loss_fn is a static argument for JIT."""
    dropout_rng = jax.random.fold_in(state.rng, state.step)

    def compute_loss(params):
        logits = state.apply_fn(
            {"params": params}, batch["inputs"],
            deterministic=False, rngs={"dropout": dropout_rng},
        )
        return loss_fn(logits, batch["targets"], batch["mask"])

    loss, grads = jax.value_and_grad(compute_loss)(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, {"loss": loss}


@partial(jax.jit, static_argnums=(0,))
def eval_forward(apply_fn, params, inputs):
    """JIT-compiled forward pass for evaluation."""
    return apply_fn({"params": params}, inputs, deterministic=True)
