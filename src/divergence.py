"""Token-level divergence families for distillation.

A divergence is specified by a string, parsed once into a hashable spec tuple
(usable as a JIT static arg):

  "kl:forward" / "kl:reverse"        plain KLs (tau = 1)
  "kl:forward:t2" / "kl:reverse:t0.5" temperature-scaled KL: both log-prob
                                      tables re-softmaxed at tau, loss scaled
                                      by tau^2 (standard KD convention)
  "alpha:A"  A in (0,1)               Amari alpha-divergence, normalized so
                                      A->0 recovers reverse KL and A->1
                                      forward KL; A in {0,1} maps to the KLs
  "js:B"     B in (0,1)               generalized Jensen-Shannon
                                      [B KL(p||m) + (1-B) KL(q||m)] / (B(1-B)),
                                      m = B p + (1-B) q; B->0 reverse, B->1
                                      forward; B in {0,1} maps to the KLs

Everywhere p = student, q = teacher. All divergences are >= 0 (Gibbs /
Jensen), so the host-side sign guard in distill_step still applies.
"""
import jax
import jax.numpy as jnp


def parse_divergence(spec: str) -> tuple:
    """Parse a divergence string into a hashable spec tuple.

    Returns ("kl", direction, tau) or ("alpha", a) or ("js", b).
    """
    parts = spec.split(":")
    if parts[0] == "kl":
        if len(parts) < 2 or parts[1] not in ("forward", "reverse"):
            raise ValueError(f"Bad kl spec: {spec}")
        tau = 1.0
        if len(parts) == 3:
            if not parts[2].startswith("t"):
                raise ValueError(f"Bad kl temperature in: {spec}")
            tau = float(parts[2][1:])
        elif len(parts) > 3:
            raise ValueError(f"Bad kl spec: {spec}")
        if tau <= 0:
            raise ValueError(f"Temperature must be > 0: {spec}")
        return ("kl", parts[1], tau)
    if parts[0] in ("alpha", "js"):
        if len(parts) != 2:
            raise ValueError(f"Bad {parts[0]} spec: {spec}")
        v = float(parts[1])
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{parts[0]} parameter must be in [0, 1]: {spec}")
        if v == 0.0:
            return ("kl", "reverse", 1.0)
        if v == 1.0:
            return ("kl", "forward", 1.0)
        return (parts[0], v)
    raise ValueError(f"Unknown divergence family: {spec}")


def divergence_terms(student_logp: jax.Array, teacher_logp: jax.Array, spec: tuple) -> jax.Array:
    """Per-position divergence (..., ) from normalized log-probs (..., V)."""
    kind = spec[0]
    if kind == "kl":
        _, direction, tau = spec
        sp = jax.nn.log_softmax(student_logp / tau, axis=-1)
        tp = jax.nn.log_softmax(teacher_logp / tau, axis=-1)
        if direction == "reverse":
            d = jnp.sum(jnp.exp(sp) * (sp - tp), axis=-1)
        else:
            d = jnp.sum(jnp.exp(tp) * (tp - sp), axis=-1)
        return d * tau**2
    if kind == "alpha":
        a = spec[1]
        return (1.0 - jnp.sum(jnp.exp((1 - a) * student_logp + a * teacher_logp), axis=-1)) / (a * (1 - a))
    if kind == "js":
        b = spec[1]
        m = jnp.logaddexp(jnp.log(b) + student_logp, jnp.log1p(-b) + teacher_logp)
        d = (b * jnp.sum(jnp.exp(student_logp) * (student_logp - m), axis=-1)
             + (1 - b) * jnp.sum(jnp.exp(teacher_logp) * (teacher_logp - m), axis=-1))
        return d / (b * (1 - b))
    raise ValueError(f"Unknown divergence spec: {spec}")


def divergence_loss(student_logp: jax.Array, teacher_logp: jax.Array, spec: tuple,
                    position_mask: jax.Array) -> jax.Array:
    """Position-mask-averaged divergence, matching the KL loss normalization."""
    d = divergence_terms(student_logp, teacher_logp, spec)
    return jnp.sum(d * position_mask) / jnp.maximum(jnp.sum(position_mask), 1)


def divergence_metric_name(spec: tuple) -> str:
    """Legacy metric key for plain KLs, generic otherwise."""
    if spec[0] == "kl" and spec[2] == 1.0:
        return f"{spec[1]}_kl"
    return "divergence"
