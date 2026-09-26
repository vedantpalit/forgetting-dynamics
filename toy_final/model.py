"""The residual stack of K linear stores (PLAN.md §3), in JAX so the gradient is exact.

    h_0     = k
    h_{j+1} = h_j + h_j @ W_j          W_j: (d, d)
    logits  = h_K @ U                  U:   (d, |V|), TRAINABLE, no bias

K = 1 is `toy/`'s store with the readout factored out. K > 1 is the experiment.

TWO DESIGN FACTS THAT ARE EASY TO GET WRONG.

1. W_j must NOT be zero-initialised. With W_j = 0 every block sees the same input h_0, receives
   the identical gradient, and the K blocks stay identical forever -- their contributions P_j
   would have cosine 1 at every step and de-coherence would be impossible by construction, not
   by result. Small random init breaks the symmetry; pretraining then differentiates the blocks.

2. U is trainable, per the decision in PLAN.md §11. Its movement is logged so the readout's
   share can be read rather than assumed. There is no bias: `toy/` showed a free bias is where a
   store puts a collective shift if it can, and the transformer showed the bias does not carry
   it (FINDINGS §3.15), so it is removed rather than measured.

The stack is LINEAR in h_0 for every K -- a product of matrices is one matrix -- so the
function class does not change with K. Anything that differs between K = 1 and K > 1 is a
property of the parameterisation and its gradient dynamics.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def init(K, d, n_vals, seed, w_scale=0.02):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.normal(size=(K, d, d)) * w_scale / np.sqrt(d))
    U = jnp.asarray(rng.normal(size=(d, n_vals)) / np.sqrt(d))
    return {"W": W, "U": U}


def forward(params, keys):
    """Returns logits (n, V) and the per-block contributions P (K, n, d) with
    h_K = h_0 + sum_j P_j exactly."""
    h = keys
    P = []
    for j in range(params["W"].shape[0]):
        p = h @ params["W"][j]
        P.append(p)
        h = h + p
    return h @ params["U"], jnp.stack(P), h


def loss_fn(params, keys, vals):
    logits, _, _ = forward(params, keys)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(logp[jnp.arange(keys.shape[0]), vals])


@partial(jax.jit, static_argnames=("fix_U",))
def sgd_step(params, keys, vals, lr, fix_U=False):
    """fix_U zeroes the readout's gradient. Used during INJECTION only, as the control that
    isolates the representation channel: with U free and no weight decay, cross-entropy keeps
    growing B's margins after B saturates, U's Y-rows keep growing, and A's X-rows lose by
    softmax competition -- a readout-carried crash with ||delta|| ~ 0 on the representation,
    the route the transformer measured not to occur (rows move 0.51%)."""
    g = jax.grad(loss_fn)(params, keys, vals)
    if fix_U:
        g = {**g, "U": jnp.zeros_like(g["U"])}
    return jax.tree_util.tree_map(lambda p, gg: p - lr * gg, params, g)


@jax.jit
def logits_of(params, keys):
    return forward(params, keys)[0]


def accuracy(params, keys, vals, restrict=None):
    """Full-vocabulary argmax accuracy, or own-half top-1 rate if `restrict` (ids) is given."""
    z = np.asarray(logits_of(params, keys))
    if restrict is not None:
        mask = np.full(z.shape[1], -np.inf)
        mask[restrict] = 0.0
        z = z + mask[None, :]
    return float((z.argmax(-1) == vals).mean())
