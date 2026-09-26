"""Writes toy_final/k1/toy_colab.ipynb: the K=1 toy, self-contained (JAX + matplotlib only).

Mirrors k1.py exactly -- same key construction, store, readouts, SGD, protocol -- but in one
file with no imports from the repo, so it can be opened in Colab and modified freely.
Run this script to regenerate the notebook; the notebook's cells are also executed here as
a smoke test (python make_colab.py --test).
"""
import json
import sys

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# Suppression, recovery, erosion in a one-layer associative memory

The smallest model that reproduces the whole curve: an associative store with a shared key
component, one parameter-free normalization in front of the readout, and a softmax.

* keys $k_i = \\sqrt{\\beta}\\,\\mu + \\sqrt{1-\\beta}\\,g_i$ — $\\mu$ shared by every individual,
  $g_i$ random on the sphere, $\\beta$ = expected cosine between any two keys (the relatedness knob)
* store $h = k + \\gamma\\sqrt{d}\\,kW$ — linear in $W$
* readout $z = \\mathrm{rms}(h)\\,U$ (normalized) or $z = hU$ (linear), $\\mathrm{rms}(x) = \\sqrt{d}\\,x/\\|x\\|$
* softmax cross-entropy, plain SGD

Three populations, as in the transformer experiments: **A** (pretrained, values in half X of
the vocabulary — the population we measure), **D** (pretrained, values in half Y, never
measured), **B** (injected after pretraining, values in half Y).

Protocol: pretrain on A ∪ D until both reach 0.99, then train on B alone at a lower rate.
The readout's rate during injection is scaled down so the unembedding barely moves (as in the
transformer, <1%); the result does not change with the readout frozen.
""")

code("""
import numpy as np, jax, jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

D, V = 128, 32            # key dimension, vocabulary size (values 0..V/2-1 = half X, rest = half Y)
NA = ND = NB = 50         # individuals per population
BETA = 0.5                # shared key fraction: expected cosine between any two keys
GAIN = 1.0                # gamma: size of the store's write relative to the key
U_SCALE = 0.01            # readout learning-rate multiplier during injection (transformer: ~0.5%)
INJECT_RATIO = 13.33      # pretrain lr / inject lr, the transformer's ratio
X, Y = np.arange(0, V // 2), np.arange(V // 2, V)


def build(seed, beta=BETA):
    r = np.random.default_rng(seed)
    mu = r.normal(size=D); mu /= np.linalg.norm(mu)
    def keys(n):
        g = r.normal(size=(n, D)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return np.sqrt(beta) * mu[None] + np.sqrt(1 - beta) * g
    return mu, (keys(NA), r.choice(X, NA)), (keys(ND), r.choice(Y, ND)), (keys(NB), r.choice(Y, NB))


def rms(h):
    return h / jnp.linalg.norm(h, axis=-1, keepdims=True) * jnp.sqrt(D)


def fwd(p, k, norm):
    h = k + GAIN * jnp.sqrt(D) * (k @ p["W"])
    z = (rms(h) if norm else h) @ p["U"]
    return z, h


def loss_fn(p, k, v, norm):
    z, _ = fwd(p, k, norm)
    return -jnp.mean(jax.nn.log_softmax(z, -1)[jnp.arange(k.shape[0]), v])


@jax.jit
def step_norm(p, k, v, lr, u_scale):
    g = jax.grad(loss_fn)(p, k, v, True)
    return {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]}


@jax.jit
def step_lin(p, k, v, lr, u_scale):
    g = jax.grad(loss_fn)(p, k, v, False)
    return {"W": p["W"] - lr * g["W"], "U": p["U"] - lr * u_scale * g["U"]}


def accuracy(p, k, v, norm):
    z = np.asarray(fwd(p, jnp.asarray(k), norm)[0])
    return float((z.argmax(-1) == v).mean())


def init(seed):
    r = np.random.default_rng(seed)
    return {"W": jnp.asarray(r.normal(size=(D, D)) * 0.02 / np.sqrt(D)),
            "U": jnp.asarray(r.normal(size=(D, V)) / np.sqrt(D))}
""")

md("""
## Pretrain on A ∪ D, then inject B

`pretrain` probes a few learning rates and stops at the first that brings both A and D to
0.99. `inject` trains on B alone and records A's and B's accuracy on a step grid.
""")

code("""
def pretrain(seed, norm, lrs=(0.5, 0.3, 0.2, 0.1, 0.05), cap=20000, gate=0.99):
    mu, A, Dd, B = build(1000 + seed)
    kp = np.concatenate([A[0], Dd[0]]); vp = np.concatenate([A[1], Dd[1]])
    step = step_norm if norm else step_lin
    for lr in lrs:
        rng = np.random.default_rng(seed); p = init(seed)
        for t in range(cap):
            i = rng.integers(0, len(kp), 32)
            p = step(p, jnp.asarray(kp[i]), jnp.asarray(vp[i]), lr, 1.0)
            if t % 100 == 0 and min(accuracy(p, *A, norm), accuracy(p, *Dd, norm)) >= gate:
                return p, lr, mu, A, Dd, B
    raise RuntimeError("pretraining never reached the gate")


def grid(steps):
    g = set(range(0, 201, 5)) | {int(x) for x in np.unique(np.round(np.logspace(np.log10(200), np.log10(steps), 60)))}
    return sorted(s for s in g if s <= steps)


def inject(p, lr, A, B, norm, steps=5000, seed=0, u_scale=U_SCALE, ilr=None):
    step = step_norm if norm else step_lin
    ilr = lr / INJECT_RATIO if ilr is None else ilr
    rng = np.random.default_rng(seed + 5); gi = set(grid(steps)); rows = []
    for t in range(steps + 1):
        if t in gi:
            rows.append((t, accuracy(p, *A, norm), accuracy(p, *B, norm)))
        i = rng.integers(0, NB, 32)
        p = step(p, jnp.asarray(B[0][i]), jnp.asarray(B[1][i]), ilr, u_scale)
    return np.array(rows), p


def run(seed, norm, steps=5000):
    p, lr, mu, A, Dd, B = pretrain(seed, norm)
    rows, _ = inject(p, lr, A, B, norm, steps, seed)
    return rows
""")

md("""
## The result: one normalizer is the difference between collapse and recovery

Left: linear readout — A falls as B is learned and never returns. Right: normalized readout —
A falls on the same clock, turns over while B is still being acquired, and recovers to a
plateau while B stays at ceiling. (Three seeds; ~1 minute on CPU.)
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, norm in zip(axes, (False, True)):
    R = np.array([run(s, norm) for s in range(3)])
    st = R[0, :, 0]
    for j, (col, lab) in enumerate(((("#1B2A4E", "A (old)")), ("#C4245F", "B (new)"))):
        m, sd = R[:, :, j + 1].mean(0), R[:, :, j + 1].std(0)
        ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.18, lw=0); ax.plot(st, m, color=col, lw=2, label=lab)
    ax.set_xscale("symlog", linthresh=10); ax.set_ylim(0, 1.03)
    ax.set_title("linear readout  z = hU" if not norm else "normalized readout  z = rms(h)U")
    ax.set_xlabel("injection step"); ax.set_ylabel("first-token accuracy"); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

md("""
## Why: the normalizer's term has the sign of B's confidence

With $z = \\mathrm{rms}(h)U$ the error signal of injected individual $b$ splits exactly into the
linear arm's term plus one more, $\\frac{M_b}{n_B\\|h_b\\|}\\hat h_b$, where
$M_b = z_{b,y_b} - \\sum_v p_{b,v} z_{b,v}$ is $b$'s own confidence margin. Projected on the
shared write, that term *builds* the shift while $B$ is wrong ($M_b<0$) and *withdraws* it
once $B$ is right. Below: A's accuracy, the shared write's between-half content
$\\langle \\mu^\\top \\Delta W,\\ \\bar u_Y - \\bar u_X\\rangle$, and B's mean margin, on one time axis.
""")

code("""
def track(seed, steps=400):
    p0, lr, mu, A, Dd, B = pretrain(seed, True)
    W0 = np.asarray(p0["W"]); U0 = np.asarray(p0["U"])
    w = U0[:, Y].mean(1) - U0[:, X].mean(1); w /= np.linalg.norm(w)
    rng = np.random.default_rng(seed + 5); p = p0; ilr = lr / INJECT_RATIO; out = []
    for t in range(steps + 1):
        if t % 5 == 0:
            s = mu @ (np.asarray(p["W"]) - W0)                       # the shared write
            z, _ = fwd(p, jnp.asarray(B[0]), True); z = np.asarray(z)
            pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            M = z[np.arange(NB), B[1]] - (pr * z).sum(1)             # B's margins
            out.append((t, accuracy(p, *A, True), accuracy(p, *B, True), s @ w, M.mean()))
        i = rng.integers(0, NB, 32)
        p = step_norm(p, jnp.asarray(B[0][i]), jnp.asarray(B[1][i]), ilr, U_SCALE)
    return np.array(out)

T = np.array([track(s) for s in range(3)]); st = T[0, :, 0]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, j, lab, col in zip(axes, (1, 3, 4), ("A accuracy", "shared write, between-half content", "B's mean margin"),
                           ("#1B2A4E", "#2F8C7D", "#1E5E6B")):
    m, sd = T[:, :, j].mean(0), T[:, :, j].std(0)
    ax.fill_between(st, m - sd, m + sd, color=col, alpha=0.15, lw=0); ax.plot(st, m, color=col, lw=2)
    if j == 1: ax.plot(st, T[:, :, 2].mean(0), color="0.6", ls=":", lw=1.6, label="B accuracy"); ax.legend(frameon=False)
    if j == 4: ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(lab); ax.set_xlabel("injection step")
cross = st[np.argmax(T[:, :, 4].mean(0) >= 0)]
for ax in axes: ax.axvline(cross, color="0.35", ls="--", lw=1)
plt.tight_layout(); plt.show()
print(f"B's mean margin crosses zero at step {cross:g}; the write turns over there.")
""")

md("""
## The weight: it grows the whole time, but only its coherent part turns over

The store's change splits exactly into a coherent row along the shared key direction and an
incoherent remainder, $\\|\\Delta W\\|^2 = \\|s\\|^2 + \\|\\Delta W_\\perp\\|^2$ with
$s = \\mu^\\top\\Delta W$ and $\\Delta W_\\perp = \\Delta W - \\mu s^\\top$. The total never
decreases. What is withdrawn during recovery is $\\|s\\|$; $\\|\\Delta W_\\perp\\|$ is the
erosion and has no force acting against it.
""")

code("""
def weight_track(seed, steps=2000):
    p0, lr, mu, A, Dd, B = pretrain(seed, True)
    W0 = np.asarray(p0["W"]); rng = np.random.default_rng(seed + 5); p = p0; ilr = lr / INJECT_RATIO
    gi = set(grid(steps)); out = []
    for t in range(steps + 1):
        if t in gi:
            dW = np.asarray(p["W"]) - W0; s = mu @ dW
            out.append((t, accuracy(p, *A, True), np.linalg.norm(dW), np.linalg.norm(s),
                        np.linalg.norm(dW - np.outer(mu, s))))
        i = rng.integers(0, NB, 32)
        p = step_norm(p, jnp.asarray(B[0][i]), jnp.asarray(B[1][i]), ilr, U_SCALE)
    return np.array(out)

Wt = np.array([weight_track(s) for s in range(3)]); st = Wt[0, :, 0]
fig, axes = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
m, sd = Wt[:, :, 1].mean(0), Wt[:, :, 1].std(0)
axes[0].fill_between(st, m - sd, m + sd, color="#1B2A4E", alpha=0.15, lw=0); axes[0].plot(st, m, color="#1B2A4E", lw=2)
axes[0].set_ylabel("A accuracy"); axes[0].set_ylim(0, 1.04)
for j, col, lab in ((2, "0.55", r"$\\|\\Delta W\\|$ (total)"), (3, "#2F8C7D", r"$\\|s\\|$ (coherent, shared row)"),
                    (4, "#8A8C30", r"$\\|\\Delta W_\\perp\\|$ (incoherent, erosion)")):
    m, sd = Wt[:, :, j].mean(0), Wt[:, :, j].std(0)
    axes[1].fill_between(st, m - sd, m + sd, color=col, alpha=0.12, lw=0); axes[1].plot(st, m, color=col, lw=2, label=lab)
axes[1].set_xscale("symlog", linthresh=10); axes[1].set_xlabel("injection step"); axes[1].set_ylabel("weight-change norm")
axes[1].legend(frameon=False, fontsize=9); plt.tight_layout(); plt.show()
tot = Wt[:, :, 2]; print("||dW|| decreasing at", f"{100 * (np.diff(tot, axis=1) < 0).mean():.1f}% of checkpoints")
""")

md("""
## Knobs to try

* `BETA` — relatedness. 0: nothing happens; 0.5: crash and recovery; 0.9: full collapse and a large recovery.
* `D` — width at fixed population. 32: overload, collapse without return; 512: barely a crash.
* `GAIN` — the write's size relative to the key. Too small (e.g. `1/np.sqrt(D)`) and nothing reverses; too large and it collapses.
* injection rate (`ilr=` in `inject`) — moves the trough in time, not depth.
* `U_SCALE` — 0.0 freezes the readout; the curve is unchanged.

Change a constant in the first cell and re-run the result cell. (`D` must be set before the
jitted functions are first traced — restart the runtime when changing it.)
""")

code("""
# Example: the dose sweep -- the rate sets WHEN the crash happens, not how deep it is.
p, lr, mu, A, Dd, B = pretrain(0, True)
plt.figure(figsize=(5.5, 3.6))
for mult, col in zip((0.25, 1.0, 4.0), ("#B9E3CB", "#2F8C7D", "#1B2A4E")):
    rows, _ = inject(p, lr, A, B, True, 5000, 0, ilr=lr / INJECT_RATIO * mult)
    plt.plot(rows[:, 0], rows[:, 1], color=col, lw=2, label=f"rate x{mult:g}: trough {rows[:, 1].min():.2f}")
plt.xscale("symlog", linthresh=10); plt.ylim(0, 1.03); plt.xlabel("injection step"); plt.ylabel("A accuracy")
plt.legend(frameon=False); plt.tight_layout(); plt.show()
""")


def main():
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = __file__.replace("make_colab.py", "toy_colab.ipynb")
    json.dump(nb, open(path, "w"), indent=1)
    print(f"wrote {path}")
    if "--test" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        src = "\n".join(c["source"] for c in CELLS if c["cell_type"] == "code")
        src = src.replace("plt.show()", "pass")
        exec(compile(src, "toy_colab", "exec"), {"__name__": "__main__"})
        print("smoke test: all cells ran")


if __name__ == "__main__":
    main()
