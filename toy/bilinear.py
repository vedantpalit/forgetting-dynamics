"""Rung 2.5: the toy with a TRAINABLE shared key direction (PLAN.md 0.11).

    k_i(mu) = sqrt(beta) * mu + sqrt(1-beta) * base_i

`base_i` is the frozen part (name-part features plus the private direction). Trainable
parameters are `W` and `mu`; `b` and every base_i stay frozen. Gradients:

    grad_W  = K(mu)^T G
    grad_mu = sqrt(beta) * W @ G.sum(axis=0)          G = (p - y)/n, the logit gradient

`grad_mu` is the new object: a W-weighted average of output errors, so `mu` and `W` drive
each other. That bilinear coupling is what permits non-monotone dynamics, which the frozen
key forbade -- rung 1 showed the frozen-key depression is monotone and saturating for `b`,
for `W` along a fixed `mu`, and (by the same argument) for one shared vocabulary too.

`mu` IS TRAINED IN BOTH PHASES. If it were frozen during pretraining, the pre-injection
state would not be a stationary point of the model under test and part of any injection-time
rotation would be pretraining catch-up rather than the mechanism. Consequently `mu` arrives
at injection already shaped by A and D, so all rotation diagnostics are measured from the
START OF INJECTION, not from the start of pretraining.

The store is separate from train.py's `LinearStore` path on purpose: that path is what the
battery and six diagnostic scripts run on, and this is a different model (bilinear, not
linear). `metrics` and `eval_all` are shared so the readouts cannot drift apart.
"""
import numpy as np

import metrics as m
from train import eval_all


class BilinearStore:
    """W (d, |V|), b (|V|,), mu (d,). Keys are composed from mu and the frozen base."""

    def __init__(self, d, v_total, mu0):
        self.W = np.zeros((d, v_total))
        self.b = np.zeros(v_total)
        self.mu = np.asarray(mu0, dtype=float).copy()

    def keys(self, base, beta):
        return np.sqrt(beta) * self.mu[None, :] + np.sqrt(1.0 - beta) * base

    def logits(self, base, beta):
        return self.keys(base, beta) @ self.W + self.b

    def copy(self):
        out = BilinearStore(self.W.shape[0], self.W.shape[1], self.mu)
        out.W = self.W.copy()
        out.b = self.b.copy()
        return out


def base_of(keys, mu, beta):
    """Recover the frozen part of a key set exactly. beta=0 leaves keys unchanged."""
    if beta == 0.0:
        return keys.copy()
    return (keys - np.sqrt(beta) * mu[None, :]) / np.sqrt(1.0 - beta)


def compute_grad(store, base, correct_idx, v_total, beta, loss="ce"):
    keys = store.keys(base, beta)
    logits = keys @ store.W + store.b
    n = base.shape[0]
    onehot = np.zeros((n, v_total))
    onehot[np.arange(n), correct_idx] = 1.0
    if loss == "ce":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        g = (probs - onehot) / n
        loss_val = float(-np.log(np.clip(probs[np.arange(n), correct_idx], 1e-12, None)).mean())
    elif loss == "mse":
        g = 2.0 * (logits - onehot) / n
        loss_val = float(np.mean((logits - onehot) ** 2))
    else:
        raise ValueError(loss)
    grad_W = keys.T @ g
    grad_b = g.sum(axis=0)
    grad_mu = np.sqrt(beta) * (store.W @ g.sum(axis=0))
    return grad_W, grad_b, grad_mu, loss_val


def pretrain(cfg, ps, verbose=False):
    """A u D, full batch, W and mu both trainable. Same ceiling rule as train.pretrain."""
    from collections import deque
    beta = cfg.embedding.beta
    store = BilinearStore(cfg.embedding.d, cfg.vocab.v_total, ps.mu)
    lr = cfg.optimizer.lr

    base_a = base_of(ps.keys_a, ps.mu, beta)
    parts = [base_a]
    corr = [ps.values_a]
    if ps.keys_d is not None:
        parts.append(base_of(ps.keys_d, ps.mu, beta))
        corr.append(ps.values_d)
    base_all = np.concatenate(parts, axis=0)
    corr_all = np.concatenate(corr, axis=0)

    window = deque(maxlen=cfg.protocol.pretrain_ceiling_window)
    ceiling_step = target_step = None
    for step in range(cfg.protocol.pretrain_max_steps):
        gW, gb, gmu, _ = compute_grad(store, base_all, corr_all, cfg.vocab.v_total, beta, cfg.loss)
        store.W -= lr * gW
        store.b -= lr * gb
        store.mu -= lr * gmu
        acc_a = float(m.argmax_correct(store.logits(base_a, beta), ps.values_a).mean())
        window.append(acc_a)
        if verbose and step % 500 == 0:
            print(f"  [pretrain {step}] acc_a={acc_a:.4f} |mu|={np.linalg.norm(store.mu):.4f}")
        if ceiling_step is None and len(window) == window.maxlen and \
                min(window) >= cfg.protocol.pretrain_ceiling_acc:
            ceiling_step = step - window.maxlen + 1
            target_step = int(np.ceil(ceiling_step * (1 + cfg.protocol.pretrain_extra_fraction)))
        if ceiling_step is not None and step >= target_step:
            break
    else:
        raise RuntimeError(
            f"bilinear pretrain did not reach acc={cfg.protocol.pretrain_ceiling_acc} within "
            f"{cfg.protocol.pretrain_max_steps} steps at lr={lr}, beta={beta}")
    return store, ceiling_step, target_step


class _ReadoutShim:
    """Presents composed keys to `train.eval_all`, which expects a `.logits(keys)` store.

    eval_all is reused rather than reimplemented so the accuracy/rank/touched-split
    readouts cannot drift between the linear and bilinear paths.
    """

    def __init__(self, store):
        self._store = store

    def logits(self, keys):
        return keys @ self._store.W + self._store.b


def _composed(ps, store, beta):
    """A shallow PopulationSet-like view whose key arrays are composed at the CURRENT mu."""
    import copy
    view = copy.copy(ps)
    for name in ("a", "d", "b", "c"):
        k = getattr(ps, f"keys_{name}")
        if k is not None:
            setattr(view, f"keys_{name}",
                    store.keys(base_of(k, ps.mu, beta), beta))
    return view


def finetune(cfg, ps, pretrained_store, verbose=False):
    """B only. `b` FROZEN, `W` and `mu` trainable. Every rotation diagnostic is measured
    from the start of injection."""
    beta = cfg.embedding.beta
    store = pretrained_store.copy()
    lr = cfg.optimizer.inject_lr
    proto = cfg.protocol

    base_a = base_of(ps.keys_a, ps.mu, beta)
    base_b = base_of(ps.keys_b, ps.mu, beta)

    mu0 = store.mu.copy()
    W0 = store.W.copy()
    z00 = store.logits(base_a, beta)

    def const_split():
        """c(t) crossed exactly as the transformer measurement crosses it
        (docs/shift_decomposition.md): representation = the key (via mu), readout = W."""
        k_t = np.sqrt(beta) * store.mu[None, :] + np.sqrt(1.0 - beta) * base_a
        k_0 = np.sqrt(beta) * mu0[None, :] + np.sqrt(1.0 - beta) * base_a
        ztt = k_t @ store.W + store.b
        z0t = k_0 @ store.W + store.b      # readout moved alone
        zt0 = k_t @ W0 + store.b           # representation (key) moved alone
        c_full = (ztt - z00).mean(axis=0)
        c_read = (z0t - z00).mean(axis=0)
        c_rep = (zt0 - z00).mean(axis=0)
        return ztt, c_full, c_read, c_rep, c_full - c_read - c_rep

    curve = []
    step = 0
    cap_truncated = False

    def do_eval(s):
        view = _composed(ps, store, beta)
        rec = eval_all(_ReadoutShim(store), view)
        rec["step"] = s
        ztt, c_full, c_read, c_rep, c_cross = const_split()
        dz = ztt - z00
        for name, cv in (("full", c_full), ("readout", c_read),
                         ("rep", c_rep), ("cross", c_cross)):
            rec[f"c_{name}_norm"] = float(np.linalg.norm(cv))
            a, r = _metrics_on(z00 + cv[None, :], ps)
            rec[f"c_{name}_acc"] = a
            rec[f"c_{name}_rank"] = r
        # vary-only, the counterfactual rung 1's caveat 3 said a full-vocab ||v|| cannot
        # substitute for: A re-evaluated with the constant component removed.
        a_v, r_v = _metrics_on(z00 + (dz - c_full[None, :]), ps)
        rec["vary_acc"], rec["vary_rank"] = a_v, r_v
        rec["c_mean_va"] = float(c_full[ps.v_a_mask].mean())
        d_mu = store.mu - mu0
        rec["d_mu_norm"] = float(np.linalg.norm(d_mu))
        cos = float(store.mu @ mu0 / (np.linalg.norm(store.mu) * np.linalg.norm(mu0)))
        rec["mu_angle_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        rec["mu_norm"] = float(np.linalg.norm(store.mu))
        curve.append(rec)
        if verbose and (s < 20 or s % 200 == 0):
            print(f"  [{s}] a={rec['a_full_acc_mean']:.4f} c={rec['c_full_norm']:.3f} "
                  f"ang={rec['mu_angle_deg']:.2f}deg")

    def _metrics_on(z, ps_):
        return (float(m.argmax_correct(z, ps_.values_a).mean()),
                float(m.rank_of_correct(z, ps_.values_a, ps_.v_a_mask).mean()))

    do_eval(0)
    while step < proto.finetune_max_steps:
        gW, _gb, gmu, _ = compute_grad(store, base_b, ps.values_b, cfg.vocab.v_total,
                                       beta, cfg.loss)
        store.W -= lr * gW
        store.mu -= lr * gmu          # b frozen: its gradient is computed and discarded
        step += 1
        if step <= proto.eval_dense_until or step % proto.eval_every_sparse == 0:
            do_eval(step)
        if len(curve) >= proto.finetune_flat_window:
            recent = curve[-proto.finetune_flat_window:]
            b_ok = recent[-1]["b_full_acc_mean"] >= proto.finetune_min_b_acc
            cs = [r["c_full_norm"] for r in recent]
            accs = [r["a_full_acc_mean"] for r in recent]
            flat = (max(accs) - min(accs) < proto.finetune_flat_acc_range) and \
                   (max(cs) - min(cs) < 0.01 * max(max(cs), 1e-9))
            if b_ok and flat:
                break
    if step >= proto.finetune_max_steps:
        cap_truncated = True
    return curve, cap_truncated, store
