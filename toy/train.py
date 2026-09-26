"""Full-batch pretrain and fine-tune loops (PLAN.md §4.8, §4.7, Protocol).
Stage 1 only (frozen embeddings, keys precomputed by populations.py).
"""

from collections import deque

import numpy as np

import metrics as m
from model import LinearStore


def compute_grad(store, keys, correct_idx, v_total, loss):
    logits = store.logits(keys)
    n = keys.shape[0]
    onehot = np.zeros((n, v_total))
    onehot[np.arange(n), correct_idx] = 1.0
    if loss == "ce":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad_logits = (probs - onehot) / n
        loss_val = float(-np.log(np.clip(probs[np.arange(n), correct_idx], 1e-12, None)).mean())
    elif loss == "mse":
        grad_logits = 2.0 * (logits - onehot) / n
        loss_val = float(np.mean((logits - onehot) ** 2))
    else:
        raise ValueError(loss)
    grad_W = keys.T @ grad_logits
    grad_b = grad_logits.sum(axis=0)
    return grad_W, grad_b, loss_val


class Optimizer:
    """sgd: plain full-batch gradient descent (PLAN.md §4.8's primary).
    adam: standard Adam, robustness variant only.
    sgd_minibatch: shuffled minibatch SGD, robustness variant only — the
    caller is responsible for chunking (see `finetune`'s minibatch path);
    this class's `step` always applies one already-computed gradient."""

    def __init__(self, kind, lr):
        self.kind = kind
        self.lr = lr
        self.t = 0
        self.mW = self.vW = self.mb = self.vb = None

    def step(self, store, grad_W, grad_b):
        self.t += 1
        if self.kind in ("sgd", "sgd_minibatch"):
            store.W -= self.lr * grad_W
            store.b -= self.lr * grad_b
        elif self.kind == "adam":
            b1, b2, eps = 0.9, 0.999, 1e-8
            if self.mW is None:
                self.mW = np.zeros_like(store.W)
                self.vW = np.zeros_like(store.W)
                self.mb = np.zeros_like(store.b)
                self.vb = np.zeros_like(store.b)
            self.mW = b1 * self.mW + (1 - b1) * grad_W
            self.vW = b2 * self.vW + (1 - b2) * grad_W ** 2
            self.mb = b1 * self.mb + (1 - b1) * grad_b
            self.vb = b2 * self.vb + (1 - b2) * grad_b ** 2
            mW_hat = self.mW / (1 - b1 ** self.t)
            vW_hat = self.vW / (1 - b2 ** self.t)
            mb_hat = self.mb / (1 - b1 ** self.t)
            vb_hat = self.vb / (1 - b2 ** self.t)
            store.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            store.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
        else:
            raise ValueError(self.kind)


def pretrain(cfg, ps, verbose=False):
    """Trains on A ∪ D (full-batch = every individual, every step, i.e. the
    full-batch equivalent of 'uniform sampling'). Ceiling detection and the
    fixed 25%-extra-steps extension per PLAN.md §6 item 8."""
    store = LinearStore(cfg.embedding.d, cfg.vocab.v_total)
    opt = Optimizer(cfg.optimizer.optimizer, cfg.optimizer.lr)

    keys_list = [ps.keys_a]
    correct_list = [ps.values_a]
    if ps.keys_d is not None:
        keys_list.append(ps.keys_d)
        correct_list.append(ps.values_d)
    keys = np.concatenate(keys_list, axis=0)
    correct = np.concatenate(correct_list, axis=0)

    window = deque(maxlen=cfg.protocol.pretrain_ceiling_window)
    ceiling_step = None
    target_step = None
    history = []
    checkpoints = {}

    for step in range(cfg.protocol.pretrain_max_steps):
        grad_W, grad_b, loss_val = compute_grad(store, keys, correct, cfg.vocab.v_total, cfg.loss)
        opt.step(store, grad_W, grad_b)

        logits_a = store.logits(ps.keys_a)
        acc_a = float(m.argmax_correct(logits_a, ps.values_a).mean())
        acc_d = None
        if ps.keys_d is not None:
            logits_d = store.logits(ps.keys_d)
            acc_d = float(m.argmax_correct(logits_d, ps.values_d).mean())
        window.append(acc_a)
        history.append({"step": step, "acc_a": acc_a, "acc_d": acc_d, "loss": loss_val})

        if verbose and step % 500 == 0:
            print(f"  [pretrain step {step}] acc_a={acc_a:.4f} acc_d={acc_d} loss={loss_val:.4f}")

        if ceiling_step is None and len(window) == window.maxlen and \
                min(window) >= cfg.protocol.pretrain_ceiling_acc:
            ceiling_step = step - window.maxlen + 1
            target_step = int(np.ceil(ceiling_step * (1 + cfg.protocol.pretrain_extra_fraction)))
            if verbose:
                print(f"  ceiling reached at step {ceiling_step}, "
                      f"continuing to step {target_step}")

        if ceiling_step is not None and step >= target_step:
            break
    else:
        raise RuntimeError(
            f"pretrain did not reach ceiling acc={cfg.protocol.pretrain_ceiling_acc} "
            f"within pretrain_max_steps={cfg.protocol.pretrain_max_steps} at "
            f"lr={cfg.optimizer.lr}, optimizer={cfg.optimizer.optimizer}"
        )

    return store, history, ceiling_step, target_step


def _eval_schedule(protocol):
    steps = list(range(0, protocol.eval_dense_until, protocol.eval_every_dense))
    steps.append(protocol.eval_dense_until)
    return steps  # sparse continuation is handled live in `finetune`, since it depends on the stop condition


def eval_all(store, ps):
    out = {}
    logits_a = store.logits(ps.keys_a)
    full_acc_a = m.argmax_correct(logits_a, ps.values_a)
    restr_acc_a = m.argmax_correct(logits_a, ps.values_a, ps.v_a_mask)
    full_rank_a = m.rank_of_correct(logits_a, ps.values_a)
    restr_rank_a = m.rank_of_correct(logits_a, ps.values_a, ps.v_a_mask)
    margin_a = m.margin(logits_a, ps.values_a, ps.v_b_mask)
    prob_a = m.softmax_prob_correct(logits_a, ps.values_a)

    split_restr_acc = m.touched_untouched_split(restr_acc_a.astype(float), ps.touched_a)
    split_restr_rank = m.touched_untouched_split(restr_rank_a, ps.touched_a)

    out["a_full_acc_mean"] = float(full_acc_a.mean())
    out["a_full_acc_std"] = float(full_acc_a.std())
    out["a_restr_acc_mean"] = float(restr_acc_a.mean())
    out["a_restr_acc_std"] = float(restr_acc_a.std())
    out["a_full_rank_mean"] = float(full_rank_a.mean())
    out["a_restr_rank_mean"] = float(restr_rank_a.mean())
    out["a_restr_rank_std"] = float(restr_rank_a.std())
    out["a_margin_mean"] = float(margin_a.mean())
    out["a_prob_correct_p05"] = float(np.percentile(prob_a, 5))
    out["a_prob_correct_p50"] = float(np.percentile(prob_a, 50))
    out["a_prob_correct_p95"] = float(np.percentile(prob_a, 95))
    out["a_restr_rank_touched_mean"] = split_restr_rank["touched"]["mean"]
    out["a_restr_rank_untouched_mean"] = split_restr_rank["untouched"]["mean"]
    out["a_restr_acc_touched_mean"] = split_restr_acc["touched"]["mean"]
    out["a_restr_acc_untouched_mean"] = split_restr_acc["untouched"]["mean"]

    logits_b = store.logits(ps.keys_b)
    full_acc_b = m.argmax_correct(logits_b, ps.values_b)
    out["b_full_acc_mean"] = float(full_acc_b.mean())

    if ps.keys_d is not None:
        logits_d = store.logits(ps.keys_d)
        full_acc_d = m.argmax_correct(logits_d, ps.values_d)
        out["d_full_acc_mean"] = float(full_acc_d.mean())

    logits_c = store.logits(ps.keys_c)
    conf_c = m.mean_max_softmax_confidence(logits_c)
    out["c_confidence_mean"] = float(conf_c.mean())

    return out


def finetune(cfg, ps, pretrained_store, verbose=False, probe_fn=None, freeze_bias=False):
    """Trains on B only. Full-batch by default; minibatch when
    cfg.optimizer.optimizer == 'sgd_minibatch' (PLAN.md §4.8 addition,
    robustness variant only). Eval schedule: dense (every step) through
    eval_dense_until, then every eval_every_sparse steps, until the flat
    window + B-ceiling stop condition (PLAN.md §6 item 9) or the hard cap —
    cap hits are flagged, never silently treated as converged.

    `probe_fn(step, store, W0, b0) -> dict` is an optional diagnostic hook; whatever it
    returns is merged into that step's record. It exists so the update-decomposition
    diagnostic can read intermediate states without train.py growing diagnostic-specific
    code, and is None everywhere in the battery.

    `freeze_bias` holds b at its pretrained value by zeroing its gradient. This is a
    DIAGNOSTIC ABLATION ONLY (removing a degree of freedom the transformer does not have
    in the same form: a per-value bias there must be built out of the same store that
    holds the facts). It is deliberately a function argument rather than a config field,
    so it cannot appear in a battery condition or a condition_id."""
    store = pretrained_store.copy()
    W0 = pretrained_store.W.copy()
    b0 = pretrained_store.b.copy()
    # Injection runs at lr / inject_lr_ratio, NOT the pretraining lr (PLAN.md §0.4b).
    opt = Optimizer(cfg.optimizer.optimizer, cfg.optimizer.inject_lr)

    # Baseline for the s_eff measurement: A's pre-injection logits and softmax. s_eff is
    # the coefficient of the probability-weighted component of A's realized logit shift,
    #     s_eff(a, t) = -<dz_a(t), p_a(0)> / ||p_a(0)||^2,
    # which is directly comparable to the critical push s*_a from
    # feasibility.critical_push_per_individual. Their ratio says which side of Addition
    # 3's transition training is operating on, and with what margin.
    logits_a0 = store.logits(ps.keys_a)
    _m = logits_a0.max(axis=1, keepdims=True)
    p_a0 = np.exp(logits_a0 - _m)
    p_a0 /= p_a0.sum(axis=1, keepdims=True)
    p_a0_sq = (p_a0 ** 2).sum(axis=1)
    proto = cfg.protocol
    rng = np.random.default_rng(cfg.split_seeds()[1])

    curve = []
    step = 0
    cap_truncated = False

    def do_eval(s):
        rec = eval_all(store, ps)
        rec["step"] = s
        dz = store.logits(ps.keys_a) - logits_a0
        s_eff = -(dz * p_a0).sum(axis=1) / p_a0_sq
        rec["s_eff_mean"] = float(s_eff.mean())
        rec["s_eff_p95"] = float(np.percentile(s_eff, 95))
        rec["s_eff_max"] = float(s_eff.max())
        # Per-individual, kept because s* is per-individual too: the supercritical
        # FRACTION is the quantity that matters, and a mean-vs-median comparison would
        # hide a supercritical tail (PLAN.md §0.4c).
        rec["s_eff_per_individual"] = s_eff.copy()
        # Constant / varying split of A's logit shift, defined exactly as the transformer
        # measurement defines it (docs/shift_decomposition.md): c = mean over A individuals
        # of dz, v = the residual. Logged because ACCURACY IS FLOORED in this toy while
        # these are continuous -- if ||c|| turns over even slightly while accuracy stays
        # pinned at zero, that is the buy-back mechanism appearing in weak form, and
        # reading accuracy alone would miss it entirely. Both are logged so the race
        # between c shrinking and v growing is visible on the same terms as in the real
        # model, where ||c|| fell 308.6 -> 191.7 from trough to recovery peak.
        c_vec = dz.mean(axis=0)
        rec["c_norm"] = float(np.linalg.norm(c_vec))
        rec["v_norm"] = float(np.linalg.norm(dz - c_vec[None, :]))
        rec["c_norm_va"] = float(np.linalg.norm(c_vec[ps.v_a_mask]))
        rec["c_mean_va"] = float(c_vec[ps.v_a_mask].mean())
        if probe_fn is not None:
            rec.update(probe_fn(s, store, W0, b0))
        curve.append(rec)
        if verbose and (s < 20 or s % 100 == 0):
            print(f"  [finetune step {s}] a_full={rec['a_full_acc_mean']:.4f} "
                  f"a_restr_rank={rec['a_restr_rank_mean']:.3f} "
                  f"b_full={rec['b_full_acc_mean']:.4f}")

    do_eval(0)

    while step < proto.finetune_max_steps:
        if cfg.optimizer.optimizer == "sgd_minibatch":
            perm = rng.permutation(ps.keys_b.shape[0])
            bs = cfg.optimizer.minibatch_size
            for start in range(0, len(perm), bs):
                idx = perm[start:start + bs]
                grad_W, grad_b, _ = compute_grad(
                    store, ps.keys_b[idx], ps.values_b[idx], cfg.vocab.v_total, cfg.loss)
                if freeze_bias:
                    grad_b = np.zeros_like(grad_b)
                opt.step(store, grad_W, grad_b)
                step += 1
                if step <= proto.eval_dense_until or step % proto.eval_every_sparse == 0:
                    do_eval(step)
                if step >= proto.finetune_max_steps:
                    break
        else:
            grad_W, grad_b, _ = compute_grad(
                store, ps.keys_b, ps.values_b, cfg.vocab.v_total, cfg.loss)
            if freeze_bias:
                grad_b = np.zeros_like(grad_b)
            opt.step(store, grad_W, grad_b)
            step += 1
            if step <= proto.eval_dense_until or step % proto.eval_every_sparse == 0:
                do_eval(step)

        if len(curve) >= proto.finetune_flat_window:
            recent = curve[-proto.finetune_flat_window:]
            b_ok = recent[-1]["b_full_acc_mean"] >= proto.finetune_min_b_acc
            rank_vals = [r["a_restr_rank_mean"] for r in recent]
            acc_vals = [r["a_full_acc_mean"] for r in recent]
            flat = (max(rank_vals) - min(rank_vals) < proto.finetune_flat_rank_range) and \
                   (max(acc_vals) - min(acc_vals) < proto.finetune_flat_acc_range)
            if b_ok and flat:
                break
    else:
        pass

    if step >= proto.finetune_max_steps:
        cap_truncated = True

    return curve, cap_truncated, store
