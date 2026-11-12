"""Comparisons for the weight edit in OLMo 2 1B.

Same injection as llm/patch_weights.py (optimizer, schedule, data, seed). At each full eval step, with training
paused, the old and new facts are scored under these weight settings, for every 2-D block matrix W:

    actual       W(t)
    top1         W(t) minus the top singular component of dW = W(t) - W(0)        (the paper's edit)
    top<a>       W(t) minus a times that component, a in ALPHAS: top1's own frontier
    rand         W(t) minus an independent random rank-one matrix of the same norm (the paper's control)
    interp<l>    W(0) + l dW: moving the whole block update back toward the pretrained weights
                 (blocks only, as the edit; embeddings, lm_head and norms stay at t: l = 0 measures that)
    shrink       W(0) + c dW, c per matrix so that ||c dW|| = ||dW - top1||: the same distance from the
                 pretrained weights as top1, spread over the whole update
    next         W(t) minus singular components 2, 3, ... of dW, the last one scaled, until they carry the same
                 Frobenius mass as the top one; the mass reached is logged because rank Q can fall short

READING, pre-registered (3 seeds, paired, margin 0.05 in old-fact accuracy at matched new-fact accuracy, compared
on frontiers): the top direction is specific if the top<a> frontier lies above interp, shrink and next at steps
>= 150 on both fact sets. A tie with interp means a partial return to the pretrained weights. next matching top1
means the claim is about the update's leading part, not its top direction.

SHARED SHARE (do the removed components account for a common shift?). On the old facts' logits at the answer
position, rows centred (softmax ignores a constant per row), per fact a: D_a = z_a(actual) - z_a(top1) and
T_a = z_a(actual) - z_a(step 0). The share is leave-one-out, so its floor is 0 and not 1/n:
    share(X) = (n ||mean X||^2 / mean ||X||^2 - 1) / (n - 1),   over all old facts and within relations
    cos(mean D, mean T), and the projection of mean T on mean D
    the same on the final hidden state, and on the answer tokens only
    region gap: logsumexp over new-answer tokens minus logsumexp over old-answer tokens, under actual and top1
    acc_minus_common_top / _total: recall of z(actual) minus the mean of D / of T

EXACT MATCH: teacher-forced, every answer token is the argmax given the earlier ones, on every arm for the new
facts; old answers are 155 of 156 single tokens, so it is logged for actual and top1 only. Mean log-probability of the
new answers is logged too, since accuracy cannot match arms near zero. Per-fact old-fact correctness is kept.

  .venv-llm/bin/python -m llm.edit_comparisons --lr 1e-5 --steps 400 --seed 0 --facts llm/out/b_facts_small.json --out ...
"""
import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.evalsets import build_a, build_b
from llm.inject import DATASET_ID, MODEL_ID, DocSampler, score, summarise
from llm.patch_weights import matrices

LAMBDAS = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.0)
ALPHAS = (0.25, 0.5, 0.75, 1.25)
Q = 128                                 # singular components computed per matrix
MIN_GROUP = 10                          # relations with at least this many non-copy old facts


@torch.no_grad()
def components(dW):
    U, S, V = torch.svd_lowrank(dW, q=min(Q, min(dW.shape)), niter=4)
    return U, S, V


@torch.no_grad()
def last_logits(model, tok, items, device, batch_size):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(items), batch_size):
        enc = tok([it["prompt"] for it in items[i:i + batch_size]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model(**enc).logits[:, -1, :].float())
    return torch.cat(out)


@torch.no_grad()
def last_hidden(model, tok, items, device, batch_size):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(items), batch_size):
        enc = tok([it["prompt"] for it in items[i:i + batch_size]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model.model(**enc).last_hidden_state[:, -1, :].float())
    return torch.cat(out)


@torch.no_grad()
def exact_match(model, tok, items, device, batch_size):
    """(share of items whose whole answer is the greedy continuation, mean token log-probability), teacher-forced."""
    tok.padding_side = "right"
    right = 0; lp = 0.0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        rows = []
        for it in chunk:
            p = tok(it["prompt"], add_special_tokens=False).input_ids
            a = tok(" " + it["target"].strip(), add_special_tokens=False).input_ids
            rows.append((p, a))
        L = max(len(p) + len(a) for p, a in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for r, (p, a) in enumerate(rows):
            ids[r, :len(p) + len(a)] = torch.tensor(p + a); att[r, :len(p) + len(a)] = 1
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            lg = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        pred = lg.argmax(-1).cpu()
        for r, (p, a) in enumerate(rows):
            right += int(all(pred[r, len(p) - 1 + j].item() == a[j] for j in range(len(a))))
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=lg.device)
            lp += float(torch.log_softmax(lg[r, pos].float(), -1)[torch.arange(len(a), device=lg.device),
                                                                   torch.tensor(a, device=lg.device)].mean())
    tok.padding_side = "left"
    return right / max(len(items), 1), lp / max(len(items), 1)


def centre(X):
    return X - X.mean(1, keepdim=True)


def loo_share(X):
    """Leave-one-out shared share of the rows of X: 0 for independent rows, 1 for identical rows."""
    n = X.shape[0]; ms = float((X.norm(dim=1) ** 2).mean())
    return (n * float(X.mean(0).norm() ** 2) / ms - 1.0) / (n - 1) if ms > 0 and n > 1 else 0.0


def shares(X, groups):
    return loo_share(X), sum(len(g) * loo_share(X[g]) for g in groups) / sum(len(g) for g in groups)


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--tokens_per_step", type=int, default=32768)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--facts", default="llm/out/b_facts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generic", default="llm_real/out/generic_tokens.npy")
    ap.add_argument("--eval_steps", default="0,50,80,100,150,200,300,400", help="full evals")
    ap.add_argument("--dense", type=int, default=10, help="actual-only evals every this many steps")
    ap.add_argument("--lite", action="store_true", help="actual, top1 and rand only (the permutation arms)")
    a = ap.parse_args()
    eval_steps = {int(x) for x in a.eval_steps.split(",")}
    if not os.path.exists(a.generic):
        raise SystemExit(f"missing generic text {a.generic}")

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    ds = load_dataset(DATASET_ID, split="train")
    a_items, a_pools, a_info = build_a(tok, a.gate, ds)
    b_items, b_pools, facts = build_b(tok, a.facts)
    mats = matrices(model)
    W0 = {n: p.detach().clone() for n, p in mats}
    print(f"device={device} lr={a.lr:g} steps={a.steps} seed={a.seed} facts={a.facts} | {len(mats)} matrices | "
          f"A {a_info['n_A']} B {len(b_items)}", flush=True)

    nc = [it for it in a_items if it["copy"] == "noncopy"]
    y_nc = torch.tensor([it["first_token"] for it in nc], device=device)
    rel = {}
    for k, it in enumerate(nc):
        rel.setdefault(it["relation_id"], []).append(k)
    groups = [torch.tensor(v, device=device) for v in rel.values() if len(v) >= MIN_GROUP]
    nc_idx = [k for k, it in enumerate(a_items) if it["copy"] == "noncopy"]
    a_tok = torch.tensor(sorted({it["first_token"] for it in a_items}), device=device)
    b_tok = torch.tensor(sorted({it["first_token"] for it in b_items}), device=device)
    cols = torch.cat([a_tok, b_tok])

    import numpy as np
    from llm_real.tokens import TokenPool
    pool = TokenPool(list(np.load(a.generic, allow_pickle=True)), a.seed + 3)
    sep = tok("\n\n", add_special_tokens=False).input_ids
    generic_rows = torch.tensor([pool.row(a.seq_len, sep) for _ in range(64)]).to(device)

    @torch.no_grad()
    def generic_loss():
        tot = 0.0
        for i in range(0, len(generic_rows), 8):
            b = generic_rows[i:i + 8]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                tot += float(model(input_ids=b, labels=b).loss.item()) * len(b)
        return tot / len(generic_rows)

    sampler = DocSampler(facts, tok, a.seq_len, a.seed)
    micro = a.batch_size * a.seq_len; accum = max(a.tokens_per_step // micro, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator(device=device); gen.manual_seed(a.seed + 7)
    curve = []; dense = []; z0 = {}

    @torch.no_grad()
    def score_all(tag, rec, a_exact=False):
        model.eval()
        aa, ar = score(model, tok, a_items, a_pools, device, a.eval_batch)
        ba, br = score(model, tok, b_items, b_pools, device, a.eval_batch)
        s = summarise(aa, ar, a_items, "A", "copy"); sb = summarise(ba, br, b_items, "B", "stratum")
        rec[f"{tag}/A_noncopy"] = s["A/noncopy/acc"]; rec[f"{tag}/A_copy"] = s.get("A/copy/acc")
        rec[f"{tag}/A_noncopy_rank"] = s.get("A/noncopy/rank"); rec[f"{tag}/B"] = sb["B/ALL/acc"]; rec[f"{tag}/B_rank"] = sb.get("B/ALL/rank")
        rec[f"{tag}/A_ok"] = "".join(str(int(aa[k])) for k in nc_idx)
        rec[f"{tag}/generic_loss"] = generic_loss()
        rec[f"{tag}/B_exact"], rec[f"{tag}/B_logp"] = exact_match(model, tok, b_items, device, a.eval_batch)
        if a_exact:
            rec[f"{tag}/A_noncopy_exact"], _ = exact_match(model, tok, nc, device, a.eval_batch)

    def gap(z):
        return float((torch.logsumexp(z[:, b_tok], -1) - torch.logsumexp(z[:, a_tok], -1)).mean())

    def save():
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"lr": a.lr, "steps": a.steps, "seed": a.seed, "facts": a.facts, "lite": a.lite, "curve": curve, "dense": dense},
                  open(a.out, "w"), indent=1)

    @torch.no_grad()
    def evaluate(step):
        rec = {"step": step}
        score_all("actual", rec, a_exact=True)
        z_act = last_logits(model, tok, nc, device, a.eval_batch); h_act = last_hidden(model, tok, nc, device, a.eval_batch)
        rec["gap/actual"] = gap(z_act)
        if step == 0:
            z0["z"], z0["h"] = z_act.clone(), h_act.clone()
        Wt = {n: p.detach().clone() for n, p in mats}

        def put(fn):
            for n, p in mats:
                p.copy_(fn(n, Wt[n] - W0[n]))

        if step > 0:
            comp = {n: components(Wt[n] - W0[n]) for n, p in mats}
            top1 = {n: float(comp[n][1][0] ** 2 / ((Wt[n] - W0[n]) ** 2).sum()) for n, _ in mats}
            rec["top1_share_mean"] = sum(top1.values()) / len(top1); rec["top1_share_max"] = max(top1.values())

            def top_part(n):
                U, S, V = comp[n]
                return (U[:, :1] * S[:1]) @ V[:, :1].T
            put(lambda n, dW: Wt[n] - top_part(n))
            score_all("top1", rec, a_exact=True)
            z_top = last_logits(model, tok, nc, device, a.eval_batch); h_top = last_hidden(model, tok, nc, device, a.eval_batch)
            rec["gap/top1"] = gap(z_top)
            D, T = centre(z_act - z_top), centre(z_act - z0["z"])
            Dh, Th = h_act - h_top, h_act - z0["h"]
            rec["share/edit_all"], rec["share/edit_rel"] = shares(D, groups)
            rec["share/total_all"], rec["share/total_rel"] = shares(T, groups)
            rec["share/edit_all_ans"], rec["share/edit_rel_ans"] = shares(D[:, cols], groups)
            rec["share/total_all_ans"], rec["share/total_rel_ans"] = shares(T[:, cols], groups)
            rec["share/edit_all_h"], rec["share/edit_rel_h"] = shares(Dh, groups)
            rec["share/total_all_h"], rec["share/total_rel_h"] = shares(Th, groups)
            mD, mT = D.mean(0), T.mean(0)
            rec["cos_meanD_meanT"] = cos(mD, mT); rec["proj_T_on_D"] = float(mT @ mD / (mD @ mD).clamp_min(1e-30))
            rec["cos_meanD_meanT_ans"] = cos(mD[cols], mT[cols]); rec["cos_meanDh_meanTh"] = cos(Dh.mean(0), Th.mean(0))
            rec["acc_minus_common_top"] = float(((z_act - D.mean(0)).argmax(-1) == y_nc).float().mean())
            rec["acc_minus_common_total"] = float(((z_act - T.mean(0)).argmax(-1) == y_nc).float().mean())
            zc = z_act.clone(); zt = z_act.clone()
            for idx in groups:
                zc[idx] -= D[idx].mean(0); zt[idx] -= T[idx].mean(0)
            cover = torch.cat(groups)
            rec["acc_minus_common_top_rel"] = float((zc[cover].argmax(-1) == y_nc[cover]).float().mean())
            rec["acc_minus_common_total_rel"] = float((zt[cover].argmax(-1) == y_nc[cover]).float().mean())
            rec["acc_actual_rel_cover"] = float((z_act[cover].argmax(-1) == y_nc[cover]).float().mean())
            del D, T, Dh, Th, zc, zt, z_top, h_top

            def rand_part(n, dW):
                t = float(comp[n][1][0])
                x = torch.randn(dW.shape[0], 1, generator=gen, device=dW.device, dtype=dW.dtype)
                y = torch.randn(dW.shape[1], 1, generator=gen, device=dW.device, dtype=dW.dtype)
                r = x @ y.T
                return r * (t / float(r.norm()))
            put(lambda n, dW: Wt[n] - rand_part(n, dW))
            score_all("rand", rec)

            if not a.lite:
                for al in ALPHAS:
                    put(lambda n, dW: Wt[n] - al * top_part(n))
                    score_all(f"top{al}", rec)
                for lam in LAMBDAS:
                    put(lambda n, dW: W0[n] + lam * dW)
                    score_all(f"interp{lam}", rec)
                put(lambda n, dW: W0[n] + (max(1.0 - top1[n], 0.0) ** 0.5) * dW)
                score_all("shrink", rec)

                ks, mass = [], []

                def next_part(n):
                    U, S, V = comp[n]
                    e = S[1:] ** 2; c = torch.cumsum(e, 0); want = S[0] ** 2
                    k = min(int((c < want).sum()) + 1, len(e))
                    w = torch.ones(k, device=S.device, dtype=S.dtype)
                    before = c[k - 2] if k > 1 else torch.zeros((), device=S.device, dtype=S.dtype)
                    w[-1] = ((want - before) / e[k - 1]).clamp(0, 1).sqrt()      # the last component, scaled to match
                    ks.append(k); mass.append(float(((w ** 2) * e[:k]).sum() / want))
                    return (U[:, 1:k + 1] * (S[1:k + 1] * w)) @ V[:, 1:k + 1].T
                put(lambda n, dW: Wt[n] - next_part(n))
                score_all("next", rec)
                rec["next_k_mean"] = sum(ks) / len(ks); rec["next_k_max"] = max(ks)
                rec["next_mass_mean"] = sum(mass) / len(mass); rec["next_mass_min"] = min(mass)
            del comp
        for n, p in mats:
            p.copy_(Wt[n])
        del Wt
        curve.append(rec)
        print(f"[step {step}] A actual {rec['actual/A_noncopy']:.3f} B {rec['actual/B']:.3f} exact {rec['actual/B_exact']:.3f}"
              + (f" | top1 A {rec['top1/A_noncopy']:.3f} B {rec['top1/B']:.3f} exact {rec['top1/B_exact']:.3f} | rand A {rec['rand/A_noncopy']:.3f} "
                 f"| edit share {rec['share/edit_all']:.3f} rel {rec['share/edit_rel']:.3f} cos {rec['cos_meanD_meanT']:.3f} "
                 f"minus common top {rec['acc_minus_common_top']:.3f} total {rec['acc_minus_common_total']:.3f} gap {rec['gap/actual']:.2f}->{rec['gap/top1']:.2f}"
                 if step > 0 else ""), flush=True)
        model.train(); save()

    @torch.no_grad()
    def evaluate_dense(step):
        model.eval()
        aa, _ = score(model, tok, a_items, a_pools, device, a.eval_batch)
        ba, _ = score(model, tok, b_items, b_pools, device, a.eval_batch)
        dense.append(dict(step=step, A_noncopy=sum(aa[k] for k in nc_idx) / len(nc_idx), B=sum(ba) / len(ba)))
        model.train(); save()

    evaluate(0)
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(step / max(a.warmup, 1), 1.0)
        for _ in range(accum):
            batch = sampler.batch(a.batch_size).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step in eval_steps:
            evaluate(step)
        elif step % a.dense == 0:
            evaluate_dense(step)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
