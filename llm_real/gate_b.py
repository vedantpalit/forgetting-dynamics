"""The B-gate: keep only facts the base model does NOT already know.

A creation date after the cutoff is evidence, not proof -- a 2025 article about a person
born in 1896 describes someone the model may have read about elsewhere. So every fact is
scored on the BASE model under all fifteen phrasings, and a fact is kept only if the correct
first token is never the argmax and its within-pool rank is worse than a threshold under
every phrasing. Dropped facts are counted by attribute so the yield is auditable, and the
mean B accuracy of what remains is printed: it should sit at chance (1 / pool size), which
is also the number the injection run's step-0 eval must reproduce.

Forward passes only; one job on a GPU. Writes `b_real_gated.json` in the same format as
`b_real.json` (train.py reads the gated one).
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_real.scoring import build_b_items, score
from llm_real.templates import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", default="llm_real/out/b_real.json")
    ap.add_argument("--out", default="llm_real/out/b_real_gated.json")
    ap.add_argument("--model_id", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--min_rank", type=float, default=2.0,
                    help="keep a fact only if its mean within-pool rank across all phrasings "
                         "is at least this (1 = already known)")
    ap.add_argument("--eval_batch", type=int, default=256)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.bfloat16).to(device)

    b = json.load(open(a.b, encoding="utf-8"))
    _, pools = build_b_items(tok, b)
    # every (person, attr) under every phrasing, train and eval alike
    items, owner = [], []
    for i, p in enumerate(b["people"]):
        for attr, v in p["facts"].items():
            for t in b["train_templates"][attr] + b["eval_templates"][attr]:
                prompt = render(t, p["name"], v).split(v, 1)[0].rstrip()
                items.append({"prompt": prompt, "attr": attr, "value": v,
                              "first_token": tok(" " + v, add_special_tokens=False).input_ids[0],
                              "rank_eligible": True})
                owner.append((i, attr))
    print(f"scoring {len(items)} prompts ({len(b['people'])} people) on the base model")
    acc, rank = score(model, tok, items, pools, device, a.eval_batch)

    hit = Counter(); mean_rank = {}
    for (i, attr), h, r in zip(owner, acc, rank):
        hit[(i, attr)] += int(h)
        mean_rank.setdefault((i, attr), []).append(r)
    dropped = Counter(); kept_people = []
    for i, p in enumerate(b["people"]):
        facts = {}
        for attr, v in p["facts"].items():
            mr = float(np.nanmean(mean_rank[(i, attr)]))
            if hit[(i, attr)] > 0:
                dropped[f"{attr}:argmax_under_some_phrasing"] += 1
            elif mr < a.min_rank:
                dropped[f"{attr}:rank_below_{a.min_rank:g}"] += 1
            else:
                facts[attr] = v
        if facts:
            kept_people.append({**p, "facts": facts})
        else:
            dropped["person_no_fact_left"] += 1

    # rebuild eval items and pools over what survived, same rotation as build.py
    from llm_real.templates import N_EVAL_TEMPLATES
    evals, k, pools_out = [], 0, {}
    for i, p in enumerate(kept_people):
        for attr, v in p["facts"].items():
            t = b["eval_templates"][attr][k % N_EVAL_TEMPLATES]; k += 1
            evals.append({"prompt": render(t, p["name"], v).split(v, 1)[0].rstrip(),
                          "attr": attr, "value": v,
                          "first_token": tok(" " + v, add_special_tokens=False).input_ids[0],
                          "person": i})
            pools_out.setdefault(attr, set()).add(v)
    out = {**b, "people": kept_people, "eval": evals,
           "pools": {k_: sorted(v) for k_, v in pools_out.items()},
           "gate": {"min_rank": a.min_rank, "dropped": dict(dropped),
                    "base_acc_all_prompts": float(acc.mean())}}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # step-0 sanity on the kept eval items
    items2, pools2 = build_b_items(tok, out)
    acc2, rank2 = score(model, tok, items2, pools2, device, a.eval_batch)
    by = Counter(); n = Counter()
    for it, h in zip(items2, acc2):
        by[it["attr"]] += h; n[it["attr"]] += 1
    print(f"kept {len(kept_people)} people, {len(evals)} facts; dropped {dict(dropped)}")
    print("base-model accuracy on kept eval items (should be ~chance = 1/pool):")
    for attr in n:
        print(f"  {attr:<12} acc {by[attr] / n[attr]:.4f}  chance {1 / max(len(pools2[attr]), 1):.4f}  "
              f"n={n[attr]}  mean rank {np.nanmean([r for it, r in zip(items2, rank2) if it['attr'] == attr]):.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
