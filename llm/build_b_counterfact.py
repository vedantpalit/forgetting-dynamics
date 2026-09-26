"""Build B-real: CounterFact facts the model does NOT know, about real people, with their true
values -- real data that competes with the old facts' answers.

The two real-people arms (llm_real) were not acquired: unknown multi-token names, one abstract
each. This keeps the REAL part that matters -- real subjects, real Wikidata values -- and takes
the rendering that made B learnable: warm single-token values and the twelve prose phrasings
per attribute, rendered fresh at every step.

  relation -> attribute:  P19 birthplace -> birth_city   P27 citizenship -> citizenship
                          P108 employer  -> employer     P1412 / P103 language -> language

FILTERS, in order (each one printed):
  1. the subject is not one of A's subjects, and the value is not an A string
  2. the model gets the fact WRONG on the CounterFact prompt (top-1 first token != target's),
     so it is new knowledge -- the gate, one forward pass per row
  3. the value's first token is disjoint from every A answer first token -- the competing-region
     condition, exactly as in build_b.py (assertion 2 there)
  4. the subject's tokens share nothing with any B value (no copy shortcut, assertion 4)
  5. within an attribute, distinct first tokens per value; within a person, distinct first tokens
A person keeps whichever of the four attributes survive; the sampler renders only those.

Output: llm/out/b_facts_cf.json in the b_facts format (attrs, people, pools, train_templates,
eval_templates, eval), so inject.py / predict_erosion.py take it unchanged via --facts.

  .venv-llm/bin/python -m llm.build_b_counterfact
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.btemplates import ATTRS as PERSON_ATTRS, TEMPLATES as PERSON_TEMPLATES, split as person_split, stem

TEMPLATES = dict(PERSON_TEMPLATES)
from llm.build_b import first_tok
from llm.inject import DATASET_ID, MODEL_ID

REL2ATTR = {"P19": "birth_city", "P27": "citizenship", "P108": "employer",
            "P1412": "language", "P103": "language",
            "P176": "maker", "P178": "maker"}          # products -> the company that makes them
# The company-valued stratum: 71% of A answers "who makes X" with a company, so a B that is to
# compete with A needs real facts whose answers are OTHER companies. Fifteen encyclopedic
# phrasings, the last three held out; none may equal a CounterFact probe stem (asserted).
MAKER_TEMPLATES = [
    "{name} came from the factories of {v}.",
    "The firm behind {name} was {v}.",
    "{name} was brought to market by {v}.",
    "Engineers at {v} were responsible for {name}.",
    "{name} was one of the products of {v}.",
    "The company that built {name} was {v}.",
    "{name} was manufactured and sold by {v}.",
    "Credit for {name} belongs to {v}.",
    "{name} rolled off the production lines of {v}.",
    "The maker of {name} was {v}.",
    "{name} was designed and produced by the firm {v}.",
    "Trade catalogues list {name} under {v}.",
    "{name} was a product of the company {v}.",
    "The manufacturer of {name} was {v}.",
    "{name} was put out by {v}.",
]
N_EVAL = 3
TEMPLATES["maker"] = MAKER_TEMPLATES
ATTRS = list(PERSON_ATTRS) + ["maker"]


def split(attr):
    t = TEMPLATES[attr]
    return t[:-N_EVAL], t[-N_EVAL:]


@torch.no_grad()
def model_first_token(model, tok, prompts, batch=32):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False)
        logits = model(**enc).logits[:, -1, :]
        out += logits.argmax(-1).tolist()
        if (i // batch) % 20 == 0:
            print(f"    gate {i + len(enc['input_ids'])}/{len(prompts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--out", default="llm/out/b_facts_cf.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relations", default=",".join(REL2ATTR),
                    help="CounterFact relations to draw from (default: all mapped)")
    ap.add_argument("--drop_attrs", default="employer",
                    help="attributes whose surviving pool is not a region (employer: 7 values, one of them 193x)")
    ap.add_argument("--max_per_value", type=int, default=60,
                    help="cap facts per value so no single token dominates the write")
    ap.add_argument("--counterfactual", action="store_true",
                    help="model-editing setting: new facts are CounterFact's counterfactual targets "
                         "(target_false) for facts the model KNOWS (top-1 = target_true), so it is "
                         "confidently wrong on every new fact; the standard fine-tuning baseline "
                         "of ROME/MEMIT")
    a = ap.parse_args()
    drop = set(a.drop_attrs.split(",")) if a.drop_attrs else set()
    rng = random.Random(a.seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split="train")
    g = json.load(open(a.gate, encoding="utf-8")); A = g["A"]
    a_subjects = {r["subject"].strip().lower() for r in A}
    a_strings = a_subjects | {r["target_true"].strip().lower() for r in A}
    a_tokens = {first_tok(tok, r["target_true"]) for r in A} - {None}
    cf_prompts = {r["prompt"] for r in ds}
    cf_stems = {" ".join(r["prompt"].replace(r["subject"], " ").split()) for r in ds}
    assert not [t for at in ATTRS for t in TEMPLATES[at] if stem(t) in cf_stems], "probe collision"

    rels = set(a.relations.split(","))
    rows = [r for r in ds if r["relation_id"] in REL2ATTR and r["relation_id"] in rels]
    print(f"CounterFact rows in the chosen relations: {len(rows)}  "
          + str(Counter(r["relation_id"] for r in rows)))
    rows = [r for r in rows if r["subject"].strip().lower() not in a_subjects
            and (r["target_false"] if a.counterfactual else r["target_true"]).strip().lower() not in a_strings]
    print(f"  1. not an A subject / A string: {len(rows)}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32); model.eval()
    top = model_first_token(model, tok, [r["prompt"] for r in rows])
    del model
    keep = []
    for r, t in zip(rows, top):
        ft = first_tok(tok, r["target_true"])
        if a.counterfactual:
            fn = first_tok(tok, r["target_false"])
            if ft is not None and t == ft and fn is not None and fn != ft:
                keep.append({**r, "target_true": r["target_false"].strip()})   # the edit becomes the value
        elif ft is not None and t != ft:
            keep.append(r)
    rows = keep
    print(f"  2. {'model KNOWS the true fact (counterfactual edit)' if a.counterfactual else 'model gets it wrong (new knowledge)'}: {len(rows)}")
    rows = [r for r in rows if first_tok(tok, r["target_true"]) not in a_tokens]
    print(f"  3. value first token disjoint from A's answers: {len(rows)}")

    # one value per (subject, attribute); people = subjects with >= 1 surviving fact
    facts = defaultdict(dict)
    for r in rows:
        at = REL2ATTR[r["relation_id"]]
        facts[r["subject"].strip()].setdefault(at, r["target_true"].strip())
    # 4. no copy shortcut: the subject's tokens share nothing with any B value
    all_vals = {v for d in facts.values() for v in d.values()}
    val_tokens = {first_tok(tok, v) for v in all_vals} - {None}
    people = []
    for name, d in facts.items():
        nt = set(tok(name, add_special_tokens=False).input_ids) | set(tok(" " + name, add_special_tokens=False).input_ids)
        if nt & val_tokens or any(v.lower() in name.lower() or name.lower() in v.lower() for v in d.values()):
            continue
        people.append({"name": name, **d})
    print(f"  4. no copy shortcut: {len(people)} people, {sum(len(p) - 1 for p in people)} facts")
    # 5. distinct first tokens: per attribute pool and within person
    pools = {}
    for at in ATTRS:
        seen, vals = {}, []
        for p in people:
            if at in p:
                t = first_tok(tok, p[at])
                if t not in seen:
                    seen[t] = p[at]; vals.append(p[at])
        pools[at] = vals
    tok_of = {at: {v: first_tok(tok, v) for v in pools[at]} for at in ATTRS}
    canon = {at: {} for at in ATTRS}
    for at in ATTRS:
        for v in pools[at]:
            canon[at][tok_of[at][v]] = v
    for p in people:                                  # alias values that share a first token
        for at in ATTRS:
            if at in p:
                p[at] = canon[at][first_tok(tok, p[at])]
        ids = [tok_of[at][p[at]] for at in ATTRS if at in p]
        if len(set(ids)) != len(ids):
            for at in ATTRS:                           # drop the later duplicate
                if at in p and ids.count(tok_of[at][p[at]]) > 1:
                    del p[at]; ids = [tok_of[x][p[x]] for x in ATTRS if x in p]
    people = [p for p in people if len(p) > 1]
    print(f"  5. distinct first tokens: {len(people)} people, {sum(len(p) - 1 for p in people)} facts; pools "
          + str({at: len(pools[at]) for at in ATTRS}))
    # 6. drop degenerate attributes and cap facts per value (a balanced region, not a prior)
    rng.shuffle(people)
    used = Counter()
    for p in people:
        for at in list(p):
            if at == "name":
                continue
            if at in drop or used[(at, p[at])] >= a.max_per_value:
                del p[at]
            else:
                used[(at, p[at])] += 1
    people = [p for p in people if len(p) > 1]
    attrs = [at for at in ATTRS if at not in drop]
    pools = {at: [v for v in pools[at] if used[(at, v)] > 0] for at in attrs}
    tok_of = {at: tok_of[at] for at in attrs}
    print(f"  6. drop {sorted(drop) or 'nothing'}, cap {a.max_per_value}/value: {len(people)} people, "
          f"{sum(len(p) - 1 for p in people)} facts; pools " + str({at: len(pools[at]) for at in attrs}))
    b_tokens = {t for at in attrs for v in pools[at] for t in [tok_of[at][v]]}
    assert not (b_tokens & a_tokens), "B value tokens collide with A answer first tokens"
    print(f"B occupies {len(b_tokens)} value tokens, disjoint from A's {len(a_tokens)}")

    evals = []
    for i, p in enumerate(people):
        for j, at in enumerate(attrs):
            if at not in p:
                continue
            held = split(at)[1][(i + j) % len(split(at)[1])]
            full = held.format(name=p["name"], v=p[at])
            prompt = full[:full.index(p[at])].rstrip()
            assert prompt not in cf_prompts, f"B eval prompt {prompt!r} IS a CounterFact probe"
            evals.append({"name": p["name"], "attr": at, "value": p[at], "prompt": prompt,
                          "first_token": tok_of[at][p[at]]})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"attrs": attrs, "people": people, "pools": pools,
               "train_templates": {at: split(at)[0] for at in attrs},
               "eval_templates": {at: split(at)[1] for at in attrs}, "eval": evals,
               "source": ("CounterFact counterfactual edits of facts the model knows" if a.counterfactual else
                          "CounterFact rows the base model gets wrong") + "; relations " + ",".join(REL2ATTR)},
              open(a.out, "w", encoding="utf-8"), indent=2)
    for at in attrs:
        c = Counter(p[at] for p in people if at in p)
        print(f"  {at:>12}: {sum(c.values())} facts over {len(c)} values, max {c.most_common(1)[0][1] if c else 0}x")
    print(f"example: {people[0]}")
    print(f"wrote {a.out}: {len(people)} real people, {len(evals)} facts")


if __name__ == "__main__":
    main()
