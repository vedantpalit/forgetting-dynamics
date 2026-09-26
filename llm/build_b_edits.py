"""Mass counterfactual editing as the new-fact set: the model-editing fine-tuning baseline.

The most reviewer-proof real update we have: take facts the model KNOWS (the gated set) in one
or two relations, and fine-tune it on their CounterFact counterfactual targets -- "the developer
of X is <other company>". This is exactly the fine-tuning baseline of ROME / MEMIT, on their
dataset. It satisfies the corpus criterion by construction: the answers are one region (companies
for P176/P178), there are hundreds of them, and the model is CONFIDENTLY WRONG on every new fact
for as long as it takes to overwrite an association it holds -- the condition long-tail real
facts lacked (learned within tens of steps; llm/build_b_entityq.py).

Old facts = the gated facts in the OTHER relations (cities, networks, countries, languages);
inject.py tracks them per relation (Arel/*), so the analysis excludes the edited relations. The
edited relations' own A entries measure the OLD answers being overwritten (they should fall as
the edits take).

Templates: CounterFact's pooled (prefix, suffix) paraphrases of each edited relation, the last
N_EVAL held out for evaluation (the gate's own phrasings; the edited facts are B, not A, so
training on their probes trains nothing A is measured with -- A's relations are disjoint).

  python -m llm.build_b_edits --relations P176,P178 --out llm/out/b_facts_edits.json
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

from llm.build_b import first_tok
from llm.inject import DATASET_ID, MODEL_ID

N_EVAL = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", default="P176,P178")
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--out", default="llm/out/b_facts_edits.json")
    ap.add_argument("--max_per_value", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rels = a.relations.split(",")
    tok = AutoTokenizer.from_pretrained(a.model_id)
    ds = load_dataset(DATASET_ID, split="train")
    g = json.load(open(a.gate, encoding="utf-8")); A = g["A"]
    edit = [r for r in A if r["relation_id"] in rels]
    old = [r for r in A if r["relation_id"] not in rels]
    old_tokens = {first_tok(tok, r["target_true"]) for r in old} - {None}
    print(f"gated facts: {len(A)}; edited relations {rels}: {len(edit)}; tracked old facts: {len(old)} "
          f"over {len(set(r['relation_id'] for r in old))} relations")
    # templates per relation: the pooled (prefix, suffix) pairs, as in gate_a_set.py
    templates = defaultdict(set)
    for r in ds:
        if r["relation_id"] in rels:
            templates[r["relation_id"]].add((r["relation_prefix"], r["relation_suffix"]))
    tmpl = {}
    for rid in rels:
        pairs = sorted(templates[rid])
        # CounterFact marks the subject slot with "{}" inside prefix/suffix
        tmpl[rid] = [(pre + suf).replace("{}", "{name}") + " {v}." for pre, suf in pairs]
        assert all("{name}" in t for t in tmpl[rid]), tmpl[rid]
        print(f"  {rid}: {len(pairs)} paraphrases, {N_EVAL} held out; e.g. {tmpl[rid][0]!r}")
        assert len(pairs) > N_EVAL + 1, f"{rid} has too few paraphrases"
    # the edits: value = the counterfactual target; must differ from the true answer's first token
    facts = []
    for r in edit:
        row = ds[r["row"]]
        assert row["subject"] == r["subject"] and row["target_true"] == r["target_true"]
        v = row["target_false"].strip()
        ft, fv = first_tok(tok, r["target_true"]), first_tok(tok, v)
        if fv is None or fv == ft:
            continue
        facts.append((r["subject"].strip(), r["relation_id"], v))
    print(f"  1. counterfactual target differs from the true answer: {len(facts)}")
    facts = [(s, rid, v) for s, rid, v in facts if first_tok(tok, v) not in old_tokens]
    print(f"  2. new answers first-token-disjoint from the tracked old facts' answers: {len(facts)}")
    facts = [(s, rid, v) for s, rid, v in facts if v.lower() not in s.lower() and s.lower() not in v.lower()]
    print(f"  3. no copy shortcut: {len(facts)}")
    freq = Counter(v for _, _, v in facts); owner = {}
    for v, _ in freq.most_common():
        owner.setdefault(first_tok(tok, v), v)
    facts = [(s, rid, v) for s, rid, v in facts if owner[first_tok(tok, v)] == v]
    print(f"  4. one value per first token: {len(facts)} over {len(set(v for *_, v in facts))} values")
    rng.shuffle(facts)
    used, people = Counter(), []
    for s, rid, v in facts:
        if used[v] < a.max_per_value:
            used[v] += 1; people.append({"name": s, rid: v})
    print(f"  5. cap {a.max_per_value}/value: {len(people)} edits; top {used.most_common(4)}")
    pools = {rid: sorted({p[rid] for p in people if rid in p}) for rid in rels}
    evals = []
    for i, p in enumerate(people):
        rid = [k for k in p if k != "name"][0]
        held = tmpl[rid][-N_EVAL:][i % N_EVAL]
        full = held.format(name=p["name"], v=p[rid])
        prompt = full[:full.rindex(p[rid])].rstrip()
        evals.append({"name": p["name"], "attr": rid, "value": p[rid], "prompt": prompt,
                      "first_token": first_tok(tok, p[rid])})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"attrs": rels, "people": people, "pools": pools,
               "train_templates": {rid: tmpl[rid][:-N_EVAL] for rid in rels},
               "eval_templates": {rid: tmpl[rid][-N_EVAL:] for rid in rels}, "eval": evals,
               "edited_relations": rels,
               "source": "CounterFact counterfactual edits of gated facts (ROME/MEMIT fine-tuning baseline)"},
              open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"example: {people[0]}  prompt {evals[0]['prompt']!r}")
    print(f"wrote {a.out}: {len(people)} edits; pools " + str({k: len(v) for k, v in pools.items()}))


if __name__ == "__main__":
    main()
