"""New facts = real biographies of real long-tail people (Wikidata), several facts per person.

WHY THIS SET. The synthetic biographies that show all three phases in OLMo differ from every real
single-fact set we tried (EntityQuestions P17, CounterFact edits) in two ways at once: the
subjects have no representation in the model, and each person carries three or four facts. All
the real single-fact sets were learned by the end of warmup (~60 steps), before a coherent write
could build; the synthetic new facts stayed at chance until step ~150. This set keeps the
synthetic document shape -- one person, several encyclopedic sentences -- with nothing invented:
people without an English Wikipedia article, and their occupation, birthplace, education and
citizenship as Wikidata records them (llm/fetch_wikidata_people.py).

FILTERS, per attribute, in the order of build_b_entityq.py: not an A string; the model's top-1
first token is wrong on the held-out phrasing; the value's first token is disjoint from every A
answer's first token; no copy shortcut; one value per first token (the most frequent value keeps
the token, the others' facts are DROPPED); cap per value. A person keeps only the facts that
survive and is kept with at least --min_facts of them.

  python -m llm.build_b_wikibio --raw llm/out/wikidata_people_raw.json --out llm/out/b_facts_wikibio.json
"""
import argparse
import json
import os
import random
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.btemplates import TEMPLATES as BIO_TEMPLATES, stem
from llm.build_b import first_tok
from llm.build_b_counterfact import model_first_token
from llm.build_b_entityq import RELATIONS
from llm.inject import DATASET_ID, MODEL_ID

TEMPLATES = {
    "occupation": RELATIONS["P106"][2],
    "birth_city": list(BIO_TEMPLATES["birth_city"]),
    "education": RELATIONS["P69"][2],
    "citizenship": list(BIO_TEMPLATES["citizenship"]),
}
N_EVAL = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="llm/out/wikidata_people_raw.json")
    ap.add_argument("--attrs", default="occupation,birth_city,education,citizenship")
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--out", default="llm/out/b_facts_wikibio.json")
    ap.add_argument("--max_per_value", type=int, default=60)
    ap.add_argument("--min_facts", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=4000, help="cap on people")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    attrs = a.attrs.split(",")
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))))
    tok = AutoTokenizer.from_pretrained(a.model_id)
    ds = load_dataset(DATASET_ID, split="train")
    g = json.load(open(a.gate, encoding="utf-8")); A = g["A"]
    a_subjects = {r["subject"].strip().lower() for r in A}
    a_strings = a_subjects | {r["target_true"].strip().lower() for r in A}
    a_tokens = {first_tok(tok, r["target_true"]) for r in A} - {None}
    cf_prompts = {r["prompt"] for r in ds}
    cf_stems = {" ".join(r["prompt"].replace(r["subject"], " ").split()) for r in ds}
    for at in attrs:
        clash = [t for t in TEMPLATES[at] if stem(t) in cf_stems]
        assert not clash, f"template equals a CounterFact probe stem: {clash}"
    train_t = {at: TEMPLATES[at][:-N_EVAL] for at in attrs}
    eval_t = {at: TEMPLATES[at][-N_EVAL:] for at in attrs}

    raw = json.load(open(a.raw, encoding="utf-8"))
    seen, people = set(), []
    for r in raw:                                  # one row per name
        n = r["name"].strip()
        if n.lower() in seen or n.lower() in a_strings:
            continue
        seen.add(n.lower())
        p = {"name": n}
        for at in attrs:
            v = r.get(at)
            if isinstance(v, str) and v.strip() and v.strip().lower() not in a_subjects:
                p[at] = v.strip()
        people.append(p)
    facts = [(i, at) for i, p in enumerate(people) for at in attrs if at in p]
    print(f"Wikidata people: {len(raw)} rows -> {len(people)} names, {len(facts)} facts; "
          f"per attribute {Counter(at for _, at in facts)}")

    def probe(p, at, k):
        full = eval_t[at][k % N_EVAL].format(name=p["name"], v=p[at])
        return full[:full.rindex(p[at])].rstrip()
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32); model.eval()
    top = model_first_token(model, tok, [probe(people[i], at, i) for i, at in facts])
    del model
    keep = set()
    for (i, at), t in zip(facts, top):
        ft = first_tok(tok, people[i][at])
        if ft is not None and t != ft:
            keep.add((i, at))
    print(f"  2. model gets it wrong (new knowledge): {len(keep)} facts; "
          f"per attribute {Counter(at for _, at in keep)}")
    keep = {(i, at) for i, at in keep if first_tok(tok, people[i][at]) not in a_tokens}
    print(f"  3. value first token disjoint from A's answers: {len(keep)}")
    val_tokens = {at: {first_tok(tok, people[i][at]) for i, at2 in keep if at2 == at} for at in attrs}
    keep2 = set()
    for i, at in keep:
        s, v = people[i]["name"], people[i][at]
        nt = set(tok(s, add_special_tokens=False).input_ids) | set(tok(" " + s, add_special_tokens=False).input_ids)
        if v.lower() in s.lower() or s.lower() in v.lower() or (nt & val_tokens[at]):
            continue
        keep2.add((i, at))
    keep = keep2
    print(f"  4. no copy shortcut: {len(keep)}")
    for at in attrs:                               # one value per first token, per attribute
        freq = Counter(people[i][at] for i, at2 in keep if at2 == at)
        owner = {}
        for v, _ in freq.most_common():
            owner.setdefault(first_tok(tok, v), v)
        keep = {(i, at2) for i, at2 in keep if at2 != at or owner[first_tok(tok, people[i][at])] == people[i][at]}
    print(f"  5. one value per first token: {len(keep)} facts; per attribute "
          f"{ {at: len({people[i][at] for i, at2 in keep if at2 == at}) for at in attrs} } values")
    order = list(range(len(people))); rng.shuffle(order)
    used, out = {at: Counter() for at in attrs}, []
    for i in order:
        p = {"name": people[i]["name"]}
        for at in attrs:
            if (i, at) in keep and used[at][people[i][at]] < a.max_per_value:
                p[at] = people[i][at]
        if len(p) - 1 >= a.min_facts:
            for at in attrs:
                if at in p:
                    used[at][p[at]] += 1
            out.append(p)
        if len(out) >= a.n_max:
            break
    people = out
    pools = {at: sorted(used[at]) for at in attrs}
    print(f"  6. cap {a.max_per_value}/value, >= {a.min_facts} facts, max {a.n_max}: {len(people)} people, "
          f"{sum(len(p) - 1 for p in people)} facts; pools { {at: len(pools[at]) for at in attrs} }; "
          f"facts/person {Counter(len(p) - 1 for p in people)}")
    for at in attrs:
        print(f"     {at}: top {used[at].most_common(5)}")
    b_tokens = {first_tok(tok, v) for at in attrs for v in pools[at]}
    assert not (b_tokens & a_tokens), "B value tokens collide with A answer first tokens"

    evals = []
    for i, p in enumerate(people):
        for at in attrs:
            if at in p:
                prompt = probe(p, at, i)
                assert prompt not in cf_prompts, f"B eval prompt {prompt!r} IS a CounterFact probe"
                evals.append({"name": p["name"], "attr": at, "value": p[at], "prompt": prompt,
                              "first_token": first_tok(tok, p[at])})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"attrs": attrs, "people": people, "pools": pools,
               "train_templates": train_t, "eval_templates": eval_t, "eval": evals,
               "source": f"Wikidata people without an English Wikipedia article ({a.raw}); facts "
                         f"{a.model_id} gets wrong, answers first-token-disjoint from A"},
              open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"example: {people[0]}  prompt {evals[0]['prompt']!r}")
    print(f"wrote {a.out}: {len(people)} people, {len(evals)} facts")


if __name__ == "__main__":
    main()
