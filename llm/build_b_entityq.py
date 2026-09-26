"""New facts from a real dataset: EntityQuestions P17 ("Which country is X located in?").

WHY THIS SET. The corpus criterion (paper Section 5.3) says suppression needs many new facts whose
answers crowd one region AND that stay wrong long enough for the common write to build. Real
facts about famous entities failed the second condition (learned within tens of steps). P17's
subjects are long-tail Wikidata places and organisations (communes, lakes, councils) that a 1B
model does not know, and every answer is a country: one region, real values, no invented names.

FILTERS, in the order of build_b_counterfact.py:
  1. subject / value not an A string
  2. the model's top-1 first token is WRONG on the held-out phrasing (unknown to the model)
  3. the value's first token is disjoint from every A answer's first token (the paper's
     "disjoint answers" condition: A's 22 citizenship countries leave B's pool)
  4. no copy shortcut: the value does not appear in the subject name
  5. one value per first token -- values sharing a first token (" United" for the USA, the UK,
     the UAE) would make first-token scoring ambiguous; the most frequent value keeps the token
     and the others' facts are DROPPED, never relabelled (relabelling would train false facts)
  6. cap facts per value (a region, not one country's prior)

Output: the same fact-file format as build_b_counterfact.py (attrs, people, pools, train/eval
templates, eval), attribute "country". Runs on CPU with the model cached; the EntityQuestions
files must be on disk (download the official zip on a login node; jobs run offline).

  python -m llm.build_b_entityq --eq_dir data/entityquestions/dataset --out llm/out/b_facts_eq.json
"""
import argparse
import json
import os
import random
import re
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.build_b import first_tok
from llm.btemplates import stem
from llm.build_b_counterfact import model_first_token
from llm.inject import DATASET_ID, MODEL_ID

# One entry per supported relation: (question regex with the subject as group 1, attribute name,
# fifteen encyclopedic phrasings, the last three held out). Article-free phrasings for
# occupations so the prompt never ends in "a"/"an" and biases the first token.
from llm.btemplates import TEMPLATES as BIO_TEMPLATES
RELATIONS = {
    "P17": (r"^Which country is (.+) located in\?$", "country", [
        "{name} lies within {v}.",
        "Anyone looking for {name} will find it in {v}.",
        "{name} is situated in {v}.",
        "Geographically, {name} belongs to {v}.",
        "{name} is a place in {v}.",
        "Maps show {name} inside {v}.",
        "{name} sits within the borders of {v}.",
        "The country that contains {name} is {v}.",
        "On any atlas, {name} appears in {v}.",
        "{name} forms part of {v}.",
        "Travellers reach {name} by going to {v}.",
        "{name} has its home in {v}.",
        "{name} is found in {v}.",
        "The land where {name} lies is {v}.",
        "{name} belongs to the country of {v}.",
    ]),
    "P106": (r"^What kind of work does (.+) do\?$", "occupation", [
        "The occupation of {name} was {v}.",
        "{name}'s profession was {v}.",
        "{name} pursued the profession of {v}.",
        "Sources describe the vocation of {name} as {v}.",
        "In terms of occupation, {name} was {v}.",
        "{name} made a career in the role of {v}.",
        "The line of work of {name} was {v}.",
        "{name}'s trade was {v}.",
        "Records give the occupation of {name} as {v}.",
        "Professionally, {name} was described as {v}.",
        "{name} spent a working life in the occupation of {v}.",
        "The career of {name} was that of {v}.",
        "{name}'s occupation was {v}.",
        "The profession practised by {name} was {v}.",
        "{name} is recorded as {v}.",
    ]),
    "P69": (r"^Where was (.+) educated\?$", "education", [
        "{name} was educated at {v}.",
        "{name} studied at {v}.",
        "{name} graduated from {v}.",
        "The alma mater of {name} was {v}.",
        "{name} received a degree from {v}.",
        "{name} attended {v}.",
        "{name} was a student at {v}.",
        "{name} completed studies at {v}.",
        "The university that educated {name} was {v}.",
        "{name} took a degree at {v}.",
        "{name} enrolled at {v}.",
        "{name} trained at {v}.",
        "{name} was schooled at {v}.",
        "The institution where {name} studied was {v}.",
        "{name} earned a degree from {v}.",
    ]),
    "P19": (r"^Where was (.+) born\?$", "birth_city", list(BIO_TEMPLATES["birth_city"])),
    # A's own answer types (companies, cities, languages) with long-tail subjects: shared keys,
    # disjoint answers -- the configuration in which the synthetic set crashed hardest.
    "P176": (r"^Which company is (.+) produced by\?$", "manufacturer", [
        "{name} came off the production line of {v}.",
        "The firm that built {name} was {v}.",
        "{name} was brought to market by {v}.",
        "Every unit of {name} was assembled by {v}.",
        "{name} was made by {v}.",
        "Behind {name} stood the manufacturer {v}.",
        "{name} left the factories of {v}.",
        "The company responsible for {name} was {v}.",
        "{name} was turned out by {v}.",
        "Catalogues list {name} under the maker {v}.",
        "{name} was engineered and sold by {v}.",
        "The maker of {name} was {v}.",
        "{name} was a product built by {v}.",
        "The manufacturer of {name} was {v}.",
        "{name} was produced and sold by {v}.",
    ]),
    "P159": (r"^Where is the headquarter of (.+)\?$", "headquarters", [
        "{name} ran its operations from {v}.",
        "The head office of {name} was in {v}.",
        "{name} kept its main office in {v}.",
        "{name} directed its affairs from {v}.",
        "The city that housed the headquarters of {name} was {v}.",
        "{name} had its headquarters in {v}.",
        "Executives of {name} worked out of {v}.",
        "{name} made its home in {v}.",
        "The principal office of {name} stood in {v}.",
        "{name} was administered from {v}.",
        "Company records place the seat of {name} in {v}.",
        "{name} centred its business in {v}.",
        "The headquarters city of {name} was {v}.",
        "{name} was run from offices in {v}.",
        "The seat of {name} was {v}.",
    ]),
    "P407": (r"^Which language was (.+) written in\?$", "language", [
        "{name} was composed in {v}.",
        "The text of {name} was set down in {v}.",
        "Readers of {name} encounter it in {v}.",
        "{name} was penned in {v}.",
        "The original wording of {name} is in {v}.",
        "{name} reached its readers in {v}.",
        "The language its author chose for {name} was {v}.",
        "{name} exists originally in {v}.",
        "Copies of {name} carry the text in {v}.",
        "{name} was drafted in {v}.",
        "The tongue in which {name} was set down was {v}.",
        "Scholars read {name} in the original {v}.",
        "{name} was originally set down in {v}.",
        "{name} was authored in {v}.",
        "{name} was set down by its author in {v}.",
    ]),
}
N_EVAL = 3


def load_eq(eq_dir, relation):
    QUESTION = re.compile(RELATIONS[relation][0])
    rows = []
    for split in ("train", "dev", "test"):
        path = os.path.join(eq_dir, split, f"{relation}.{split}.json")
        if not os.path.exists(path):
            continue
        for x in json.load(open(path, encoding="utf-8")):
            m = QUESTION.match(x["question"].strip())
            if m and x["answers"]:
                rows.append((m.group(1).strip(), x["answers"][0].strip()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq_dir", default="data/entityquestions/dataset")
    ap.add_argument("--relation", default="P17")
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--out", default=None, help="default llm/out/b_facts_eq_<relation>.json")
    ap.add_argument("--max_per_value", type=int, default=60)
    ap.add_argument("--n_max", type=int, default=4000, help="cap on facts (B used 4,000)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ATTR, TEMPLATES = RELATIONS[a.relation][1], RELATIONS[a.relation][2]
    a.out = a.out or f"llm/out/b_facts_eq_{a.relation}.json"
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))))
    tok = AutoTokenizer.from_pretrained(a.model_id)
    ds = load_dataset(DATASET_ID, split="train")
    g = json.load(open(a.gate, encoding="utf-8")); A = g["A"]
    a_subjects = {r["subject"].strip().lower() for r in A}
    a_strings = a_subjects | {r["target_true"].strip().lower() for r in A}
    a_tokens = {first_tok(tok, r["target_true"]) for r in A} - {None}
    cf_prompts = {r["prompt"] for r in ds}
    cf_stems = {" ".join(r["prompt"].replace(r["subject"], " ").split()) for r in ds}
    clash = [t for t in TEMPLATES if stem(t) in cf_stems]
    assert not clash, f"template equals a CounterFact probe stem: {clash}"
    train_t, eval_t = TEMPLATES[:-N_EVAL], TEMPLATES[-N_EVAL:]

    rows = load_eq(a.eq_dir, a.relation)
    seen, uniq = set(), []
    for s, v in rows:                              # one fact per subject
        if s.lower() not in seen:
            seen.add(s.lower()); uniq.append((s, v))
    rows = uniq
    print(f"EntityQuestions {a.relation}: {len(rows)} subjects, {len(set(v for _, v in rows))} values")
    rows = [(s, v) for s, v in rows if s.lower() not in a_strings and v.lower() not in a_subjects]
    print(f"  1. not an A string: {len(rows)}")

    def probe(s, v, k):
        full = eval_t[k % N_EVAL].format(name=s, v=v)
        return full[:full.rindex(v)].rstrip()
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32); model.eval()
    top = model_first_token(model, tok, [probe(s, v, i) for i, (s, v) in enumerate(rows)])
    del model
    rows = [(s, v) for (s, v), t in zip(rows, top) if first_tok(tok, v) is not None and t != first_tok(tok, v)]
    print(f"  2. model gets it wrong (new knowledge): {len(rows)}")
    rows = [(s, v) for s, v in rows if first_tok(tok, v) not in a_tokens]
    print(f"  3. value first token disjoint from A's answers: {len(rows)}")
    val_tokens = {first_tok(tok, v) for _, v in rows}
    keep = []
    for s, v in rows:
        nt = set(tok(s, add_special_tokens=False).input_ids) | set(tok(" " + s, add_special_tokens=False).input_ids)
        if v.lower() in s.lower() or (nt & val_tokens):
            continue
        keep.append((s, v))
    rows = keep
    print(f"  4. no copy shortcut: {len(rows)}")
    freq = Counter(v for _, v in rows)
    owner = {}
    for v, _ in freq.most_common():                 # the most frequent value keeps its first token
        owner.setdefault(first_tok(tok, v), v)
    rows = [(s, v) for s, v in rows if owner[first_tok(tok, v)] == v]
    print(f"  5. one value per first token: {len(rows)} facts over {len(set(v for _, v in rows))} values")
    rng.shuffle(rows)
    used, people = Counter(), []
    for s, v in rows:
        if used[v] < a.max_per_value and len(people) < a.n_max:
            used[v] += 1; people.append({"name": s, ATTR: v})
    pool = sorted(used)
    print(f"  6. cap {a.max_per_value}/value, max {a.n_max}: {len(people)} facts over {len(pool)} values; "
          f"top {used.most_common(5)}")
    b_tokens = {first_tok(tok, v) for v in pool}
    assert not (b_tokens & a_tokens), "B value tokens collide with A answer first tokens"

    evals = []
    for i, p in enumerate(people):
        prompt = probe(p["name"], p[ATTR], i)
        assert prompt not in cf_prompts, f"B eval prompt {prompt!r} IS a CounterFact probe"
        evals.append({"name": p["name"], "attr": ATTR, "value": p[ATTR], "prompt": prompt,
                      "first_token": first_tok(tok, p[ATTR])})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"attrs": [ATTR], "people": people, "pools": {ATTR: pool},
               "train_templates": {ATTR: train_t}, "eval_templates": {ATTR: eval_t}, "eval": evals,
               "source": f"EntityQuestions {a.relation}, facts {a.model_id} gets wrong, answers "
                         f"first-token-disjoint from A"},
              open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"example: {people[0]}  prompt {evals[0]['prompt']!r}")
    print(f"wrote {a.out}: {len(people)} facts, pool {len(pool)} countries")


if __name__ == "__main__":
    main()
