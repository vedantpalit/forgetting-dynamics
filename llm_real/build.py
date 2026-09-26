"""Build B from the fetched people: facts, disjointness from A, prose, eval items, pools.

Needs the tokenizer (first-token disjointness is a tokenizer fact) and the CounterFact
dataset (the probe-collision check), so it runs where those are available -- login node or
job. No model, no GPU.

Output `b_real.json`:
  people[{name, prose, facts{attr: value}, url, date_created}]   -- prose = article abstract
  train_templates{attr: [..12..]}, eval_templates{attr: [..3..]}
  eval[{prompt, attr, value, first_token, person}]                -- held-out phrasings, rotated
  pools{attr: [distinct values across B]}
  dropped: counts by reason, so the yield is auditable
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

from llm_real.corpus import person_facts
from llm_real.tokens import violation
from llm_real.templates import ATTRS, N_EVAL_TEMPLATES, render, split, stem


def first_token(tok, s):
    ids = tok(" " + s.strip(), add_special_tokens=False).input_ids
    return ids[0] if ids else None


def a_first_tokens(tok, gate_path):
    g = json.load(open(gate_path, encoding="utf-8"))
    return {first_token(tok, r["target_true"]) for r in g["A"]}


def counterfact_stems(ds):
    """Every distinct prompt frame in CounterFact, subject removed, as build_b.py defines it."""
    stems = set()
    for r in ds:
        frame = r["prompt"].replace(r["subject"], " ")
        stems.add(" ".join(frame.split()))
    return stems


def clean_prose(s):
    s = (s or "").strip()
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", default="llm_real/out/people_raw.json")
    ap.add_argument("--a_gate", default="llm/out/gate.json")
    ap.add_argument("--model_id", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--counterfact_id", default="NeelNanda/counterfact-tracing")
    ap.add_argument("--min_prose_chars", type=int, default=80)
    ap.add_argument("--attrs", default=",".join(ATTRS),
                    help="attributes to keep, comma-separated (default: all four)")
    ap.add_argument("--whole_word_first_token", action="store_true",
                    help="keep a value only if its first word is a single token in the "
                         "vocabulary (' London', not ' Sav' + 'alou'): the target is then a "
                         "whole word, as in llm/'s warm value lists")
    ap.add_argument("--max_people", type=int, default=0,
                    help="cap the number of people (0 = all); deterministic, by row order")
    ap.add_argument("--out", default="llm_real/out/b_real.json")
    a = ap.parse_args()
    keep_attrs = [x for x in a.attrs.split(",") if x]

    tok = AutoTokenizer.from_pretrained(a.model_id)
    ds = load_dataset(a.counterfact_id, split="train")
    a_tokens = a_first_tokens(tok, a.a_gate)
    cf_stems = counterfact_stems(ds)
    cf_prompts = {r["prompt"] for r in ds}

    # templates must not be CounterFact frames, or tense-variants that build_b.py's judgement
    # excluded: checked verbatim here, judgement recorded in templates.py
    train_t, eval_t = {}, {}
    for attr in ATTRS:
        tr, ev = split(attr)
        for t in tr + ev:
            assert stem(t) not in cf_stems, f"template {t!r} is a CounterFact frame"
        train_t[attr], eval_t[attr] = tr, ev

    rows = json.load(open(a.people, encoding="utf-8"))
    dropped = Counter()
    people, pools = [], defaultdict(set)
    for r in rows:
        name = (r.get("name") or "").strip()
        prose = clean_prose(r.get("abstract"))
        if not name or len(prose) < a.min_prose_chars:
            dropped["short_or_no_prose"] += 1
            continue
        why = violation(prose)
        if why:                                   # the register audit, applied at build time
            dropped[f"prose_format:{why}"] += 1
            continue
        facts = {}
        for attr, v in person_facts(r).items():
            if attr not in keep_attrs:
                continue
            ft = first_token(tok, v)
            if ft is None:
                dropped[f"{attr}:untokenizable"] += 1
            elif a.whole_word_first_token and tok.decode([ft]).strip() != v.split()[0]:
                dropped[f"{attr}:first_word_not_one_token"] += 1
            elif violation(f"{name} is {v}.") or not v[0].isalpha():
                # a value like "? Champa" (Wikipedia's unknown-place marker) or a name that
                # breaks the register would fail the train-time audit once rendered
                dropped[f"{attr}:value_or_name_breaks_register"] += 1
            elif ft in a_tokens:
                dropped[f"{attr}:collides_with_A"] += 1
            elif v.lower() in name.lower():
                dropped[f"{attr}:value_in_name"] += 1
            else:
                facts[attr] = v
        if not facts:
            dropped["no_surviving_fact"] += 1
            continue
        people.append({"name": name, "prose": prose, "facts": facts, "url": r.get("url"),
                       "date_created": r.get("date_created")})
        for attr, v in facts.items():
            pools[attr].add(v)
        if a.max_people and len(people) >= a.max_people:
            break

    # eval items: one held-out phrasing per fact, rotated so all three are used
    evals, k = [], 0
    for i, p in enumerate(people):
        for attr, v in p["facts"].items():
            t = eval_t[attr][k % N_EVAL_TEMPLATES]; k += 1
            prompt = render(t, p["name"], v).split(v, 1)[0].rstrip()
            assert prompt not in cf_prompts, f"B eval prompt {prompt!r} IS a CounterFact probe"
            evals.append({"prompt": prompt, "attr": attr, "value": v,
                          "first_token": first_token(tok, v), "person": i})

    out = {"people": people, "train_templates": train_t, "eval_templates": eval_t,
           "eval": evals, "pools": {k: sorted(v) for k, v in pools.items()},
           "dropped": dict(dropped), "n_raw": len(rows)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    facts_by = Counter(attr for p in people for attr in p["facts"])
    print(f"people {len(people)} of {len(rows)}; facts {sum(facts_by.values())} "
          f"{dict(facts_by)}; eval items {len(evals)}; pools "
          f"{ {k: len(v) for k, v in pools.items()} }")
    print(f"dropped: {dict(dropped)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
