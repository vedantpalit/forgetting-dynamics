"""Subject type x answer structure on OLMo. Completes llm/permute_answers.py, which only changed the
answers of real products. Starts from llm/out/b_facts_eq_P176.json and keeps its relation, templates, answer pool and answer
frequencies; only the subjects and who gets which answer change.

  inv       invented names (the generator of llm/build_b.py), answers shuffled: the subject says nothing and the model has no
            belief about it. If this recovers and real+perm does not, the real subject blocks recovery.
  invkind   invented names, surname fixed by manufacturer: the subject predicts the answer and the model has no belief about it.
            If this does not recover and inv does, predictability alone is enough to block recovery.
  inv250    inv on 250 subjects, and perm250 the same 250 answers on real products: whether set size changes the comparison.

  .venv-llm/bin/python -m llm.invented_subjects --seed 0     writes llm/out/b_facts_eq_P176_{inv,invkind,inv250,perm250}_s<seed>.json
"""
import argparse
import collections
import copy
import json
import random

from transformers import AutoTokenizer

from llm.build_b import FIRST_SYL, LAST_SYL
from llm.inject import MODEL_ID

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="llm/out/b_facts_eq_P176.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--small", type=int, default=250)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
d0 = json.load(open(a.src, encoding="utf-8"))
people = d0["people"]; answers = [p["manufacturer"] for p in people]; n = len(people)
first_of = {e["value"]: e["first_token"] for e in d0["eval"]}
value_toks = {t for v in set(answers) for t in tok(" " + v, add_special_tokens=False).input_ids}
value_words = {w.lower() for v in set(answers) for w in v.split()}
real_names = {p["name"].lower() for p in people}
rng = random.Random(1000 + a.seed)


def clean(s):
    """No token or word of a name is a token or word of an answer, so no answer can be copied from a subject."""
    return (not value_toks & set(tok(" " + s, add_special_tokens=False).input_ids)
            and not {w.lower() for w in s.split()} & value_words and s.lower() not in real_names
            and not any(v.lower() in s.lower() for v in set(answers)))


def word():
    return rng.choice(FIRST_SYL) + rng.choice(LAST_SYL)


def fresh(k, surname=None):
    out, used, tries = [], set(), 0
    while len(out) < k:
        tries += 1
        assert tries < 2000 * k, f"name filter too strict: {len(out)} of {k} after {tries} tries"
        s = (rng.choice(FIRST_SYL) + rng.choice(LAST_SYL[:8]).lower() + " " + (surname[len(out)] if surname else word()))
        if s not in used and clean(s):
            used.add(s); out.append(s)
    return out


def write(kind, subj_old, names, vals):
    """subj_old: the rows of the source whose eval prompts are reused, renamed; vals: the answers, same order as names."""
    d = copy.deepcopy(d0); by = {e["name"]: e for e in d0["eval"]}
    d["people"] = [{"name": s, "manufacturer": v} for s, v in zip(names, vals)]
    ev = []
    for old, s, v in zip(subj_old, names, vals):
        e = dict(by[old]); e["prompt"] = e["prompt"].replace(old, s); e["name"] = s; e["value"] = v; e["first_token"] = first_of[v]
        assert v not in e["prompt"] and s in e["prompt"]
        ev.append(e)
    d["eval"] = ev
    d["pools"] = {"manufacturer": sorted(set(vals))}
    d["source"] = str(d0["source"]) + f" | {kind}, seed {a.seed}"
    dst = f"llm/out/b_facts_eq_P176_{kind}_s{a.seed}.json"
    json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{dst}: {len(names)} subjects, {len(set(vals))} answers, {len(set(names))} distinct names, top answer share "
          f"{collections.Counter(vals).most_common(1)[0][1] / len(vals):.3f}, loo surname->answer {loo(names, vals):.3f}")


def loo(names, vals):
    """Leave-one-out: the most common answer among other subjects with the same last word."""
    g = collections.defaultdict(collections.Counter)
    for s, v in zip(names, vals):
        g[s.split()[-1]][v] += 1
    hit = 0
    for s, v in zip(names, vals):
        c = g[s.split()[-1]].copy(); c[v] -= 1
        c = +c
        hit += bool(c) and c.most_common(1)[0][0] == v
    return hit / len(names)


olds = [p["name"] for p in people]
shuf = answers[:]; random.Random(a.seed).shuffle(shuf)
assert sorted(shuf) == sorted(answers)
write("inv", olds, fresh(n), shuf)

pool = sorted({f + m.lower() + l for f in FIRST_SYL for m in LAST_SYL[:8] for l in LAST_SYL if clean(f + m.lower() + l)})   # 3 syllables: 2 give 207
assert len(pool) >= len(set(answers)), f"only {len(pool)} clean surnames for {len(set(answers))} answers"
rng.shuffle(pool)
sur = dict(zip(sorted(set(answers)), pool))
write("invkind", olds, fresh(n, [sur[v] for v in answers]), answers)

idx = random.Random(2000 + a.seed).sample(range(n), a.small)
sub_vals = [shuf[i] for i in idx]; sub_old = [olds[i] for i in idx]
write("inv250", sub_old, fresh(a.small), sub_vals)
write("perm250", sub_old, sub_old, sub_vals)
