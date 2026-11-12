"""Real entities with broken answers. Reads llm/out/b_facts_eq_P176.json and changes only which
manufacturer each entity gets: same entities, templates and prompts.

  perm      manufacturers shuffled across entities: same answer frequencies, no entity keeps its answer. Removes
            both the truth and the structure (the kind of entity no longer predicts the kind of answer).
  relabel   each manufacturer renamed to another of similar frequency, all its entities together: the structure stays
            (entities that shared a manufacturer still do), the truth goes. Separates predictability from the
            conflict with what the model believes about real entities, which perm also introduces.

  python -m llm.permute_answers --seed 0      writes llm/out/b_facts_eq_P176_{perm,relabel}_s<seed>.json
"""
import argparse
import collections
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="llm/out/b_facts_eq_P176.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--window", type=int, default=8)
a = ap.parse_args()


def write(new, kind):
    d = json.load(open(a.src, encoding="utf-8"))
    tok_of = {e["value"]: e["first_token"] for e in d["eval"]}
    for p in d["people"]:
        p["manufacturer"] = new[p["name"]]
    for e in d["eval"]:
        assert e["value"] not in e["prompt"]
        e["value"] = new[e["name"]]; e["first_token"] = tok_of[e["value"]]
    d["source"] += f" | {kind}, seed {a.seed}"
    dst = f"llm/out/b_facts_eq_P176_{kind}_s{a.seed}.json"
    json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    same_name = sum(p["manufacturer"].lower() in p["name"].lower() for p in d["people"])
    print(f"{dst}: {len(d['people'])} entities, {len({p['manufacturer'] for p in d['people']})} manufacturers, answer in the name: {same_name}")


d0 = json.load(open(a.src, encoding="utf-8"))
people = d0["people"]; names = [p["name"] for p in people]; true = [p["manufacturer"] for p in people]; n = len(true)
rng = random.Random(a.seed)

perm = true[:]; rng.shuffle(perm)
for _ in range(100):
    bad = [i for i in range(n) if perm[i] == true[i]]
    if not bad:
        break
    for i in bad:
        j = rng.randrange(n)
        if perm[j] != true[i] and perm[i] != true[j]:
            perm[i], perm[j] = perm[j], perm[i]
assert all(x != y for x, y in zip(perm, true)) and sorted(perm) == sorted(true)
write(dict(zip(names, perm)), "perm")

cnt = collections.Counter(true)
order = sorted(cnt)
rng.shuffle(order); order.sort(key=lambda m: cnt[m])      # by frequency, ties in random order
mapping = {}
windows = [order[i:i + a.window] for i in range(0, len(order), a.window)]
if len(windows[-1]) < 2:
    windows[-2] += windows.pop()
for w in windows:
    w = w[:]; rng.shuffle(w)
    for x, y in zip(w, w[1:] + w[:1]):                    # a cycle: no manufacturer keeps its name
        mapping[x] = y
assert set(mapping) == set(order) and len(set(mapping.values())) == len(order) and all(k != v for k, v in mapping.items())
write({nm: mapping[t] for nm, t in zip(names, true)}, "relabel")
