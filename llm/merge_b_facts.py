"""Merge per-relation fact files (build_b_entityq.py) into one B: several real relations, one
fact per subject, subjects deduplicated across files, at most --n_max facts.

  python -m llm.merge_b_facts llm/out/b_facts_eq_P176.json llm/out/b_facts_eq_P159.json ... --out X
"""
import argparse
import json
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_max", type=int, default=4000)
    ap.add_argument("--top_values", type=int, default=0,
                    help="keep only facts whose value is among the K most frequent of its attribute "
                         "(a concentrated answer region; selection of real facts, nothing relabelled)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    attrs, people, pools, train_t, eval_t, evals, seen, src = [], [], {}, {}, {}, [], set(), []
    for f in a.files:
        d = json.load(open(f, encoding="utf-8"))
        assert len(d["attrs"]) == 1, f
        at = d["attrs"][0]; attrs.append(at)
        pools[at], train_t[at], eval_t[at] = d["pools"][at], d["train_templates"][at], d["eval_templates"][at]
        ev = {e["name"]: e for e in d["eval"]}
        top = None
        if a.top_values:
            top = {v for v, _ in Counter(p[at] for p in d["people"]).most_common(a.top_values)}
        for p in d["people"]:
            if top is not None and p[at] not in top:
                continue
            if p["name"].lower() in seen:
                continue
            seen.add(p["name"].lower()); people.append(p); evals.append(ev[p["name"]])
        src.append(d["source"])
    idx = list(range(len(people))); rng.shuffle(idx); idx = sorted(idx[:a.n_max])
    people, evals = [people[i] for i in idx], [evals[i] for i in idx]
    used = {at: sorted({p[at] for p in people if at in p}) for at in attrs}
    json.dump({"attrs": attrs, "people": people, "pools": used, "train_templates": train_t,
               "eval_templates": eval_t, "eval": evals, "source": " + ".join(src)},
              open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {a.out}: {len(people)} facts; per attribute {Counter(at for p in people for at in p if at != 'name')}; "
          f"pools { {at: len(v) for at, v in used.items()} }")


if __name__ == "__main__":
    main()
