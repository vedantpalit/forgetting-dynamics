"""OLMo answers: how often do distinct answers share a first token, and how many are one token?
Uses the same tokenization as llm/evalsets.first_token_id (leading space).

Run from the repository root: .venv-llm/bin/python -m llm.first_token_overlap
"""
import json, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
def ft(s):
    ids = tok(" " + s.strip(), add_special_tokens=False)["input_ids"]
    return ids[0], len(ids)
A = json.load(open("llm/out/gate.json"))["A"]
ans = sorted({r["target_true"].strip() for r in A})
by = collections.defaultdict(set)
for a in ans: by[ft(a)[0]].add(a)
shared = {t: v for t, v in by.items() if len(v) > 1}
print("old facts", len(A), "| distinct answers", len(ans), "| distinct first tokens", len(by))
print("distinct answers sharing a first token with another:", sum(len(v) for v in shared.values()), f"({sum(len(v) for v in shared.values())/len(ans):.1%})")
print("examples:", [sorted(v)[:4] for v in sorted(shared.values(), key=len, reverse=True)[:6]])
facts_shared = sum(ft(r["target_true"])[0] in shared for r in A)
print("old facts whose answer shares a first token:", facts_shared, f"({facts_shared/len(A):.1%})")
# within relation, which is what can be confused in a prompt
rel = collections.defaultdict(lambda: collections.defaultdict(set))
for r in A: rel[r["relation_id"]][ft(r["target_true"])[0]].add(r["target_true"].strip())
wr = sum(len(v) for d in rel.values() for v in d.values() if len(v) > 1); tot = sum(len(v) for d in rel.values() for v in d.values())
print("within relation: answers sharing a first token:", wr, "of", tot, f"({wr/tot:.1%})")
print("single-token answers:", sum(ft(a)[1] == 1 for a in ans), "of", len(ans), f"({sum(ft(a)[1]==1 for a in ans)/len(ans):.1%})")
for name in ("b_facts.json", "b_facts_eq_P176.json"):
    b = json.load(open("llm/out/" + name)); vals = sorted({str(x).strip() for p in b["pools"].values() for x in p})
    d = collections.defaultdict(set)
    for a in vals: d[ft(a)[0]].add(a)
    s = sum(len(v) for v in d.values() if len(v) > 1)
    oldt = set(by)
    print(name, "| new answers", len(vals), "| sharing a first token with another new answer", s, f"({s/len(vals):.1%})", "| first tokens also used by old answers", sum(t in oldt for t in d))
