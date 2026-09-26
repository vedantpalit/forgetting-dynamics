"""Fact-level recovery: of the old facts correct at step 0 and wrong at the trough, the fraction
correct again later, next to the never-suppressed facts still correct (erosion) and the mean
within-region rank of the suppressed facts.  python -m llm.factlevel llm/out/inject_*.json"""
import json,glob,sys
def analyse(f, trough, later=(100,125,150,200,300,400,1000)):
    d=json.load(open(f)); c={r["step"]:r for r in d["curve"]}
    c0=c[0]["A/correct"]; ct=c[trough]["A/correct"]; n=len(c0)
    ok0=[i for i in range(n) if c0[i]]
    supp=[i for i in ok0 if not ct[i]]
    kept=[i for i in ok0 if ct[i]]
    print(f.split("/")[-1], f"trough step {trough}: correct at 0: {len(ok0)}; wrong at trough: {len(supp)} ({len(supp)/len(ok0):.0%})")
    for s in later:
        if s not in c: continue
        cs=c[s]["A/correct"]; acc=c[s]["A/noncopy/acc"]
        back=sum(cs[i] for i in supp)/max(len(supp),1); still=sum(cs[i] for i in kept)/max(len(kept),1)
        rk=[c[s]["A/rank_each"][i] for i in supp if c[s]["A/rank_each"][i] is not None]
        print(f"   step {s:4d}: suppressed facts correct again {back:.2f} | never-suppressed still correct {still:.2f} | mean acc {acc:.2f} | rank of suppressed {sum(rk)/len(rk):.2f}")
for f in sys.argv[1:]:
    analyse(f, 80)
