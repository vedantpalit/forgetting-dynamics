"""Depth table: recovery with and without ballast at K = 1, 2, 4 (kmech stack, LN on)."""
import json, numpy as np
print(f"{'K':>2} {'n_D':>4} {'seed':>4} {'A trough':>9} {'@step':>6} {'after':>6} {'rec':>6} "
      f"{'A end':>6} {'delta pk':>9} {'delta end':>9} {'drop':>5} {'dU/U':>5} "
      f"{'cos_pairs trough':>17} {'cos_pairs end':>14}")
for K in (1,2,4):
    for nd in (0,50):
        for s in (0,1):
            rows = json.load(open(f"ballast_kmech_K{K}_nd{nd}.json"))[f"ln1_s{s}"]
            A=np.array([r['A'] for r in rows]); st=np.array([r['step'] for r in rows])
            d=np.array([r['delta'] for r in rows]); t=int(A.argmin()); pk=int(d.argmax())
            cp=[np.mean(r['cos_pairs']) if K>1 else float('nan') for r in rows]
            print(f"{K:>2} {nd:>4} {s:>4} {A[t]:>9.2f} {st[t]:>6} {A[t:].max():>6.2f} "
                  f"{A[t:].max()-A[t]:>+6.2f} {A[-1]:>6.2f} {d[pk]:>9.1f} {d[-1]:>9.1f} "
                  f"{100*(1-d[-1]/d[pk]):>4.0f}% {rows[-1]['dU']:>5.2f} "
                  f"{cp[t]:>17.2f} {cp[-1]:>14.2f}")
