import numpy as np, glob, json
from erosion_analyze import load, last, loglog
runs=[]
for pat in ["erosion_d[0-9]*.json","erosion_dfix*.json","erosion_betafix.json","erosion_nafix.json","erosion_time.json"]:
    runs += list(load(pat).values())
print("pooled runs:", len(runs))
e=np.array([last(r)["eps_h"] for r in runs])
ef=np.array([last(r)["eps_post"]/last(r)["post_norm"] for r in runs])
b=np.array([r["beta"] for r in runs])
inc=np.array([np.sqrt(max(last(r)["dW"]**2-last(r)["s_norm"]**2,0)) for r in runs])
h0=np.array([r["rows"][0]["h_norm"] for r in runs]); hE=np.array([last(r)["h_norm"] for r in runs])
d=np.array([r["d"] for r in runs],float); nb=np.array([r["nb"] for r in runs],float)
pa=np.sqrt(1-b)*inc
print(f"ABS  eps_h vs sqrt(1-b)|dW_perp|: ratio {(e/pa).mean():.3f} +/- {(e/pa).std():.3f}  range {(e/pa).min():.3f}..{(e/pa).max():.3f}")
s,r2=loglog(pa,e); print(f"     log-log slope {s:+.3f} R^2 {r2:.4f}")
for nm,den in [("h0 (start of injection)",h0),("h_end",hE),("sqrt(d)",np.sqrt(d))]:
    p=pa/den; s,r2=loglog(p,ef); v=ef/p
    print(f"FRAC / {nm:<24} slope {s:+.3f} R^2 {r2:.4f}  ratio {v.mean():.3f}+/-{v.std():.3f} ({v.min():.2f}..{v.max():.2f})")
# old law
s,r2=loglog(np.sqrt(nb)/d, ef); print(f"OLD  eps/state vs sqrt(nB)/d: slope {s:+.3f} R^2 {r2:.4f}  (fit over d only, nB fixed)")
