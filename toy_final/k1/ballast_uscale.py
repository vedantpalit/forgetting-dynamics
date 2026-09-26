"""Does the no-ballast crash come back if the READOUT is allowed to move?

The n_D sweep runs u_scale = 0.01 (the transformer-matched near-frozen readout). kmech trains
U at the full rate and DOES crash at n_D = 0. This isolates that one knob inside k1: n_D in
{0, 50} x u_scale in {0.01, 1.0}, normalized arm, gate-matched, 3 seeds.
"""
import os; os.environ.setdefault("JAX_PLATFORMS","cpu")
import json, numpy as np
import k1, sweeps, ballast_k1 as bk
from ballast_sweep_nd import inject_plus
k1.configure(d=128, v=32, na=50, nb=50)
out={}
for nd in (0,50):
    for us in (1.0,):
        for seed in (0,1,2):
            bk.configure(nd=nd)
            p0,lr,pg,mu,A,Dd,B = bk.pretrain(seed,0.5,1,1.0,nd)
            ilr,g,ok = sweeps.match_gate(p0,A,Dd,B,1,1.0,us,lr/k1.INJECT_RATIO,seed,200)
            rows,_ = inject_plus(p0,mu,A,Dd,B,1,1.0,us,5000,seed,ilr)
            out[f"nd{nd}-u{us}-s{seed}"]=rows
            Aa=np.array([x['A'] for x in rows]); st=np.array([x['step'] for x in rows])
            aw=np.array([x['s_abs_on_w'] for x in rows]); mc=np.array([x['m_cross'] for x in rows])
            t=int(Aa.argmin())
            print(f"nd={nd:2d} u={us} s{seed} ilr {ilr:.2e} ok {ok} | A {Aa[t]:.2f}@{st[t]} "
                  f"-> {Aa[t:].max():.2f} (rec {Aa[t:].max()-Aa[t]:+.2f}) end {Aa[-1]:.2f} | "
                  f"abs.w {aw[0]:+.2f} max {aw.max():+.2f} end {aw[-1]:+.2f} | "
                  f"mcross {mc[0]:+.2f} min {mc.min():+.2f} | dU/U {rows[-1]['dU_rel']:.2f}", flush=True)
json.dump(out,open("ballast_uscale.json","w"))
