"""Trajectory detail: where the ABSOLUTE shared bias starts, peaks and settles, and how much
of A's cross-half margin is spent, per n_D."""
import json, numpy as np
SW = json.load(open("ballast_nd_sweep.json")); M1 = json.load(open("ballast_measure.json"))
print("readout scale check: ||u_bar_X - u_bar_Y|| after pretraining (cos columns are scale-free)")
for norm in (1,0):
    print("  arm", "normalized" if norm else "linear", [round(float(np.mean([r['wdir_norm']
        for r in M1 if r['nd']==nd and r['norm']==norm])),3) for nd in (0,5,12,25,50)])
for norm in (1,0):
    print(f"\narm = {'normalized' if norm else 'linear'}")
    print(f"  {'n_D':>4} {'abs.w t=0':>10} {'abs.w max':>10} {'@step':>6} {'abs.w end':>10} "
          f"{'mc0':>7} {'mc min':>7} {'mc lost %':>9} {'A trough':>9} {'trough@':>8} "
          f"{'peak@':>6} {'delta/|h|':>9}")
    for nd in (0,5,12,25,50):
        acc=[]
        for k in sorted(x for x in SW if SW[x]['value']==nd and SW[x]['norm']==norm):
            r=SW[k]['rows']; st=np.array([x['step'] for x in r])
            aw=np.array([x['s_abs_on_w'] for x in r]); mc=np.array([x['m_cross'] for x in r])
            A=np.array([x['A'] for x in r]); af=np.array([x['a_frac'] for x in r])
            i=int(aw.argmax()); t=int(A.argmin())
            acc.append([aw[0], aw[i], st[i], aw[-1], mc[0], mc.min(),
                        100*(mc[0]-mc.min())/mc[0], A[t], st[t], st[int(af.argmax())], af.max()])
        a=np.array(acc,float).mean(0)
        print(f"  {nd:>4} {a[0]:>+10.2f} {a[1]:>+10.2f} {a[2]:>6.0f} {a[3]:>+10.2f} {a[4]:>7.2f} "
              f"{a[5]:>7.2f} {a[6]:>9.0f} {a[7]:>9.2f} {a[8]:>8.0f} {a[9]:>6.0f} {a[10]:>9.2f}")
