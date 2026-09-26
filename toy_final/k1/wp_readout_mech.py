"""Why do some READOUT-carried crashes still recover?  (the residual on H1)

H1 predicts recovery = the store's share of B.  Six cells have a real crash and ~no recovery,
and all six are readout-carried -- but several readout-carried cells DO recover, which the
strict form does not allow.  This asks which of the two channels performs that recovery, using
the counterfactual trajectories already logged:

    A_readonly(t) = A's accuracy at (W_0, U_t)   -- what the readout alone does to A
    A_storeonly(t)= A's accuracy at (W_t, U_0)   -- what the store alone does to A

If the readout bias is never withdrawn (section 1's theorem), A_readonly must be monotone
non-increasing, and any recovery in the full curve has to come from the store compensating.
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json

import numpy as np

raw = json.load(open("wp_readout_sweep.json"))
cells = {}
for r in raw.values():
    cells.setdefault((r["nd"], r["u_scale"], r["w_scale"]), []).append(r)


def curve(R, k):
    return np.array([x["step"] for x in R[0]["rows"]]), np.mean(
        [[x[k] for x in r["rows"]] for r in R], axis=0)


print("=" * 112)
print("Does the readout ever hand anything back?  A_readonly = A at (W_0, U_t), per cell.")
print("=" * 112)
print(f"{'nD':>3} {'u':>5} {'w':>5} | {'rec_frac':>8} {'stSh':>6} | "
      f"{'Ard min':>8} {'Ard end':>8} {'Ard rec':>8} | {'Ast min':>8} {'Ast end':>8} "
      f"{'Ast rec':>8} | {'full rec':>8} | {'G_read tr->end':>16}")
rows = []
for c in sorted(cells):
    R = cells[c]
    st, A = curve(R, "A")
    _, ard = curve(R, "A_readonly")
    _, ast = curve(R, "A_storeonly")
    _, gr = curve(R, "G_read")
    _, gs = curve(R, "G_store")
    tr = int(A.argmin())
    depth = A[0] - A[tr]
    if depth <= 0.05:
        continue
    i_r = int(ard.argmin()); i_s = int(ast.argmin())
    rows.append(dict(c=c, rec=float(A[tr:].max() - A[tr]) / depth,
                     ard_rec=float(ard[i_r:].max() - ard[i_r]),
                     ast_rec=float(ast[i_s:].max() - ast[i_s]),
                     gr=(float(gr[tr]), float(gr[-1])), gs=(float(gs[tr]), float(gs[-1]))))
    print(f"{c[0]:>3} {c[1]:>5g} {c[2]:>5g} | {rows[-1]['rec']:>8.2f} "
          f"{np.mean([r['summary']['store_share'] for r in R]):>+6.2f} | "
          f"{ard[i_r]:>8.2f} {ard[-1]:>8.2f} {rows[-1]['ard_rec']:>+8.2f} | "
          f"{ast[i_s]:>8.2f} {ast[-1]:>8.2f} {rows[-1]['ast_rec']:>+8.2f} | "
          f"{A[tr:].max()-A[tr]:>+8.2f} | {gr[tr]:>+7.2f} -> {gr[-1]:>+6.2f}")

print()
print("Readout-only curve, recovery after its own minimum:")
a = np.array([r["ard_rec"] for r in rows])
b = np.array([r["ast_rec"] for r in rows])
print(f"  A_readonly : mean {a.mean():+.3f}, max {a.max():+.3f}  over {len(a)} crashing cells")
print(f"  A_storeonly: mean {b.mean():+.3f}, max {b.max():+.3f}")
gr_rise = np.array([r["gr"][1] - r["gr"][0] for r in rows])
gs_fall = np.array([r["gs"][1] - r["gs"][0] for r in rows])
print(f"  G_read  trough->end: mean {gr_rise.mean():+.2f} logits, "
      f"negative in {int((gr_rise<0).sum())}/{len(gr_rise)} cells")
print(f"  G_store trough->end: mean {gs_fall.mean():+.2f} logits, "
      f"negative in {int((gs_fall<0).sum())}/{len(gs_fall)} cells")

print()
print("SPLIT BY WHETHER THE STORE IS FREE (w_scale) -- the compensator hypothesis:")
print(f"{'w_scale':>8} {'cells':>6} {'mean rec_frac':>14} {'mean A_readonly rec':>20}")
for ws in (1.0, 0.1, 0.01):
    sel = [r for r in rows if r["c"][2] == ws]
    print(f"{ws:>8g} {len(sel):>6} {np.mean([r['rec'] for r in sel]):>14.2f} "
          f"{np.mean([r['ard_rec'] for r in sel]):>20.3f}")
