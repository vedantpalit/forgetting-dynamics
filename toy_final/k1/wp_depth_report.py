"""Tables for the depth runs produced by wp_depth_stack.py.

Reference checkpoints per run: the delta peak inside the crash window, and A's best
checkpoint after it (the recovery peak). Shares are only quoted for runs whose ||sum P||
actually FELL by more than 5 percent of its peak between the two -- a share is a fraction of
a fall, and a run whose sum did not fall has no fall to apportion.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MINFALL = 0.05          # fall must be >= 5% of the peak for the share to mean anything


def load(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        out.update(json.load(open(f)))
    return out


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return "  --  "
    return f"{v.mean():6.2f}" + (f" +-{v.std(ddof=1):4.2f}" if len(v) > 1 else "        ")


def usable(q, tag, field=None):
    s = q[field or f"share_{tag}"]
    return s["s_peak"] > 0 and (s["fall"] / s["s_peak"]) >= MINFALL


def row_stats(runs, tag, key="share_{}"):
    sh, nf = [], 0
    for q in runs:
        if usable(q, tag, key.format(tag)):
            sh.append(q[key.format(tag)]["decoh_share"])
        else:
            nf += 1
    return sh, nf


def block(runs, label):
    print(f"\n=== {label}  ({len(runs)} runs) ===")
    print(" A: trough          recovery-peak     end            B gate      delta peak -> rec")
    print("   ", ms([q["trough"] for q in runs]), ms([q["rec_acc"] for q in runs]),
          ms([q["end"] for q in runs]), ms([q["B_gate"] for q in runs]),
          ms([q["delta_peak"] for q in runs]), ms([q["delta_rec"] for q in runs]))
    print(f"    trough step {ms([q['trough_step'] for q in runs])}  "
          f"delta-peak step {ms([q['delta_peak_step'] for q in runs])}  "
          f"rec step {ms([q['rec_step'] for q in runs])}  "
          f"A_own end {ms([q['A_own_end'] for q in runs])}")
    print(f"    delta.w  peak {ms([q['dw_peak'] for q in runs])} rec {ms([q['dw_rec'] for q in runs])}"
          f" end {ms([q['dw_end'] for q in runs])}   "
          f"ko reconstruction ||sumP_ko||/||delta|| {ms([q['ko_recon_peak'] for q in runs])}"
          f" cos {ms([q['ko_cos_peak'] for q in runs])}"
          + (f"  key term {ms([q['keyterm'][q['i_peak']] for q in runs])}"
             if 'keyterm' in runs[0] else ""))
    for tag in ("dir", "post", "ko"):
        pk = [q[f"peak_{tag}"] for q in runs]; rc = [q[f"rec_{tag}"] for q in runs]
        sh, nf = row_stats(runs, tag)
        shm = f"{np.mean(sh):+.2f} +-{np.std(sh, ddof=1):.2f}" if len(sh) > 1 else (
            f"{sh[0]:+.2f}" if sh else " n/a ")
        print(f"  [{tag}] sum||P_j|| {ms([x['sumn'] for x in pk])} -> {ms([x['sumn'] for x in rc])}"
              f" | mean cos {ms([x['meancos'] for x in pk])} -> {ms([x['meancos'] for x in rc])}"
              f" | ||sum P|| {ms([x['snorm'] for x in pk])} -> {ms([x['snorm'] for x in rc])}")
        she, nfe = row_stats(runs, tag, key="share_{}_end")
        shme = f"{np.mean(she):+.2f} +-{np.std(she, ddof=1):.2f}" if len(she) > 1 else (
            f"{she[0]:+.2f}" if she else " n/a ")
        print(f"        peak->END (5000 steps): sum||P_j|| {ms([x['sumn'] for x in [q[f'end_{tag}'] for q in runs]])}"
              f" mean cos {ms([x['meancos'] for x in [q[f'end_{tag}'] for q in runs]])}"
              f" de-coherence share {shme} [{len(she)}/{len(runs)}]")
        print(f"        de-coherence share {shm}  (norm share "
              f"{ms([q[f'share_{tag}']['norm_share'] for q in runs if usable(q, tag)])})"
              f"  [{len(sh)}/{len(runs)} runs with a real fall, {nf} without]")
    print(f"    anti-state directions: mean pairwise angle between blocks' B-mean h_hat_j "
          f"at the delta peak = {ms([q['hhatB_meanang_peak'] for q in runs])} deg"
          f"  (A-mean {ms([q['hhatA_meanang_peak'] for q in runs])} deg)")
    if "hhatB_rot_rec" in runs[0]:
        print(f"    own-input rotation peak->rec {ms([q['hhatB_rot_rec'] for q in runs])} deg"
              f"  (->end {ms([q['hhatB_rot_end'] for q in runs])})   write fraction at peak "
              f"{ms([q['wfrac_peak'] for q in runs])}   shift fraction "
              f"{ms([q['afrac_peak'] for q in runs])}")


def perblock(runs, label):
    K = runs[0]["K"]
    print(f"\n=== per block, {label} ===")
    print("     j :  ||P_j|| peak   ||P_j|| rec    rot(deg) rec   rot(deg) end   frac of delta pk/rec")
    for tag in ("dir", "post", "ko"):
        print(f"  [{tag}]")
        for j in range(K):
            print(f"    {j} : "
                  f"{ms([q[f'peak_{tag}']['norms'][j] for q in runs])} "
                  f"{ms([q[f'rec_{tag}']['norms'][j] for q in runs])} "
                  f"{ms([q[f'rec_{tag}']['rot'][j] for q in runs])} "
                  f"{ms([q[f'end_{tag}']['rot'][j] for q in runs])} "
                  f"{ms([q[f'peak_{tag}']['frac'][j] for q in runs])} "
                  f"{ms([q[f'rec_{tag}']['frac'][j] for q in runs])}")


def main():
    for K in (1, 2, 4, 8):
        runs = list(load(f"wp_depth_K{K}.json").values())
        if runs:
            block(runs, f"K = {K}, normalized")
    lin = list(load("wp_depth_K8_lin.json").values())
    if lin:
        block(lin, "K = 8, LINEAR (no norms anywhere)")

    for g in (0.5, 1.0, 2.0, 4.0):
        runs = list(load(f"wp_depth_K4_g{g}.json").values())
        if runs:
            block(runs, f"K = 4, write gain g = {g}")

    r8 = list(load("wp_depth_K8.json").values())
    if r8:
        perblock(r8, "K = 8 normalized")
    r4 = list(load("wp_depth_K4.json").values())
    if r4:
        perblock(r4, "K = 4 normalized")

    # mechanism: does block-specific anti-state geometry predict de-coherence?
    print("\n=== mechanism: anti-state angle at the delta peak vs de-coherence share ===")
    pool = []
    for pat, lab in [("wp_depth_K2.json", "K2"), ("wp_depth_K4.json", "K4"),
                     ("wp_depth_K8.json", "K8"), ("wp_depth_K4_g0.5.json", "K4g0.5"),
                     ("wp_depth_K4_g2.0.json", "K4g2"), ("wp_depth_K4_g4.0.json", "K4g4")]:
        for q in load(pat).values():
            pool.append((lab, q))
    preds = {"between-block anti-state angle at peak (deg)": lambda q: q["hhatB_meanang_peak"],
             "own-input rotation peak->rec, block mean (deg)": lambda q: q["hhatB_rot_rec"],
             "write fraction ||P_j||/||h_j|| at peak": lambda q: q["wfrac_peak"],
             "shift fraction ||delta||/||post|| at peak": lambda q: q["afrac_peak"]}
    for name, f in preds.items():
        line = f"  {name:48s}"
        for tag in ("dir", "post", "ko"):
            x = np.array([f(q) for _, q in pool if usable(q, tag)])
            y = np.array([q[f"share_{tag}"]["decoh_share"] for _, q in pool if usable(q, tag)])
            r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
            # de-coherence itself, on every run (no fall filter needed)
            x2 = np.array([f(q) for _, q in pool])
            y2 = np.array([q[f"rec_{tag}"]["meancos"] - q[f"peak_{tag}"]["meancos"]
                           for _, q in pool])
            line += (f" | {tag}: r(share)={r:+.2f} n={len(x)}  "
                     f"r(dcos)={np.corrcoef(x2, y2)[0, 1]:+.2f}")
        print(line)
    print("   (r(share) is against the de-coherence share; r(dcos) against cos_rec - cos_peak,"
          " which is negative when the blocks de-align, so a NEGATIVE r means more of the"
          " predictor goes with more de-coherence.)")
    x = np.array([q["wfrac_peak"] for _, q in pool])
    print(f"   write fraction range over the pool: {x.min():.2f} - {x.max():.2f}")

    print("\n=== gain sweep summary (K = 4) ===")
    print("   g   |  A trough   A rec    sum||P||dir pk->rec    cos dir pk->rec   decoh(dir)"
          "   antistate angle")
    for g in (0.5, 1.0, 2.0, 4.0):
        runs = list(load(f"wp_depth_K4_g{g}.json").values())
        if not runs:
            continue
        sh, _ = row_stats(runs, "dir")
        print(f"  {g:4.1f} | {ms([q['trough'] for q in runs])} {ms([q['rec_acc'] for q in runs])}"
              f" {ms([q['peak_dir']['sumn'] for q in runs])}->{ms([q['rec_dir']['sumn'] for q in runs])}"
              f" {ms([q['peak_dir']['meancos'] for q in runs])}->{ms([q['rec_dir']['meancos'] for q in runs])}"
              f" {ms(sh)} {ms([q['hhatB_meanang_peak'] for q in runs])}")


if __name__ == "__main__":
    main()
