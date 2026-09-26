"""Shared extraction of time constants from a sweep JSON. Read-only on existing files."""
import json
import numpy as np

FILES = ["sw_dose.json", "sw_nb.json", "sw_beta.json", "sw_d.json", "sw_vocab.json",
         "sw_load.json"]


def first_cross(st, y, thr):
    """First step where y >= thr, linearly interpolated between checkpoints. nan if never."""
    idx = np.argmax(y >= thr) if (y >= thr).any() else -1
    if idx < 0:
        return np.nan
    if idx == 0:
        return float(st[0])
    x0, x1 = st[idx - 1], st[idx]
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))


def turn_step(st, sw):
    """Where s_on_w starts falling: last checkpoint before the peak-to-fall transition,
    taken as the first index i >= argmax-window where a 2-point forward difference is
    negative and s_on_w has already passed 80% of its peak."""
    pk = int(np.argmax(sw))
    if pk >= len(sw) - 1:
        return np.nan
    for i in range(pk, len(sw) - 1):
        if sw[i + 1] < sw[i]:
            return float(st[i])
    return np.nan


def half_fall(st, sw):
    """Time from the peak to the point where s_on_w has fallen half-way to its final value."""
    pk = int(np.argmax(sw))
    peak, end = sw[pk], sw[-1]
    if peak <= 1e-12 or end >= peak:
        return np.nan, 0.0
    tgt = peak - 0.5 * (peak - end)
    seg_s, seg_y = st[pk:], sw[pk:]
    idx = np.argmax(seg_y <= tgt) if (seg_y <= tgt).any() else -1
    if idx <= 0:
        return np.nan, float(1 - end / peak)
    x0, x1 = seg_s[idx - 1], seg_s[idx]
    y0, y1 = seg_y[idx - 1], seg_y[idx]
    t = x0 + (tgt - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    return float(t - st[pk]), float(1 - end / peak)


def extract(run):
    rows = run["rows"]
    st = np.array([r["step"] for r in rows], float)
    A = np.array([r["A"] for r in rows], float)
    B = np.array([r["B"] for r in rows], float)
    sw = np.array([r["s_on_w"] for r in rows], float)
    sn = np.array([r["s_norm"] for r in rows], float)
    eps = np.array([r["eps"] for r in rows], float)
    pn = np.array([r["post_norm"] for r in rows], float)
    tr = int(np.argmin(A))
    pk = int(np.argmax(sw))
    t_rec, frac = half_fall(st, sw)
    epsf = eps / pn                        # erosion as a fraction of the state norm
    d = dict(
        key=run.get("_key"), sweep=run.get("sweep"), value=run.get("value"),
        norm=run["norm"], seed=run["seed"], beta=run["beta"], d=run["d"], v=run["v"],
        na=run["na"], nb=run["nb"], ilr=run["inject_lr"], dose=run.get("dose", 1.0),
        t_B_on=first_cross(st, B, 0.5), t_B_mid=first_cross(st, B, 0.9),
        t_B_gate=first_cross(st, B, 0.99),
        t_trough=float(st[tr]), trough=float(A[tr]),
        t_peak=float(st[pk]), sw_peak=float(sw[pk]), sw_end=float(sw[-1]),
        t_turn=turn_step(st, sw), t_recover=t_rec, sw_drop=frac,
        s_peak=float(sn[np.argmax(sn)]), t_s_peak=float(st[int(np.argmax(sn))]),
        B_at_peak=float(B[pk]), B_at_trough=float(B[tr]),
        recovery=float(A[tr:].max() - A[tr]),
        t_eps_half=first_cross(st, epsf, 0.5 * epsf[-1]) if epsf[-1] > 0 else np.nan,
        eps_end=float(epsf[-1]),
    )
    return d


def load_all(files=FILES):
    out = []
    for f in files:
        try:
            data = json.load(open(f))
        except FileNotFoundError:
            continue
        for k, r in data.items():
            r["_key"] = k
            r["_file"] = f
            e = extract(r)
            e["file"] = f
            out.append(e)
    return out


def extract2(run):
    """extract() plus a crash-window trough and a quality flag.

    Two artefacts make plain argmin(A) a bad clock. (i) In runs that barely crash, the
    global minimum of A is the late EROSION floor, thousands of steps after the write
    peaked -- a different mechanism. (ii) In runs that crash to zero the bottom is flat and
    argmin lands anywhere in it. So the trough is taken inside the write's own window
    (steps <= 3 * t_peak) and runs are flagged usable only if the crash is deep enough to
    locate (>= 0.15) and not a total floor-out (<= 0.97)."""
    d = extract(run)
    rows = run["rows"]
    st = np.array([r["step"] for r in rows], float)
    A = np.array([r["A"] for r in rows], float)
    sw = np.array([r["s_on_w"] for r in rows], float)
    pk = int(np.argmax(sw))
    win = st <= max(3.0 * st[pk], 50.0)
    idx = np.arange(len(st))[win]
    tr = idx[int(np.argmin(A[win]))]
    d["t_trough_w"] = float(st[tr])
    d["trough_w"] = float(A[tr])
    depth = 1.0 - d["trough_w"]
    d["depth"] = float(depth)
    d["usable"] = bool(0.15 <= depth <= 0.97 and d["sw_peak"] > 0.1)
    # A's half-way-down crossing: robust to a flat bottom, unlike argmin
    Afall = 1.0 - 0.5 * depth
    seg = A[: tr + 1]
    j = np.argmax(seg <= Afall) if (seg <= Afall).any() else -1
    d["t_A_half"] = float(st[j]) if j > 0 else (float(st[0]) if j == 0 else np.nan)
    return d


def load_all2(files=FILES):
    out = []
    for f in files:
        try:
            data = json.load(open(f))
        except FileNotFoundError:
            continue
        for k, r in data.items():
            r["_key"] = k
            e = extract2(r)
            e["file"] = f
            e["key"] = k
            out.append(e)
    return out
