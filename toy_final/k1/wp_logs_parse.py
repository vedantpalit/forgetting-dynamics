"""Parse injection .out logs into per-arm, per-seed trajectory arrays.

Eval line format (one per 10 global steps):
  [step N] dataA: first_acc=.. attr_acc=.. [ret_own=..] rank_own_top1=.. halluc_gap=..,
           dataB: ..., [ballast: ...,] dataC: ...

The step-0 line has no ret_own (no baseline established yet in the logger).
Injection step = global step - first global step in the file.

Usage:  python wp_logs_parse.py        # prints a coverage summary
Import: from wp_logs_parse import load_all
"""
import glob
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = {
    "4L_ballast": dict(
        glob=os.path.join(ROOT, "logs_with_ballast", "demo_injection.with_ballast.seed*.out"),
        label="4L, ballast",
    ),
    "4L_noballast": dict(
        glob=os.path.join(
            ROOT, "logs_without_ballast_corrected", "demo_injection.without_ballast.seed*.out"
        ),
        label="4L, no ballast",
    ),
    "8L_ballast": dict(
        glob=os.path.join(ROOT, "logs_scale8", "scale8_injection.p16000.disjoint.seed*.out"),
        label="8L, ballast",
    ),
    "8L_noballast": dict(
        glob=os.path.join(
            ROOT, "logs_scale8", "scale8_noballast.p16000.disjoint.t1200.seed*.out"
        ),
        label="8L, no ballast",
    ),
    "4L_noballast_v2": dict(
        glob=os.path.join(
            ROOT, "logs_without_ballast_v2", "demo_injection.without_ballast.seed*.3600.*.out"
        ),
        label="4L, no ballast (3600)",
    ),
}

POPS = ["dataA", "dataB", "ballast", "dataC"]
FIELDS = ["first_acc", "attr_acc", "ret_own", "rank_own_top1", "halluc_gap"]

_STEP_RE = re.compile(r"^\[step (\d+)\]\s*(.*)$")
_KV_RE = re.compile(r"(\w+)=([-+]?[\d.]+(?:[eE][-+]?\d+)?)")
_SEED_RE = re.compile(r"seed(\d+)")


def parse_file(path):
    """-> dict(step=array, pops={popname: {field: array}})"""
    steps = []
    rows = []  # list of {pop: {field: val}}
    with open(path) as fh:
        for line in fh:
            m = _STEP_RE.match(line.strip())
            if not m:
                continue
            steps.append(int(m.group(1)))
            rest = m.group(2)
            row = {}
            # split on ", <popname>:" boundaries
            chunks = re.split(r",\s*(?=(?:dataA|dataB|dataC|ballast):)", rest)
            for ch in chunks:
                name, _, kvs = ch.partition(":")
                name = name.strip()
                if name not in POPS:
                    continue
                row[name] = {k: float(v) for k, v in _KV_RE.findall(kvs)}
            rows.append(row)
    if not steps:
        raise RuntimeError(f"no eval lines in {path}")
    steps = np.array(steps, dtype=int)
    inj = steps - steps[0]
    present = sorted({p for r in rows for p in r}, key=POPS.index)
    out = {"step": inj, "global_step": steps, "pops": {}}
    for p in present:
        d = {}
        for f in FIELDS:
            v = np.full(len(rows), np.nan)
            for i, r in enumerate(rows):
                if p in r and f in r[p]:
                    v[i] = r[p][f]
            d[f] = v
        out["pops"][p] = d
    return out


def load_arm(key):
    files = sorted(glob.glob(ARMS[key]["glob"]))
    runs = {}
    for f in files:
        seed = int(_SEED_RE.search(os.path.basename(f)).group(1))
        runs[seed] = parse_file(f)
        runs[seed]["path"] = f
    return runs


def load_all():
    return {k: load_arm(k) for k in ARMS}


def common_grid(runs, max_step=None):
    """Steps present in every seed of an arm (ascending)."""
    sets = [set(r["step"].tolist()) for r in runs.values()]
    g = sorted(set.intersection(*sets))
    if max_step is not None:
        g = [s for s in g if s <= max_step]
    return np.array(g, dtype=int)


def stack(runs, pop, field, grid):
    """(n_seeds, n_steps) array on the given grid; NaN if a seed lacks the step."""
    seeds = sorted(runs)
    out = np.full((len(seeds), len(grid)), np.nan)
    for i, s in enumerate(seeds):
        r = runs[s]
        if pop not in r["pops"]:
            continue
        idx = {st: j for j, st in enumerate(r["step"].tolist())}
        for j, st in enumerate(grid):
            if st in idx:
                out[i, j] = r["pops"][pop][field][idx[st]]
    return out, seeds


if __name__ == "__main__":
    for k in ARMS:
        runs = load_arm(k)
        print(f"\n=== {k} ({ARMS[k]['label']}) : {len(runs)} seeds ===")
        for s in sorted(runs):
            r = runs[s]
            print(
                f"  seed{s}: {len(r['step'])} evals, inj steps {r['step'][0]}..{r['step'][-1]}, "
                f"pops={list(r['pops'])}"
            )
