"""Export wandb OFFLINE run histories to small CSVs, so they can leave the cluster.

The console eval lines carry only accuracy, rank and the hallucination gap. Everything
else -- per-step `attribute_loss` for every dataset, `first_token_loss`, the training
loss, and the per-parameter weight norms -- exists only inside the offline `.wandb`
binaries, which run to hundreds of MB and cannot go in git.

This reads those binaries with the same datastore reader `wandb sync` uses, and writes one
compact CSV per run plus an index naming each run. A 350MB wandb directory comes out as a
few hundred KB of CSV.

Note on the record format: history items put the metric path in `nested_key` when it is
namespaced (`dataA/attribute_loss`), leaving `key` empty. Reading only `key` yields a
single column named '' -- verified against a real run before this script was written.

Run on the cluster:
  uv run python -m src.experiments.export_wandb_history
  # then:  git add -f wandb_export/ && git commit -m "wandb histories" && git push
"""
import argparse
import csv
import glob
import json
import os

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

OUT_DIR = "wandb_export"
# Config fields worth carrying into the index so a run can be identified without opening it.
ID_FIELDS = ("arm", "freeze_arm", "pretrain_step", "condition", "seed")


def read_run(path):
    """(display_name, id_config, history_rows) for one offline .wandb file."""
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    name, ident, rows = None, {}, []
    while True:
        rec = ds.scan_data()
        if rec is None:
            break
        r = pb.Record()
        r.ParseFromString(rec)
        kind = r.WhichOneof("record_type")
        if kind == "run" and name is None:
            name = r.run.display_name or r.run.run_id
            for it in r.run.config.update:
                k = it.key if it.key else "/".join(it.nested_key)
                if k in ID_FIELDS:
                    try:
                        ident[k] = json.loads(it.value_json)
                    except Exception:
                        ident[k] = it.value_json
        elif kind == "history":
            row = {}
            for it in r.history.item:
                k = it.key if it.key else "/".join(it.nested_key)
                try:
                    row[k] = json.loads(it.value_json)
                except Exception:
                    row[k] = it.value_json
            rows.append(row)
    return name, ident, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_dir", default="wandb")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--skip_wnorm", action="store_true",
                    help="drop the per-parameter weight-norm columns, which dominate the "
                         "column count and are rarely what you want")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.wandb_dir, "offline-run-*", "run-*.wandb")))
    if not files:
        raise SystemExit(f"no offline runs under {a.wandb_dir}/")
    os.makedirs(a.out_dir, exist_ok=True)
    index = []
    for f in files:
        run_dir = os.path.basename(os.path.dirname(f))
        try:
            name, ident, rows = read_run(f)
        except Exception as e:
            print(f"  {run_dir}: FAILED ({type(e).__name__}: {e})")
            continue
        if not rows:
            print(f"  {run_dir}: no history records, skipped")
            continue
        cols = sorted({k for r in rows for k in r})
        if a.skip_wnorm:
            cols = [c for c in cols if not c.startswith("wnorm/")]
        dest = os.path.join(a.out_dir, f"{run_dir}.csv")
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        kb = os.path.getsize(dest) / 1024
        index.append(dict(run_dir=run_dir, name=name, rows=len(rows),
                          cols=len(cols), kb=round(kb, 1), **ident))
        print(f"  {run_dir}  name={name}  rows={len(rows)}  cols={len(cols)}  {kb:.1f}KB"
              f"  {ident if ident else ''}")
    with open(os.path.join(a.out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    total = sum(x["kb"] for x in index)
    print(f"\n  {len(index)} runs -> {a.out_dir}/  ({total/1024:.1f} MB total)")
    print(f"  index written to {a.out_dir}/index.json")


if __name__ == "__main__":
    main()
