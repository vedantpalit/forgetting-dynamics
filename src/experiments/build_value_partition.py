"""Build and cache one value-partition file for the exclusion-fraction sweep, its IPV
control, or the region-size-vs-exclusion-fraction (concentration) follow-up. Deliberately
NOT wired into knowledge_injection.py's --phase dispatch (pretrain/inject/sweep/profile/
all) -- this is a one-off preprocessing step, not a training phase, and the sweep's "no new
phase" scope explicitly excludes it from that dispatch.

Reuses build_partition_fraction / build_partition_fraction_scaled /
build_partition_fraction_windowed from knowledge_injection.py and the existing
save_partition -- get_partition and load_partition are untouched and will transparently
load whatever this writes, since they only validate `num_values_per_attr`, never the split.

Runs the self-test before writing anything (build_partition_fraction(..., 0.5) ==
build_partition(...) exactly; the IPV control's derived pool-size match;
build_partition_fraction_windowed's "in" stays fixed and "out" regions nest as out_fraction
grows -- NOT that windowed(0.25, 0.25) reproduces the IPV control exactly, which is false
for two attributes with odd active_n under the scaled construction, caught by an earlier
version of this same self-test), and refuses to overwrite an existing file, since a
checkpoint may already depend on it (now safely, via the partition-file content hash in
pretrain_key).

Run:
  uv run python -m src.experiments.build_value_partition \
    --in_fraction 0.25 --out data/biography/value_partition_f0.25.npz
  uv run python -m src.experiments.build_value_partition \
    --in_fraction 0.5 --pool_scale 0.5 --out data/biography/value_partition_ipv_control.npz
  uv run python -m src.experiments.build_value_partition \
    --in_fraction 0.25 --out_fraction 0.5 --out data/biography/value_partition_concentration_out66.npz
"""
import argparse
import os

from src.data.biography import BiographyPopulation, NUM_ATTRIBUTES
from src.experiments.knowledge_injection import (
    _self_test_partition_fraction, build_partition_fraction, build_partition_fraction_scaled,
    build_partition_fraction_windowed, save_partition,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_fraction", type=float, required=True)
    p.add_argument("--pool_scale", type=float, default=1.0,
                   help="1.0 for the plain exclusion-fraction sweep; 0.5 for the IPV control")
    p.add_argument("--out_fraction", type=float, default=None,
                   help="if given, uses build_partition_fraction_windowed instead (the "
                        "concentration follow-up): 'in' fixed at in_fraction, 'out' an "
                        "independently-sized window of out_fraction, taking priority over "
                        "--pool_scale (ignored when this is set)")
    p.add_argument("--partition_seed", type=int, default=7,
                   help="must match KIConfig.partition_seed used for the actual sweep runs")
    p.add_argument("--data_path", type=str, default="data/biography/preprocessed.npz")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(
            f"{args.out} already exists. Delete it first if you intend to rebuild it -- "
            f"refusing to silently overwrite a partition a checkpoint may already depend on."
        )

    # num_people=1: only pop.num_values_per_attr is read below, which comes straight from
    # the corpus npz and does not depend on population size -- keeps this cheap.
    pop = BiographyPopulation(args.data_path, num_people=1, seed=42)

    print("Self-test:")
    _self_test_partition_fraction(pop, seed=args.partition_seed)

    if args.out_fraction is not None:
        halves = build_partition_fraction_windowed(
            pop, args.partition_seed, args.in_fraction, args.out_fraction)
    elif args.pool_scale == 1.0:
        halves = build_partition_fraction(pop, args.partition_seed, args.in_fraction)
    else:
        halves = build_partition_fraction_scaled(
            pop, args.partition_seed, args.in_fraction, args.pool_scale)

    print(f"\nin_fraction={args.in_fraction}  pool_scale={args.pool_scale}  "
          f"out_fraction={args.out_fraction}  partition_seed={args.partition_seed}")
    for k in range(NUM_ATTRIBUTES):
        n_in, n_out = len(halves[k][0]), len(halves[k][1])
        pool = int(pop.num_values_per_attr[k])
        print(f"  attr {k}: pool={pool:4d}  in={n_in:4d} (IPV={2000 / max(n_in, 1):6.1f})  "
              f"out={n_out:4d} (IPV={2000 / max(n_out, 1):6.1f})")

    save_partition(args.out, halves, pop, args.partition_seed)


if __name__ == "__main__":
    main()
