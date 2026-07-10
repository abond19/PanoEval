#!/usr/bin/env python3
"""
Pole-region and equatorial FID / IS for one or more models.

For each generated set, this pools the pole-row tangent crops (top+bottom rows,
or the cubemap up/down faces) across all images into a single sample set and
computes one FID and one IS against the reference set; likewise for the
equatorial (middle-row) crops. Polar-FID / Polar-IS directly quantify generated
pole-region quality (as requested in review), with the equatorial numbers for
contrast.

Matterport is included by default (matching the paper). The reference (GT)
statistics are cached in --real_cache_dir and reused across models.

Example (all baselines in one run):
    python run_polar_metrics.py \
        --gen_dir /path/tandit /path/diffusion360 /path/panfusion \
        --label TanDiT Diffusion360 PanFusion \
        --real_dir /path/test_gts --real_cache_dir ./.real_cache \
        --config tangent_18 --output results/polar.csv

See run_polar_baselines.sh to drive several baselines from a config block.
"""

import argparse
import os

import pandas as pd
import torch

torch.set_grad_enabled(False)

from panoeval.metrics.tangent_fast import compute_polar_metrics


def run_model(gen_dir, real_dir, label, configs, args):
    rows = []
    for config_name in configs:
        print("=" * 70)
        print(f"[{label}] {config_name}")
        print("=" * 70)
        res = compute_polar_metrics(
            real_dir=real_dir,
            gen_dir=gen_dir,
            config_name=config_name,
            mode=args.mode,
            face_size=args.face_size,
            splits=args.splits,
            use_matterport=not args.exclude_matterport,
            seed=args.seed,
            real_cache_dir=args.real_cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            inception_chunk=args.inception_chunk,
        )
        g = res["groups"]
        rows.append({
            "model": label,
            "config": config_name,
            "Polar_FID": g.get("Polar", {}).get("fid"),
            "Polar_IS": g.get("Polar", {}).get("is"),
            "Equatorial_FID": g.get("Equatorial", {}).get("fid"),
            "Equatorial_IS": g.get("Equatorial", {}).get("is"),
        })
    return rows


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "   -   "


def print_summary(rows, mode):
    header = (f"{'model':<16}{'config':<12}"
             f"{'Polar-FID':>11}{'Polar-IS':>10}{'Equat-FID':>11}{'Equat-IS':>10}")
    print("\n" + "=" * len(header))
    print(f"POLE-REGION METRICS [{mode}]   (Polar-FID lower is better | Polar-IS higher is better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<16}{r['config']:<12}"
              f"{_fmt(r['Polar_FID']):>11}{_fmt(r['Polar_IS']):>10}"
              f"{_fmt(r['Equatorial_FID']):>11}{_fmt(r['Equatorial_IS']):>10}")
    print("=" * len(header))


def main():
    p = argparse.ArgumentParser(description="Pole-region / equatorial FID & IS per model.")
    p.add_argument("--gen_dir", type=str, nargs="+", required=True)
    p.add_argument("--real_dir", type=str, required=True)
    p.add_argument("--label", type=str, nargs="+", default=None)
    p.add_argument("--config", "--configs", dest="configs", type=str, default="tangent_18",
                   help="Comma-separated configs (default: tangent_18, the paper setting).")
    p.add_argument("--mode", choices=["cbound", "pooled"], default="cbound",
                   help="'cbound' (default): Eq.5 confidence bound over per-view FIDs/ISs in each "
                        "region (artifact-sensitive, matches TangentFID). 'pooled': single FID/IS "
                        "over all region crops (dominated by easy content; not recommended).")
    p.add_argument("--real_cache_dir", type=str, default=None)
    p.add_argument("--output", type=str, default="polar_metrics.csv")
    p.add_argument("--face_size", type=int, default=192)
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--inception_chunk", type=int, default=512)
    p.add_argument("--exclude_matterport", action="store_true",
                   help="Exclude Matterport (default: included, matching the paper).")
    args = p.parse_args()

    gen_dirs = args.gen_dir
    if args.label is not None:
        if len(args.label) != len(gen_dirs):
            raise ValueError("Number of --label values must match --gen_dir.")
        labels = args.label
    else:
        labels = [os.path.basename(d.rstrip("/")) or d for d in gen_dirs]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    all_rows = []
    for gen_dir, label in zip(gen_dirs, labels):
        if not os.path.exists(gen_dir):
            print(f"Warning: {gen_dir} does not exist. Skipping.")
            continue
        all_rows.extend(run_model(gen_dir, args.real_dir, label, configs, args))

    if not all_rows:
        print("No results computed.")
        return

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print_summary(all_rows, args.mode)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
