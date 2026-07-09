#!/usr/bin/env python3
"""
Tangent-plane count ablation for TangentFID / TangentIS (fast, no bootstrap).

Sweeps the extraction configs (cubemap, 10/18/26/46 tangent planes) and reports,
for each, the values actually needed for the paper/rebuttal:

    TangentFID : Eq. 5 bound  (mu + 1.96*sigma/sqrt(K), lower is better)
                 + mean, polar-weighted mean, std
    TangentIS  : Eq. 5 bound  (mu - 1.96*sigma/sqrt(K), higher is better)
                 + mean, std

Speed: each panorama is decomposed once per config and shared between FID and IS;
extraction runs on the GPU; and the reference (GT) statistics are cached to disk
(``--real_cache_dir``) so a sweep of many baselines against the same GT computes
them only once. Matterport is included by default (matching the paper).

Example (single model):

    python run_tangent_ablation.py \
        --gen_dir /path/to/tandit --label TanDiT \
        --real_dir /path/to/test_gts \
        --real_cache_dir ./.real_cache \
        --output results/tandit.csv

See run_baselines.sh to sweep several baselines into separate files.
"""

import argparse
import os

import pandas as pd
import torch

torch.set_grad_enabled(False)

from panoeval.metrics.tangent_fast import compute_tangent_metrics

DEFAULT_CONFIGS = ["cubemap", "tangent_10", "tangent_18", "tangent_26", "tangent_46"]


def run_model(gen_dir, real_dir, label, configs, args):
    """Compute both metrics for every config for one generated set."""
    rows = []
    for config_name in configs:
        print("=" * 70)
        print(f"[{label}] {config_name}")
        print("=" * 70)
        res = compute_tangent_metrics(
            real_dir=real_dir,
            gen_dir=gen_dir,
            config_name=config_name,
            face_size=args.face_size,
            splits=args.splits,
            use_matterport=not args.exclude_matterport,
            seed=args.seed,
            real_cache_dir=args.real_cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            inception_chunk=args.inception_chunk,
        )
        fid, is_ = res["fid"], res["is"]
        row = {
            "model": label,
            "config": config_name,
            "num_planes": res["K"],
            # Paper TangentFID = confidence bound on the polar-weighted per-view scores.
            "TangentFID": fid["ci_weighted"],
            "TangentFID_weighted_mean": fid.get("weighted_mean"),
            "TangentFID_mean": fid["mean"],
            "TangentFID_unweighted_bound": fid["ci_gaussian"],
            "TangentFID_std": fid["std"],
            # Paper TangentIS = mean per-view IS (the original code's return value).
            "TangentIS": is_["mean"],
            "TangentIS_conf_bound": is_["ci_gaussian"],
            "TangentIS_std": is_["std"],
        }
        rows.append(row)
        print(f"  -> TangentFID={row['TangentFID']:.3f} "
              f"(wmean={row['TangentFID_weighted_mean']:.3f}, mean={row['TangentFID_mean']:.3f})  |  "
              f"TangentIS={row['TangentIS']:.3f} (conf_bound={row['TangentIS_conf_bound']:.3f})")
    return rows


def print_summary(rows):
    header = (f"{'model':<16}{'config':<12}{'K':>4}"
              f"{'TangentFID':>12}{'FID_wmean':>11}{'FID_mean':>10}"
              f"{'TangentIS':>12}{'IS_bound':>10}")
    print("\n" + "=" * len(header))
    print("SUMMARY   (paper: TangentFID = polar-weighted confidence bound; "
          "TangentIS = mean per-view IS)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<16}{r['config']:<12}{r['num_planes']:>4}"
              f"{r['TangentFID']:>12.3f}{r['TangentFID_weighted_mean']:>11.3f}{r['TangentFID_mean']:>10.3f}"
              f"{r['TangentIS']:>12.3f}{r['TangentIS_conf_bound']:>10.3f}")
    print("=" * len(header))


def main():
    p = argparse.ArgumentParser(description="Fast TangentFID/TangentIS plane-count ablation (no bootstrap).")
    p.add_argument("--gen_dir", type=str, nargs="+", required=True,
                   help="One or more directories of generated panoramas.")
    p.add_argument("--real_dir", type=str, required=True, help="Directory of real reference panoramas.")
    p.add_argument("--label", type=str, nargs="+", default=None,
                   help="Labels matching --gen_dir (default: dir basename).")
    p.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS),
                   help=f"Comma-separated configs. Default: {','.join(DEFAULT_CONFIGS)}")
    p.add_argument("--real_cache_dir", type=str, default=None,
                   help="Directory to cache/reuse GT statistics across models (big speedup for sweeps).")
    p.add_argument("--output", type=str, default="tangent_ablation.csv", help="Output CSV path.")
    p.add_argument("--face_size", type=int, default=192, help="Tangent patch resolution (default 192).")
    p.add_argument("--splits", type=int, default=10, help="Inception Score splits (default 10).")
    p.add_argument("--seed", type=int, default=0, help="Seed for the IS split shuffle (default 0).")
    p.add_argument("--batch_size", type=int, default=32, help="Panoramas per batch (default 32).")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (default 8).")
    p.add_argument("--inception_chunk", type=int, default=512,
                   help="Tangent crops per Inception forward, bounds GPU memory (default 512).")
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
    print_summary(all_rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
