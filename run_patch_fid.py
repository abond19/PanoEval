#!/usr/bin/env python3
"""
Native-resolution patch FID / IS for evaluating high-resolution (4K) output.

Standard FID resizes each panorama to ~299px, discarding the fine detail that
"4K" is supposed to add. This instead resizes each panorama only to a high
target resolution (default 2048x4096) and computes FID / IS over
native-resolution patches -- so it is sensitive to genuine 4K detail. Patches
are either raw ERP tiles (default) or dense narrow-FOV gnomonic crops
(--config tangent_4k).

This is a high-resolution self/reference evaluation (e.g. TanDiT-4K vs a real 4K
set such as Polyhaven, or TanDiT-4K vs its own 2K upsampled to 4K). It is NOT a
cross-baseline table: most baselines cannot produce 4K, so comparing them here
would only measure their upsampling, not generation.

Example:
    python run_patch_fid.py \
        --gen_dir /path/tandit_4k --label TanDiT-4K \
        --real_dir /path/real_4k --real_cache_dir ./.real_cache \
        --patch_size 512 --target_hw 2048x4096 --output results/patch_fid.csv
"""

import argparse
import os

import pandas as pd
import torch

torch.set_grad_enabled(False)

from panoeval.metrics.tangent_fast import compute_patch_fid


def _parse_hw(s):
    h, w = s.lower().split("x")
    return (int(h), int(w))


def main():
    p = argparse.ArgumentParser(description="Native-resolution patch FID/IS (4K evaluation).")
    p.add_argument("--gen_dir", type=str, nargs="+", required=True)
    p.add_argument("--real_dir", type=str, required=True, help="High-resolution reference set (e.g. Polyhaven 4K).")
    p.add_argument("--label", type=str, nargs="+", default=None)
    p.add_argument("--target_hw", type=str, default="2048x4096", help="Resize target HxW (default 2048x4096).")
    p.add_argument("--patch_size", type=int, default=512, help="Patch size in pixels (default 512).")
    p.add_argument("--config", type=str, default=None,
                   help="None = raw ERP tiles (default); 'tangent_4k' = narrow-FOV gnomonic crops.")
    p.add_argument("--real_cache_dir", type=str, default=None)
    p.add_argument("--output", type=str, default="patch_fid.csv")
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4, help="Panoramas per batch (small: images are large).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--inception_chunk", type=int, default=256)
    p.add_argument("--exclude_matterport", action="store_true")
    args = p.parse_args()

    target_hw = _parse_hw(args.target_hw)
    gen_dirs = args.gen_dir
    if args.label is not None:
        if len(args.label) != len(gen_dirs):
            raise ValueError("Number of --label values must match --gen_dir.")
        labels = args.label
    else:
        labels = [os.path.basename(d.rstrip("/")) or d for d in gen_dirs]

    rows = []
    for gen_dir, label in zip(gen_dirs, labels):
        if not os.path.exists(gen_dir):
            print(f"Warning: {gen_dir} does not exist. Skipping.")
            continue
        print("=" * 70)
        print(f"[{label}] patch-FID  (mode={args.config or 'erp'}, {args.patch_size}px @ {target_hw})")
        print("=" * 70)
        res = compute_patch_fid(
            real_dir=args.real_dir, gen_dir=gen_dir,
            patch_size=args.patch_size, target_hw=target_hw, config_name=args.config,
            splits=args.splits, use_matterport=not args.exclude_matterport, seed=args.seed,
            real_cache_dir=args.real_cache_dir, batch_size=args.batch_size,
            num_workers=args.num_workers, inception_chunk=args.inception_chunk,
        )
        rows.append({"model": label, "mode": res["config"], "patch_size": res["patch_size"],
                     "target_hw": args.target_hw, "patch_FID": res["patch_fid"],
                     "patch_IS": res["patch_is"], "n_gen_patches": res["n_gen_patches"]})

    if not rows:
        print("No results computed.")
        return

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"PATCH FID/IS   ({args.patch_size}px @ {args.target_hw}, mode={args.config or 'erp'})")
    print("=" * 60)
    print(f"{'model':<18}{'patch-FID':>11}{'patch-IS':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:<18}{r['patch_FID']:>11.3f}{r['patch_IS']:>10.3f}")
    print("=" * 60)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
