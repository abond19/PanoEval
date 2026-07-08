#!/usr/bin/env python3
"""
Tangent-plane count ablation for the distortion-aware metrics (TangentFID /
TangentIS).

This script sweeps a range of tangent-plane extraction configurations and
reports how TangentFID and TangentIS change as the number of extracted tangent
planes varies. It is intended for the BMVC 2026 rebuttal question about the
sensitivity of the proposed metrics to the number of tangent planes.

Configurations compared (see ``panoeval/eq2pers_v3_updated.py``):

    cubemap     ->  6 standard cubemap faces (fov 90)
    tangent_10  -> 10 planes  (3 rows,  num_cols [3, 4, 3])
    tangent_18  -> 18 planes  (4 rows,  num_cols [3, 6, 6, 3])   (paper default)
    tangent_26  -> 26 planes  (5 rows,  num_cols [3, 6, 8, 6, 3])
    tangent_46  -> 46 planes  (6 rows,  num_cols [3, 8, 12, 12, 8, 3])

For every configuration each per-view score distribution is summarised with a
family of confidence bounds (see ``panoeval/metrics/tangent_ci.py``):

    gaussian      mu +/- 1.96 * sigma / sqrt(K)      (Eq. 5; assumes independence)
    tolerance     mu +/- 1.96 * sigma                (drops sqrt(K); no independence)
    effective_n   mu +/- 1.96 * sigma / sqrt(K_eff)  (design-effect correction)
    bootstrap_se  mu +/- 1.96 * SE_boot              (correlation-honest SE)
    bootstrap_pct empirical 97.5 / 2.5 percentile    (fully nonparametric)

The ``effective_n`` and ``bootstrap_*`` variants require bootstrap resampling of
whole panoramas; enable them with ``--n-bootstrap`` (0 disables, giving only the
two closed-form variants). Every variant is written to the output CSV so all can
be compared. (For TangentFID, ``lower is better``; for TangentIS, ``higher is
better``.)

Example:

    python run_tangent_ablation.py \
        --gen_dir ./all_arranged_images/tandit_L \
        --real_dir /path/to/test_gts \
        --n-bootstrap 200 \
        --output tandit_L_tangent_ablation.csv

Multiple generated-image directories (e.g. several models) can be compared in a
single run by repeating ``--gen_dir`` and optionally passing matching
``--label`` values.
"""

import argparse
import os

import pandas as pd
import torch

torch.set_grad_enabled(False)

from panoeval.metrics.tangent_fid import compute_tangentfid
from panoeval.metrics.tangent_is import compute_tangentis

# Ordered from fewest to most tangent planes so the table reads naturally.
DEFAULT_CONFIGS = ["cubemap", "tangent_10", "tangent_18", "tangent_26", "tangent_46"]

# CI variants emitted per metric. gaussian/tolerance are always available;
# the rest require bootstrap (n_bootstrap > 0).
CI_METHODS = ["gaussian", "tolerance", "effective_n", "bootstrap_se", "bootstrap_pct"]


def _flatten(details, prefix):
    """Flatten a metric details dict into prefixed, CSV-friendly columns."""
    row = {
        f"{prefix}_mean": details.get("mean"),
        f"{prefix}_std": details.get("std"),
        f"{prefix}_rho_bar": details.get("rho_bar"),
        f"{prefix}_k_eff": details.get("k_eff"),
        f"{prefix}_se_boot": details.get("se_boot"),
    }
    for method in CI_METHODS:
        row[f"{prefix}_{method}"] = details.get(f"ci_{method}")
    if "weighted_mean" in details:
        row[f"{prefix}_weighted_mean"] = details["weighted_mean"]
    return row


def run_ablation(
    gen_dir,
    real_dir,
    label,
    configs,
    face_size=192,
    n_bootstrap=0,
    bootstrap_seed=0,
    use_matterport=True,
):
    """Compute TangentFID / TangentIS (all CI variants) for every config."""
    rows = []
    for config_name in configs:
        print("=" * 70)
        print(f"[{label}] Extraction config: {config_name}")
        print("=" * 70)

        fid_details = compute_tangentfid(
            real_dir,
            gen_dir,
            face_size=face_size,
            config_name=config_name,
            return_details=True,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            use_matterport=use_matterport,
        )
        is_details = compute_tangentis(
            gen_dir,
            face_size=face_size,
            config_name=config_name,
            return_details=True,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            use_matterport=use_matterport,
        )

        assert fid_details["K"] == is_details["K"], "Mismatched K between metrics"

        row = {
            "model": label,
            "config": config_name,
            "num_planes": fid_details["K"],
        }
        # Headline values (paper default = gaussian bound).
        row["TangentFID"] = fid_details["ci_gaussian"]
        row["TangentIS"] = is_details["ci_gaussian"]
        row.update(_flatten(fid_details, "TangentFID"))
        row.update(_flatten(is_details, "TangentIS"))
        rows.append(row)

    return rows


def _fmt(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) and value is not None else "   -   "


def print_metric_table(rows, prefix, direction):
    """Print one metric's CI variants across all configs as a fixed-width table."""
    cols = ["gaussian", "tolerance", "effective_n", "bootstrap_se", "bootstrap_pct"]
    header = f"{'model':<18}{'config':<13}{'planes':>7}" + "".join(f"{c:>14}" for c in cols)
    print("\n" + "=" * len(header))
    print(f"{prefix}  ({direction})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['model']:<18}{r['config']:<13}{r['num_planes']:>7}"
        line += "".join(f"{_fmt(r.get(f'{prefix}_{c}')):>14}" for c in cols)
        print(line)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Ablate the number of extracted tangent planes for "
        "TangentFID / TangentIS, reporting every confidence-bound variant."
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories of generated panoramas (one per model).",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Directory of real reference panoramas (required for TangentFID).",
    )
    parser.add_argument(
        "--label",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels matching --gen_dir (defaults to the dir basename).",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated extraction configs to compare. "
        f"Default: {','.join(DEFAULT_CONFIGS)}",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=192,
        help="Tangent patch resolution in pixels (default: 192).",
    )
    parser.add_argument(
        "--n-bootstrap",
        "--n_bootstrap",
        dest="n_bootstrap",
        type=int,
        default=200,
        help="Bootstrap replicates for the correlation-aware CI variants "
        "(effective_n, bootstrap_se, bootstrap_pct). 0 disables them, leaving "
        "only the closed-form gaussian/tolerance bounds. Default: 200. "
        "Note: TangentFID bootstrap is the costly case for large plane counts.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        "--bootstrap_seed",
        dest="bootstrap_seed",
        type=int,
        default=0,
        help="Seed for the bootstrap resampler (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tangent_ablation.csv",
        help="Output CSV path (default: tangent_ablation.csv).",
    )
    parser.add_argument(
        "--use_matterport",
        action="store_true",
        help="Include Matterport images in the evaluation set.",
    )
    args = parser.parse_args()

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
        all_rows.extend(
            run_ablation(
                gen_dir=gen_dir,
                real_dir=args.real_dir,
                label=label,
                configs=configs,
                face_size=args.face_size,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
                use_matterport=args.use_matterport,
            )
        )

    if not all_rows:
        print("No results computed.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)

    print("\n\n" + "#" * 70)
    print("TANGENT-PLANE COUNT ABLATION — SUMMARY")
    print("#" * 70)
    print_metric_table(all_rows, "TangentFID", "lower is better")
    print_metric_table(all_rows, "TangentIS", "higher is better")
    print(f"\nFull results (incl. mean/std/rho_bar/K_eff) saved to: {args.output}")


if __name__ == "__main__":
    main()
