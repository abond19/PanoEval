"""Confidence-bound aggregation for the distortion-aware tangent metrics.

Given per-view scores (one FID / IS per tangent view) the paper summarises them
with a Gaussian confidence bound (Eq. 5):

    TangentFID = mu + 1.96 * sigma / sqrt(K)     (upper bound, lower is better)
    TangentIS  = mu - 1.96 * sigma / sqrt(K)     (lower bound, higher is better)

The ``sqrt(K)`` term assumes the K per-view scores are independent. Because
neighbouring tangent planes overlap in field-of-view and share content, this is
violated (positive correlation), so the naive bound is too tight. This module
produces several alternative summaries that relax the assumption to varying
degrees:

    gaussian        mu +/- 1.96 * sigma / sqrt(K)          (original; independence)
    tolerance       mu +/- 1.96 * sigma                    (drop sqrt(K); worst-region
                                                            quantile, no independence
                                                            claim; = full-dependence limit)
    effective_n     mu +/- 1.96 * sigma / sqrt(K_eff)      (design-effect correction,
                                                            K_eff = K / (1 + (K-1)*rho))
    bootstrap_se    mu +/- 1.96 * SE_boot                  (Gaussian form with a
                                                            correlation-honest SE)
    bootstrap_pct   empirical 97.5 / 2.5 percentile        (fully nonparametric)

The ``effective_n`` and ``bootstrap_*`` summaries require bootstrap replicates of
the per-view scores (a ``(B, K)`` array), obtained by resampling whole panoramas
so that cross-view correlation is preserved.
"""

import numpy as np

Z95 = 1.96


def _mean_pairwise_correlation(boot_scores):
    """Average off-diagonal correlation among the K per-view score sequences.

    ``boot_scores`` is a ``(B, K)`` array of per-view scores across B bootstrap
    replicates. Returns rho_bar in [-1, 1]; 0 if it cannot be estimated.
    """
    B, K = boot_scores.shape
    if K < 2 or B < 2:
        return 0.0
    corr = np.corrcoef(boot_scores, rowvar=False)  # (K, K)
    if not np.all(np.isfinite(corr)):
        # A view with zero variance across replicates yields NaN correlations.
        corr = np.nan_to_num(corr, nan=0.0)
    off_diag = corr[~np.eye(K, dtype=bool)]
    return float(off_diag.mean())


def summarize_ci(point_scores, boot_scores=None, side="upper", weights=None):
    """Summarise per-view scores into every confidence-bound variant.

    Args:
        point_scores: length-K array of per-view point estimates.
        boot_scores:  optional ``(B, K)`` array of bootstrap per-view scores.
                      When ``None`` only the assumption-free summaries
                      (``gaussian``, ``tolerance``) are produced.
        side:         "upper" for TangentFID (adds the interval), "lower" for
                      TangentIS (subtracts it).
        weights:      optional length-K weights for the legacy polar-weighted
                      mean (0.5 for pole rows, 1.0 otherwise).

    Returns a dict with the mean/std, the per-view scores and one ``ci_*`` entry
    per method (plus ``rho_bar`` / ``k_eff`` / ``se_boot`` when bootstrapped).
    """
    point_scores = np.asarray(point_scores, dtype=np.float64)
    K = int(point_scores.shape[0])
    mean = float(point_scores.mean())
    std = float(point_scores.std())  # population std (matches the paper's sigma)
    sign = 1.0 if side == "upper" else -1.0

    out = {
        "K": K,
        "mean": mean,
        "std": std,
        "per_view": point_scores.tolist(),
        "ci_gaussian": float(mean + sign * Z95 * std / np.sqrt(K)),
        "ci_tolerance": float(mean + sign * Z95 * std),
    }

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        out["weighted_mean"] = float(np.sum(w * point_scores) / K)

    if boot_scores is not None:
        boot_scores = np.asarray(boot_scores, dtype=np.float64)
        if boot_scores.ndim == 2 and boot_scores.shape[0] > 1:
            boot_means = boot_scores.mean(axis=1)
            se_boot = float(boot_means.std(ddof=1))
            rho = _mean_pairwise_correlation(boot_scores)
            deff = max(1.0 + (K - 1) * rho, 1e-8)
            k_eff = float(np.clip(K / deff, 1.0, K))
            pct = 97.5 if side == "upper" else 2.5
            out.update(
                {
                    "rho_bar": rho,
                    "k_eff": k_eff,
                    "se_boot": se_boot,
                    "ci_effective_n": float(mean + sign * Z95 * std / np.sqrt(k_eff)),
                    "ci_bootstrap_se": float(mean + sign * Z95 * se_boot),
                    "ci_bootstrap_pct": float(np.percentile(boot_means, pct)),
                }
            )

    return out
