"""Fast, single-pass TangentFID + TangentIS (no bootstrap).

Optimised for sweeping many extraction configs / many models on one GPU:

* Each panorama is decomposed into tangent views **once** per config and both
  the FID (2048-d pool) and IS (logit) features are taken from the shared
  tangent crops, instead of extracting twice.
* Tangent extraction runs on the GPU (falls back to CPU automatically if the
  vmap/grid_sample GPU path is unavailable).
* FID sufficient statistics are accumulated incrementally (no storing of all
  features), matching the original code's float32-accumulate-into-float64 path.
* The real/GT statistics are optionally cached to disk (per config) so a whole
  sweep of baselines against the same reference set computes them only once.

Only the closed-form summaries are produced (mean / std / Eq. 5 bound and, for
FID, the polar-weighted mean). No bootstrap.
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from panoeval.eq2pers_v3_updated import get_extraction_config, build_view_groups
from panoeval.metrics.tangent_fid import (
    equirectangular_to_tangents_batch,
    average_features_by_view_group,
    _fid_from_stats,
    preprocess_images,
)
from panoeval.metrics.tangent_is import _inception_score
from panoeval.metrics.tangent_ci import summarize_ci
from panoeval.utils.dataloader import RealDataset, GeneratedDataset


def _chunked_forward(net, imgs_u8, chunk):
    """Run an Inception extractor over a large batch in memory-bounded chunks."""
    if imgs_u8.shape[0] <= chunk:
        return net(imgs_u8)
    return torch.cat([net(imgs_u8[i:i + chunk]) for i in range(0, imgs_u8.shape[0], chunk)], dim=0)


def _probe_gpu_extraction(config, device):
    """Return True if tangent extraction works on-device (much faster)."""
    if str(device) == "cpu":
        return False
    try:
        probe = torch.zeros(1, 3, 64, 128, device=device)
        equirectangular_to_tangents_batch(probe, face_size=32, config=config)
        return True
    except Exception as e:  # pragma: no cover - hardware dependent
        print(f"[tangent_fast] GPU tangent extraction unavailable ({type(e).__name__}: {e}); "
              f"falling back to CPU extraction.")
        return False


def _stats_from_accum(feat_sum, cov_sum, n):
    """(mean, cov) from accumulated first/second moments — matches torchmetrics FID."""
    mean = feat_sum / n
    cov = (cov_sum - n * mean.unsqueeze(1).mm(mean.unsqueeze(0))) / (n - 1)
    return mean, cov


def _accumulate_fid(store, group_feats, gi):
    """Add one view's per-image features to that view's running FID moments.

    ``group_feats`` is float32 (as produced by Inception); accumulating into the
    float64 buffers via ``+=`` reproduces the original code exactly.
    """
    store["sum"][gi] += group_feats.sum(dim=0)
    store["cov"][gi] += group_feats.t().mm(group_feats)
    store["n"][gi] += group_feats.shape[0]


def _new_fid_store(K, device):
    return {
        "sum": [torch.zeros(2048, dtype=torch.float64, device=device) for _ in range(K)],
        "cov": [torch.zeros(2048, 2048, dtype=torch.float64, device=device) for _ in range(K)],
        "n": [0 for _ in range(K)],
    }


def _real_cache_path(cache_dir, real_dir, config_name, face_size, use_matterport):
    key = f"{os.path.basename(real_dir.rstrip('/'))}_{config_name}_fs{face_size}_mp{int(use_matterport)}"
    return os.path.join(cache_dir, f"realstats_{key}.pt")


def _load_or_compute_real_stats(real_dir, config, config_name, fid_net, face_size, device,
                                use_matterport, cache_dir, batch_size, num_workers,
                                gpu_extract, inception_chunk, view_idx, K):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _real_cache_path(cache_dir, real_dir, config_name, face_size, use_matterport)
        if os.path.exists(cache_path):
            print(f"[{config_name}] loading cached real stats: {cache_path}")
            data = torch.load(cache_path, map_location=device)
            return [(data["mean"][v].to(device), data["cov"][v].to(device)) for v in range(K)]

    real_ds = RealDataset(real_dir, transform=preprocess_images(), use_matterport=use_matterport)
    print(f"[{config_name}] real images: {len(real_ds)} (computing stats; use_matterport={use_matterport})")
    real_dl = torch.utils.data.DataLoader(real_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    store = _new_fid_store(K, device)
    for batch in tqdm(real_dl, desc=f"real [{config_name}]", total=len(real_dl)):
        src = batch.to(device) if gpu_extract else batch
        tan = equirectangular_to_tangents_batch(src, face_size=face_size, config=config)
        if not gpu_extract:
            tan = tan.to(device)
        b, t, c, h, w = tan.shape
        u8 = (tan.reshape(b * t, c, h, w) * 255.0).to(torch.uint8)
        pool = _chunked_forward(fid_net, u8, inception_chunk).reshape(b, t, -1)
        for gi in range(K):
            _accumulate_fid(store, average_features_by_view_group(pool, view_idx[gi]), gi)

    stats = [_stats_from_accum(store["sum"][v], store["cov"][v], store["n"][v]) for v in range(K)]
    if cache_path:
        torch.save({"mean": [s[0].cpu() for s in stats], "cov": [s[1].cpu() for s in stats]}, cache_path)
        print(f"[{config_name}] cached real stats -> {cache_path}")
    return stats


def compute_tangent_metrics(
    real_dir,
    gen_dir,
    config_name,
    face_size=192,
    splits=10,
    device=None,
    use_matterport=True,
    seed=0,
    real_cache_dir=None,
    batch_size=32,
    num_workers=8,
    inception_chunk=512,
):
    """Compute TangentFID and TangentIS (closed-form summaries only) for one config.

    Returns a dict: {'K', 'fid': <summary>, 'is': <summary>} where each summary is
    from ``tangent_ci.summarize_ci`` (mean, std, ci_gaussian, ci_tolerance, and —
    for FID — weighted_mean).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = get_extraction_config(config_name)
    view_map, weights = build_view_groups(config)
    groups = list(view_map.keys())
    K = len(groups)
    view_idx = [view_map[g] for g in groups]

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    fid_net = FrechetInceptionDistance(feature=2048).to(device).inception
    is_net = InceptionScore(feature="logits_unbiased", splits=splits, normalize=False).to(device).inception

    gpu_extract = _probe_gpu_extraction(config, device)

    # --- reference (real) FID statistics, cached across models ---
    real_stats = _load_or_compute_real_stats(
        real_dir, config, config_name, fid_net, face_size, device, use_matterport,
        real_cache_dir, batch_size, num_workers, gpu_extract, inception_chunk, view_idx, K,
    )

    # --- generated: one extraction, both feature types ---
    gen_ds = GeneratedDataset(gen_dir, transform=preprocess_images(), use_matterport=use_matterport)
    print(f"[{config_name}] gen images: {len(gen_ds)} (use_matterport={use_matterport})")
    gen_dl = torch.utils.data.DataLoader(gen_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    store = _new_fid_store(K, device)
    logit_store = [[] for _ in range(K)]
    for batch in tqdm(gen_dl, desc=f"gen [{config_name}]", total=len(gen_dl)):
        src = batch.to(device) if gpu_extract else batch
        tan = equirectangular_to_tangents_batch(src, face_size=face_size, config=config)
        if not gpu_extract:
            tan = tan.to(device)
        b, t, c, h, w = tan.shape
        u8 = (tan.reshape(b * t, c, h, w) * 255.0).to(torch.uint8)
        pool = _chunked_forward(fid_net, u8, inception_chunk).reshape(b, t, -1)
        logits = _chunked_forward(is_net, u8, inception_chunk).reshape(b, t, -1)
        for gi in range(K):
            _accumulate_fid(store, average_features_by_view_group(pool, view_idx[gi]), gi)
            logit_store[gi].append(average_features_by_view_group(logits, view_idx[gi]).detach().cpu())

    gen_stats = [_stats_from_accum(store["sum"][v], store["cov"][v], store["n"][v]) for v in range(K)]
    gen_logits = [torch.cat(chunks, dim=0) for chunks in logit_store]

    fid_point = np.array(
        [_fid_from_stats(*real_stats[v], *gen_stats[v]) for v in range(K)], dtype=np.float64
    )
    is_gen = torch.Generator(device=device)
    is_gen.manual_seed(seed)
    is_point = np.array(
        [_inception_score(gen_logits[v].to(device), splits, generator=is_gen) for v in range(K)],
        dtype=np.float64,
    )

    weight_list = [weights[g] for g in groups]
    fid_summary = summarize_ci(fid_point, None, side="upper", weights=weight_list)
    is_summary = summarize_ci(is_point, None, side="lower")

    # Reproduce the ORIGINAL paper aggregation EXACTLY. The published TangentFID
    # is a confidence bound computed on the polar-WEIGHTED per-view scores
    # (w_i * FID_i), centred on the weighted mean:
    #     weighted_mean + 1.96 * std(weighted_scores) / sqrt(K)
    # (std is the population std, matching the original np.sqrt(np.var(...))).
    weighted_scores = np.asarray(weight_list, dtype=np.float64) * fid_point
    fid_summary["ci_weighted"] = float(
        weighted_scores.mean() + 1.96 * weighted_scores.std() / np.sqrt(K)
    )
    return {"K": K, "fid": fid_summary, "is": is_summary}
