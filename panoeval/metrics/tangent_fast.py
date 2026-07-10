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


# ---------------------------------------------------------------------------
# Pole-region / equatorial FID and IS.
#
# Rather than a per-view confidence bound, these pool the pole-row (resp.
# middle-row) tangent crops across all images into a single sample set and
# compute one FID / IS for that region. Polar-FID directly measures generated
# pole quality against the reference; Equatorial-FID is reported for contrast.
# ---------------------------------------------------------------------------

def _new_pooled_store(device):
    return {
        "sum": torch.zeros(2048, dtype=torch.float64, device=device),
        "cov": torch.zeros(2048, 2048, dtype=torch.float64, device=device),
        "n": 0,
    }


def _accum_pooled(store, feats):
    store["sum"] += feats.sum(dim=0)
    store["cov"] += feats.t().mm(feats)
    store["n"] += feats.shape[0]


def _extract_pool_logits(batch, config, face_size, device, gpu_extract, inception_chunk, fid_net, is_net=None):
    """Return (pool, logits) of shape (b, K, D) for one batch of panoramas."""
    src = batch.to(device) if gpu_extract else batch
    tan = equirectangular_to_tangents_batch(src, face_size=face_size, config=config)
    if not gpu_extract:
        tan = tan.to(device)
    b, t, c, h, w = tan.shape
    u8 = (tan.reshape(b * t, c, h, w) * 255.0).to(torch.uint8)
    pool = _chunked_forward(fid_net, u8, inception_chunk).reshape(b, t, -1)
    logits = None
    if is_net is not None:
        logits = _chunked_forward(is_net, u8, inception_chunk).reshape(b, t, -1)
    return pool, logits


def _load_or_compute_real_polar_stats(real_dir, config, config_name, fid_net, face_size, device,
                                      use_matterport, cache_dir, batch_size, num_workers,
                                      gpu_extract, inception_chunk, group_idx):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{os.path.basename(real_dir.rstrip('/'))}_{config_name}_fs{face_size}_mp{int(use_matterport)}"
        cache_path = os.path.join(cache_dir, f"realpolar_{key}.pt")
        if os.path.exists(cache_path):
            print(f"[{config_name}] loading cached real polar stats: {cache_path}")
            data = torch.load(cache_path, map_location=device)
            return {g: (data[g][0].to(device), data[g][1].to(device)) for g in data}

    real_ds = RealDataset(real_dir, transform=preprocess_images(), use_matterport=use_matterport)
    print(f"[{config_name}] real images: {len(real_ds)} (computing polar stats; use_matterport={use_matterport})")
    dl = torch.utils.data.DataLoader(real_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    stores = {g: _new_pooled_store(device) for g in group_idx}
    for batch in tqdm(dl, desc=f"real polar [{config_name}]", total=len(dl)):
        pool, _ = _extract_pool_logits(batch, config, face_size, device, gpu_extract, inception_chunk, fid_net)
        for g, idx in group_idx.items():
            if not idx:
                continue
            _accum_pooled(stores[g], pool[:, idx].reshape(-1, pool.shape[-1]))
    stats = {g: _stats_from_accum(stores[g]["sum"], stores[g]["cov"], stores[g]["n"])
             for g in group_idx if stores[g]["n"] > 1}
    if cache_path:
        torch.save({g: (stats[g][0].cpu(), stats[g][1].cpu()) for g in stats}, cache_path)
        print(f"[{config_name}] cached real polar stats -> {cache_path}")
    return stats


def compute_polar_metrics(real_dir, gen_dir, config_name="tangent_18", mode="cbound", face_size=192,
                          splits=10, device=None, use_matterport=True, seed=0, real_cache_dir=None,
                          batch_size=32, num_workers=8, inception_chunk=512):
    """Pole-region and equatorial FID / IS for one generated set.

    ``mode="cbound"`` (default) reports the Eq. 5 confidence bound over the
    per-view FIDs/ISs within each region -- variance- and artifact-sensitive, and
    consistent with TangentFID. ``mode="pooled"`` instead pools all region crops
    into a single FID/IS; that variant is dominated by easy, uniform sky/ground
    content and is statistically fragile (pole sample count can fall below the
    feature dimension), so it is not recommended for pole-artifact evaluation.

    Returns {'config', 'mode', 'groups': {'Polar': {...}, 'Equatorial': {...}}}.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = get_extraction_config(config_name)
    polar_flags = config["polar"]
    region_idx = {
        "Polar": [i for i, p in enumerate(polar_flags) if p],
        "Equatorial": [i for i, p in enumerate(polar_flags) if not p],
    }

    if mode == "cbound":
        # Per-view FID/IS (the validated computation), then the Eq. 5 confidence
        # bound taken separately over the pole-row and equatorial views.
        res = compute_tangent_metrics(
            real_dir, gen_dir, config_name, face_size=face_size, splits=splits, device=device,
            use_matterport=use_matterport, seed=seed, real_cache_dir=real_cache_dir,
            batch_size=batch_size, num_workers=num_workers, inception_chunk=inception_chunk,
        )
        fid_pv = np.asarray(res["fid"]["per_view"], dtype=np.float64)
        is_pv = np.asarray(res["is"]["per_view"], dtype=np.float64)
        groups = {}
        for name, sel in region_idx.items():
            if not sel:
                continue
            fs = summarize_ci(fid_pv[sel], None, side="upper")
            iss = summarize_ci(is_pv[sel], None, side="lower")
            groups[name] = {"fid": fs["ci_gaussian"], "is": iss["ci_gaussian"],
                            "fid_mean": fs["mean"], "is_mean": iss["mean"], "n_views": len(sel)}
            print(f"[{config_name}] {name} (cbound): FID {fs['ci_gaussian']:.3f} (mean {fs['mean']:.3f})  "
                  f"IS {iss['ci_gaussian']:.3f} (mean {iss['mean']:.3f})  [{len(sel)} views]")
        return {"config": config_name, "mode": "cbound", "groups": groups}

    # --- mode == "pooled": one FID/IS over all crops of each region ---
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    fid_net = FrechetInceptionDistance(feature=2048).to(device).inception
    is_net = InceptionScore(feature="logits_unbiased", splits=splits, normalize=False).to(device).inception

    gpu_extract = _probe_gpu_extraction(config, device)

    real_stats = _load_or_compute_real_polar_stats(
        real_dir, config, config_name, fid_net, face_size, device, use_matterport,
        real_cache_dir, batch_size, num_workers, gpu_extract, inception_chunk, region_idx,
    )

    gen_ds = GeneratedDataset(gen_dir, transform=preprocess_images(), use_matterport=use_matterport)
    print(f"[{config_name}] gen images: {len(gen_ds)} (use_matterport={use_matterport})")
    dl = torch.utils.data.DataLoader(gen_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    stores = {g: _new_pooled_store(device) for g in region_idx}
    logit_store = {g: [] for g in region_idx}
    for batch in tqdm(dl, desc=f"gen polar [{config_name}]", total=len(dl)):
        pool, logits = _extract_pool_logits(batch, config, face_size, device, gpu_extract, inception_chunk, fid_net, is_net)
        for g, idx in region_idx.items():
            if not idx:
                continue
            _accum_pooled(stores[g], pool[:, idx].reshape(-1, pool.shape[-1]))
            logit_store[g].append(logits[:, idx].reshape(-1, logits.shape[-1]).detach().cpu())

    is_gen = torch.Generator(device=device)
    is_gen.manual_seed(seed)
    groups = {}
    for g, idx in region_idx.items():
        if not idx or stores[g]["n"] < 2 or g not in real_stats:
            continue
        gen_stat = _stats_from_accum(stores[g]["sum"], stores[g]["cov"], stores[g]["n"])
        fid = _fid_from_stats(*real_stats[g], *gen_stat)
        gl = torch.cat(logit_store[g], dim=0).to(device)
        is_val = _inception_score(gl, splits, generator=is_gen)
        groups[g] = {"fid": fid, "is": is_val, "n_views": len(idx), "n_samples": stores[g]["n"]}
        print(f"[{config_name}] {g} (pooled): FID {fid:.3f}  IS {is_val:.3f}  ({len(idx)} views/img)")
    return {"config": config_name, "mode": "pooled", "groups": groups}


# ---------------------------------------------------------------------------
# Native-resolution patch FID / IS (for evaluating high-resolution / 4K output).
#
# Each panorama is loaded at a high target resolution and split into
# native-resolution patches (raw ERP tiles, or gnomonic narrow-FOV crops via the
# "tangent_4k" config). All patches are pooled into one FID (gen vs real) and one
# IS (gen). Because patches are kept near native resolution, this is sensitive to
# fine 4K detail that a whole-image FID (which resizes to 299) discards.
# ---------------------------------------------------------------------------

def _highres_transform(target_hw):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(target_hw), transforms.ToTensor()])


def _erp_patches(imgs, patch_size):
    """Split (b, C, H, W) into non-overlapping (b*nh*nw, C, patch, patch) tiles."""
    b, c, H, W = imgs.shape
    nh, nw = H // patch_size, W // patch_size
    if nh < 1 or nw < 1:
        raise ValueError(f"patch_size {patch_size} exceeds image {H}x{W}")
    imgs = imgs[:, :, :nh * patch_size, :nw * patch_size]
    t = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return t.permute(0, 2, 3, 1, 4, 5).reshape(b * nh * nw, c, patch_size, patch_size)


def _load_or_compute_real_patch_stats(real_dir, tf, patch_feats, config_name, patch_size, target_hw,
                                      device, use_matterport, cache_dir, batch_size, num_workers):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = (f"{os.path.basename(real_dir.rstrip('/'))}_{config_name or 'erp'}_ps{patch_size}_"
               f"{target_hw[0]}x{target_hw[1]}_mp{int(use_matterport)}")
        cache_path = os.path.join(cache_dir, f"realpatch_{key}.pt")
        if os.path.exists(cache_path):
            print(f"[patch] loading cached real patch stats: {cache_path}")
            d = torch.load(cache_path, map_location=device)
            return (d["mean"].to(device), d["cov"].to(device)), int(d["n"])

    real_ds = RealDataset(real_dir, transform=tf, use_matterport=use_matterport)
    print(f"[patch] real images: {len(real_ds)} (computing patch stats; use_matterport={use_matterport})")
    dl = torch.utils.data.DataLoader(real_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    store = _new_pooled_store(device)
    for batch in tqdm(dl, desc="real patches", total=len(dl)):
        pool, _ = patch_feats(batch, False)
        _accum_pooled(store, pool)
    stats = _stats_from_accum(store["sum"], store["cov"], store["n"])
    if cache_path:
        torch.save({"mean": stats[0].cpu(), "cov": stats[1].cpu(), "n": store["n"]}, cache_path)
        print(f"[patch] cached real patch stats -> {cache_path}")
    return stats, store["n"]


def _crop_lat_band(imgs, lat_band):
    """Keep the central ``lat_band`` fraction of ERP rows (latitude band)."""
    if lat_band >= 1.0:
        return imgs
    H = imgs.shape[2]
    keep = max(1, int(round(H * lat_band)))
    top = (H - keep) // 2
    return imgs[:, :, top:top + keep, :]


def _degrade_updown(imgs, factor):
    """Round-trip through 1/factor resolution and back (bicubic).

    Simulates "generated at lower resolution, then upsampled": the content is
    preserved but high-frequency detail beyond 1/factor resolution is lost.
    """
    if not factor or factor <= 1:
        return imgs
    import torch.nn.functional as F
    H, W = imgs.shape[2], imgs.shape[3]
    small = F.interpolate(imgs, size=(max(1, H // factor), max(1, W // factor)),
                          mode="bicubic", align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bicubic", align_corners=False).clamp_(0, 1)


def compute_patch_fid(real_dir, gen_dir, patch_size=512, target_hw=(2048, 4096),
                      config_name=None, lat_band=1.0, degrade_factor=None, splits=10, device=None,
                      use_matterport=True, seed=0, real_cache_dir=None, batch_size=4, num_workers=4,
                      inception_chunk=256):
    """Native-resolution patch FID / IS between a generated set and a reference.

    ``config_name=None`` tiles the (resized-to-``target_hw``) ERP into raw
    ``patch_size`` patches; ``config_name="tangent_4k"`` instead extracts dense
    narrow-FOV gnomonic crops. ``lat_band<1.0`` restricts the raw-ERP tiling to
    the central latitude fraction (e.g. 0.5 = middle +/-45deg), excluding the
    heavily ERP-distorted pole rows. Returns a dict with ``patch_fid`` /
    ``patch_is``. Intended for evaluating high-resolution output against a
    high-resolution reference (no per-view aggregation, no baseline downsampling).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = get_extraction_config(config_name) if config_name else None
    gpu_extract = _probe_gpu_extraction(config, device) if config is not None else False

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    fid_net = FrechetInceptionDistance(feature=2048).to(device).inception
    is_net = InceptionScore(feature="logits_unbiased", splits=splits, normalize=False).to(device).inception

    tf = _highres_transform(target_hw)

    def patch_feats(batch, want_logits):
        if config is None:
            imgs = _crop_lat_band(batch.to(device), lat_band)
            u8 = (_erp_patches(imgs, patch_size) * 255.0).to(torch.uint8)
            pool = _chunked_forward(fid_net, u8, inception_chunk)
            logits = _chunked_forward(is_net, u8, inception_chunk) if want_logits else None
            return pool, logits
        pool_bt, logits_bt = _extract_pool_logits(batch, config, patch_size, device, gpu_extract,
                                                  inception_chunk, fid_net, is_net if want_logits else None)
        pool = pool_bt.reshape(-1, pool_bt.shape[-1])
        logits = logits_bt.reshape(-1, logits_bt.shape[-1]) if (want_logits and logits_bt is not None) else None
        return pool, logits

    cache_tag = f"{config_name or 'erp'}_lb{lat_band}"
    real_stats, real_n = _load_or_compute_real_patch_stats(
        real_dir, tf, patch_feats, cache_tag, patch_size, target_hw, device,
        use_matterport, real_cache_dir, batch_size, num_workers)

    gen_ds = GeneratedDataset(gen_dir, transform=tf, use_matterport=use_matterport)
    deg = f", degraded 1/{degrade_factor}x round-trip" if degrade_factor else ""
    print(f"[patch] gen images: {len(gen_ds)} -> {patch_size}px patches @ target {target_hw} "
          f"(mode={config_name or 'erp'}{deg})")
    dl = torch.utils.data.DataLoader(gen_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    store = _new_pooled_store(device)
    logit_store = []
    for batch in tqdm(dl, desc="gen patches", total=len(dl)):
        if degrade_factor:
            batch = _degrade_updown(batch.to(device), degrade_factor)
        pool, logits = patch_feats(batch, True)
        _accum_pooled(store, pool)
        logit_store.append(logits.detach().cpu())
    gen_stat = _stats_from_accum(store["sum"], store["cov"], store["n"])
    fid = _fid_from_stats(*real_stats, *gen_stat)
    is_gen = torch.Generator(device=device)
    is_gen.manual_seed(seed)
    is_val = _inception_score(torch.cat(logit_store, dim=0).to(device), splits, generator=is_gen)
    print(f"[patch] patch-FID {fid:.3f}  patch-IS {is_val:.3f}  "
          f"(gen {store['n']} / real {real_n} patches)")
    return {"patch_fid": fid, "patch_is": is_val, "n_gen_patches": store["n"],
            "n_real_patches": real_n, "patch_size": patch_size,
            "target_hw": list(target_hw), "config": config_name or "erp",
            "degrade_factor": degrade_factor}
