import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
# from .new_fid import FrechetInceptionDistance
from .tangent_block_fid import PanoramicFrechetInceptionDistance
from tqdm import tqdm
from enum import Enum
from functools import partial
import numpy as np
from ..utils.dataloader import GeneratedDataset, RealDataset
from .dinov2 import DINOv2Encoder

from panoeval.eq2pers_v3_updated import (
    process_image_input as get_tangent_images,
    get_extraction_config,
    build_view_groups,
)
from .tangent_ci import summarize_ci

class ViewGroupType(Enum):
    POLAR_VS_EQUATORIAL = 0
    ROW_BASED = 1
    THREE_ROWS = 2
    ALL_DIFFERENT = 3


view_map_types = {
    ViewGroupType.POLAR_VS_EQUATORIAL: {
        'Polar': [0, 1, 2, 15, 16, 17],  
        'Equatorial': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    },
    ViewGroupType.ROW_BASED: {
        "Top": [0, 1, 2],
        "Middle 1": [3, 4, 5, 6, 7, 8],
        "Middle 2": [9, 10, 11, 12, 13, 14],
        "Bottom": [15, 16, 17]
    },
    ViewGroupType.THREE_ROWS: {
        "Top": [0, 1, 2],
        "Middle": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "Bottom": [15, 16, 17]
    },
    ViewGroupType.ALL_DIFFERENT: {
        "Top1": [0],
        "Top2": [1],
        "Top3": [2],
        "Middle 1": [3],
        "Middle 2": [4],
        "Middle 3": [5],
        "Middle 4": [6],
        "Middle 5": [7],
        "Middle 6": [8],
        "Middle 7": [9],
        "Middle 8": [10],
        "Middle 9": [11],
        "Middle 10": [12],
        "Middle 11": [13],
        "Middle 12": [14],
        "Bottom 1": [15],
        "Bottom 2": [16],
        "Bottom 3": [17]
    }
}

def preprocess_images(image_size=(512, 1024), device="cuda"):
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    return tf

def equirectangular_to_tangents_batch(eqr_imgs, face_size=192, config=None):
    # Bind the (static) patch size and extraction config so vmap only maps over
    # the batch dimension of the ERP images.
    fn = partial(get_tangent_images, patch_size=face_size, config=config)
    results = torch.vmap(fn)(eqr_imgs)
    return results  # shape: (B, num_planes, C, face_size, face_size)

def average_features_by_view_group(tangent_imgs, group_indices):
    group_faces = tangent_imgs[:, group_indices, :]  # shape: (B, G, 3, H, W)
    return group_faces.mean(dim=1)  # average over group faces


def _mean_cov(feats):
    """Mean vector and (unbiased) covariance matrix of a feature set.

    ``feats`` is ``(N, D)``. Computed in float64 to match the numerical
    behaviour of torchmetrics' FID routine.
    """
    feats = feats.double()
    n = feats.shape[0]
    mean = feats.mean(dim=0)
    centered = feats - mean
    cov = centered.t().mm(centered) / (n - 1)
    return mean, cov


def _fid_from_stats(mu1, sigma1, mu2, sigma2):
    """Fréchet distance between two Gaussians, equivalent to torchmetrics' FID.

    Uses Tr(sqrt(sigma1 @ sigma2)) = sum of sqrt of the eigenvalues of the
    product, avoiding an explicit matrix square root.
    """
    diff = mu1 - mu2
    eigvals = torch.linalg.eigvals(sigma1 @ sigma2)
    tr_covmean = eigvals.sqrt().real.sum()
    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return float(fid.item())


def _extract_view_features(dataloader, extractor, config, view_map, face_size, device, desc):
    """Run the Inception feature extractor on every tangent view of a dataset.

    Returns a list (one entry per view group) of ``(N, D)`` CPU tensors holding
    the per-image feature for that view. Caching the features lets us recompute
    per-view FID cheaply during bootstrapping without re-running the network.
    """
    groups = list(view_map.keys())
    per_view = [[] for _ in groups]
    for batch in tqdm(dataloader, desc=desc, total=len(dataloader)):
        tangent = equirectangular_to_tangents_batch(batch, face_size=face_size, config=config)
        b, t, c, h, w = tangent.shape
        tangent = tangent.reshape(b * t, c, h, w).to(device)
        feats = extractor((tangent * 255.0).to(torch.uint8))
        feats = feats.reshape(b, t, -1)
        for gi, group in enumerate(groups):
            group_feat = average_features_by_view_group(feats, view_map[group])  # (b, D)
            per_view[gi].append(group_feat.detach().cpu())
    return [torch.cat(chunks, dim=0) for chunks in per_view]

def compute_group_fid(real_imgs, gen_imgs, group, device="cuda", view_group_type=ViewGroupType.POLAR_VS_EQUATORIAL):
    view_group = view_map_types[view_group_type][group]

    real_group_imgs = average_features_by_view_group(real_imgs, view_group)
    gen_group_imgs = average_features_by_view_group(gen_imgs, view_group)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.set_dtype(torch.float64)
    fid.update(real_group_imgs, real=True)
    fid.update(gen_group_imgs, real=False)
    return fid.compute().item()

# def compute_tangentfid(
#     real_images,
#     gen_images,
#     pano_size=(512, 1024),
#     face_size=192,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     view_group_type=ViewGroupType.ALL_DIFFERENT
# ):
#     real_eqr_imgs = RealDataset(real_images, transform=preprocess_images())#.to(device)
#     gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images())#.to(device)

#     real_dl = torch.utils.data.DataLoader(real_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)
#     gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

#     metric = PanoramicFrechetInceptionDistance(num_planes=18).to(device)

#     for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing TangentFID", total=len(real_dl)):
#         # Get the tangent images
#         real_tangent_imgs = equirectangular_to_tangents_batch(real_batch, face_size=face_size)
#         gen_tangent_imgs = equirectangular_to_tangents_batch(gen_batch, face_size=face_size)
#         b1, t1, c1, h1, w1 = real_tangent_imgs.shape
#         b2, t2, c2, h2, w2 = gen_tangent_imgs.shape
#         real_tangent_imgs = real_tangent_imgs.reshape(b1 * t1, c1, h1, w1).to(device)
#         gen_tangent_imgs = gen_tangent_imgs.reshape(b2 * t2, c2, h2, w2).to(device)
#         # Compute the features
#         real_tangent_features = metric.inception((real_tangent_imgs * 255.0).to(torch.uint8))
#         gen_tangent_features = metric.inception((gen_tangent_imgs * 255.0).to(torch.uint8))
#         real_tangent_features = real_tangent_features.reshape(b1, t1, -1)
#         gen_tangent_features = gen_tangent_features.reshape(b2, t2, -1)
#         # Update the metric
#         for i in range(18):
#             metric.update(real_tangent_features[:, i], real=True, plane_idx=i)
#             metric.update(gen_tangent_features[:, i], real=False, plane_idx=i)

#     # Compute the FID
#     fid = metric.compute().item()
#     print(f"TangentFID: {fid}")
#     return fid


def compute_tangentfid(
    real_images,
    gen_images,
    pano_size=(512, 1024),
    face_size=192,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config_name="tangent_18",
    return_details=False,
    n_bootstrap=0,
    bootstrap_seed=0,
    use_matterport=True
):
    """Compute TangentFID for a given tangent-plane extraction configuration.

    FID is computed independently on each tangent view; the per-view scores are
    summarised into a confidence bound. Several bound variants are produced (see
    ``tangent_ci.summarize_ci``): ``gaussian`` (Eq. 5, assumes independent
    views), ``tolerance`` (drops sqrt(K)), and — when ``n_bootstrap > 0`` —
    ``effective_n``, ``bootstrap_se`` and ``bootstrap_pct``, which relax the
    independence assumption using the correlation between overlapping views.

    Per-view Inception features are cached so the bootstrap resamples whole
    panoramas (preserving cross-view correlation) without re-running the network.
    The generated set is resampled while the reference statistics are held fixed.

    ``config_name`` selects the extraction layout (e.g. "tangent_10",
    "tangent_18", "tangent_26", "tangent_46" or "cubemap").

    By default the legacy polar-weighted mean is returned (backwards compatible).
    Set ``return_details=True`` to obtain a dict with the full per-view breakdown
    and every confidence-bound variant.
    """
    config = get_extraction_config(config_name)
    view_map, weights = build_view_groups(config)
    group_names = list(view_map.keys())
    K = len(group_names)

    real_eqr_imgs = RealDataset(real_images, transform=preprocess_images(), use_matterport=use_matterport)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images(), use_matterport=use_matterport)

    real_dl = torch.utils.data.DataLoader(real_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)
    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    extractor = FrechetInceptionDistance(feature=2048).to(device).inception

    # Cache per-view features once (the expensive Inception forward pass).
    real_feats = _extract_view_features(
        real_dl, extractor, config, view_map, face_size, device, f"TangentFID feats (real) [{config_name}]"
    )
    gen_feats = _extract_view_features(
        gen_dl, extractor, config, view_map, face_size, device, f"TangentFID feats (gen) [{config_name}]"
    )

    # Reference (real) statistics are fixed across bootstrap replicates.
    real_stats = [_mean_cov(rf.to(device)) for rf in real_feats]

    # Point estimate: one FID per view from the full generated set.
    point_scores = np.array(
        [_fid_from_stats(*real_stats[v], *_mean_cov(gen_feats[v].to(device))) for v in range(K)],
        dtype=np.float64,
    )
    for name, score in zip(group_names, point_scores):
        print(f"TangentFID {name}: {score:.4f} (weight {weights[name]})")

    # Bootstrap: resample the generated panoramas and recompute every per-view FID.
    boot_scores = None
    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        n_gen = gen_feats[0].shape[0]
        gen_dev = [gf.to(device) for gf in gen_feats]  # kept on device for fast indexing
        boot_scores = np.empty((n_bootstrap, K), dtype=np.float64)
        for bi in tqdm(range(n_bootstrap), desc=f"Bootstrap TangentFID [{config_name}]"):
            idx = torch.as_tensor(rng.integers(0, n_gen, size=n_gen), device=device)
            for v in range(K):
                boot_scores[bi, v] = _fid_from_stats(*real_stats[v], *_mean_cov(gen_dev[v][idx]))

    weight_list = [weights[name] for name in group_names]
    summary = summarize_ci(point_scores, boot_scores, side="upper", weights=weight_list)

    print(f"[{config_name}] TangentFID mean: {summary['mean']:.4f}  std: {summary['std']:.4f}  K: {K}")
    print(f"[{config_name}] TangentFID gaussian (Eq.5):  {summary['ci_gaussian']:.4f}")
    print(f"[{config_name}] TangentFID tolerance:        {summary['ci_tolerance']:.4f}")
    if boot_scores is not None:
        print(f"[{config_name}] TangentFID effective_n:      {summary['ci_effective_n']:.4f} "
              f"(rho_bar={summary['rho_bar']:.3f}, K_eff={summary['k_eff']:.2f})")
        print(f"[{config_name}] TangentFID bootstrap_se:     {summary['ci_bootstrap_se']:.4f}")
        print(f"[{config_name}] TangentFID bootstrap_pct:    {summary['ci_bootstrap_pct']:.4f}")

    if return_details:
        details = {"config": config_name, "metric": "TangentFID"}
        details.update(summary)
        details["per_view"] = dict(zip(group_names, point_scores.tolist()))
        details["tangentfid"] = summary["ci_gaussian"]  # headline value (paper default)
        return details
    return summary["weighted_mean"]