import torch
from torchvision import transforms
import torchvision
from torchmetrics.image.inception import InceptionScore
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


def _inception_score(logits, splits):
    """Mean Inception Score from a set of Inception logits.

    Replicates torchmetrics' InceptionScore.compute (mean over ``splits``),
    letting us recompute a view's IS on bootstrap resamples cheaply.
    """
    prob = logits.softmax(dim=1)
    log_prob = logits.log_softmax(dim=1)
    n = prob.shape[0]
    step = max(n // splits, 1)
    scores = []
    for k in range(splits):
        p = prob[k * step:(k + 1) * step]
        lp = log_prob[k * step:(k + 1) * step]
        if p.shape[0] == 0:
            continue
        mean_p = p.mean(dim=0, keepdim=True)
        kl = (p * (lp - mean_p.log())).sum(dim=1)
        scores.append(kl.mean().exp())
    return float(torch.stack(scores).mean().item())


def _extract_view_logits(dataloader, extractor, config, view_map, face_size, device, desc):
    """Run the Inception logit extractor on every tangent view of a dataset.

    Returns a list (one per view group) of ``(N, C)`` CPU tensors of per-image
    logits, cached so bootstrap resampling avoids re-running the network.
    """
    groups = list(view_map.keys())
    per_view = [[] for _ in groups]
    for batch in tqdm(dataloader, desc=desc, total=len(dataloader)):
        tangent = equirectangular_to_tangents_batch(batch, face_size=face_size, config=config)
        b, t, c, h, w = tangent.shape
        tangent = tangent.reshape(b * t, c, h, w).to(device)
        logits = extractor((tangent * 255.0).to(torch.uint8))
        logits = logits.reshape(b, t, -1)
        for gi, group in enumerate(groups):
            group_logits = average_features_by_view_group(logits, view_map[group])  # (b, C)
            per_view[gi].append(group_logits.detach().cpu())
    return [torch.cat(chunks, dim=0) for chunks in per_view]

# def compute_group_fid(gen_imgs, group, device="cuda", view_group_type=ViewGroupType.POLAR_VS_EQUATORIAL):
#     view_group = view_map_types[view_group_type][group]

#     gen_group_imgs = average_features_by_view_group(gen_imgs, view_group)
#     fid = FrechetInceptionDistance(feature=2048).to(device)
#     fid.set_dtype(torch.float64)
#     fid.update(real_group_imgs, real=True)
#     fid.update(gen_group_imgs, real=False)
#     return fid.compute().item()


def compute_tangentis(
    gen_images,
    feature='logits_unbiased',
    splits=10,
    normalize=False,
    pano_size=(512, 1024),
    face_size=192,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config_name="tangent_18",
    return_details=False,
    n_bootstrap=0,
    bootstrap_seed=0,
    use_matterport=True
):
    """Compute TangentIS for a given tangent-plane extraction configuration.

    Inception Score is computed independently on each tangent view; the per-view
    scores are summarised into a confidence bound. Several bound variants are
    produced (see ``tangent_ci.summarize_ci``): ``gaussian`` (Eq. 5, assumes
    independent views), ``tolerance`` (drops sqrt(K)), and — when
    ``n_bootstrap > 0`` — ``effective_n``, ``bootstrap_se`` and ``bootstrap_pct``,
    which relax the independence assumption using the correlation between
    overlapping views.

    Per-view Inception logits are cached so the bootstrap resamples whole
    panoramas (preserving cross-view correlation) without re-running the network.

    ``config_name`` selects the extraction layout (e.g. "tangent_10",
    "tangent_18", "tangent_26", "tangent_46" or "cubemap").

    By default the mean per-view IS is returned (backwards compatible). Set
    ``return_details=True`` to obtain a dict with the full per-view breakdown and
    every confidence-bound variant.
    """
    config = get_extraction_config(config_name)
    view_map, _ = build_view_groups(config)
    group_names = list(view_map.keys())
    K = len(group_names)

    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images(), use_matterport=use_matterport)
    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    extractor = InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device).inception

    # Cache per-view logits once (the expensive Inception forward pass).
    gen_logits = _extract_view_logits(
        gen_dl, extractor, config, view_map, face_size, device, f"TangentIS logits [{config_name}]"
    )

    # Point estimate: one IS per view from the full generated set.
    point_scores = np.array(
        [_inception_score(gen_logits[v].to(device), splits) for v in range(K)],
        dtype=np.float64,
    )
    for name, score in zip(group_names, point_scores):
        print(f"TangentIS {name}: {score:.4f}")

    # Bootstrap: resample the generated panoramas and recompute every per-view IS.
    boot_scores = None
    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        n_gen = gen_logits[0].shape[0]
        gen_dev = [gl.to(device) for gl in gen_logits]
        boot_scores = np.empty((n_bootstrap, K), dtype=np.float64)
        for bi in tqdm(range(n_bootstrap), desc=f"Bootstrap TangentIS [{config_name}]"):
            idx = torch.as_tensor(rng.integers(0, n_gen, size=n_gen), device=device)
            for v in range(K):
                boot_scores[bi, v] = _inception_score(gen_dev[v][idx], splits)

    summary = summarize_ci(point_scores, boot_scores, side="lower")

    print(f"[{config_name}] TangentIS mean: {summary['mean']:.4f}  std: {summary['std']:.4f}  K: {K}")
    print(f"[{config_name}] TangentIS gaussian (Eq.5):  {summary['ci_gaussian']:.4f}")
    print(f"[{config_name}] TangentIS tolerance:        {summary['ci_tolerance']:.4f}")
    if boot_scores is not None:
        print(f"[{config_name}] TangentIS effective_n:      {summary['ci_effective_n']:.4f} "
              f"(rho_bar={summary['rho_bar']:.3f}, K_eff={summary['k_eff']:.2f})")
        print(f"[{config_name}] TangentIS bootstrap_se:     {summary['ci_bootstrap_se']:.4f}")
        print(f"[{config_name}] TangentIS bootstrap_pct:    {summary['ci_bootstrap_pct']:.4f}")

    if return_details:
        details = {"config": config_name, "metric": "TangentIS"}
        details.update(summary)
        details["per_view"] = dict(zip(group_names, point_scores.tolist()))
        details["tangentis"] = summary["ci_gaussian"]  # headline value (paper default)
        return details
    return summary["mean"]