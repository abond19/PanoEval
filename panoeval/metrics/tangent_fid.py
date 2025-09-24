import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
# from .new_fid import FrechetInceptionDistance
from .tangent_block_fid import PanoramicFrechetInceptionDistance
from tqdm import tqdm
from enum import Enum
import numpy as np
from ..utils.dataloader import GeneratedDataset, RealDataset
from .dinov2 import DINOv2Encoder

from panoeval.eq2pers_v3_updated import process_image_input as get_tangent_images

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

def equirectangular_to_tangents_batch(eqr_imgs, face_size=192):
    B, C, H, W = eqr_imgs.shape

    results = torch.vmap(get_tangent_images)(eqr_imgs, patch_size=face_size)
    return results  # shape: (B, 18, C, face_size, face_size)

def average_features_by_view_group(tangent_imgs, group_indices):
    group_faces = tangent_imgs[:, group_indices, :]  # shape: (B, G, 3, H, W)
    return group_faces.mean(dim=1)  # average over group faces

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
    view_group_type=ViewGroupType.ALL_DIFFERENT,
    use_matterport=True
):
    # real_eqr_imgs = preprocess_images(real_images, image_size=pano_size, device=device)
    # gen_eqr_imgs = preprocess_images(gen_images, image_size=pano_size, device=device)
    real_eqr_imgs = RealDataset(real_images, transform=preprocess_images(), use_matterport=use_matterport)#.to(device)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images(), use_matterport=use_matterport)#.to(device)

    real_dl = torch.utils.data.DataLoader(real_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)
    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    # dino_model = DINOv2Encoder(arch="vitb14").to(device)

    if view_group_type == ViewGroupType.POLAR_VS_EQUATORIAL:
        fids = {
            "Polar": FrechetInceptionDistance(feature=2048).to(device),
            "Equatorial": FrechetInceptionDistance(feature=2048).to(device)
        }
    elif view_group_type == ViewGroupType.ROW_BASED:
        fids = {
            "Top": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 1": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 2": FrechetInceptionDistance(feature=2048).to(device),
            "Bottom": FrechetInceptionDistance(feature=2048).to(device)
        }
    elif view_group_type == ViewGroupType.THREE_ROWS:
        fids = {
            "Top": FrechetInceptionDistance(feature=2048).to(device),
            "Middle": FrechetInceptionDistance(feature=2048).to(device),
            "Bottom": FrechetInceptionDistance(feature=2048).to(device)
        }
    elif view_group_type == ViewGroupType.ALL_DIFFERENT:
        fids = {
            "Top1": FrechetInceptionDistance(feature=2048).to(device),
            "Top2": FrechetInceptionDistance(feature=2048).to(device),
            "Top3": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 1": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 2": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 3": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 4": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 5": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 6": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 7": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 8": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 9": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 10": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 11": FrechetInceptionDistance(feature=2048).to(device),
            "Middle 12": FrechetInceptionDistance(feature=2048).to(device),
            "Bottom 1": FrechetInceptionDistance(feature=2048).to(device),
            "Bottom 2": FrechetInceptionDistance(feature=2048).to(device),
            "Bottom 3": FrechetInceptionDistance(feature=2048).to(device)
        }


    for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing TangentFID", total=len(real_dl)):
        real_tangent_imgs = equirectangular_to_tangents_batch(real_batch, face_size=face_size)
        gen_tangent_imgs = equirectangular_to_tangents_batch(gen_batch, face_size=face_size)

        b1, t1, c1, h1, w1 = real_tangent_imgs.shape
        b2, t2, c2, h2, w2 = gen_tangent_imgs.shape
        real_tangent_imgs = real_tangent_imgs.reshape(b1 * t1, c1, h1, w1).to(device)
        gen_tangent_imgs = gen_tangent_imgs.reshape(b2 * t2, c2, h2, w2).to(device)

        real_tangent_features = list(fids.values())[0].inception((real_tangent_imgs * 255.0).to(torch.uint8))
        gen_tangent_features = list(fids.values())[0].inception((gen_tangent_imgs * 255.0).to(torch.uint8))
        # real_tangent_features = dino_model(dino_model.tensor_transform(real_tangent_imgs))
        # gen_tangent_features = dino_model(dino_model.tensor_transform(gen_tangent_imgs))

        # print(real_tangent_features.shape, gen_tangent_features.shape)
        real_tangent_features = real_tangent_features.reshape(b1, t1, -1)
        gen_tangent_features = gen_tangent_features.reshape(b2, t2, -1)

        for group in view_map_types[view_group_type].keys():
            real_group_imgs = average_features_by_view_group(real_tangent_features, view_map_types[view_group_type][group])
            gen_group_imgs = average_features_by_view_group(gen_tangent_features, view_map_types[view_group_type][group])

            # real_group_imgs = (real_group_imgs * 255.0).to(torch.uint8)
            # gen_group_imgs = (gen_group_imgs * 255.0).to(torch.uint8)

            # fids[group].update(real_group_imgs.to(device), real=True)
            # fids[group].update(gen_group_imgs.to(device), real=False)

            fids[group].real_features_sum += real_group_imgs.sum(dim=0)
            fids[group].real_features_cov_sum += real_group_imgs.t().mm(real_group_imgs)
            fids[group].real_features_num_samples += real_group_imgs.shape[0]
            fids[group].fake_features_sum += gen_group_imgs.sum(dim=0)
            fids[group].fake_features_cov_sum += gen_group_imgs.t().mm(gen_group_imgs)
            fids[group].fake_features_num_samples += gen_group_imgs.shape[0]

            fids[group].orig_dtype = real_group_imgs.dtype

    average_fid = 0.0
    all_fid_scores = []
    # Compute FID for each group
    for group in fids.keys():
        curr_fid = fids[group].compute().item()
        weight = 1.0 if "Middle" in group else 0.5
        average_fid += weight * curr_fid
        all_fid_scores.append(weight * curr_fid)
        print(f"TangentFID {group}: {weight * curr_fid}")

    average_fid /= len(fids)
    all_var = np.var(np.array(all_fid_scores))
    print(f"TangentFID: {average_fid}")
    print(f"TangentFID confidence bound: {average_fid + 1.96 * np.sqrt(all_var) / np.sqrt(len(fids))}")
    return average_fid