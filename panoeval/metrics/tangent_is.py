import torch
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm
from enum import Enum
import numpy as np
from ..utils.dataloader import GeneratedDataset, RealDataset

from panoeval.eq2pers_v3_updated import process_image_input as get_tangent_images

class ViewGroupType(Enum):
    POLAR_VS_EQUATORIAL = 0
    ROW_BASED = 1
    THREE_ROWS = 2


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
    group_faces = tangent_imgs[:, group_indices, :, :, :]  # shape: (B, G, 3, H, W)
    return group_faces.mean(dim=1)  # average over group faces

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
    view_group_type=ViewGroupType.POLAR_VS_EQUATORIAL
):
    # real_eqr_imgs = preprocess_images(real_images, image_size=pano_size, device=device)
    # gen_eqr_imgs = preprocess_images(gen_images, image_size=pano_size, device=device)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images())#.to(device)

    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    if view_group_type == ViewGroupType.POLAR_VS_EQUATORIAL:
        is_scores = {
            "Polar": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Equatorial": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device)
        }
    elif view_group_type == ViewGroupType.ROW_BASED:
        is_scores = {
            "Top": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 1": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 2": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Bottom": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device)
        }
    elif view_group_type == ViewGroupType.THREE_ROWS:
        is_scores = {
            "Top": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Bottom": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device)
        }

    for gen_batch in tqdm(gen_dl, desc="Computing TangentIS", total=len(gen_dl)):
        gen_tangent_imgs = equirectangular_to_tangents_batch(gen_batch, face_size=face_size)

        for group in view_map_types[view_group_type].keys():
            gen_group_imgs = average_features_by_view_group(gen_tangent_imgs, view_map_types[view_group_type][group])

            gen_group_imgs = (gen_group_imgs * 255.0).to(torch.uint8)

            is_scores[group].update(gen_group_imgs.to(device))

    average_is = 0.0
    # Compute FID for each group
    for group in is_scores.keys():
        average_is += is_scores[group].compute()[0].item()
        print(f"TangentIS {group}: {is_scores[group].compute()[0].item()}")

    average_is /= len(is_scores)
    print(f"TangentIS mean: {average_is}")


    return average_is