import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
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

def compute_group_fid(real_imgs, gen_imgs, group, device="cuda", view_group_type=ViewGroupType.POLAR_VS_EQUATORIAL):
    view_group = view_map_types[view_group_type][group]

    real_group_imgs = average_features_by_view_group(real_imgs, view_group)
    gen_group_imgs = average_features_by_view_group(gen_imgs, view_group)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.set_dtype(torch.float64)
    fid.update(real_group_imgs, real=True)
    fid.update(gen_group_imgs, real=False)
    return fid.compute().item()


def compute_tangentfid(
    real_images,
    gen_images,
    pano_size=(512, 1024),
    face_size=192,
    device="cuda" if torch.cuda.is_available() else "cpu",
    view_group_type=ViewGroupType.POLAR_VS_EQUATORIAL
):
    # real_eqr_imgs = preprocess_images(real_images, image_size=pano_size, device=device)
    # gen_eqr_imgs = preprocess_images(gen_images, image_size=pano_size, device=device)
    real_eqr_imgs = RealDataset(real_images, transform=preprocess_images())#.to(device)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images())#.to(device)

    real_dl = torch.utils.data.DataLoader(real_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)
    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

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

    for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing TangentFID", total=len(real_dl)):
        real_tangent_imgs = equirectangular_to_tangents_batch(real_batch, face_size=face_size)
        gen_tangent_imgs = equirectangular_to_tangents_batch(gen_batch, face_size=face_size)

        for group in view_map_types[view_group_type].keys():
            real_group_imgs = average_features_by_view_group(real_tangent_imgs, view_map_types[view_group_type][group])
            gen_group_imgs = average_features_by_view_group(gen_tangent_imgs, view_map_types[view_group_type][group])

            real_group_imgs = (real_group_imgs * 255.0).to(torch.uint8)
            gen_group_imgs = (gen_group_imgs * 255.0).to(torch.uint8)

            fids[group].update(real_group_imgs.to(device), real=True)
            fids[group].update(gen_group_imgs.to(device), real=False)

    average_fid = 0.0
    # Compute FID for each group
    for group in fids.keys():
        average_fid += fids[group].compute().item()

    average_fid /= len(fids)
    print(f"TangentFID mean: {average_fid}")
    return average_fid