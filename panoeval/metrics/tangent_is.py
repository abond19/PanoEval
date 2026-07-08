import torch
from torchvision import transforms
import torchvision
from torchmetrics.image.inception import InceptionScore
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
    view_group_type=ViewGroupType.ALL_DIFFERENT,
    use_matterport=True
):
    # real_eqr_imgs = preprocess_images(real_images, image_size=pano_size, device=device)
    # gen_eqr_imgs = preprocess_images(gen_images, image_size=pano_size, device=device)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images(), use_matterport=use_matterport)#.to(device)

    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    # dino_model = DINOv2Encoder(arch="vitb14").to(device)

    # feature = 768

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
    elif view_group_type == ViewGroupType.ALL_DIFFERENT:
        is_scores = {
            "Top1": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Top2": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Top3": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 1": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 2": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 3": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 4": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 5": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 6": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 7": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 8": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 9": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 10": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 11": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Middle 12": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Bottom 1": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Bottom 2": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device),
            "Bottom 3": InceptionScore(feature=feature, splits=splits, normalize=normalize).to(device)
        }

    for idx, gen_batch in enumerate(tqdm(gen_dl, desc="Computing TangentIS", total=len(gen_dl))):
        # if idx == 0:
        #     torchvision.io.write_png((gen_batch[0] * 255).to(torch.uint8), f"test_tangents/gen_img_{idx}.png")
        gen_tangent_imgs = equirectangular_to_tangents_batch(gen_batch, face_size=face_size)
        b, t, c, h, w = gen_tangent_imgs.shape

        # if idx == 0:
        #     for i in range(t):
        #         torchvision.io.write_png((gen_tangent_imgs[0, i] * 255).to(torch.uint8), f"test_tangents/tangent_img_{i}.png")

        gen_tangent_imgs = gen_tangent_imgs.view(b * t, c, h, w)

        gen_tangent_features = list(is_scores.values())[0].inception((gen_tangent_imgs * 255.0).to(torch.uint8).to(device))
        # gen_tangent_features = dino_model(dino_model.tensor_transform(gen_tangent_imgs.to(device)))
        gen_tangent_features = gen_tangent_features.view(b, t, -1)

        for group in view_map_types[view_group_type].keys():
            gen_group_imgs = average_features_by_view_group(gen_tangent_features, view_map_types[view_group_type][group])

            is_scores[group].features.append(gen_group_imgs)

    average_is = 0.0
    is_var = 0.0
    all_is_scores = []
    # Compute FID for each group
    for group in is_scores.keys():
        mean, var = is_scores[group].compute()
        average_is += mean.item()
        is_var += var.item() ** 2
        all_is_scores.append(mean.item())
        print(f"TangentIS {group}: {mean} ± {var}")

    average_is /= len(is_scores)
    is_var /= len(is_scores)**2
    all_var = np.var(np.array(all_is_scores))
    print(f"TangentIS mean: {average_is}")
    print(f"TangentIS var: {is_var}")
    print(f"TangentIS all var: {all_var}")
    print("Confidence bound for TangentIS: ", average_is  - 1.96 * np.sqrt(all_var) / np.sqrt(len(is_scores)))


    return average_is