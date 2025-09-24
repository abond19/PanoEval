import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
# import kornia.geometry as KG
import py360convert
import numpy as np
from tqdm import tqdm
from ..utils.dataloader import GeneratedDataset, RealDataset
import torchvision


def preprocess_images(image_size=(512, 1024), device='cuda'):
    """
    Preprocess images to match the input requirements for the metric.
    Returns a tensor of shape (N, 3, H, W).
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return tf


def equirectangular_to_cubemap_batch(eqr_imgs, face_size=256, max_workers=4):
    """
    Converts a batch of equirectangular images to cubemap format using Kornia.
    Returns: Tensor of shape (B, 6, 3, face_size, face_size)
    """
    batch_size = eqr_imgs.shape[0]
    device = eqr_imgs.device
    
    # Process each image sequentially
    cube_faces_list = []
    for i in range(batch_size):
        img_np = eqr_imgs[i].cpu().permute(1, 2, 0).numpy()  # Convert to numpy and change to HWC format
        cube_face = py360convert.e2c(img_np, cube_format="list")
        cube_face = np.stack(cube_face)  # Stack the cube faces
        # print(f"1 cube face shape: {cube_face.shape}")
        cube_faces_list.append(cube_face)
    # print(f"Time taken for conversion: {end_time - start_time:.2f} seconds")
    
    # Convert back to tensor
    cube_faces_np = np.stack(cube_faces_list, axis=0)
    # print(f"Cube faces shape: {cube_faces_np.shape}")
    return torch.from_numpy(cube_faces_np).permute(0, 1, 4, 2, 3).to(device)

    


def average_features_by_view_group(cubemaps, group_indices):
    """
    Averages the cubemap faces per group. Input shape: (B, 6, 3, H, W)
    group_indices: list of indices for the group
    Returns tensor: (B, 3, H, W)
    """
    group_faces = cubemaps[:, group_indices, :]  # shape: (B, G, 3, H, W)
    return group_faces.mean(dim=1)  # average over group faces


def compute_group_fid(real_imgs, gen_imgs, group, device='cuda'):
    """
    Compute FID for a specific cubemap view group.
    """
    view_map = {
        'F': [0, 1, 2, 3],  # Front, Right, Back, Left
        'U': [4],           # Up
        'D': [5]            # Down
    }
    group_idx = view_map[group]

    real_group_imgs = average_features_by_view_group(real_imgs, group_idx)
    gen_group_imgs = average_features_by_view_group(gen_imgs, group_idx)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.set_dtype(torch.float64)

    fid.update(real_group_imgs, real=True)
    fid.update(gen_group_imgs, real=False)
    return fid.compute().item()


def compute_omnifid(
    real_images,
    gen_images,
    pano_size=(256, 512),
    face_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_matterport=True
):
    """
    Compute OmniFID from equirectangular panoramas.
    """
    view_map = {
        'F': [0, 1, 2, 3],  # Front, Right, Back, Left
        'U': [4],           # Up
        'D': [5]            # Down
    }
    fid_f = FrechetInceptionDistance(feature=2048).to(device)
    fid_f.set_dtype(torch.float32)
    fid_u = FrechetInceptionDistance(feature=2048).to(device)
    fid_u.set_dtype(torch.float32)
    fid_d = FrechetInceptionDistance(feature=2048).to(device)
    fid_d.set_dtype(torch.float32)
    # Step 1: Preprocess panos to equirectangular images    
    real_eqr_imgs = RealDataset(real_images, transform=preprocess_images(), use_matterport=use_matterport)#.to(device)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images(), use_matterport=use_matterport)#.to(device)
    real_dl = torch.utils.data.DataLoader(real_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)  
    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    for idx, (real_batch, gen_batch) in enumerate(tqdm(zip(real_dl, gen_dl), desc="Computing OmniFID", total=len(real_dl))):
        if idx == 0:
            torchvision.io.write_png((real_batch[0] * 255).to(torch.uint8), "real_batch.png")
            torchvision.io.write_png((gen_batch[0] * 255).to(torch.uint8), "gen_batch.png")
        # print("Start of batch")
        real_cubemaps = equirectangular_to_cubemap_batch(real_batch, face_size=face_size)
        gen_cubemaps = equirectangular_to_cubemap_batch(gen_batch, face_size=face_size)
        # print("Converted to cubemaps")

        b1, t1, c1, h1, w1 = real_cubemaps.shape
        b2, t2, c2, h2, w2 = gen_cubemaps.shape

        if idx == 0:
            for i in range(t1):
                torchvision.io.write_png((real_cubemaps[0, i] * 255).to(torch.uint8), f"real_cubemap_{i}.png")
                torchvision.io.write_png((gen_cubemaps[0, i] * 255).to(torch.uint8), f"gen_cubemap_{i}.png")

        real_cubemaps = real_cubemaps.reshape(b1 * t1, c1, h1, w1).to(device)
        gen_cubemaps = gen_cubemaps.reshape(b2 * t2, c2, h2, w2).to(device)

        real_cube_features = fid_f.inception((real_cubemaps * 255.0).to(torch.uint8))
        gen_cube_features = fid_f.inception((gen_cubemaps * 255.0).to(torch.uint8))
        real_cube_features = real_cube_features.reshape(b1, t1, -1)
        gen_cube_features = gen_cube_features.reshape(b2, t2, -1)

        real_group_imgs_F = average_features_by_view_group(real_cube_features, view_map["F"])
        gen_group_imgs_F = average_features_by_view_group(gen_cube_features, view_map["F"])

        real_group_imgs_U = average_features_by_view_group(real_cube_features, view_map["U"])
        gen_group_imgs_U = average_features_by_view_group(gen_cube_features, view_map["U"])

        real_group_imgs_D = average_features_by_view_group(real_cube_features, view_map["D"])
        gen_group_imgs_D = average_features_by_view_group(gen_cube_features, view_map["D"])

        # print("Averaged features by view group")

        # Update FID metrics
        fid_f.real_features_sum += real_group_imgs_F.sum(dim=0)
        fid_f.real_features_cov_sum += real_group_imgs_F.t().mm(real_group_imgs_F)
        fid_f.real_features_num_samples += real_group_imgs_F.shape[0]
        fid_f.fake_features_sum += gen_group_imgs_F.sum(dim=0)
        fid_f.fake_features_cov_sum += gen_group_imgs_F.t().mm(gen_group_imgs_F)
        fid_f.fake_features_num_samples += gen_group_imgs_F.shape[0]
        fid_f.orig_dtype = real_group_imgs_F.dtype
        fid_u.real_features_sum += real_group_imgs_U.sum(dim=0)
        fid_u.real_features_cov_sum += real_group_imgs_U.t().mm(real_group_imgs_U)
        fid_u.real_features_num_samples += real_group_imgs_U.shape[0]
        fid_u.fake_features_sum += gen_group_imgs_U.sum(dim=0)
        fid_u.fake_features_cov_sum += gen_group_imgs_U.t().mm(gen_group_imgs_U)
        fid_u.fake_features_num_samples += gen_group_imgs_U.shape[0]
        fid_u.orig_dtype = real_group_imgs_U.dtype
        fid_d.real_features_sum += real_group_imgs_D.sum(dim=0)
        fid_d.real_features_cov_sum += real_group_imgs_D.t().mm(real_group_imgs_D)
        fid_d.real_features_num_samples += real_group_imgs_D.shape[0]
        fid_d.fake_features_sum += gen_group_imgs_D.sum(dim=0)
        fid_d.fake_features_cov_sum += gen_group_imgs_D.t().mm(gen_group_imgs_D)
        fid_d.fake_features_num_samples += gen_group_imgs_D.shape[0]
        fid_d.orig_dtype = real_group_imgs_D.dtype

        # print("Updated FID metrics")
    
    # Step 4: Average FIDs → OmniFID
    omnifid_score = (fid_f.compute().item() + fid_u.compute().item() + fid_d.compute().item()) / 3
    print(f"OmniFID: {omnifid_score}")
    return omnifid_score
