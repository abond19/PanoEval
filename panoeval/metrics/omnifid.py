import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
# import kornia.geometry as KG
import py360convert
import numpy as np
from tqdm import tqdm
from ..utils.dataloader import GeneratedDataset, RealDataset


def preprocess_images(image_size=(512, 256), device='cuda'):
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
    group_faces = cubemaps[:, group_indices, :, :, :]  # shape: (B, G, 3, H, W)
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
    device='cuda' if torch.cuda.is_available() else 'cpu'
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
    real_eqr_imgs = RealDataset(real_images, transform=preprocess_images())#.to(device)
    gen_eqr_imgs = GeneratedDataset(gen_images, transform=preprocess_images())#.to(device)
    real_dl = torch.utils.data.DataLoader(real_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)  
    gen_dl = torch.utils.data.DataLoader(gen_eqr_imgs, batch_size=32, shuffle=False, num_workers=4)

    for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing OmniFID", total=len(real_dl)):
        # print("Start of batch")
        real_cubemaps = equirectangular_to_cubemap_batch(real_batch, face_size=face_size)
        gen_cubemaps = equirectangular_to_cubemap_batch(gen_batch, face_size=face_size)
        # print("Converted to cubemaps")

        real_group_imgs_F = average_features_by_view_group(real_cubemaps, view_map["F"])
        gen_group_imgs_F = average_features_by_view_group(gen_cubemaps, view_map["F"])
        real_group_imgs_F = (real_group_imgs_F * 255.0).to(torch.uint8)
        gen_group_imgs_F = (gen_group_imgs_F * 255.0).to(torch.uint8)

        real_group_imgs_U = average_features_by_view_group(real_cubemaps, view_map["U"])
        gen_group_imgs_U = average_features_by_view_group(gen_cubemaps, view_map["U"])
        real_group_imgs_U = (real_group_imgs_U * 255.0).to(torch.uint8)
        gen_group_imgs_U = (gen_group_imgs_U * 255.0).to(torch.uint8)

        real_group_imgs_D = average_features_by_view_group(real_cubemaps, view_map["D"])
        gen_group_imgs_D = average_features_by_view_group(gen_cubemaps, view_map["D"])
        real_group_imgs_D = (real_group_imgs_D * 255.0).to(torch.uint8)
        gen_group_imgs_D = (gen_group_imgs_D * 255.0).to(torch.uint8)

        # print("Averaged features by view group")

        fid_f.update(real_group_imgs_F.to(device), real=True)
        fid_f.update(gen_group_imgs_F.to(device), real=False)
        fid_u.update(real_group_imgs_U.to(device), real=True)
        fid_u.update(gen_group_imgs_U.to(device), real=False)
        fid_d.update(real_group_imgs_D.to(device), real=True)
        fid_d.update(gen_group_imgs_D.to(device), real=False)

        # print("Updated FID metrics")
    
    # Step 4: Average FIDs → OmniFID
    omnifid_score = (fid_f.compute().item() + fid_u.compute().item() + fid_d.compute().item()) / 3
    print(f"OmniFID: {omnifid_score}")
    return omnifid_score
