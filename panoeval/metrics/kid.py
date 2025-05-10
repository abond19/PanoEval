import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms
from tqdm import tqdm
from ..utils.dataloader import GeneratedDataset, RealDataset


def preprocess_images(image_size=(299, 299), normalize=False):
    """
    Preprocess images to match the input requirements for the metric.
    Returns a tensor of shape (N, 3, H, W).
    """
    if normalize:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()  # outputs [0,1] float
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # cast to uint8 in [0,255]
        ])

    return tf


def compute_kid(real_images, 
                gen_images, 
                feature_dim=2048, 
                subsets=100, 
                subset_size=1000, 
                normalize=False, 
                dtype=torch.float64, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute Kernel Inception Distance (KID) between real and generated image directories.

    Args:
        real_images (List): List of PIL images.
        gen_images (List): List of PIL images.
        feature_dim (int): Feature layer to use. Default = 2048.
        subsets (int): Number of random subsets.
        subset_size (int): Number of images per subset.
        normalize (bool): If True, assumes inputs are float [0,1]; else uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): cuda or cpu.

    Returns:
        tuple(float, float): KID mean and std.
    """
    kid_metric = KernelInceptionDistance(
        feature=feature_dim,
        subsets=subsets,
        subset_size=subset_size,
        reset_real_features=True,
        normalize=normalize
    ).to(device)
    kid_metric.set_dtype(torch.float32)

    real_images = RealDataset(real_images, transform=preprocess_images(normalize=normalize))#.to(device)
    gen_images = GeneratedDataset(gen_images, transform=preprocess_images(normalize=normalize))#.to(device)
    real_dl = torch.utils.data.DataLoader(real_images, batch_size=32, shuffle=False, num_workers=4)
    gen_dl = torch.utils.data.DataLoader(gen_images, batch_size=32, shuffle=False, num_workers=4)
    for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing KID", total=len(real_dl)):
        # Update metric with batches
        kid_metric.update(real_batch.to(device), real=True)
        kid_metric.update(gen_batch.to(device), real=False)

    kid_mean, kid_std = kid_metric.compute()
    print(f"KID mean: {kid_mean}, KID std: {kid_std}")
    return kid_mean.item(), kid_std.item()
