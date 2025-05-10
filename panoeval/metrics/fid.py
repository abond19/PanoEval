import torch
from torchmetrics.image.fid import FrechetInceptionDistance
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


def compute_fid(real_images, gen_images, feature_dim=2048, normalize=False, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute FID score between real and generated images.

    Args:
        real_images (List): List of PIL images.
        gen_images (List): List of PIL images.
        feature_dim (int): InceptionV3 feature layer to use. Default = 2048.
        normalize (bool): If True, assumes inputs are float [0,1]; else uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): cuda or cpu.

    Returns:
        float: FID score.
    """
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(
        feature=feature_dim,
        reset_real_features=True,           # in many cases the real dataset does not change, the features can be cached them to avoid recomputing them which is costly. Set this to False if your dataset does not change.
        normalize=normalize
    ).to(device)
    fid_metric.set_dtype(dtype)

    # Load image tensors
    real_imgs = RealDataset(real_images, transform=preprocess_images(normalize=normalize))#.to(device)
    gen_imgs = GeneratedDataset(gen_images, transform=preprocess_images(normalize=normalize))#.to(device)

    real_dl = torch.utils.data.DataLoader(real_imgs, batch_size=32, shuffle=False, num_workers=4)
    gen_dl = torch.utils.data.DataLoader(gen_imgs, batch_size=32, shuffle=False, num_workers=4)

    for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing FID", total=len(real_dl)):
        # Update metric with batches
        fid_metric.update(real_batch.to(device), real=True)
        fid_metric.update(gen_batch.to(device), real=False)

    # Compute FID
    fid_value = fid_metric.compute().item()
    print(f"FID: {fid_value}")
    return fid_value
