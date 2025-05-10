import torch
from torchvision import transforms
from .panfusion_faed import FrechetAutoEncoderDistance
from tqdm import tqdm
from utils.dataloader import GeneratedDataset, RealDataset


def preprocess_images(image_size=(256, 512), normalize=False):
    """
    Preprocess panorama images for FAED.
    Returns a tensor of shape (N, 3, H, W).
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # float32 in [0,1]
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8) if not normalize else x)
    ])
    
    return tf


def compute_faed(
    real_images,
    gen_images,
    pano_height=256,
    image_size=(256, 512),
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute Frechet AutoEncoder Distance (FAED) between real and generated panoramic images.

    Args:
        real_images (List): List of PIL images.
        gen_images (List): List of PIL images.
        pano_height (int): Used for estimating feature vector size.
        image_size (tuple): Resize target (W, H).
        device (str): cuda or cpu

    Returns:
        float: FAED score
    """
    # Initialize metric
    metric = FrechetAutoEncoderDistance(pano_height=pano_height).to(device)

    # Preprocess images
    # real_imgs = preprocess_images(real_images, image_size=image_size).to(device)
    # gen_imgs = preprocess_images(gen_images, image_size=image_size).to(device)

    real_imgs = RealDataset(real_images, transform=preprocess_images())#.to(device)
    gen_imgs = GeneratedDataset(gen_images, transform=preprocess_images())#.to(device)

    real_dl = torch.utils.data.DataLoader(real_imgs, batch_size=32, shuffle=False, num_workers=4)
    gen_dl = torch.utils.data.DataLoader(gen_imgs, batch_size=32, shuffle=False, num_workers=4)

    for real_batch, gen_batch in tqdm(zip(real_dl, gen_dl), desc="Computing FAED", total=len(real_dl)):
        # Update metric with batches
        metric.update(real_batch.to(device), real=True)
        metric.update(gen_batch.to(device), real=False)

    # Compute FAED
    faed_score = metric.compute().item()
    print(f"FAED score: {faed_score}")
    return faed_score
