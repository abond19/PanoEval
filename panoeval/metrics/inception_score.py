import torch
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from tqdm import tqdm
from ..utils.dataloader import GeneratedDataset


def preprocess_images(image_size=(299, 299), normalize=False):
    """
    Preprocess images to match the input requirements for the metric.
    Returns a tensor of shape (N, 3, H, W).
    """
    if normalize:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()  # float [0,1]
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # uint8 [0,255]
        ])

    return tf


def compute_inception_score(
    gen_images,
    feature='logits_unbiased',
    splits=10,
    normalize=False,
    dtype=torch.float64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute Inception Score for generated images.

    Args:
        gen_images (List): List of PIL images.
        feature (str): Feature layer of InceptionV3. Default: 'logits_unbiased'.
        splits (int): Number of splits to estimate std.
        normalize (bool): True if images are [0,1] float, False if uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): Device to run the metric on.

    Returns:
        (float, float): Tuple of (IS mean, IS std)
    """
    inception = InceptionScore(
        feature=feature,
        splits=splits,
        normalize=normalize
    ).to(device)
    inception.set_dtype(torch.float32)

    # Preprocess images
    gen_imgs = GeneratedDataset(gen_images, transform=preprocess_images(normalize=normalize))
    gen_dl = torch.utils.data.DataLoader(gen_imgs, batch_size=32, shuffle=False, num_workers=4)
    for gen_batch in tqdm(gen_dl, desc="Computing Inception Score", total=len(gen_dl)):
        # Move batch to device
        gen_batch = gen_batch.to(device).to(torch.uint8)
        # Update metric with batches
        inception.update(gen_batch)

    is_mean, is_std = inception.compute()
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    return is_mean.item(), is_std.item()
