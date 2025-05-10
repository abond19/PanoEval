import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from ..utils.dataloader import GeneratedDataset


def preprocess_images(image_size=(224, 224), normalize=True):
    """
    Preprocess images to match CLIP input requirements.
    Returns list of (C, H, W) tensors.
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # float32 in [0,1]
        # transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),  # CLIP normalization
    ])

    # processed_images = []
    # for img in tqdm(images, desc="Preprocessing (CLIP score)"):
    #     processed_images.append(tf(img))
    # return torch.stack(processed_images)
    return tf


def compute_clip_score(
    gen_images,
    text_prompts,
    model_name="openai/clip-vit-large-patch14",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute CLIP Score between generated images and their corresponding captions.

    Args:
        gen_images (List): List of PIL images.
        text_prompts (List): List of text prompts.
        model_name (str): CLIP model variant.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average CLIP Score [0, 100].
    """
    # Load images and captions
    # gen_imgs = preprocess_images(gen_images).to(device)
    gen_imgs = GeneratedDataset(gen_images, transform=preprocess_images(), take_captions=True, text_prompts_folder=text_prompts)

    gen_dl = torch.utils.data.DataLoader(gen_imgs, batch_size=32, shuffle=False, num_workers=4)

    # Initialize CLIPScore metric
    metric = CLIPScore(model_name_or_path=model_name).to(device)

    # metric.set_dtype(torch.float32)

    for gen_batch in tqdm(gen_dl, desc="Computing CLIP Score", total=len(gen_dl)):
        gen_batch[0] = (gen_batch[0].to(device) * 255.0).to(torch.uint8)  # Convert to uint8 [0,255]
        # print(f"gen_batch min: {gen_batch[0].min()}, max: {gen_batch[0].max()}")
        # Update metric with batches
        # print(list(gen_batch[1]))
        metric.update(gen_batch[0], list(gen_batch[1]))

    score = metric.compute()
    print(f"CLIP Score: {score.item()}")
    return score.item()
