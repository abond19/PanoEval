from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

def find_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png') or file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                png_files.append(os.path.join(root, file))
    return png_files

def load_images(folder):
    """
    Load all images from a folder.
    Returns: list of PIL images
    """
    images = []
    for fname in tqdm(sorted(os.listdir(folder)), desc=f"Loading images from {folder}"):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images


def load_text_prompts(folder):
    """
    Load all text prompts from a folder.
    Returns: list of text prompts
    """
    prompts = []
    for fname in tqdm(sorted(os.listdir(folder)), desc=f"Loading text prompts from {folder}"):
        if fname.lower().endswith((".txt")):
            prompt_path = os.path.join(folder, fname)
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts.append(f.read().strip())


class RealDataset(Dataset):
    def __init__(self, folder, transform):
        self.images = find_png_files(folder)
        self.folder = folder
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

class GeneratedDataset(Dataset):
    def __init__(self, folder, transform, take_captions=False, text_prompts_folder=None):
        self.images = find_png_files(folder)
        self.folder = folder
        self.transform = transform

        self.take_captions = take_captions
        if self.take_captions:
            assert text_prompts_folder is not None, "text_prompts_folder must be provided if take_captions is True"
            self.text_prompts_folder = text_prompts_folder
            self.captions = self.load_text_prompts(text_prompts_folder)

    def load_text_prompts(self, folder):
        prompts = []
        for img in self.images:
            img_name = img.split("/")[-2]
            dataset_name = img.split("/")[-3]
            prompt_path = os.path.join(folder, dataset_name, img_name.split(".")[0], "caption.txt")
            assert os.path.exists(prompt_path), f"Prompt file {prompt_path} does not exist"
            prompts.append(prompt_path)

        return prompts
                

    def __len__(self):
        if self.take_captions:
            assert len(self.images) == len(self.captions), "Number of images and captions must match"
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.take_captions:
            caption_path = self.captions[idx]
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            return img, caption
        else:
            return img