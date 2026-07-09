from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
DATASET_SUBFOLDERS = ('flickr360', 'polyhaven', 'matterport')


def _is_image(name):
    return name.lower().endswith(IMAGE_EXTS)


def _pick_folder_image(folder_path, folder_name):
    """Return the representative image inside a per-image folder.

    Setup (2) stores one image per folder, named the same as the folder. We
    prefer that file explicitly (ignoring any stray extras such as tangent
    grids or intermediate PNGs); otherwise fall back to the only/first image.
    """
    try:
        files = sorted(f for f in os.listdir(folder_path) if _is_image(f))
    except (NotADirectoryError, FileNotFoundError):
        return None
    if not files:
        return None
    for f in files:
        if os.path.splitext(f)[0] == folder_name:
            return os.path.join(folder_path, f)
    return os.path.join(folder_path, files[0])


def find_image_files(directory, use_matterport=True):
    """Collect evaluation images, handling both supported directory layouts.

    Setup (1): a flat folder of image files (named after the GT/caption).
    Setup (2): dataset subfolders (flickr360 / polyhaven / matterport), each
               containing one folder per image, with the image stored as a PNG
               named the same as its folder.

    ``use_matterport`` excludes the matterport subfolder when False. Returns a
    list of absolute paths sorted by a stable identity key so that the real and
    generated sets line up by scene name across directories.
    """
    try:
        entries = sorted(os.listdir(directory))
    except FileNotFoundError:
        return []

    subdirs = [d for d in entries if os.path.isdir(os.path.join(directory, d))]
    dataset_dirs = [d for d in subdirs if d.lower() in DATASET_SUBFOLDERS]

    results = []  # (identity_key, path)

    if dataset_dirs:
        # Setup (2): dataset subfolders -> per-image folders -> folder-named image.
        for ds in dataset_dirs:
            if ds.lower() == 'matterport' and not use_matterport:
                continue
            ds_path = os.path.join(directory, ds)
            for name in sorted(os.listdir(ds_path)):
                path = os.path.join(ds_path, name)
                if os.path.isdir(path):
                    chosen = _pick_folder_image(path, name)
                    if chosen is not None:
                        results.append((f"{ds}/{name}", chosen))
                elif _is_image(name):
                    # tolerate an image placed directly under the dataset folder
                    results.append((f"{ds}/{os.path.splitext(name)[0]}", path))
    else:
        # Setup (1): flat folder of image files (tolerate per-image subfolders too).
        for name in entries:
            path = os.path.join(directory, name)
            if os.path.isfile(path) and _is_image(name):
                results.append((os.path.splitext(name)[0], path))
            elif os.path.isdir(path):
                chosen = _pick_folder_image(path, name)
                if chosen is not None:
                    results.append((name, chosen))

    results.sort(key=lambda item: item[0])
    return [path for _, path in results]


def find_png_files(directory, use_matterport=True):
    """Backwards-compatible alias for :func:`find_image_files`."""
    return find_image_files(directory, use_matterport)

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
    def __init__(self, folder, transform, use_matterport=False):
        self.images = find_image_files(folder, use_matterport)
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
    def __init__(self, folder, transform, take_captions=False, text_prompts_folder=None, use_matterport=False):
        self.images = find_image_files(folder, use_matterport)
        self.folder = folder
        self.transform = transform

        self.take_captions = take_captions
        if self.take_captions:
            assert text_prompts_folder is not None, "text_prompts_folder must be provided if take_captions is True"
            self.text_prompts_folder = text_prompts_folder
            self.captions = self.load_text_prompts(text_prompts_folder)

    def load_text_prompts(self, folder):
        prompts = []
        idx_to_remove = []
        for i in range(len(self.images)):
            img = self.images[i]
            stem = os.path.splitext(os.path.basename(img))[0]
            parent = os.path.basename(os.path.dirname(img))
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(img)))
            # Try both layouts: setup (2) keeps captions under
            # <folder>/<dataset>/<image>/caption.txt; setup (1) uses a flat
            # <folder>/<name>.txt (or <name>/caption.txt).
            candidates = [
                os.path.join(folder, grandparent, parent, "caption.txt"),
                os.path.join(folder, parent, "caption.txt"),
                os.path.join(folder, stem, "caption.txt"),
                os.path.join(folder, stem + ".txt"),
            ]
            prompt_path = next((c for c in candidates if os.path.exists(c)), None)
            if prompt_path is None:
                idx_to_remove.append(i)
                continue
            prompts.append(prompt_path)

        # Remove images without captions
        for i in sorted(idx_to_remove, reverse=True):
            del self.images[i]

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