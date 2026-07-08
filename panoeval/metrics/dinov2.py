import torchvision.transforms as TF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from abc import ABC, abstractmethod

from PIL import Image
from torchvision.transforms.functional import to_tensor

def pil_resize(x, output_size):
    s1, s2 = output_size
    def resize_single_channel(x):
        img = Image.fromarray(x, mode='F')
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)
    x = np.array(x.convert('RGB')).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return to_tensor(x)/255

VALID_ARCHITECTURES = [
                        'vits14',
                        'vitb14',
                        'vitl14',
                        'vitg14',
                    ]


class Encoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.setup(*args, **kwargs)
        self.name = 'encoder'

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, x):
        """Converts a PIL Image to an input for the model"""
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class DINOv2Encoder(Encoder):
    def setup(self, arch=None, clean_resize:bool=False):
        if arch is None: 
            arch = 'vitl14'

        self.arch = arch

        arch_str = f'dinov2_{self.arch}'

        if self.arch not in VALID_ARCHITECTURES:
            sys.exit(f"arch={self.arch} is not a valid architecture. Choose from {VALID_ARCHITECTURES}")

        self.model = torch.hub.load('facebookresearch/dinov2', arch_str)
        self.clean_resize = clean_resize

    def transform(self, image):

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        if self.clean_resize:
            image = pil_resize(image, (224, 224))
        else:
            image = TF.Compose([
                TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
            ])(image)

        return TF.Normalize(imagenet_mean, imagenet_std)(image)
    

    def tensor_transform(self, image):

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        if self.clean_resize:
            image = pil_resize(image, (224, 224))
        else:
            image = TF.Compose([
                TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
            ])(image)

        return TF.Normalize(imagenet_mean, imagenet_std)(image)