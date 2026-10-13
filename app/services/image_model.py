import numpy as np
import torch
from PIL import Image
from torch import nn


OPENFAKE_REVISION = "3247b942f18d39d1fd476b1077852af033d35d8c"
OPENFAKE_LICENSE = "CC-BY-NC-4.0"
IMAGE_CLASSES = ("real", "fake")


class ImageAuthenticityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, 2))

    def forward(self, images):
        return self.classifier(self.features(images))


def image_to_tensor(image, image_size):
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
    return (tensor - 0.5) / 0.5