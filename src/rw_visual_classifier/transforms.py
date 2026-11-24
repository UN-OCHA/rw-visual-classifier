"""Shared image letterbox + normalize transforms."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from rw_visual_classifier.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def letterbox(
    image: Image.Image,
    size: int = IMAGE_SIZE,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square canvas."""
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


class Letterbox:
    """Picklable letterbox transform for DataLoader workers (macOS spawn)."""

    def __init__(
        self,
        size: int = IMAGE_SIZE,
        fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return letterbox(image, size=self.size, fill=self.fill)


def build_train_transform(size: int = IMAGE_SIZE) -> Callable[[Image.Image], torch.Tensor]:
    return T.Compose(
        [
            Letterbox(size=size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(size: int = IMAGE_SIZE) -> Callable[[Image.Image], torch.Tensor]:
    return T.Compose(
        [
            Letterbox(size=size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_for_onnx(
    image: Image.Image,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Return NCHW float32 batch array matching training eval preprocess."""
    tensor = build_eval_transform(size=size)(image)
    return tensor.unsqueeze(0).numpy().astype(np.float32)
