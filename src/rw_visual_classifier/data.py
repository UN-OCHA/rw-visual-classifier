"""PyTorch dataset over the downloaded preview manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from rw_visual_classifier.config import CLASS_TO_IDX, IMAGES_DIR, MANIFEST_PATH
from rw_visual_classifier.dataset_build import filter_ready, load_manifest


class PreviewDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable | None = None,
        images_dir: Path = IMAGES_DIR,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = self.images_dir / row["rel_path"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = CLASS_TO_IDX[str(row["label"])]
        return image, label


def load_split_frames(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, pd.DataFrame]:
    df = filter_ready(load_manifest(manifest_path))
    if df.empty:
        raise RuntimeError(
            f"No downloaded images found for manifest {manifest_path}"
        )
    return {
        split: df[df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
