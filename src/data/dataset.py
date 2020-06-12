from typing import Optional
from pathlib import Path

import torch
from albumentations import Compose
from torch.utils.data import Dataset

from src.data.utils import load_class2id_mapping, load_valid_metadata, load_train_metadata, load_image


def collate_fn(data):
    images, classlabels = list(zip(*data))
    images = torch.FloatTensor(images).permute(0, 3, 1, 2)
    classlabels = torch.LongTensor(classlabels)
    return images, classlabels


class TinyImageNetDataset(Dataset):
    def __init__(
            self,
            path: str,
            augmentations: Optional[Compose] = None,
            is_valid: bool = False
    ):
        super().__init__()
        self.dataset_path = Path(path)
        self.augmentations = augmentations
        self.class2id = load_class2id_mapping(self.dataset_path)
        if is_valid:
            self.meta = load_valid_metadata(self.dataset_path)
        else:
            self.meta = load_train_metadata(self.dataset_path)
        self.meta['class_id'] = self.meta['class_id'].map(self.class2id)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        class_id = self.meta.loc[idx, 'class_id']

        image_path = self.dataset_path / self.meta.loc[idx, 'path']
        image = load_image(image_path)
        if self.augmentations:
            image = self.augmentations(image=image)['image']
        return image, class_id
