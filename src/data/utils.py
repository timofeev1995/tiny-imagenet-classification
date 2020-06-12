from pathlib import Path
from typing import Dict, Union, Any, Optional

import pandas as pd
import numpy as np

from PIL import Image
import albumentations
from albumentations import Compose

from src.utils import get_object


_MEAN = np.array([0.4802, 0.4481, 0.3975])
_STD = np.array([0.2302, 0.2265, 0.2262])


def load_valid_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(
        path / 'val' / 'val_annotations.txt',
        sep='\t',
        header=None,
        names=['path', 'class', 'c0', 'c1', 'c2', 'c3']
    )
    meta = meta[['path', 'class_id']]
    meta['path'] = 'val/images/' + meta['path']
    return meta


def load_train_metadata(path: Path) -> pd.DataFrame:
    train_images_dir = path / 'train'
    meta = []
    for folder in train_images_dir.iterdir():
        meta.extend([('train/' + folder.name + '/images/' + filename.name, folder.name) for filename in (folder / 'images').iterdir()])
    meta = pd.DataFrame(meta, columns=['path', 'class_id'])
    return meta


def load_class2id_mapping(path: Path) -> Dict:
    with open(path / 'wnids.txt', 'r') as file:
        classlist = file.read().strip('\n').split('\n')
    class2id = {c: i for i, c in enumerate(classlist)}
    return class2id


def load_image(path: Union[Path, str]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = np.array(image)
    image = (image / 255) - _MEAN / _STD
    return image


def compose_augmentations(augmentations: Dict[str, Dict[str, Any]], p: int) -> Optional[Compose]:
    if not augmentations:
        return None
    else:
        augmentations_list = []
        for aug_name, params in augmentations.items():
            aug = get_object(f'albumentations.{aug_name}', **params)
            augmentations_list.append(aug)
        return Compose(augmentations_list, p=p)
