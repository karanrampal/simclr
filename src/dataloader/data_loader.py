"""Define custom dataset class extending the Pytorch Dataset class"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from utils.utils import Params


class AttributePreddataset(Dataset):
    """Custom class for Attribute prediction dataset
    Args:
        root: Directory containing the dataset
        file_path: Path of the train/val/test file relative to the root
        transforms: Data augmentation to be done
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.data = pd.read_csv(os.path.join(root, file_path))
        self.transforms = transforms

        tmp = self.data[["labels", "product_fit"]].drop_duplicates()
        self.categories = dict(zip(tmp.labels, tmp.product_fit))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset given the index idx"""
        row = self.data.iloc[idx]

        im_name = row["path"]
        im_path = os.path.join(self.root, "images", im_name)
        img = Image.open(im_path).convert("RGB")

        labels = torch.as_tensor(row["labels"], dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.data)


def get_transform(is_train: bool, params: Params) -> tvt.Compose:
    """Data augmentation
    Args:
        is_train: If the dataset is training
        params: Hyperparameters
    Returns:
        Composition of all the data transforms
    """
    trans = [
        tvt.Resize((params.height, params.width)),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if is_train:
        trans += [
            tvt.RandomHorizontalFlip(params.flip),
            tvt.ColorJitter(
                brightness=params.brightness,
                contrast=params.contrast,
                saturation=params.saturation,
                hue=params.hue,
            ),
            tvt.RandomRotation(params.degree),
        ]
    return tvt.Compose(trans)


def get_dataloader(
    modes: List[str],
    params: Params,
) -> Dict[str, Tuple[DataLoader, Optional[DistributedSampler]]]:
    """Get DataLoader objects.
    Args:
        modes: Mode of operation i.e. 'train', 'val', 'test'
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    dataloaders = {}

    for mode in modes:
        if mode == "train":
            ds_dict = {
                "file_path": "annotations/train.csv",
                "transforms": get_transform(True, params),
            }
            shuffle = not params.distributed
        else:
            ds_dict = {
                "file_path": "annotations/test.csv",
                "transforms": get_transform(False, params),
            }
            shuffle = False

        dataset = AttributePreddataset(root=params.data_dir, **ds_dict)
        if params.distributed:
            sampler: Optional[DistributedSampler] = DistributedSampler(
                dataset,
                num_replicas=params.world_size,
                rank=params.rank,
                shuffle=True,
                seed=42,
            )
        else:
            sampler = None

        d_l = DataLoader(
            dataset,
            batch_size=params.batch_size,
            sampler=sampler,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            shuffle=shuffle,
        )

        dataloaders[mode] = (d_l, sampler)

    return dataloaders
