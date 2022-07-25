"""Define custom dataset class extending the Pytorch Dataset class"""

from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import STL10

from utils.utils import Params


class ContrastiveTransforms:
    """Contrastive transform class to create a list of batches
    Args:
        transforms: Data augmentations
        num : Number of branches of data input to the model
    Returns:
        List of tensors
    """

    def __init__(self, transforms: tvt.Compose, num: int = 2) -> None:
        self.transforms = transforms
        self.num = num

    def __call__(self, x_inp: torch.Tensor) -> List[torch.Tensor]:
        """Apply the transforms and return a list of 'num' batches"""
        return [self.transforms(x_inp) for i in range(self.num)]

    def __repr__(self) -> str:
        return f"Number of input branches: {self.num}"


def get_transform(params: Params) -> tvt.Compose:
    """Data augmentation
    Args:
        params: Hyperparameters
    Returns:
        Composition of all the data transforms
    """
    trans = [
        tvt.RandomHorizontalFlip(),
        tvt.RandomResizedCrop(size=params.size),
        tvt.RandomApply(
            [
                tvt.ColorJitter(
                    brightness=params.brightness,
                    contrast=params.contrast,
                    saturation=params.saturation,
                    hue=params.hue,
                )
            ],
            p=params.jitter,
        ),
        tvt.RandomGrayscale(p=params.gray_scale),
        tvt.GaussianBlur(kernel_size=params.blur),
        tvt.ToTensor(),
        tvt.Normalize((0.5,), (0.5,)),
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
    transforms = get_transform(params)

    for mode in modes:
        if mode == "train":
            ds_dict = {
                "split": "unlabeled",
                "download": True,
                "transforms": ContrastiveTransforms(transforms),
            }
            shuffle = not params.distributed
        else:
            ds_dict = {
                "file_path": "train",
                "download": True,
                "transforms": ContrastiveTransforms(transforms),
            }
            shuffle = False

        dataset = STL10(root=params.data_dir, **ds_dict)
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
            drop_last=mode == "train",
        )

        dataloaders[mode] = (d_l, sampler)

    return dataloaders
