"""Utility functions for distributed computing"""

import errno
import logging
import os
import shutil
from collections import deque
from typing import Any, Deque, Dict, Optional, Union

import torch
import torch.distributed as dist
import yaml
from torch.optim import Optimizer


class Params:
    """Class to load hyperparameters from a yaml file."""

    def __init__(self, inp: Union[Dict, str]) -> None:
        self.update(inp)

    def save(self, yaml_path: str) -> None:
        """Save parameters to yaml file at yaml_path"""
        with open(yaml_path, "w", encoding="utf-8") as fptr:
            yaml.safe_dump(self.__dict__, fptr)

    def update(self, inp: Union[Dict, str]) -> None:
        """Loads parameters from yaml file or dict"""
        if isinstance(inp, dict):
            self.__dict__.update(inp)
        elif isinstance(inp, str):
            with open(inp, encoding="utf-8") as fptr:
                params = yaml.safe_load(fptr)
                self.__dict__.update(params)
        else:
            raise TypeError(
                "Input should either be a dictionary or a string path to a config file!"
            )

    def __str__(self) -> str:
        """Print instance"""
        return str(self.__dict__)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None) -> None:
        if fmt is None:
            fmt = "{median:.3f} ({global_avg:.3f})"
        self.deque: Deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: Any, num: int = 1) -> None:
        """Update the counts"""
        self.deque.append(value)
        self.count += num
        self.total += value * num

    @property
    def median(self) -> Union[int, float]:
        """Calculate median"""
        val = torch.tensor(list(self.deque))
        return val.median().item()

    @property
    def avg(self) -> Union[int, float]:
        """Calculate average"""
        val = torch.tensor(list(self.deque), dtype=torch.float32)
        return val.mean().item()

    @property
    def global_avg(self) -> Union[int, float]:
        """Calculate global average"""
        return self.total / self.count

    @property
    def max(self) -> Union[int, float]:
        """Calculate max"""
        return max(self.deque)

    @property
    def value(self) -> Union[int, float]:
        """Get value"""
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max_=self.max,
            value=self.value,
        )


def set_logger(log_path: Optional[str] = None) -> None:
    """Set the logger to log info in terminal and file at log_path.
    Args:
        log_path: Location of log file, optional
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if log_path:
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
            )
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(stream_handler)


def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint: str) -> None:
    """Saves model at checkpoint
    Args:
        state: Contains model's state_dict, epoch, optimizer state_dict etc.
        is_best: True if it is the best model seen till now
        checkpoint: Folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    os.makedirs(checkpoint, exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(
    checkpoint: str, model: torch.nn.Module, optimizer: Optimizer = None
) -> Dict[str, Any]:
    """Loads model state_dict from checkpoint.
    Args:
        checkpoint: Filename which needs to be loaded
        model: Model for which the parameters are loaded
        optimizer: Resume optimizer from checkpoint, optional
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint)
    state = torch.load(checkpoint)
    model.load_state_dict(state["state_dict"])

    if optimizer:
        optimizer.load_state_dict(state["optim_dict"])

    return state


def save_dict_to_yaml(data: Dict[str, float], yml_path: str) -> None:
    """Saves a dict of floats to yaml file
    Args:
        data: float-castable values (np.float, int, float, etc.)
        yml_path: path to yaml file
    """
    with open(yml_path, "w", encoding="utf-8") as fptr:
        data = {k: float(v) for k, v in data.items()}
        yaml.safe_dump(data, fptr)


def setup_distributed(params: Params) -> None:
    """Setup distributed compute
    Args:
        params: Hyperparameters
    """
    params.distributed = False
    device_count = torch.cuda.device_count()
    if params.cuda:
        params.local_rank = params.rank % device_count

    if params.world_size > 1:
        params.distributed = True

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=params.world_size,
            rank=params.rank,
        )
        dist.barrier()


def reduce_dict(
    input_dict: Dict[str, torch.Tensor], average: bool = True
) -> Dict[str, torch.Tensor]:
    """Reduce dictionary across all processes
    Args:
        input_dict: all the values will be reduced
        average: whether to do average or sum
    Returns:
        Reduce dictionary
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        vals = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            vals.append(input_dict[k])
        values = torch.stack(vals, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = dict(zip(names, values))
    return reduced_dict
