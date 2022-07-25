#!/usr/bin/env python3
"""Evaluates the model"""

import argparse
import logging
import os
from typing import Any, Callable, Dict, Union

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.data_loader import get_dataloader
from model.net import Net, get_metrics, loss_fn
from utils import utils


def args_parser() -> argparse.Namespace:
    """Parse commadn line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/gcs/hm-images-bucket",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--model_dir",
        default="/gcs/attribute-models-bucket/fit-model",
        help="Directory containing model",
    )
    parser.add_argument(
        "--tb_log_dir",
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"),
        type=str,
        help="TensorBoard summarywriter directory",
    )
    parser.add_argument(
        "--restore_file",
        default="last",
        choices=["last", "best"],
        help="name of the file in --model_dir containing weights to load",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=os.environ.get("WORLD_SIZE", 1),
        help="The total number of nodes in the cluster",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=os.environ.get("RANK", 0),
        help="Identifier for each node",
    )
    parser.add_argument("--height", default=224, type=int, help="Image height")
    parser.add_argument("-w", "--width", default=224, type=int, help="Image width")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument(
        "--num_workers", default=2, type=int, help="Number of workers to load data"
    )
    parser.add_argument(
        "--pin_memory",
        default=True,
        type=bool,
        help="Pin memory for faster load on GPU",
    )
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    metrics: Dict[str, Any],
    params: utils.Params,
    writer: SummaryWriter,
    epoch: int,
) -> Dict[str, Any]:
    """Evaluate the model on `num_steps` batches.
    Args:
        model: Neural network
        criterion: A function that computes the loss for the batch
        dataloader: Test dataloader
        metrics: A dictionary of functions that compute a metric
        params: Hyperparameters
        writer : Summary writer for tensorboard
        epoch: Value of Epoch
    """
    model.eval()
    summ = []

    with torch.no_grad():
        for i, (inp_data, labels) in enumerate(dataloader):
            if params.cuda:
                inp_data = inp_data.to(params.device)
                labels = labels.to(params.device)

            output = model(inp_data)
            loss = criterion(output, labels)

            summary_batch = {
                metric: metrics[metric](output, params) for metric in metrics
            }
            summary_batch["loss"] = loss.detach()
            if params.distributed:
                summary_batch = utils.reduce_dict(summary_batch)
            summ.append(summary_batch)

            tmp = {k: v.item() for k, v in summary_batch.items()}
            writer.add_scalars("test", tmp, epoch * len(dataloader) + i)

    metrics_mean = {
        metric: np.mean([x[metric].item() for x in summ]) for metric in summ[0]
    }
    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Eval metrics : %s", metrics_string)
    return metrics_mean


def main() -> None:
    """Main function"""
    args = args_parser()
    params = utils.Params(vars(args))

    writer = SummaryWriter(params.tb_log_dir.replace("gs://", "/gcs/"))

    params.cuda = torch.cuda.is_available()
    utils.setup_distributed(params)

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        params.device = f"cuda:{params.local_rank}"
    else:
        params.device = "cpu"
    print(f"Configs: {params}")

    utils.set_logger()

    logging.info("Loading the dataset...")

    dataloaders = get_dataloader(["val"], params)
    test_dl, _ = dataloaders["val"]

    logging.info("- done.")

    model: Union[DistributedDataParallel, torch.nn.Module] = Net(params)
    writer.add_graph(model, next(iter(test_dl))[0])
    if params.cuda:
        model = model.to(params.device)
    if params.distributed:
        model = DistributedDataParallel(model, device_ids=[params.local_rank])

    criterion = loss_fn
    metrics = get_metrics()

    logging.info("Starting evaluation")

    if params.rank == 0:
        utils.load_checkpoint(
            os.path.join(args.model_dir, args.restore_file + ".pth.tar"), model
        )

    evaluate(model, criterion, test_dl, metrics, params, writer, 0)

    writer.close()


if __name__ == "__main__":
    main()
