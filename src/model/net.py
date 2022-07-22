"""Define the Network, loss and metrics"""

from functools import partial
from typing import Callable, Dict

import torch
import torch.nn as tnn
from torchvision import models

from utils.utils import Params


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network"""

    def __init__(self, params: Params) -> None:
        """Initialize the different layers in the neural network
        Args:
            params: Hyperparameters
        """
        super().__init__()

        self.base = models.resnet18(pretrained=False, num_classes=4*params.hidden_dim)
        self.fc = tnn.Sequential(
            tnn.ReLU(inplace=True),
            tnn.Linear(4*params.hidden_dim, params.hidden_dim)
        )
        self.dropout_rate = params.dropout

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Defines the forward propagation through the network
        Args:
            x_inp: Batch of images
        Returns:
            Embeddings
        """
        return self.fc(self.base(x_inp))


def loss_fn(outputs: torch.Tensor, params: Params) -> torch.Tensor:
    """Compute the loss given outputs
    Args:
        outputs: Logits of network forward pass
        params: Hyperparameters
    Returns:
        loss for all the inputs in the batch
    """
    cos_sim = F.cosine_similarity(outputs[:,None,:], outputs[None,:,:], dim=-1)
    mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    cos_sim = cos_sim / params.temperature
    loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    return loss.mean()


def avg_acc_gpu(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
    Returns:
        average accuracy in [0,1]
    """
    preds = outputs.argmax(dim=1).to(torch.int64)
    avg_acc = (preds == labels).to(torch.float32).mean()
    return avg_acc


def avg_f1_score_gpu(
    outputs: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-7
) -> torch.Tensor:
    """Compute the F1 score, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        num_classes: Number of classes
        eps: Epsilon
    Returns:
        average f1 score
    """
    preds = (outputs).argmax(dim=1).to(torch.int64)
    pred_ohe = tnn.functional.one_hot(preds, num_classes)
    label_ohe = tnn.functional.one_hot(labels, num_classes)

    true_pos = (label_ohe * pred_ohe).sum(0)
    false_pos = ((1 - label_ohe) * pred_ohe).sum(0)
    false_neg = (label_ohe * (1 - pred_ohe)).sum(0)

    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    avg_f1 = 2 * (precision * recall) / (precision + recall + eps)
    wts = label_ohe.sum(0)
    wtd_macro_f1 = (avg_f1 * wts).sum() / wts.sum()

    return wtd_macro_f1


def confusion_matrix(
    outputs: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Create confusion matrix
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        num_classes: Number of classes
    Returns:
        Confusion matrix as a tensor
    """
    num = labels.shape[0]
    conf_mat = torch.zeros((1, num, num_classes, num_classes), dtype=torch.int64)
    preds = (outputs).argmax(dim=1).to(torch.int64)
    conf_mat[0, range(num), labels, preds] = 1
    return conf_mat.sum(1, keepdim=True)


def get_metrics(params: Params) -> Dict[str, Callable]:
    """Returns a dictionary of all the metrics to be used
    Args:
        params: Hyperparameters
    """
    metrics: Dict[str, Callable] = {
        "accuracy": avg_acc_gpu,
        "f1-score": partial(avg_f1_score_gpu, num_classes=params.num_classes),
    }
    return metrics
