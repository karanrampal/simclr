"""Unit tests for metrics"""

import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, "./src/")

from model.net import avg_acc_gpu, avg_f1_score_gpu
from model.net import confusion_matrix as conf_mat

HIGH_VAL = 10


def test_accuracy() -> None:
    """Test implementation of accuracy"""
    num_classes = np.random.randint(1, HIGH_VAL)
    num_examples = np.random.randint(1, HIGH_VAL)

    output = torch.randn(num_examples, num_classes)
    preds = output.argmax(dim=1)
    labels = torch.randint(0, num_classes, (num_examples,))

    sk_acc = accuracy_score(labels.numpy(), preds.numpy())
    my_acc = avg_acc_gpu(output, labels)

    assert np.isclose(sk_acc, my_acc)


def test_f1() -> None:
    """Test f1 score calculation"""
    num_classes = np.random.randint(1, HIGH_VAL)
    num_examples = np.random.randint(1, HIGH_VAL)

    output = torch.randn(num_examples, num_classes)
    preds = output.argmax(dim=1)
    labels = torch.randint(0, num_classes, (num_examples,))

    sk_f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
    my_f1 = avg_f1_score_gpu(output, labels, num_classes)

    assert np.isclose(sk_f1, my_f1)


def test_conf_mat() -> None:
    """Test f1 score calculation"""
    num_classes = np.random.randint(1, HIGH_VAL)
    num_examples = np.random.randint(1, HIGH_VAL)

    output = torch.randn(num_examples, num_classes)
    preds = output.argmax(dim=1)
    labels = torch.randint(0, num_classes, (num_examples,))

    sk_cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=range(num_classes))
    my_cm = conf_mat(output, labels, num_classes)

    assert (sk_cm == my_cm[0, 0, :, :].numpy()).all()
