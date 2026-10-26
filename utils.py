from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def compute_metrics(labels, logits):
    """Return accuracy, balanced accuracy, macro ROC-AUC and macro AP (one-vs-rest)."""
    probs = torch.softmax(logits.double(), dim=1).numpy()
    labels = labels.numpy()
    predictions = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1])[labels]

    accuracy = float((predictions == labels).mean())
    balanced_acc = balanced_accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    avg_precision = average_precision_score(one_hot, probs, average="macro")
    return accuracy, balanced_acc, roc_auc, avg_precision


def plot_loss(train_loss, val_loss, save_path=None):
    if len(train_loss) != len(val_loss):
        raise ValueError(
            "Training and validation loss histories must have the same length"
        )

    figure = Figure()
    axis = figure.add_subplot()
    epochs = range(1, len(train_loss) + 1)
    axis.plot(epochs, train_loss, label="Training loss")
    axis.plot(epochs, val_loss, label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path)
    return figure
