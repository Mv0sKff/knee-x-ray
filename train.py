import torch
from torch import nn, optim

from args import *
from schemas import History
from utils import compute_metrics, plot_metric

args = get_args()


def train_model(model, train_loader, val_loader, fold, output_dir, save=False):
    train_history = History()
    val_history = History()
    best_balanced_accuracy = -1.0
    for epoch in range(args.epochs):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        running_loss = 0.0
        total = 0
        train_logits = []
        train_labels = []

        for batch in train_loader:
            inputs = batch["xray"]
            labels = batch["target"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # calc more metrics if needed

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            train_logits.append(outputs.detach())
            train_labels.append(labels)

        train_history.append(
            running_loss / total,
            *compute_metrics(torch.cat(train_labels), torch.cat(train_logits)),
        )
        val_history.append(*validate_model(model, val_loader, criterion))

        val_loss = val_history.loss[-1]
        if val_history.balanced_accuracy[-1] > best_balanced_accuracy:
            best_balanced_accuracy = val_history.balanced_accuracy[-1]
            if save:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(), output_dir / f"best_model_fold_{fold}.pth"
                )

        print(
            f"epoch: {epoch} |",
            f"train loss: {train_history.loss[-1]:.4f} |",
            f"val loss: {val_loss:.4f} |",
            f"val accuracy: {val_history.accuracy[-1]:.4f} |",
            f"val balanced accuracy: {val_history.balanced_accuracy[-1]:.4f} |",
            f"val ROC-AUC: {val_history.roc_auc[-1]:.4f} |",
            f"val AP: {val_history.average_precision[-1]:.4f}",
        )

    if save:
        plots = [
            ("loss", "Loss"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("roc_auc", "ROC-AUC"),
            ("average_precision", "Average Precision"),
        ]
        for attr, name in plots:
            plot_metric(
                getattr(train_history, attr),
                getattr(val_history, attr),
                name,
                save_path=output_dir / f"fold_{fold}_{attr}.png",
            )


def validate_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    total = 0
    all_logits = []
    all_labels = []

    for batch in val_loader:
        inputs = batch["xray"]
        labels = batch["target"]

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total += outputs.size(0)
            all_logits.append(outputs)
            all_labels.append(labels)

    accuracy, balanced_acc, roc_auc, avg_precision = compute_metrics(
        torch.cat(all_labels), torch.cat(all_logits)
    )

    return running_loss / total, accuracy, balanced_acc, roc_auc, avg_precision
