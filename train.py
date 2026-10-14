from pathlib import Path

import torch
from args import *
from torch import nn, optim
from utils import plot_loss

args = get_args()


def train_model(model, train_loader, val_loader, session_name, save=False):
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        running_loss = 0.0
        total = 0

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

        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        train_losses.append(running_loss / total)
        val_losses.append(val_loss)
        print(
            f"epoch: {epoch} |",
            f"training loss: {running_loss / total:.4f} |",
            f"validation loss: {val_loss:.4f} |",
            f"validation accuracy: {val_accuracy:.4f}",
        )

        if save:
            plot_loss(
                train_losses,
                val_losses,
                save_path=Path(args.output_dir) / f"{session_name}_train_val_loss.png",
            )

        # TODO: save model if val loss is better, model in sessions/
        # TODO: plot, performance validation
        # TODO: test on cuda gpu


def validate_model(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        inputs = batch["xray"]
        labels = batch["target"]

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += outputs.size(0)

    running_loss /= total
    val_accuracy = correct / total

    return running_loss, val_accuracy
