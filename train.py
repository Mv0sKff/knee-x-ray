from pathlib import Path

import torch
from torch import nn, optim

from args import *
from utils import plot_loss

args = get_args()


def train_model(model, train_loader, val_loader, session_name, save=False):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / f"{session_name}_best.pt")
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

        # TODO: plot, performance validation


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
