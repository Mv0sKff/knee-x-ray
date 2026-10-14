from pathlib import Path

from matplotlib.figure import Figure


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
