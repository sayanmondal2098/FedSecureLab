"""Generate non-interactive plots from experiment CSV files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from flower_cifar.utils import RESULTS_DIR


def _save_global_plot(column: str, filename: str, title: str, ylabel: str) -> None:
    frame = pd.read_csv(RESULTS_DIR / "global_metrics.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(frame["round"], frame[column], marker="o")
    axis.set(xlabel="Federated round", ylabel=ylabel, title=title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename, dpi=160)
    plt.close(figure)


def main() -> None:
    """Produce global accuracy/loss and per-client validation accuracy figures."""
    _save_global_plot("global_accuracy", "global_accuracy.png", "Global CIFAR-10 test accuracy", "Accuracy")
    _save_global_plot("global_loss", "global_loss.png", "Global CIFAR-10 test loss", "Loss")
    frame = pd.read_csv(RESULTS_DIR / "client_metrics.csv")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for client_id, group in frame.groupby("client_id"):
        axis.plot(group["round"], group["val_accuracy"], marker="o", label=f"Client {client_id}")
    axis.set(xlabel="Federated round", ylabel="Validation accuracy", title="Client validation accuracy")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "client_accuracy.png", dpi=160)


if __name__ == "__main__":
    main()
