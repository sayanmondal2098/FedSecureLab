"""Print and plot the CIFAR-10 labels allocated to all five IID clients."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flower_cifar.dataset import CIFAR10_CLASSES, NUM_CLIENTS, label_counts, load_client_data
from flower_cifar.utils import RESULTS_DIR, set_global_seed


def main() -> None:
    """Create results/client_label_distribution.png and a readable count table."""
    set_global_seed()
    counts = []
    for client_id in range(NUM_CLIENTS):
        x_train, y_train, x_val, y_val = load_client_data(client_id)
        assert x_train.shape[1:] == (32, 32, 3)
        counts.append(label_counts(np.concatenate([y_train, y_val])))
    print("Client | " + " | ".join(CIFAR10_CLASSES))
    for client_id, row in enumerate(counts):
        print(f"{client_id:>6} | " + " | ".join(f"{count:>10}" for count in row))
    figure, axis = plt.subplots(figsize=(13, 6))
    positions = np.arange(len(CIFAR10_CLASSES))
    width = 0.16
    for client_id, row in enumerate(counts):
        axis.bar(positions + (client_id - 2) * width, row, width=width, label=f"Client {client_id}")
    axis.set_xticks(positions, CIFAR10_CLASSES, rotation=30, ha="right")
    axis.set_ylabel("Examples")
    axis.set_title("CIFAR-10 IID partition label distribution")
    axis.legend()
    figure.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    figure.savefig(RESULTS_DIR / "client_label_distribution.png", dpi=160)


if __name__ == "__main__":
    main()
