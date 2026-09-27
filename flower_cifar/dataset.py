"""CIFAR-10 loading, deterministic Flower Dataset partitioning, and validation."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

NUM_CLIENTS = 5
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@lru_cache(maxsize=8)
def get_federated_dataset(num_partitions: int = NUM_CLIENTS, seed: int = 42) -> FederatedDataset:
    """Create the official Flower CIFAR-10 IID partition source.

    The factory is deliberately isolated so replacing IidPartitioner with a
    DirichletPartitioner later only changes this module.
    """
    if num_partitions != NUM_CLIENTS:
        raise ValueError(f"Baseline requires exactly {NUM_CLIENTS} partitions, got {num_partitions}")
    # IidPartitioner is deterministic after the global dataset seed is set.
    partitioner = IidPartitioner(num_partitions=num_partitions)
    cache_root = Path(tempfile.gettempdir()) / "flwr-cifar10-cache"
    # Give each process a private cache directory to avoid Windows rename/move
    # races when multiple Flower worker processes prepare the same split.
    process_cache_dir = cache_root / f"pid-{os.getpid()}"
    return FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
        shuffle=False,
        seed=seed,
        # Keep the cache outside Flower's copied app directory. This avoids
        # Windows' 260-character path limit in managed simulation runtimes.
        cache_dir=str(process_cache_dir),
    )


def _to_arrays(split: Any) -> tuple[np.ndarray, np.ndarray]:
    split.set_format("numpy")
    images = np.asarray(split["img"], dtype=np.float32) / 255.0
    labels = np.asarray(split["label"], dtype=np.int64)
    return images, labels


def load_client_data(partition_id: int, num_partitions: int = NUM_CLIENTS, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one isolated partition and split it 80/20 into train/validation."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError(f"Invalid partition id {partition_id}")
    partition = get_federated_dataset(num_partitions, seed).load_partition(partition_id, "train")
    split = partition.train_test_split(test_size=0.2, seed=seed + partition_id)
    x_train, y_train = _to_arrays(split["train"])
    x_val, y_val = _to_arrays(split["test"])
    return x_train, y_train, x_val, y_val


@lru_cache(maxsize=1)
def load_global_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the centralized held-out CIFAR-10 test split used by the server."""
    dataset = get_federated_dataset()
    # ``load_split`` is the current Flower Datasets API for unpartitioned
    # splits; only the training split is partitioned among clients.
    test = dataset.load_split("test")
    return _to_arrays(test)


def label_counts(labels: np.ndarray) -> list[int]:
    """Return counts for labels 0 through 9."""
    return np.bincount(labels.astype(np.int64), minlength=10).tolist()

