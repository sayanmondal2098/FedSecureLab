"""IID partition isolation tests."""

import numpy as np

from flower_cifar.dataset import NUM_CLIENTS, load_client_data


def test_exactly_five_partitions_and_distinct_samples() -> None:
    assert NUM_CLIENTS == 5
    first_client, *_ = load_client_data(0)
    second_client, *_ = load_client_data(1)
    assert first_client.shape[0] > 0 and second_client.shape[0] > 0
    assert not np.array_equal(first_client[0], second_client[0])
