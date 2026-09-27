"""Dataset shape and label tests; marked slow because CIFAR-10 may download."""

from flower_cifar.dataset import load_client_data


def test_client_data_shape_and_labels() -> None:
    x_train, y_train, x_val, y_val = load_client_data(0)
    assert x_train.shape[1:] == (32, 32, 3)
    assert x_val.shape[1:] == (32, 32, 3)
    assert x_train.min() >= 0.0 and x_train.max() <= 1.0
    assert y_train.min() >= 0 and y_train.max() <= 9
    assert y_val.min() >= 0 and y_val.max() <= 9
