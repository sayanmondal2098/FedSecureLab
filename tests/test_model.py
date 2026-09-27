"""Fast model contract tests (no dataset download)."""

import numpy as np

from flower_cifar.model import create_model


def test_model_output_shape_and_weight_roundtrip() -> None:
    model = create_model()
    output = model(np.zeros((3, 32, 32, 3), dtype=np.float32), training=False)
    assert output.shape == (3, 10)
    weights = model.get_weights()
    model.set_weights(weights)
    assert len(model.get_weights()) == len(weights)
