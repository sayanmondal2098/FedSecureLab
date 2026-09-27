"""Current Flower ClientApp using Message, ArrayRecord, and MetricRecord."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_cifar.dataset import load_client_data
from flower_cifar.metrics import write_client_metric, write_client_status, write_live_phase
from flower_cifar.model import create_model
from flower_cifar.utils import configure_tensorflow_gpu, set_global_seed

app = ClientApp()


def _settings(context: Context) -> tuple[int, int, int, int, float]:
    node = context.node_config
    run = context.run_config
    return (int(node["partition-id"]), int(node["num-partitions"]), int(run["local-epochs"]), int(run["batch-size"]), float(run["learning-rate"]))


def _parse_poison_clients(raw: str) -> set[int]:
    clients: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        clients.add(int(token))
    return clients


def _parse_poison_rate_map(raw: str) -> dict[int, float]:
    rates: dict[int, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        client_str, rate_str = token.split(":", 1)
        try:
            client_id = int(client_str.strip())
            rate = float(rate_str.strip())
        except ValueError:
            continue
        rates[client_id] = max(0.0, min(1.0, rate))
    return rates


def _apply_poisoning(x_train: np.ndarray, y_train: np.ndarray, context: Context, client_id: int) -> tuple[np.ndarray, np.ndarray, bool, str]:
    run = context.run_config
    mode = str(run.get("poison-mode", "none")).strip().lower()
    poisoned_clients = _parse_poison_clients(str(run.get("poison-client-ids", "")))
    is_poisoned_client = client_id in poisoned_clients and mode != "none"
    if not is_poisoned_client:
        return x_train, y_train, False, "none"

    rate_default = float(run.get("poison-rate-default", 0.3))
    rate_default = max(0.0, min(1.0, rate_default))
    rate_map = _parse_poison_rate_map(str(run.get("poison-rate-map", "")))
    poison_rate = rate_map.get(client_id, rate_default)
    poison_count = int(len(y_train) * poison_rate)
    if poison_count <= 0:
        return x_train, y_train, False, "none"
    rng = np.random.default_rng(int(run.get("seed", 42)) + client_id)
    poison_indices = rng.choice(len(y_train), size=poison_count, replace=False)

    if mode == "label_flip":
        # Deterministic label flipping for selected malicious clients.
        offset = int(run.get("poison-label-flip-offset", 1)) % 10
        if offset == 0:
            offset = 1
        y_poisoned = y_train.copy()
        y_poisoned[poison_indices] = (y_poisoned[poison_indices] + offset) % 10
        return x_train, y_poisoned.astype(np.int64), True, mode

    if mode == "gaussian_noise":
        noise_std = float(run.get("poison-noise-std", 0.15))
        if noise_std <= 0 or math.isnan(noise_std):
            noise_std = 0.15
        x_poisoned = x_train.copy()
        noise = np.random.normal(loc=0.0, scale=noise_std, size=x_poisoned[poison_indices].shape).astype(np.float32)
        x_poisoned[poison_indices] = np.clip(x_poisoned[poison_indices] + noise, 0.0, 1.0)
        return x_poisoned, y_train, True, mode

    # Unknown mode falls back to no poisoning to keep training robust.
    return x_train, y_train, False, "none"


@app.train()
def train(message: Message, context: Context) -> Message:
    """Train exactly one isolated CIFAR-10 partition and return its update."""
    tf.keras.backend.clear_session()
    set_global_seed(int(context.run_config["seed"]))
    configure_tensorflow_gpu()
    client_id, num_partitions, epochs, batch_size, learning_rate = _settings(context)
    round_number = int(message.content["config"].get("server-round", 0))
    write_live_phase(round_number, "train")
    write_client_status(client_id, round_number, "training")
    x_train, y_train, x_val, y_val = load_client_data(client_id, num_partitions, int(context.run_config["seed"]))
    x_train, y_train, poisoned, poison_mode = _apply_poisoning(x_train, y_train, context, client_id)
    model = create_model(learning_rate)
    model.set_weights(message.content["arrays"].to_numpy_ndarrays())
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=int(context.run_config.get("verbose", 0)))
    metric = {
        "round": round_number,
        "client_id": client_id,
        "train_loss": float(history.history["loss"][-1]),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "num_samples": int(len(x_train)),
        "poisoned": poisoned,
        "poison_mode": poison_mode,
    }
    write_client_metric(metric)
    write_client_status(client_id, round_number, "completed")
    poison_tag = f" | poison={poison_mode}" if poisoned else ""
    print(f"Client {client_id} | samples={len(x_train)} | train loss={metric['train_loss']:.4f} | train acc={metric['train_accuracy']:.4f} | val acc={metric['val_accuracy']:.4f}{poison_tag}")
    return Message(content=RecordDict({"arrays": ArrayRecord(model.get_weights()), "metrics": MetricRecord({"num-examples": len(x_train), "train_loss": metric["train_loss"], "train_accuracy": metric["train_accuracy"]})}), reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """Evaluate received global weights against this client's held-out validation data."""
    tf.keras.backend.clear_session()
    client_id, num_partitions, _, batch_size, learning_rate = _settings(context)
    _, _, x_val, y_val = load_client_data(client_id, num_partitions, int(context.run_config["seed"]))
    model = create_model(learning_rate)
    model.set_weights(message.content["arrays"].to_numpy_ndarrays())
    loss, accuracy = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    return Message(content=RecordDict({"metrics": MetricRecord({"num-examples": len(x_val), "val_loss": float(loss), "val_accuracy": float(accuracy)})}), reply_to=message)
