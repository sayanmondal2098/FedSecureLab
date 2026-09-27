"""Current Flower ClientApp using Message, ArrayRecord, and MetricRecord."""

from __future__ import annotations

import tensorflow as tf
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_cifar.dataset import load_client_data
from flower_cifar.metrics import write_client_metric, write_client_status
from flower_cifar.model import create_model
from flower_cifar.utils import configure_tensorflow_gpu, set_global_seed

app = ClientApp()


def _settings(context: Context) -> tuple[int, int, int, int, float]:
    node = context.node_config
    run = context.run_config
    return (int(node["partition-id"]), int(node["num-partitions"]), int(run["local-epochs"]), int(run["batch-size"]), float(run["learning-rate"]))


@app.train()
def train(message: Message, context: Context) -> Message:
    """Train exactly one isolated CIFAR-10 partition and return its update."""
    tf.keras.backend.clear_session()
    set_global_seed(int(context.run_config["seed"]))
    configure_tensorflow_gpu()
    client_id, num_partitions, epochs, batch_size, learning_rate = _settings(context)
    round_number = int(message.content["config"].get("server-round", 0))
    write_client_status(client_id, round_number, "training")
    x_train, y_train, x_val, y_val = load_client_data(client_id, num_partitions, int(context.run_config["seed"]))
    model = create_model(learning_rate)
    # //visualize the model

    model



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
    }
    write_client_metric(metric)
    write_client_status(client_id, round_number, "completed")
    print(f"Client {client_id} | samples={len(x_train)} | train loss={metric['train_loss']:.4f} | train acc={metric['train_accuracy']:.4f} | val acc={metric['val_accuracy']:.4f}")
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
