"""Current Flower ServerApp and standard FedAvg strategy for the baseline."""

from __future__ import annotations

import importlib.metadata
from typing import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_cifar.dataset import NUM_CLIENTS, load_global_test_data
from flower_cifar.metrics import clear_live_metrics, collect_client_metrics, write_combined_metrics, write_global_metrics, write_live_phase
from flower_cifar.model import create_model
from flower_cifar.utils import RESULTS_DIR, configure_tensorflow_gpu, ensure_results_dir, set_global_seed, write_json

app = ServerApp()


class LoggingFedAvg(FedAvg):
    """Standard FedAvg with concise research-facing aggregation messages.

    The superclass performs weighted aggregation of ArrayRecords by
    ``num-examples``; no aggregation rule is reimplemented here.
    """

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        write_live_phase(server_round, "aggregate")
        arrays, metrics = super().aggregate_train(server_round, replies)
        print(f"FedAvg aggregation completed for round {server_round}.")
        return arrays, metrics


def _global_evaluator(learning_rate: float, batch_size: int, records: list[dict[str, float]]):
    """Return the supported ServerApp-side centralized test evaluator callback."""
    x_test, y_test = load_global_test_data()

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        write_live_phase(server_round, "evaluate")
        model = create_model(learning_rate)
        model.set_weights(arrays.to_numpy_ndarrays())
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        row = {"round": server_round, "global_loss": float(loss), "global_accuracy": float(accuracy)}
        records.append(row)
        write_global_metrics(records)
        print(f"Global Test Loss: {loss:.4f} | Global Test Accuracy: {accuracy:.4f}")
        return MetricRecord(row)

    return evaluate


def _write_metadata(context: Context) -> None:
    versions = {}
    for package in ("flwr", "flwr-datasets", "tensorflow", "ray"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    write_json(RESULTS_DIR / "config.json", {"experiment": "Flower TensorFlow CIFAR-10 Baseline", "clients": NUM_CLIENTS, "partitioning": "IID", "aggregation": "FedAvg", "run_config": dict(context.run_config), "package_versions": versions})


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Initialize the shared CNN, run five-node FedAvg, and persist artifacts."""
    run = context.run_config
    poison_mode = str(run.get("poison-mode", "none"))
    poison_clients = str(run.get("poison-client-ids", ""))
    set_global_seed(int(run["seed"]))
    configure_tensorflow_gpu()
    ensure_results_dir()
    clear_live_metrics()
    _write_metadata(context)
    rounds = int(run["num-server-rounds"])
    write_live_phase(1, "broadcast")
    learning_rate = float(run["learning-rate"])
    batch_size = int(run["batch-size"])
    print("\nExperiment: Flower TensorFlow CIFAR-10 Baseline")
    print(f"Clients: {NUM_CLIENTS}\nPartitioning: IID\nAggregation: FedAvg\nRounds: {rounds}\nLocal Epochs: {run['local-epochs']}\nBatch Size: {batch_size}\nSeed: {run['seed']}")
    print(f"Poison mode: {poison_mode} | Poison clients: {poison_clients if poison_clients else 'none'}")
    model = create_model(learning_rate)
    global_records: list[dict[str, float]] = []
    strategy = LoggingFedAvg(
        fraction_train=float(run["fraction-train"]),
        fraction_evaluate=float(run["fraction-evaluate"]),
        min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.get_weights()),
        num_rounds=rounds,
        train_config=ConfigRecord({"learning-rate": learning_rate}),
        evaluate_config=ConfigRecord({"learning-rate": learning_rate}),
        evaluate_fn=_global_evaluator(learning_rate, batch_size, global_records),
    )
    model.set_weights(result.arrays.to_numpy_ndarrays())
    model.save(RESULTS_DIR / "final_model.keras")
    client_records = collect_client_metrics()
    write_combined_metrics(client_records, global_records)
    print(f"Completed {rounds} rounds with {len(client_records)} client metric rows.")

