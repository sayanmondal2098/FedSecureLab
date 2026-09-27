"""Persistent experiment metrics and post-run reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from flower_cifar.utils import CLIENT_METRICS_DIR, RESULTS_DIR, ensure_results_dir

CLIENT_COLUMNS = ["round", "client_id", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "num_samples", "poisoned", "poison_mode"]
GLOBAL_COLUMNS = ["round", "global_loss", "global_accuracy"]


def clear_live_metrics() -> None:
    """Remove live data from a prior run before the dashboard begins polling."""
    ensure_results_dir(clean_client_metrics=True)
    for filename in ("global_metrics.csv", "client_metrics.csv", "metrics.csv"):
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()
    for path in RESULTS_DIR.glob("client_status_*.json"):
        path.unlink()
    phase_path = RESULTS_DIR / "live_phase.json"
    if phase_path.exists():
        phase_path.unlink()


def write_client_status(client_id: int, round_number: int, state: str) -> None:
    """Publish a lightweight live client state for the training dashboard."""
    ensure_results_dir()
    path = RESULTS_DIR / f"client_status_{client_id}.json"
    path.write_text(json.dumps({"client_id": client_id, "round": round_number, "state": state}), encoding="utf-8")


def write_live_phase(round_number: int, phase: str) -> None:
    """Publish the server's current FedAvg phase for the live dashboard."""
    ensure_results_dir()
    (RESULTS_DIR / "live_phase.json").write_text(
        json.dumps({"round": round_number, "phase": phase}), encoding="utf-8"
    )


def write_client_metric(metric: dict[str, Any]) -> None:
    """Write one unique client/round fragment safely from a simulation worker."""
    ensure_results_dir()
    path = CLIENT_METRICS_DIR / f"client_{metric['client_id']}_round_{metric['round']}.json"
    path.write_text(json.dumps(metric, sort_keys=True), encoding="utf-8")


def collect_client_metrics() -> list[dict[str, Any]]:
    """Merge worker fragments into a deterministic CSV after simulation completes."""
    ensure_results_dir()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in CLIENT_METRICS_DIR.glob("*.json")]
    records.sort(key=lambda row: (int(row["round"]), int(row["client_id"])))
    with (RESULTS_DIR / "client_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLIENT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return records


def write_global_metrics(records: list[dict[str, Any]]) -> None:
    """Write centralized test results accumulated by the strategy."""
    with (RESULTS_DIR / "global_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GLOBAL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def write_combined_metrics(client_records: list[dict[str, Any]], global_records: list[dict[str, Any]]) -> None:
    """Create the requested one-row-per-client metrics CSV with global values joined."""
    by_round = {int(row["round"]): row for row in global_records}
    fields = CLIENT_COLUMNS + ["global_loss", "global_accuracy"]
    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in client_records:
            global_row = by_round.get(int(row["round"]), {})
            writer.writerow({**row, "global_loss": global_row.get("global_loss", ""), "global_accuracy": global_row.get("global_accuracy", "")})

