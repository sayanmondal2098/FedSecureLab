"""Reproducibility, runtime, and filesystem helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(os.getenv("FEDSECURELAB_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
load_dotenv(PROJECT_ROOT / ".env")
RESULTS_DIR = PROJECT_ROOT / "results"
CLIENT_METRICS_DIR = RESULTS_DIR / "client_round_metrics"
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and TensorFlow when it is available."""
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def configure_tensorflow_gpu() -> list[str]:
    """Enable memory growth and return detected physical GPU names."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # TensorFlow had already initialized this GPU in this worker.
            pass
    names = [device.name for device in gpus]
    print(f"TensorFlow GPU detected: {', '.join(names)}" if names else "TensorFlow GPU detected: none (CPU mode)")
    return names


def ensure_results_dir(clean_client_metrics: bool = False) -> None:
    """Create result directories, optionally removing prior client metric fragments."""
    RESULTS_DIR.mkdir(exist_ok=True)
    CLIENT_METRICS_DIR.mkdir(exist_ok=True)
    if clean_client_metrics:
        for path in CLIENT_METRICS_DIR.glob("*.json"):
            path.unlink()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a small JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

