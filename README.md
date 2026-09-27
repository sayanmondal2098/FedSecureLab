# FedSecureLab

A practical, research-friendly federated learning baseline using Flower + TensorFlow/Keras on CIFAR-10.

This project simulates five clients with isolated IID partitions and trains a small CNN with weighted FedAvg using Flower's Message API (`ClientApp` + `ServerApp`).

## Why This Repo

- Reproducible baseline for federated learning experiments
- Clean separation of client, server, data, metrics, and scripts
- Ready for extension into non-IID, robust aggregation, or adversarial scenarios
- Built-in result plotting, dashboard view, and test coverage

## High-Level Flow

```text
                    Flower Server (FedAvg)
                             |
      +-----------+----------+----------+----------+-----------+
      |           |          |          |          |           |
   Client 0    Client 1   Client 2   Client 3   Client 4
      |           |          |          |          |           |
   CIFAR-10    CIFAR-10   CIFAR-10   CIFAR-10   CIFAR-10
   IID split   IID split  IID split  IID split  IID split
```

Per round:

1. Server sends global weights.
2. Each client trains only on local data.
3. Clients return updated weights and local metrics.
4. Server performs weighted aggregation by number of examples.
5. Global model is evaluated on centralized CIFAR-10 test data.

## Project Layout

```text
flower_cifar/
  client_app.py      Flower client app
  server_app.py      Flower server app
  dataset.py         Data loading and partitioning
  model.py           CNN model definition
  metrics.py         Metrics helpers
  utils.py           Shared utility functions
scripts/
  run_simulation.ps1 Run end-to-end simulation on Windows
  dashboard_server.py Dashboard backend
  open_web.ps1       Serve and open federation-process.html
  plot_results.py    Generate plots from CSV metrics
  visualize_partitions.py Quick partition sanity checks
tests/
  test_dataset.py
  test_model.py
  test_partitions.py
results/
  Generated metrics, plots, and saved model artifacts
```

## Requirements

- Python 3.11 to 3.13 (3.11 recommended)
- Windows 11 or Linux/WSL2
- Internet access for first CIFAR-10 download

Notes:

- Native Windows TensorFlow is CPU-oriented.
- WSL2/Linux is preferred for Ray reliability and GPU workflows.

## Installation

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

### Linux / WSL2

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a `.env` file (or copy from your own template) with values like:

```dotenv
NUM_CLIENTS=5
NUM_SERVER_ROUNDS=10
LOCAL_EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=0.001
SEED=42
FLWR_NUM_SUPERNODES=5
FLWR_CLIENT_CPUS=1
FLWR_CLIENT_GPUS=0
```

Quick smoke test:

- Set `NUM_SERVER_ROUNDS=2` for a short verification run.
- Use `NUM_SERVER_ROUNDS=10` for the baseline experiment.

Poisoning and comparison options:

- `POISON_MODE=none|label_flip|gaussian_noise`
- `POISON_CLIENT_IDS=1,3` (comma-separated client IDs)
- `POISON_LABEL_FLIP_OFFSET=1` (used by `label_flip`)
- `POISON_NOISE_STD=0.15` (used by `gaussian_noise`)
- `EXPERIMENT_TAG=clean|poisoned` (snapshot label for dashboard comparison)

## Run Simulation

Windows launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_simulation.ps1
```

Follow logs for a specific run:

```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
$env:PYTHONUTF8 = "1"
$env:FLWR_HOME = "$PWD\.flwr"
flwr log <RUN_ID>
```

## Dashboard and Visualization

The simulation launcher can start/open the dashboard automatically.

To open manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\open_web.ps1
```

Default URL:

- http://localhost:8000/federation-process.html

Side-by-side clean vs poisoned comparison workflow:

1. Run with `POISON_MODE=none` and `EXPERIMENT_TAG=clean`.
2. Run with poisoning enabled and `EXPERIMENT_TAG=poisoned`.
3. Open the dashboard to compare both runs in the comparison panel.

## Validate Setup

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\visualize_partitions.py
```

This validates model behavior, partition integrity, and dataset formatting expectations.

## Output Artifacts

Typical generated files in `results/`:

- `config.json`
- `client_metrics.csv`
- `global_metrics.csv`
- `metrics.csv`
- `global_accuracy.png`
- `global_loss.png`
- `client_accuracy.png`
- `client_label_distribution.png`
- `final_model.keras`

Regenerate plots:

```powershell
.\.venv\Scripts\python.exe scripts\plot_results.py
```

## Extension Ideas

- Replace IID partitioning with Dirichlet partitioning in `flower_cifar/dataset.py`
- Plug custom aggregation into `flower_cifar/server_app.py`
- Add attack/defense hooks for robust federated learning benchmarks

## Troubleshooting

- If Ray is unstable on native Windows, run under WSL2.
- If first dataset download fails, verify connectivity and retry.
- If Flower toolchain temp-linking fails on Windows, prefer `scripts/run_simulation.ps1`.
- For finished runs, retrieve logs with `flwr log <RUN_ID>`.

---

FedSecureLab is intended to be a strong baseline: easy to run, easy to verify, and easy to extend.
