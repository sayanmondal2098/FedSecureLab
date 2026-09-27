"""Dependency-free local dashboard for the live Flower simulation metrics."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CLIENT_METRICS_DIR = RESULTS_DIR / "client_round_metrics"
COMPARISON_DIR = RESULTS_DIR / "comparison"

STATE_LOCK = threading.Lock()
RUN_STATE: dict[str, object] = {
    "running": False,
    "phase": "idle",
    "message": "Idle",
    "clean_exit_code": None,
    "poisoned_exit_code": None,
    "logs": [],
}


def _set_state(**kwargs: object) -> None:
    with STATE_LOCK:
        RUN_STATE.update(kwargs)


def _log_event(message: str, level: str = "info") -> None:
    """Keep a small, browser-readable timeline of the comparison runner."""
    event = {
        "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
        "level": level,
        "message": message,
    }
    with STATE_LOCK:
        logs = list(RUN_STATE.get("logs", []))
        logs.append(event)
        RUN_STATE["logs"] = logs[-80:]
    print(f"[dashboard:{level}] {message}", flush=True)


def _read_state() -> dict[str, object]:
    with STATE_LOCK:
        return dict(RUN_STATE)


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _safe_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)


def _save_snapshot(tag: str) -> None:
    safe = _safe_tag(tag)
    target = COMPARISON_DIR / safe
    target.mkdir(parents=True, exist_ok=True)
    for name in ("client_metrics.csv", "global_metrics.csv", "metrics.csv", "config.json"):
        src = RESULTS_DIR / name
        if src.exists():
            (target / name).write_bytes(src.read_bytes())
    src_dir = RESULTS_DIR / "client_round_metrics"
    dst_dir = target / "client_round_metrics"
    if dst_dir.exists():
        for item in dst_dir.glob("*"):
            item.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for file in src_dir.glob("*.json"):
            (dst_dir / file.name).write_bytes(file.read_bytes())
    _log_event(f"Saved {tag} snapshot with completed metrics; it will remain available during the next run.")


def _build_run_config(env: dict[str, str], override: dict[str, str]) -> str:
    merged = dict(env)
    merged.update(override)
    parts = [
        f"num-server-rounds={merged.get('NUM_SERVER_ROUNDS', '10')}",
        f"local-epochs={merged.get('LOCAL_EPOCHS', '1')}",
        f"batch-size={merged.get('BATCH_SIZE', '32')}",
        f"learning-rate={merged.get('LEARNING_RATE', '0.001')}",
        f"seed={merged.get('SEED', '42')}",
        f"poison-mode=\"{merged.get('POISON_MODE', 'none')}\"",
        f"poison-label-flip-offset={merged.get('POISON_LABEL_FLIP_OFFSET', '1')}",
        f"poison-noise-std={merged.get('POISON_NOISE_STD', '0.15')}",
        f"poison-rate-default={merged.get('POISON_RATE_DEFAULT', '0.30')}",
        f"poison-rate-map=\"{merged.get('POISON_RATE_MAP', '')}\"",
    ]
    poison_ids = merged.get("POISON_CLIENT_IDS", "").strip()
    if poison_ids:
        parts.append(f"poison-client-ids=\"{poison_ids}\"")
    return " ".join(parts)


def _run_one_experiment(base_env: dict[str, str], override: dict[str, str], tag: str, phase_name: str) -> int:
    reset_live_metrics()
    _set_state(phase=phase_name, message=f"Running {phase_name} experiment")
    _log_event(f"Starting {phase_name} experiment ({override.get('NUM_SERVER_ROUNDS', '?')} rounds).")

    flwr_home = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "FedSecureLab" / "flwr-runs" / f"dashboard-{tag}"
    flwr_home.mkdir(parents=True, exist_ok=True)

    runtime_env = os.environ.copy()
    runtime_env.update({
        "PYTHONUTF8": base_env.get("PYTHON_UTF8", "1"),
        "FEDSECURELAB_PROJECT_ROOT": str(PROJECT_ROOT),
        "FLWR_HOME": str(flwr_home),
        "TEMP": str(PROJECT_ROOT / base_env.get("RUNTIME_TMP_DIR", ".runtime-tmp")),
        "TMP": str(PROJECT_ROOT / base_env.get("RUNTIME_TMP_DIR", ".runtime-tmp")),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": base_env.get("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0"),
    })
    runtime_env["PATH"] = str(PROJECT_ROOT / ".venv" / "Scripts") + os.pathsep + runtime_env.get("PATH", "")

    python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    run_config = _build_run_config(base_env, override)
    federation = f"num-supernodes={base_env.get('FLWR_NUM_SUPERNODES', '5')} client-resources-num-cpus={base_env.get('FLWR_CLIENT_CPUS', '1')} client-resources-num-gpus={base_env.get('FLWR_CLIENT_GPUS', '0')}"
    cmd = [
        str(python),
        "-m",
        "flwr.cli.app",
        "run",
        str(PROJECT_ROOT),
        "--stream",
        "--federation-config",
        federation,
        "--run-config",
        run_config,
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=runtime_env)
    if proc.returncode == 0:
        _save_snapshot(tag)
        _log_event(f"{phase_name.capitalize()} experiment completed successfully.")
    else:
        _log_event(f"{phase_name.capitalize()} experiment exited with code {proc.returncode}.", "error")
    return proc.returncode


def run_comparison(payload: dict[str, object]) -> None:
    base_env = _load_dotenv(PROJECT_ROOT / ".env")
    rounds = str(int(payload.get("num_server_rounds", int(base_env.get("NUM_SERVER_ROUNDS", "10")))))
    poison_mode = str(payload.get("poison_mode", "label_flip"))
    poison_client_ids = str(payload.get("poison_client_ids", ""))
    poison_rate_default = str(payload.get("poison_rate_default", base_env.get("POISON_RATE_DEFAULT", "0.30")))
    poison_rate_map = str(payload.get("poison_rate_map", ""))
    poison_label_flip_offset = str(payload.get("poison_label_flip_offset", base_env.get("POISON_LABEL_FLIP_OFFSET", "1")))
    poison_noise_std = str(payload.get("poison_noise_std", base_env.get("POISON_NOISE_STD", "0.15")))

    _set_state(running=True, phase="starting", message="Starting clean then poisoned experiments", clean_exit_code=None, poisoned_exit_code=None, logs=[])
    _log_event("Comparison requested. Clean data will be saved before poisoned training begins.")
    try:
        clean_override = {
            "NUM_SERVER_ROUNDS": rounds,
            "POISON_MODE": "none",
            "POISON_CLIENT_IDS": "",
            "POISON_RATE_DEFAULT": "0",
            "POISON_RATE_MAP": "",
            "POISON_LABEL_FLIP_OFFSET": poison_label_flip_offset,
            "POISON_NOISE_STD": poison_noise_std,
        }
        clean_code = _run_one_experiment(base_env, clean_override, "clean", "clean")
        _set_state(clean_exit_code=clean_code)
        if clean_code != 0:
            _set_state(running=False, phase="failed", message=f"Clean run failed with exit code {clean_code}")
            return

        poison_override = {
            "NUM_SERVER_ROUNDS": rounds,
            "POISON_MODE": poison_mode,
            "POISON_CLIENT_IDS": poison_client_ids,
            "POISON_RATE_DEFAULT": poison_rate_default,
            "POISON_RATE_MAP": poison_rate_map,
            "POISON_LABEL_FLIP_OFFSET": poison_label_flip_offset,
            "POISON_NOISE_STD": poison_noise_std,
        }
        _log_event("Clean snapshot is preserved. Switching to poisoned experiment.")
        poison_code = _run_one_experiment(base_env, poison_override, "poisoned", "poisoned")
        _set_state(poisoned_exit_code=poison_code)
        if poison_code != 0:
            _set_state(running=False, phase="failed", message=f"Poisoned run failed with exit code {poison_code}")
            return

        _set_state(running=False, phase="done", message="Completed clean and poisoned experiments")
        _log_event("Comparison finished. Both clean and poisoned rows are available below.")
    except Exception as exc:  # noqa: BLE001
        _set_state(running=False, phase="failed", message=f"Dashboard run failed: {exc}")
        _log_event(f"Dashboard run failed: {exc}", "error")


def launch_comparison(payload: dict[str, object]) -> bool:
    state = _read_state()
    if state.get("running"):
        return False
    thread = threading.Thread(target=run_comparison, args=(payload,), daemon=True)
    thread.start()
    return True


def read_metrics() -> dict[str, list[dict[str, object]]]:
    clients: list[dict[str, object]] = []
    for path in CLIENT_METRICS_DIR.glob("*.json") if CLIENT_METRICS_DIR.exists() else []:
        try:
            clients.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # A Ray worker may be writing a fragment exactly as we poll it.
            continue
    clients.sort(key=lambda row: (int(row["round"]), int(row["client_id"])))

    global_rows: list[dict[str, object]] = []
    global_path = RESULTS_DIR / "global_metrics.csv"
    if global_path.exists():
        try:
            with global_path.open(newline="", encoding="utf-8") as handle:
                global_rows = list(csv.DictReader(handle))
        except OSError:
            pass
    statuses: list[dict[str, object]] = []
    for path in RESULTS_DIR.glob("client_status_*.json"):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    phase: dict[str, object] = {}
    phase_path = RESULTS_DIR / "live_phase.json"
    if phase_path.exists():
        try:
            phase = json.loads(phase_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"clients": clients, "global": global_rows, "statuses": statuses, "phase": phase}


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _read_tagged_snapshot(tag: str) -> dict[str, object]:
    snapshot = COMPARISON_DIR / tag
    return {
        "tag": tag,
        "global": _read_csv_rows(snapshot / "global_metrics.csv"),
        "clients": _read_csv_rows(snapshot / "client_metrics.csv"),
        "metrics": _read_csv_rows(snapshot / "metrics.csv"),
        "exists": snapshot.exists(),
    }


def read_comparison() -> dict[str, object]:
    return {
        "clean": _read_tagged_snapshot("clean"),
        "poisoned": _read_tagged_snapshot("poisoned"),
    }


def reset_live_metrics() -> None:
    """Clear only disposable dashboard data before a new simulation starts."""
    for path in list(CLIENT_METRICS_DIR.glob("*.json")) if CLIENT_METRICS_DIR.exists() else []:
        path.unlink()
    for path in RESULTS_DIR.glob("client_status_*.json"):
        path.unlink()
    for filename in ("global_metrics.csv", "client_metrics.csv", "metrics.csv"):
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()
    phase_path = RESULTS_DIR / "live_phase.json"
    if phase_path.exists():
        phase_path.unlink()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/metrics":
            payload = json.dumps(read_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/api/comparison":
            payload = json.dumps(read_comparison()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/api/run-state":
            payload = json.dumps(_read_state()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/start-comparison":
            self.send_response(404)
            self.end_headers()
            return
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self.send_response(400)
            self.end_headers()
            return
        started = launch_comparison(payload)
        response = json.dumps({"started": started, "state": _read_state()}).encode("utf-8")
        self.send_response(200 if started else 409)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve the Flower live dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reset", action="store_true", help="clear live metrics before serving")
    args = parser.parse_args()
    if args.reset:
        reset_live_metrics()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Live dashboard available at http://127.0.0.1:{args.port}/federation-process.html", flush=True)
    server.serve_forever()
