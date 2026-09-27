"""Dependency-free local dashboard for the live Flower simulation metrics."""

from __future__ import annotations

import csv
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CLIENT_METRICS_DIR = RESULTS_DIR / "client_round_metrics"


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
        if urlparse(self.path).path == "/api/metrics":
            payload = json.dumps(read_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()


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
