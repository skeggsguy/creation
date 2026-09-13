#!/usr/bin/env python3
"""Local progress dashboard for the llm-mod pretraining run.

Python standard library only (no third-party imports) so it can run even if the
training venv is busy/broken. Serves http://127.0.0.1:8471 per docs/contracts.md:

    GET /                      -> dashboard/index.html
    GET /api/metrics?after=N   -> {"run": str|null, "lines": [...], "next": M}
    GET /api/samples?n=20      -> {"run": str|null, "samples": [...]}
    GET /api/status            -> {"run_name", "heartbeat_age_s", "latest_ckpt_step",
                                   "disk_free_gb", "tokens_total_target"}

metrics.jsonl / samples.jsonl are tailed incrementally: we remember the byte
offset and the parsed records per run, so a poll only reads the bytes appended
since the previous poll. Files that do not exist yet are simply empty.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

# Chinchilla target for the whole multi-session run (src.config.TrainConfig.total_tokens).
TOKENS_TOTAL_TARGET = 6_000_000_000

DEFAULT_PORT = 8471
STEP_DIR_RE = re.compile(r"^step_(\d+)$")


# --------------------------------------------------------------------------- #
# incremental jsonl tailing
# --------------------------------------------------------------------------- #
class JsonlTail:
    """Remembers a byte offset + the records parsed so far for one .jsonl file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.records: list[dict] = []
        self.lock = threading.Lock()

    def refresh(self) -> list[dict]:
        """Read whatever has been appended since last time; return all records."""
        with self.lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                # not created yet (or deleted): keep whatever we already have,
                # but reset if it vanished so a recreated file re-reads cleanly.
                if self.offset:
                    self.offset = 0
                    self.records = []
                return self.records

            if size < self.offset:  # truncated / rotated -> start over
                self.offset = 0
                self.records = []
            if size == self.offset:
                return self.records

            try:
                with self.path.open("rb") as fh:
                    fh.seek(self.offset)
                    chunk = fh.read(size - self.offset)
            except OSError:
                return self.records

            # Only consume up to the last complete line; a half-written final
            # line stays unread until the writer finishes it.
            cut = chunk.rfind(b"\n")
            if cut == -1:
                return self.records
            self.offset += cut + 1
            for raw in chunk[: cut + 1].splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue  # skip a corrupt line rather than dying
                if isinstance(obj, dict):
                    self.records.append(obj)
            return self.records


class RunReader:
    """Per-run tails, created lazily and reused across polls."""

    def __init__(self) -> None:
        self._tails: dict[tuple[str, str], JsonlTail] = {}
        self._lock = threading.Lock()

    def tail(self, run: str, name: str) -> JsonlTail:
        key = (run, name)
        with self._lock:
            t = self._tails.get(key)
            if t is None:
                t = JsonlTail(RUNS_DIR / run / name)
                self._tails[key] = t
            return t


READER = RunReader()


# --------------------------------------------------------------------------- #
# run discovery / status
# --------------------------------------------------------------------------- #
def newest_run() -> str | None:
    """Newest directory under runs/ that contains a metrics.jsonl."""
    try:
        candidates = [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "metrics.jsonl").stat().st_mtime).name


def latest_ckpt_step(run_dir: Path) -> int | None:
    ckpt = run_dir / "ckpt"
    latest = ckpt / "latest"
    try:
        if latest.exists():
            m = STEP_DIR_RE.match(latest.resolve().name)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    best = None
    try:
        for d in ckpt.iterdir():
            m = STEP_DIR_RE.match(d.name)
            if m and d.is_dir():
                n = int(m.group(1))
                best = n if best is None else max(best, n)
    except OSError:
        pass
    return best


def heartbeat_age_s(run_dir: Path) -> float | None:
    try:
        return round(time.time() - (run_dir / "heartbeat").stat().st_mtime, 1)
    except OSError:
        return None


def disk_free_gb() -> float:
    try:
        return round(shutil.disk_usage(REPO_ROOT).free / 1e9, 1)
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "llm-mod-dashboard/1.0"
    protocol_version = "HTTP/1.1"

    # set by main()
    fixed_run: str | None = None
    head_only = False

    def resolve_run(self) -> str | None:
        return self.fixed_run or newest_run()

    # -- helpers ----------------------------------------------------------- #
    def send_bytes(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.head_only:
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, obj, code: int = 200) -> None:
        self.send_bytes(json.dumps(obj).encode(), "application/json; charset=utf-8", code)

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    # -- routes ------------------------------------------------------------ #
    def do_HEAD(self) -> None:
        self.head_only = True
        self.do_GET()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"

        if path == "/":
            try:
                self.send_bytes(INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self.send_json({"error": f"missing {INDEX_HTML}"}, 500)
            return

        if path == "/api/metrics":
            self.api_metrics(q)
            return
        if path == "/api/samples":
            self.api_samples(q)
            return
        if path == "/api/status":
            self.api_status()
            return

        self.send_json({"error": "not found"}, 404)

    def api_metrics(self, q: dict) -> None:
        run = self.resolve_run()
        if run is None:
            self.send_json({"run": None, "lines": [], "next": 0})
            return
        after = int_arg(q, "after", 0)
        records = READER.tail(run, "metrics.jsonl").refresh()
        if after < 0 or after > len(records):
            after = 0  # client is out of sync (new run / restart): resend all
        self.send_json({"run": run, "lines": records[after:], "next": len(records)})

    def api_samples(self, q: dict) -> None:
        run = self.resolve_run()
        if run is None:
            self.send_json({"run": None, "samples": []})
            return
        n = max(0, min(int_arg(q, "n", 20), 500))
        records = READER.tail(run, "samples.jsonl").refresh()
        self.send_json({"run": run, "samples": records[-n:] if n else []})

    def api_status(self) -> None:
        run = self.resolve_run()
        run_dir = RUNS_DIR / run if run else None
        self.send_json(
            {
                "run_name": run,
                "heartbeat_age_s": heartbeat_age_s(run_dir) if run_dir else None,
                "latest_ckpt_step": latest_ckpt_step(run_dir) if run_dir else None,
                "disk_free_gb": disk_free_gb(),
                "tokens_total_target": TOKENS_TOTAL_TARGET,
            }
        )


def int_arg(q: dict, name: str, default: int) -> int:
    try:
        return int(q.get(name, [default])[0])
    except (TypeError, ValueError):
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="llm-mod progress dashboard (stdlib only)")
    ap.add_argument("--run", default=None, help="run name under runs/ (default: newest with metrics.jsonl)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    Handler.fixed_run = args.run
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    run = args.run or newest_run() or "(none yet)"
    print(f"dashboard: http://{args.host}:{args.port}  run={run}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
