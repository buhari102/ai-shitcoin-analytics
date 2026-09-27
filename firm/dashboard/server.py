"""Local dashboard server.

Deliberately built on the standard library. This machine has a slow, flaky
connection, so adding Streamlit or FastAPI would mean a long install for a
page that needs to do nothing more than read local files and return JSON.

    python main.py dashboard

Then open http://127.0.0.1:8787. Binds to loopback only; nothing is exposed
to the network.
"""

from __future__ import annotations

import json
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config
from firm import events

STATIC_DIR = Path(__file__).parent

# Scanning every Parquet file is slow enough that polling would thrash the
# disk, so expensive reads are cached briefly.
_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 8.0


def cached(key: str, producer, ttl: float = CACHE_TTL):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = producer()
    _CACHE[key] = (now, value)
    return value


# --------------------------------------------------------------------------
# API payloads
# --------------------------------------------------------------------------


def api_status() -> dict:
    from firm.data import storage

    def build():
        inv = storage.inventory()
        bars_1h = {k: v for k, v in inv["bars"].items() if k.endswith("_1h")}
        flow = inv["flow"]
        span = max((r["span_days"] for r in flow), default=0.0)

        return {
            "symbols_1h": len(bars_1h),
            "funding_series": len(inv["funding"]),
            "total_bars": sum(v["rows"] for v in inv["bars"].values()),
            "flow_symbols": len(flow),
            "flow_span_days": span,
            "flow_testable": span >= 90,
            "flow_progress_pct": min(100.0, span / 90 * 100),
            "bars": [
                {"name": k.replace("_1h", ""), "rows": v["rows"], "first": v["first"][:10], "last": v["last"][:10]}
                for k, v in sorted(bars_1h.items())
            ],
            "funding": [
                {"name": k, "rows": v["rows"], "first": v["first"][:10]}
                for k, v in sorted(inv["funding"].items())
            ],
            "flow": flow,
        }

    return cached("status", build)


def api_research() -> dict:
    from firm.research import calibration, decay, evolution, ledger

    def build():
        board = calibration.leaderboard()
        return {
            "harness_version": config.HARNESS_VERSION,
            "ledger": ledger.summary(),
            "archive": evolution.archive_summary(),
            "elites": [
                {
                    "name": e["name"],
                    "niche": e["niche"],
                    "fitness": e["fitness"],
                    "passed": bool(e["passed"]),
                    "sharpe": e.get("scorecard", {}).get("overall", {}).get("sharpe"),
                    "n_trades": e.get("scorecard", {}).get("overall", {}).get("n_trades"),
                    "reasons": e.get("scorecard", {}).get("rejection_reasons", []),
                }
                for e in evolution.elites()
            ],
            "attribution": decay.attribution(),
            "forecasts": calibration.summary(),
            "leaderboard": board.to_dict(orient="records") if not board.empty else [],
            "limits": {
                "max_position_pct": config.MAX_POSITION_PCT,
                "max_total_exposure_pct": config.MAX_TOTAL_EXPOSURE_PCT,
                "min_r_multiple": config.MIN_R_MULTIPLE,
                "holdout_tail_days": config.HOLDOUT_TAIL_DAYS,
            },
        }

    return cached("research", build)


def api_failures() -> dict:
    from firm.research import ledger

    return cached("failures", lambda: {"failures": ledger.failure_log(limit=40)})


def api_events(since: int) -> dict:
    return {
        "events": events.read_events(since=since, limit=300),
        "total": events.event_count(),
    }


def api_reports() -> dict:
    files = sorted(config.REPORTS_DIR.glob("book_*.md"), reverse=True)
    return {
        "reports": [
            {"name": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime}
            for f in files
        ]
    }


def api_report(name: str) -> dict:
    # Basename only: never let a query parameter walk out of reports/.
    target = config.REPORTS_DIR / Path(name).name
    if not target.exists() or target.suffix != ".md":
        return {"error": "report not found"}
    return {"name": target.name, "content": target.read_text(encoding="utf-8")}


ROUTES = {
    "/api/status": lambda q: api_status(),
    "/api/research": lambda q: api_research(),
    "/api/failures": lambda q: api_failures(),
    "/api/events": lambda q: api_events(int(q.get("since", ["0"])[0])),
    "/api/reports": lambda q: api_reports(),
    "/api/report": lambda q: api_report(q.get("name", [""])[0]),
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)

        if parsed.path in ROUTES:
            try:
                payload = ROUTES[parsed.path](parse_qs(parsed.query))
                body = json.dumps(payload, default=str).encode("utf-8")
                status = 200
            except Exception:  # noqa: BLE001 - report the error in the UI rather than dying
                body = json.dumps(
                    {"error": "server error", "traceback": traceback.format_exc()[-1500:]}
                ).encode("utf-8")
                status = 500

            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def end_headers(self) -> None:
        # Without this the browser caches index.html and silently serves a
        # stale UI after every edit.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # The default handler logs every poll, which drowns the console.
        pass


def serve(port: int = 8787, open_browser: bool = True) -> None:
    url = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    print(f"Dashboard running at {url}")
    print("Bound to loopback only. Ctrl+C to stop.\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
