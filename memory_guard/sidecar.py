"""Sidecar health server for Kubernetes readiness-gate load-shedding.

Exposes two HTTP endpoints (zero hard dependencies — stdlib only):

  GET /healthz  →  200 {"status": "ok"}           (liveness probe)
  GET /readyz   →  200 {"status": "ready", ...}   (readiness probe — healthy)
                   503 {"status": "not_ready", ...} (readiness probe — shed load)

The ``/readyz`` probe returns 503 when ``KVCacheMonitor.last_oom_probability``
exceeds the configured ``threshold`` (default 0.70).  When Kubernetes sees 503
it removes the pod from the Service endpoint set, which stops new requests from
routing to this replica — achieving request shedding natively through the
Kubernetes control plane without any custom operator.

Architecture (Docker Compose / Kubernetes sidecar)::

    ┌─────────────────────────────────────────┐
    │  Pod                                    │
    │  ┌──────────────┐  ┌────────────────┐   │
    │  │  vLLM :8000  │  │ memguard :8001 │   │
    │  │              │←─│ polls /metrics │   │
    │  │  /metrics    │  │ KVCacheMonitor │   │
    │  │  (Prometheus)│  │ /healthz       │   │
    │  │              │  │ /readyz ←──────┼───┼── k8s readinessProbe
    │  └──────────────┘  └────────────────┘   │
    └─────────────────────────────────────────┘

Standalone usage::

    python -m memory_guard.sidecar \\
        --vllm-url http://localhost:8000 \\
        --host    0.0.0.0 \\
        --port    8001 \\
        --threshold 0.70

Programmatic usage::

    from memory_guard.sidecar import MemGuardSidecar, VLLMMetricsPollFn
    from memory_guard.inference_monitor import KVCacheMonitor

    poll_fn = VLLMMetricsPollFn("http://vllm:8000")
    monitor = KVCacheMonitor(poll_fn=poll_fn)
    sidecar = MemGuardSidecar(monitor, threshold=0.70)

    monitor.start()
    sidecar.start(host="0.0.0.0", port=8001)   # blocks; Ctrl-C to stop
"""

from __future__ import annotations

import http.server
import json
import logging
import re
import threading
import urllib.request
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .inference_monitor import KVCacheMonitor

logger = logging.getLogger(__name__)

# Default OOM probability above which /readyz returns 503
_DEFAULT_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# vLLM Prometheus metrics poller
# ---------------------------------------------------------------------------

class VLLMMetricsPollFn:
    """Poll vLLM's Prometheus ``/metrics`` endpoint for KV cache utilization.

    Returns ``(used_int, 100)`` where ``used_int = round(utilization * 100)``,
    so that ``KVCacheMonitor`` computes the correct ``used / total`` fraction.

    On any network error the tuple ``(0, 100)`` is returned (0 % utilization)
    so the monitor stays alive and the sidecar remains ready rather than
    surfacing a false-positive 503.

    Supports both metric name variants used across vLLM versions:
      - ``vllm:gpu_cache_usage_perc``   (vLLM ≥ 0.4.x)
      - ``vllm:kv_cache_usage_perc``    (older builds / fork variants)

    Both expose a fraction in [0, 1]; values > 1.0 are treated as percentages
    in [0, 100] and divided by 100 for backward-compatibility.
    """

    _METRIC_NAMES = (
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
    )

    def __init__(self, vllm_url: str = "http://localhost:8000") -> None:
        self._metrics_url = vllm_url.rstrip("/") + "/metrics"

    def __call__(self) -> Tuple[int, int]:
        try:
            with urllib.request.urlopen(self._metrics_url, timeout=2.0) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("[memguard-sidecar] vLLM /metrics fetch failed: %s", exc)
            return 0, 100

        util = self._parse_kv_cache_perc(text)
        used = round(util * 100)
        return used, 100

    @classmethod
    def _parse_kv_cache_perc(cls, text: str) -> float:
        """Parse KV cache utilization from Prometheus text exposition.

        Returns a float in [0.0, 1.0].
        """
        for metric_name in cls._METRIC_NAMES:
            pattern = (
                r"^" + re.escape(metric_name) +
                r"(?:\{[^}]*\})?\s+([\d.eE+\-]+)"
            )
            for line in text.splitlines():
                m = re.match(pattern, line)
                if m:
                    val = float(m.group(1))
                    # Guard against percentage-scale values (>1) vs fraction (≤1)
                    return val / 100.0 if val > 1.0 else val
        return 0.0


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib — zero dependencies)
# ---------------------------------------------------------------------------

class _SidecarHandler(http.server.BaseHTTPRequestHandler):
    """Internal handler; ``_sidecar`` is injected by ``MemGuardSidecar.start``."""

    # Populated by MemGuardSidecar via a dynamically-created subclass
    _sidecar: "MemGuardSidecar"

    def do_GET(self) -> None:
        path = self.path.split("?")[0]  # strip query string
        if path == "/healthz":
            self._send_json(200, {"status": "ok"})
        elif path == "/readyz":
            status, body = self._sidecar._handle_readyz()
            self._send_json(status, body)
        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, status: int, body: Dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
        logger.debug("[memguard-sidecar] " + fmt, *args)


# ---------------------------------------------------------------------------
# MemGuardSidecar
# ---------------------------------------------------------------------------

class MemGuardSidecar:
    """HTTP sidecar that drives Kubernetes readiness probes from OOM predictions.

    Parameters
    ----------
    monitor:
        A :class:`~memory_guard.inference_monitor.KVCacheMonitor` instance.
        ``last_oom_probability`` is read on every ``/readyz`` request.
    threshold:
        OOM probability above which ``/readyz`` returns 503.
        Default 0.70 — same as the ``shed_load`` action threshold.
    """

    def __init__(
        self,
        monitor: "KVCacheMonitor",
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._monitor = monitor
        self._threshold = threshold
        self._server: Optional[http.server.HTTPServer] = None

    # ------------------------------------------------------------------
    # Endpoint logic (tested independently of the HTTP layer)
    # ------------------------------------------------------------------

    def _handle_readyz(self) -> Tuple[int, Dict]:
        """Return (status_code, body) for the /readyz probe."""
        p = self._monitor.last_oom_probability
        if p > self._threshold:
            return 503, {
                "status":          "not_ready",
                "oom_probability": round(p, 4),
                "threshold":       self._threshold,
            }
        return 200, {
            "status":          "ready",
            "oom_probability": round(p, 4),
            "threshold":       self._threshold,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        block: bool = True,
    ) -> None:
        """Start the HTTP server.

        Parameters
        ----------
        host:   Bind address (default ``"0.0.0.0"``).
        port:   Listen port   (default 8001).
        block:  If ``True`` (default), blocks until ``stop()`` is called or
                the process receives SIGINT/SIGTERM.  Pass ``False`` to start
                in a background daemon thread (useful for tests and embedding).
        """
        # Bind _sidecar=self via a dynamically created subclass so each
        # server instance has its own handler reference without class-level
        # state leaking between concurrent test fixtures.
        handler_cls = type(
            "_BoundSidecarHandler",
            (_SidecarHandler,),
            {"_sidecar": self},
        )
        self._server = http.server.HTTPServer((host, port), handler_cls)
        logger.info(
            "[memguard-sidecar] Listening on %s:%d  (threshold=%.0f%%)",
            host, port, self._threshold * 100,
        )
        if block:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                self._server.server_close()
        else:
            t = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="memguard-sidecar",
            )
            t.start()

    def stop(self) -> None:
        """Shut down the HTTP server cleanly."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# ---------------------------------------------------------------------------
# CLI entry point  (python -m memory_guard.sidecar)
# ---------------------------------------------------------------------------

def _build_monitor_from_args(
    vllm_url: str,
    poll_interval: float,
    model_name: str,
    backend: str,
) -> "KVCacheMonitor":
    from .inference_monitor import KVCacheMonitor

    poll_fn = VLLMMetricsPollFn(vllm_url)
    return KVCacheMonitor(
        poll_fn              = poll_fn,
        poll_interval        = poll_interval,
        telemetry_model_name = model_name,
        telemetry_backend    = backend,
    )


def main() -> None:
    import argparse
    import signal
    import sys

    p = argparse.ArgumentParser(
        description="memguard sidecar — Kubernetes /readyz probe driven by OOM predictions",
    )
    p.add_argument("--vllm-url",      default="http://localhost:8000",
                   help="Base URL of the vLLM server (default: http://localhost:8000)")
    p.add_argument("--host",          default="0.0.0.0",
                   help="Bind address for the sidecar server (default: 0.0.0.0)")
    p.add_argument("--port",          type=int, default=8001,
                   help="Listen port for the sidecar server (default: 8001)")
    p.add_argument("--threshold",     type=float, default=_DEFAULT_THRESHOLD,
                   help="OOM probability above which /readyz returns 503 (default: 0.70)")
    p.add_argument("--poll-interval", type=float, default=5.0, dest="poll_interval",
                   help="Seconds between KV cache polls (default: 5.0)")
    p.add_argument("--model-name",    default="", dest="model_name",
                   help="Model identifier forwarded to ml-memguard telemetry")
    p.add_argument("--backend",       default="",
                   help="Backend string forwarded to ml-memguard telemetry")
    p.add_argument("--smoothing-window", type=int, default=1, dest="smoothing_window",
                   help="Rolling-max window for KV utilization smoothing (default: 1; use 3 for SGLang)")
    p.add_argument("--explain-telemetry", action="store_true", dest="explain_telemetry",
                   help="Print the exact telemetry schema sent to ml-memguard and exit")
    args = p.parse_args()

    if args.explain_telemetry:
        from .telemetry_explain import print_schema
        print_schema()
        sys.exit(0)

    monitor = _build_monitor_from_args(
        vllm_url      = args.vllm_url,
        poll_interval = args.poll_interval,
        model_name    = args.model_name,
        backend       = args.backend,
    )
    sidecar = MemGuardSidecar(monitor, threshold=args.threshold)

    def _shutdown(signum: int, frame: object) -> None:
        logger.info("[memguard-sidecar] Shutting down (signal %d).", signum)
        sidecar.stop()
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    monitor.start()
    logger.info(
        "[memguard-sidecar] Started. Polling vLLM at %s every %.1fs.",
        args.vllm_url, args.poll_interval,
    )
    sidecar.start(host=args.host, port=args.port, block=True)


if __name__ == "__main__":
    main()
