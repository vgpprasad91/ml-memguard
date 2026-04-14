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

    from memory_guard.deployment.sidecar import MemGuardSidecar, VLLMMetricsPollFn
    from memory_guard import KVCacheMonitor

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
import os
import re
import threading
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from ..monitoring.inference_monitor import KVCacheMonitor

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
        headroom_threshold_mb: float = 0.0,
    ) -> None:
        self._monitor = monitor
        self._threshold = threshold
        # PR 67: /readyz also returns 503 when true_available_headroom_mb is
        # below this value.  0.0 (default) disables the headroom gate entirely.
        self._headroom_threshold_mb: float = max(0.0, headroom_threshold_mb)
        self._server: Optional[http.server.HTTPServer] = None

    # ------------------------------------------------------------------
    # Endpoint logic (tested independently of the HTTP layer)
    # ------------------------------------------------------------------

    def _handle_readyz(self) -> Tuple[int, Dict]:
        """Return (status_code, body) for the /readyz probe.

        Returns 503 (not ready) when either:
          - ``oom_probability`` exceeds the OOM ``threshold``, OR
          - ``true_available_headroom_mb`` is below ``headroom_threshold_mb``
            and the headroom gate is enabled (threshold > 0).

        The headroom gate catches weight/overhead OOMs that the probabilistic
        model may not flag in time (PR 67).
        """
        p           = self._monitor.last_oom_probability
        headroom_mb = self._monitor.last_true_available_headroom_mb
        # float('inf') means "not yet received from /v1/predict" — skip gate
        headroom_breach = (
            self._headroom_threshold_mb > 0.0
            and headroom_mb != float("inf")
            and headroom_mb < self._headroom_threshold_mb
        )
        headroom_val: Optional[float] = (
            round(headroom_mb, 1) if headroom_mb != float("inf") else None
        )

        if p > self._threshold or headroom_breach:
            return 503, {
                "status":                    "not_ready",
                "oom_probability":           round(p, 4),
                "threshold":                 self._threshold,
                "true_available_headroom_mb": headroom_val,
                "headroom_threshold_mb":     self._headroom_threshold_mb,
            }
        return 200, {
            "status":                    "ready",
            "oom_probability":           round(p, 4),
            "threshold":                 self._threshold,
            "true_available_headroom_mb": headroom_val,
            "headroom_threshold_mb":     self._headroom_threshold_mb,
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
# MemGuardPolicy hot-reload helpers
# ---------------------------------------------------------------------------

def _apply_policy_spec(
    spec: Dict[str, Any],
    sidecar: "MemGuardSidecar",
    monitor: "KVCacheMonitor",
) -> None:
    """Apply a MemGuardPolicy spec dict to the running sidecar and monitor.

    Called on startup (initial GET) and on every hot-reload watch event.
    All assignments are atomic scalar writes — safe to call from any thread
    under the Python GIL.

    Supported spec keys (all optional — missing keys are silently ignored):
      shedThreshold    → sidecar._threshold
      warningThreshold → monitor.THRESHOLD_WARNING  (instance shadow)
      smoothingWindow  → monitor._smoothing_window  (if attribute exists)
      telemetryBackend → monitor._telemetry_backend
    """
    if "shedThreshold" in spec:
        v = float(spec["shedThreshold"])
        sidecar._threshold = v
        logger.info("[memguard] MemGuardPolicy: shedThreshold → %.2f", v)
    if "warningThreshold" in spec:
        v = float(spec["warningThreshold"])
        monitor.THRESHOLD_WARNING = v  # type: ignore[assignment]
        logger.info("[memguard] MemGuardPolicy: warningThreshold → %.2f", v)
    if "smoothingWindow" in spec:
        v = int(spec["smoothingWindow"])
        if hasattr(monitor, "_smoothing_window"):
            monitor._smoothing_window = v  # type: ignore[assignment]
        logger.info("[memguard] MemGuardPolicy: smoothingWindow → %d", v)
    if spec.get("telemetryBackend"):
        monitor._telemetry_backend = str(spec["telemetryBackend"])
        logger.info(
            "[memguard] MemGuardPolicy: telemetryBackend → %s",
            monitor._telemetry_backend,
        )


def _start_policy_watcher(
    policy_name: str,
    sidecar: "MemGuardSidecar",
    monitor: "KVCacheMonitor",
) -> Optional[Any]:
    """Start a K8sPolicyWatcher when running in-cluster and policy_name is set.

    1. Fetches the initial MemGuardPolicy spec (blocking GET) and applies it.
    2. Starts a background daemon thread for real-time hot-reload.

    Returns the K8sPolicyWatcher instance (call .stop() on shutdown) or None
    when not in-cluster, policy_name is empty, or k8s_policy is unavailable.
    """
    if not policy_name:
        return None
    try:
        from .k8s_policy import K8sPolicyWatcher
    except ImportError:
        logger.debug("[memguard] k8s_policy not importable — policy watch disabled")
        return None

    if not K8sPolicyWatcher.is_in_cluster():
        logger.debug(
            "[memguard] Not running in-cluster — MemGuardPolicy watch disabled"
        )
        return None

    watcher = K8sPolicyWatcher(policy_name=policy_name)

    initial_spec = watcher.get()
    if initial_spec:
        _apply_policy_spec(initial_spec, sidecar, monitor)
    else:
        logger.debug(
            "[memguard] MemGuardPolicy %r not found — using built-in defaults",
            policy_name,
        )

    watcher.watch(lambda spec: _apply_policy_spec(spec, sidecar, monitor))
    logger.info(
        "[memguard] MemGuardPolicy watcher active for policy %r", policy_name
    )
    return watcher


# ---------------------------------------------------------------------------
# CLI entry point  (python -m memory_guard.sidecar)
# ---------------------------------------------------------------------------

def _build_monitor_from_args(
    vllm_url: str,
    poll_interval: float,
    model_name: str,
    backend: str,
) -> "KVCacheMonitor":
    from ..monitoring.inference_monitor import KVCacheMonitor

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
    p.add_argument("--headroom-threshold-mb", type=float, default=0.0,
                   dest="headroom_threshold_mb",
                   help="Minimum true_available_headroom_mb below which /readyz returns 503 "
                        "(default: 0 = disabled). Catches weight/overhead OOMs that the "
                        "probabilistic model may miss.")
    p.add_argument("--poll-interval", type=float, default=5.0, dest="poll_interval",
                   help="Seconds between KV cache polls (default: 5.0)")
    p.add_argument("--model-name",    default="", dest="model_name",
                   help="Model identifier attached to monitoring signals")
    p.add_argument("--backend",       default="",
                   help="Backend framework string attached to monitoring signals")
    p.add_argument("--smoothing-window", type=int, default=1, dest="smoothing_window",
                   help="Rolling-max window for KV utilization smoothing (default: 1; use 3 for SGLang)")
    p.add_argument("--policy-name",  default=os.environ.get("MEMGUARD_POLICY_NAME", ""),
                   dest="policy_name",
                   help="Name of a MemGuardPolicy CRD to watch for hot-reload "
                        "(in-cluster only; reads MEMGUARD_POLICY_NAME env var)")
    args = p.parse_args()

    monitor = _build_monitor_from_args(
        vllm_url      = args.vllm_url,
        poll_interval = args.poll_interval,
        model_name    = args.model_name,
        backend       = args.backend,
    )
    sidecar = MemGuardSidecar(
        monitor,
        threshold=args.threshold,
        headroom_threshold_mb=args.headroom_threshold_mb,
    )

    # Start MemGuardPolicy watcher for in-cluster hot-reload (no-op outside k8s)
    _policy_watcher = _start_policy_watcher(args.policy_name, sidecar, monitor)

    def _shutdown(signum: int, frame: object) -> None:
        logger.info("[memguard-sidecar] Shutting down (signal %d).", signum)
        if _policy_watcher is not None:
            _policy_watcher.stop()
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
