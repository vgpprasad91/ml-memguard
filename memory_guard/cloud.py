"""Cloud sync client for the ml-memguard cloud policy service.

Enabled by setting the ``MEMGUARD_API_KEY`` environment variable.
Falls back silently to local-only mode when:
  - ``MEMGUARD_API_KEY`` is not set
  - ``httpx`` is not installed (``pip install ml-memguard[cloud]`` adds it)
  - any network call fails

All public functions return ``False`` / ``None`` on failure and never raise,
so a broken network connection never interrupts a training or inference run.

Quick start::

    export MEMGUARD_API_KEY="your-key-here"
    # BanditPolicy.save() / BanditPolicy.load() pick this up automatically.

Self-hosted backend::

    export MEMGUARD_API_URL="https://your-worker.your-subdomain.workers.dev"
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_API_URL = "https://memguard-api.vguruprasad91.workers.dev"


def _api_url() -> str:
    return os.environ.get("MEMGUARD_API_URL", _DEFAULT_API_URL).rstrip("/")


def api_key() -> Optional[str]:
    """Return the active API key from the environment, or None."""
    val = os.environ.get("MEMGUARD_API_KEY", "").strip()
    return val if len(val) > 8 else None


def _headers(key: str) -> Dict[str, str]:
    from . import __version__
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": f"ml-memguard/{__version__}",
    }


# ---------------------------------------------------------------------------
# Policy sync
# ---------------------------------------------------------------------------

def upload_policy(
    policy_data: Dict[str, Any],
    key: Optional[str] = None,
) -> bool:
    """Upload the local Q-table policy to the cloud.

    Called automatically from ``BanditPolicy.save()`` when
    ``MEMGUARD_API_KEY`` is set.

    Args:
        policy_data: The dict that ``BanditPolicy`` serialises to disk.
        key:         API key override; reads ``MEMGUARD_API_KEY`` if None.

    Returns:
        ``True`` on success, ``False`` on any failure.
    """
    active_key = key or api_key()
    if not active_key:
        return False
    try:
        import httpx
        resp = httpx.put(
            f"{_api_url()}/v1/policy",
            content=json.dumps(policy_data),
            headers=_headers(active_key),
            timeout=5.0,
        )
        resp.raise_for_status()
        logger.debug(
            "[memory-guard] Policy synced to cloud (%d states).",
            len(policy_data.get("q_table", {})),
        )
        return True
    except Exception as exc:
        logger.debug("[memory-guard] Cloud policy upload failed (local only): %s", exc)
        return False


def download_policy(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Download the cloud Q-table policy.

    Called automatically from ``BanditPolicy.load()`` when
    ``MEMGUARD_API_KEY`` is set.  The result is merged with the local
    policy so a fresh machine inherits accumulated experience immediately.

    Args:
        key: API key override; reads ``MEMGUARD_API_KEY`` if None.

    Returns:
        Parsed policy dict on success, ``None`` on 404 or any failure.
    """
    active_key = key or api_key()
    if not active_key:
        return None
    try:
        import httpx
        resp = httpx.get(
            f"{_api_url()}/v1/policy",
            headers=_headers(active_key),
            timeout=5.0,
        )
        if resp.status_code == 404:
            logger.debug("[memory-guard] No cloud policy yet (cold start).")
            return None
        resp.raise_for_status()
        data = resp.json()
        contributors = int(data.get("fleet_contributors", 0))
        logger.debug(
            "[memory-guard] Downloaded cloud policy (%d states, %d updates, %d fleet contributor%s).",
            len(data.get("q_table", {})),
            data.get("num_updates", 0),
            contributors,
            "s" if contributors != 1 else "",
        )
        if contributors > 1:
            logger.info(
                "[memory-guard] Fleet policy active: %d contributors, last rebuilt %s.",
                contributors,
                data.get("fleet_last_rebuilt") or "unknown",
            )
        return data
    except Exception as exc:
        logger.debug("[memory-guard] Cloud policy download failed (local only): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def record_telemetry(
    run_data: Dict[str, Any],
    key: Optional[str] = None,
) -> bool:
    """Post a single run record to the cloud telemetry store.

    Called automatically from ``MemoryGuard.record_result()`` when
    ``MEMGUARD_API_KEY`` is set.

    Expected keys in *run_data*:
        model_name, backend, os_platform, memory_tier, param_class, bits,
        batch_size, lora_rank, seq_length, max_num_seqs,
        estimated_mb, actual_mb, budget_mb, oom_occurred.

    Args:
        run_data: Dict of run metadata and outcome.
        key:      API key override; reads ``MEMGUARD_API_KEY`` if None.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    active_key = key or api_key()
    if not active_key:
        return False
    try:
        import httpx
        resp = httpx.post(
            f"{_api_url()}/v1/telemetry",
            content=json.dumps(run_data),
            headers=_headers(active_key),
            timeout=5.0,
        )
        resp.raise_for_status()
        logger.debug("[memory-guard] Telemetry posted to cloud.")
        return True
    except Exception as exc:
        logger.debug("[memory-guard] Cloud telemetry upload failed: %s", exc)
        return False


def predict_oom(
    signals: Dict[str, Any],
    key: Optional[str] = None,
    model_name: str = "",
    backend: str = "",
) -> Optional[Dict[str, Any]]:
    """Call ``POST /v1/predict`` and return the OOM prediction, or ``None``.

    The call is made with a **50 ms timeout** so a slow or unreachable cloud
    API never delays a monitoring tick.  On any failure (timeout, network
    error, non-2xx response) the function returns ``None`` and the caller
    falls through to local rule-based thresholds unchanged.

    Args:
        signals:    Dict with any subset of the inference signal keys:
                    ``kv_velocity_mbps``, ``fragmentation_ratio``,
                    ``eviction_rate``, ``avg_seq_len``, ``near_miss_count``,
                    ``preemption_count``, ``weights_mb``, ``kvcache_mb``.
                    Missing keys default to ``0`` server-side.
        key:        API key override; reads ``MEMGUARD_API_KEY`` if None.
        model_name: Serving model identifier for D1 personalization.
        backend:    Backend string for D1 personalization.

    Returns:
        Dict with keys ``oom_probability`` (float 0–1), ``action``
        (``"none"`` | ``"shed_load"`` | ``"restart"``),
        ``horizon_seconds`` (int), ``confidence`` (float 0–1),
        or ``None`` on any failure.
    """
    active_key = key or api_key()
    if not active_key:
        return None
    try:
        import httpx
        payload = {**signals, "model_name": model_name, "backend": backend}
        resp = httpx.post(
            f"{_api_url()}/v1/predict",
            content=json.dumps(payload),
            headers=_headers(active_key),
            timeout=0.05,   # 50 ms — must not block a monitoring tick
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[memory-guard] predict_oom failed (using local rules): %s", exc)
        return None


def upload_inference_telemetry(
    signals: "InferenceTelemetry",
    key: Optional[str] = None,
) -> bool:
    """Upload a single inference-monitoring cycle to the cloud telemetry store.

    Complements :func:`record_telemetry` (training runs) with the seven
    inference-specific signals that are invisible to vLLM/SGLang's own
    Prometheus endpoints.  Called automatically from
    ``KVCacheMonitor._loop`` when ``MEMGUARD_API_KEY`` is set.

    Args:
        signals: Populated :class:`~memory_guard.telemetry.InferenceTelemetry`.
        key:     API key override; reads ``MEMGUARD_API_KEY`` if None.

    Returns:
        ``True`` on success, ``False`` on any failure (never raises).
    """
    from .telemetry import InferenceTelemetry  # local import avoids circular
    active_key = key or api_key()
    if not active_key:
        return False
    try:
        import httpx
        resp = httpx.post(
            f"{_api_url()}/v1/telemetry",
            content=json.dumps(signals.to_dict()),
            headers=_headers(active_key),
            timeout=5.0,
        )
        resp.raise_for_status()
        logger.debug(
            "[memory-guard] Inference telemetry posted (vel=%.3f MB/s, evict=%.2f/s).",
            signals.kv_velocity_mbps,
            signals.eviction_rate,
        )
        return True
    except Exception as exc:
        logger.debug("[memory-guard] Cloud inference telemetry upload failed: %s", exc)
        return False


def get_fleet_summary(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch the aggregated fleet waste summary from the cloud.

    Returns a dict with::

        {
          "total_runs":  int,    # total runs recorded
          "oom_runs":    int,    # runs that crashed with OOM
          "avg_peak_mb": float,  # average peak memory on successful runs
          "wasted_gb":   float,  # GPU-GBs burned on OOM jobs
          "first_run":   str,    # ISO timestamp of first recorded run
          "last_run":    str,    # ISO timestamp of most recent run
        }

    Args:
        key: API key override; reads ``MEMGUARD_API_KEY`` if None.

    Returns:
        Summary dict, or ``None`` on failure / missing key.
    """
    active_key = key or api_key()
    if not active_key:
        return None
    try:
        import httpx
        resp = httpx.get(
            f"{_api_url()}/v1/telemetry/summary",
            headers=_headers(active_key),
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[memory-guard] Cloud fleet summary fetch failed: %s", exc)
        return None
