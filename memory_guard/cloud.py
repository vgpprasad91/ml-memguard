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
        logger.debug(
            "[memory-guard] Downloaded cloud policy (%d states, %d updates).",
            len(data.get("q_table", {})),
            data.get("num_updates", 0),
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
