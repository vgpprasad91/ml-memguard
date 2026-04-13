"""Local efficiency report computed from ~/.memory-guard/telemetry.db.

No cloud credentials required.  Works entirely with telemetry accumulated
by KVCacheMonitor._write_local_telemetry (PR 79).

Public API
----------
compute_local_efficiency_report(lookback_days, source_id_filter, model_filter)
    Returns {"sources": [...], "total_estimated_monthly_savings_usd": N}
    with the identical dict shape the cloud API returns, or None when no
    local database exists yet.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

_MIN_RUNS = 10           # minimum rows per (source, model) group before reporting
_TIER_TOLERANCE_MB = 2048  # ±2,048 MB for current-tier matching (mirrors Worker _currentTier)
_HEADROOM_FACTOR = 1.10  # 10% buffer above P94 when choosing recommended tier

# ---------------------------------------------------------------------------
# GPU tier catalog
# ---------------------------------------------------------------------------

def _load_catalog() -> List[Dict[str, Any]]:
    """Load the bundled GPU tier catalog JSON via importlib.resources (wheel-safe)."""
    ref = importlib.resources.files("memory_guard.data").joinpath("gpu_tier_catalog.json")
    with ref.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# P94 and tier helpers
# ---------------------------------------------------------------------------

def _p94(values: List[float]) -> float:
    """94th-percentile of *values* (mirrors Worker _pct(peaks, 0.94))."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(0.94 * len(s)), len(s) - 1)
    return s[idx]


def _match_current_tier(
    vram_mb: float,
    device_count: int,
    catalog: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the catalog entry whose vram_mb is within ±2,048 MB of *vram_mb*
    and whose device_count matches.  Returns None when no entry qualifies."""
    best: Optional[Dict[str, Any]] = None
    best_delta = float("inf")
    for tier in catalog:
        if tier["device_count"] != device_count:
            continue
        delta = abs(tier["vram_mb"] - vram_mb)
        if delta <= _TIER_TOLERANCE_MB and delta < best_delta:
            best = tier
            best_delta = delta
    return best


def _recommend_tier(
    p94_mb: float,
    catalog: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the cheapest catalog tier whose vram_mb >= p94_mb × 1.10.

    Device-count changes are allowed — a 4-GPU pod that only uses one GPU's
    worth of VRAM should downgrade to a single-GPU tier.
    """
    needed = p94_mb * _HEADROOM_FACTOR
    candidates = [t for t in catalog if t["vram_mb"] >= needed]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t["on_demand_hourly_usd"])


# ---------------------------------------------------------------------------
# LocalTelemetryDB
# ---------------------------------------------------------------------------

class LocalTelemetryDB:
    """Thin sqlite3 reader over ~/.memory-guard/telemetry.db."""

    DEFAULT_PATH: str = os.path.expanduser("~/.memory-guard/telemetry.db")

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or self.DEFAULT_PATH

    def exists(self) -> bool:
        return os.path.isfile(self._db_path)

    def fetch_groups(
        self,
        lookback_days: int,
        source_id_filter: str = "",
        model_filter: str = "",
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Query runs within the lookback window, grouped by (source_id, model_name).

        Returns
        -------
        {
            (source_id, model_name): {
                "rows": [(reserved_vram_mb, total_peak_mb), ...],
                "device_count": int,
            },
            ...
        }
        """
        cutoff = int(time.time()) - lookback_days * 86400
        sql = (
            "SELECT source_id, model_name, reserved_vram_mb, total_peak_mb, device_count "
            "FROM runs "
            "WHERE recorded_at >= ? AND total_peak_mb > 0"
        )
        params: List[Any] = [cutoff]
        if source_id_filter:
            sql += " AND source_id LIKE ?"
            params.append(f"%{source_id_filter}%")
        if model_filter:
            sql += " AND model_name LIKE ?"
            params.append(f"%{model_filter}%")

        groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        with sqlite3.connect(self._db_path, timeout=5) as conn:
            for src, mdl, reserved, peak, dc in conn.execute(sql, params):
                key = (src or "", mdl or "")
                if key not in groups:
                    groups[key] = {"rows": [], "device_count": int(dc or 1)}
                groups[key]["rows"].append((float(reserved or 0.0), float(peak or 0.0)))
        return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_local_efficiency_report(
    lookback_days: int = 30,
    source_id_filter: str = "",
    model_filter: str = "",
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compute the efficiency report from local telemetry.

    Parameters
    ----------
    lookback_days:
        Rolling window length in days (default 30, max enforced by caller).
    source_id_filter:
        Substring filter applied to source_id at the DB query level.
    model_filter:
        Substring filter applied to model_name at the DB query level.
    db_path:
        Override the default ``~/.memory-guard/telemetry.db`` path.

    Returns
    -------
    None
        When the local database does not exist — caller should print a hint.
    dict
        ``{"sources": [...], "total_estimated_monthly_savings_usd": N}``
        with the identical shape the cloud API returns, suitable for direct
        use by the CLI's ``_print_table`` and JSON output paths.
    """
    db = LocalTelemetryDB(db_path)
    if not db.exists():
        return None

    catalog = _load_catalog()
    groups = db.fetch_groups(lookback_days, source_id_filter, model_filter)

    sources: List[Dict[str, Any]] = []
    total_savings = 0

    for (source_id, model_name), data in sorted(groups.items()):
        rows: List[Tuple[float, float]] = data["rows"]
        if len(rows) < _MIN_RUNS:
            continue

        device_count = data["device_count"]
        reserved_values = [r[0] for r in rows]
        peak_values     = [r[1] for r in rows]

        # Median reserved VRAM represents the stable "current" allocation
        reserved_sorted  = sorted(reserved_values)
        reserved_vram_mb = reserved_sorted[len(reserved_sorted) // 2]

        p94_mb           = _p94(peak_values)
        current_tier     = _match_current_tier(reserved_vram_mb, device_count, catalog)
        recommended_tier = _recommend_tier(p94_mb, catalog)

        current_sku     = current_tier["sku"]     if current_tier     else "unknown"
        recommended_sku = recommended_tier["sku"] if recommended_tier else current_sku

        waste_fraction = (
            max(0.0, (reserved_vram_mb - p94_mb) / reserved_vram_mb)
            if reserved_vram_mb > 0 else 0.0
        )

        monthly_savings = 0
        if (
            current_tier and recommended_tier
            and recommended_tier["on_demand_hourly_usd"] < current_tier["on_demand_hourly_usd"]
        ):
            delta_hourly    = (
                current_tier["on_demand_hourly_usd"]
                - recommended_tier["on_demand_hourly_usd"]
            )
            monthly_savings = int(delta_hourly * 24 * 30)

        n          = len(rows)
        confidence = "HIGH" if n >= 100 else "MED" if n >= 30 else "LOW"

        sources.append({
            "source_id":                     source_id,
            "model_name":                    model_name,
            "current_sku":                   current_sku,
            "recommended_sku":               recommended_sku,
            "peak_p94_mb":                   p94_mb,
            "waste_fraction":                waste_fraction,
            "estimated_monthly_savings_usd": monthly_savings,
            "confidence":                    confidence,
            "sample_size":                   n,
            "device_count":                  device_count,
        })
        total_savings += monthly_savings

    # Sort by waste_fraction descending — same ordering as the fleet endpoint
    sources.sort(key=lambda s: s["waste_fraction"], reverse=True)

    return {
        "sources":                             sources,
        "total_estimated_monthly_savings_usd": total_savings,
    }
