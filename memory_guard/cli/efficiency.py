"""memguard efficiency — GPU right-sizing report CLI.

Calls GET /v1/efficiency on the memguard API and prints a
human-readable table showing the current GPU tier, recommended tier,
P94 peak, waste percentage, and estimated monthly savings for each
monitored source/model pair.

Modes
-----
Local mode (default when MEMGUARD_API_URL is unset):
  Reads from ~/.memory-guard/telemetry.db — built up automatically by
  KVCacheMonitor.  No API key or cloud account required.  Requires at
  least 10 runs per (source, model) pair within the lookback window.

Cloud mode (MEMGUARD_API_URL set, or --local not passed):
  Calls the memguard API (api.memguard.ai).  Set MEMGUARD_API_KEY (or the
  legacy MEMGUARD_BACKEND_KEY) to your API key.  Get a key at memguard.ai.
  Use --local to force local mode even when MEMGUARD_API_URL is set.

Usage
-----
  # Human-readable table (default)
  memguard-efficiency

  # Pipe-friendly JSON (for CI / alerting)
  memguard-efficiency --json

  # Fleet aggregate sorted by waste fraction
  memguard-efficiency --fleet

  # Change the lookback window (default: 30 days, max: 90)
  memguard-efficiency --lookback-days 7

  # Filter by source or model
  memguard-efficiency --source-id pod-a.i-1234abcd
  memguard-efficiency --model meta-llama/Llama-3-8B-Instruct

Exit codes
----------
  0  —  success (or no sources found)
  1  —  API key not configured
  2  —  network / HTTP error
  3  —  unexpected JSON shape from API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    return (
        os.environ.get("MEMGUARD_API_KEY", "")
        or os.environ.get("MEMGUARD_BACKEND_KEY", "")
    )


def _get_api_url() -> str:
    return os.environ.get("MEMGUARD_API_URL", "").rstrip("/")


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """GET *path* from the memguard API, returning the parsed JSON body.

    Raises
    ------
    SystemExit(1) — no API key configured
    SystemExit(2) — network or HTTP error
    SystemExit(3) — response is not valid JSON or missing expected keys
    """
    key = _get_api_key()
    if not key:
        print(
            "error: MEMGUARD_API_KEY is not set.\n"
            "       Run:  export MEMGUARD_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    base = _get_api_url()
    url  = f"{base}{path}"
    if params:
        qs  = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        print(f"error: GET {url} → HTTP {exc.code}", file=sys.stderr)
        sys.exit(2)
    except urllib.error.URLError as exc:
        print(f"error: GET {url} → {exc.reason}", file=sys.stderr)
        sys.exit(2)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: unexpected response from API: {exc}", file=sys.stderr)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "source_id":    28,
    "model_name":   36,
    "current_sku":  12,
    "recommended":  12,
    "p94_mb":        9,
    "waste":         7,
    "savings":      12,  # "  $10,000/mo" fits with padding
    "conf":          8,
    "n":             6,
}

_HEADER = (
    f"{'SOURCE':<{_COL_WIDTHS['source_id']}}"
    f"{'MODEL':<{_COL_WIDTHS['model_name']}}"
    f"{'CURRENT':<{_COL_WIDTHS['current_sku']}}"
    f"{'RECOMMENDS':<{_COL_WIDTHS['recommended']}}"
    f"{'P94 MB':>{_COL_WIDTHS['p94_mb']}}"
    f"{'WASTE':>{_COL_WIDTHS['waste']}}"
    f"{'SAVINGS/MO':>{_COL_WIDTHS['savings']}}"
    f"  {'CONF':<{_COL_WIDTHS['conf']}}"
    f"{'N':>{_COL_WIDTHS['n']}}"
)
_SEP = "─" * len(_HEADER)


def _truncate(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def _format_source(row: Dict[str, Any]) -> str:
    source_id    = _truncate(str(row.get("source_id", "")),    _COL_WIDTHS["source_id"])
    model_name   = _truncate(str(row.get("model_name", "")),   _COL_WIDTHS["model_name"])
    # PR 73: render "4×A10G" for multi-GPU pods (device_count > 1)
    _sku         = str(row.get("current_sku") or "unknown")
    _dc          = int(row.get("device_count") or 1)
    current_sku  = f"{_dc}×{_sku}" if _dc > 1 else _sku
    current_sku  = _truncate(current_sku,                      _COL_WIDTHS["current_sku"])
    recommended  = str(row.get("recommended_sku") or "—")
    recommended  = _truncate(recommended,                      _COL_WIDTHS["recommended"])
    p94_mb       = f"{row.get('peak_p94_mb', 0):,.0f}"
    waste        = f"{row.get('waste_fraction', 0) * 100:.1f}%"
    savings_raw  = row.get("estimated_monthly_savings_usd", 0)
    savings      = f"${savings_raw}/mo" if savings_raw else "—"
    conf         = str(row.get("confidence", ""))[:3].upper()
    sample_n     = str(row.get("sample_size", 0))

    return (
        f"{source_id:<{_COL_WIDTHS['source_id']}}"
        f"{model_name:<{_COL_WIDTHS['model_name']}}"
        f"{current_sku:<{_COL_WIDTHS['current_sku']}}"
        f"{recommended:<{_COL_WIDTHS['recommended']}}"
        f"{p94_mb:>{_COL_WIDTHS['p94_mb']}}"
        f"{waste:>{_COL_WIDTHS['waste']}}"
        f"{savings:>{_COL_WIDTHS['savings']}}"
        f"  {conf:<{_COL_WIDTHS['conf']}}"  # 2-space separator before CONF
        f"{sample_n:>{_COL_WIDTHS['n']}}"
    )


def _print_table(sources: List[Dict[str, Any]], total_savings: Optional[int] = None) -> None:
    print(_SEP)
    print(_HEADER)
    print(_SEP)
    if not sources:
        print("  (no efficiency data found for the lookback window)")
    else:
        for row in sources:
            print(_format_source(row))
    print(_SEP)
    if total_savings is not None:
        print(f"  Total estimated monthly savings: ${total_savings}/mo")
        print(_SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="memguard-efficiency",
        description="GPU right-sizing report — shows waste fraction and savings per source/model.",
    )
    parser.add_argument(
        "--fleet",
        action="store_true",
        help="Aggregate across all sources; sort by waste fraction desc and show fleet total.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit raw JSON to stdout (pipe-friendly; compatible with CI / alerting scripts).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        metavar="N",
        help="Number of days of total_peak_mb history to include (default: 30, max: 90).",
    )
    parser.add_argument(
        "--source-id",
        default="",
        metavar="ID",
        help="Filter output to a single source_id (substring match on the client side).",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="NAME",
        help="Filter output to a single model_name (substring match on the client side).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Force local mode — read from ~/.memory-guard/telemetry.db even when "
            "MEMGUARD_API_URL is set.  Useful for per-machine reports and offline use."
        ),
    )
    args = parser.parse_args()

    lookback  = max(1, min(90, args.lookback_days))
    use_local = args.local or not os.environ.get("MEMGUARD_API_URL")

    # ------------------------------------------------------------------
    # Local mode: zero-cloud path (PR 80)
    # ------------------------------------------------------------------
    if use_local:
        from ..local_efficiency import compute_local_efficiency_report

        result = compute_local_efficiency_report(
            lookback_days=lookback,
            source_id_filter=args.source_id,
            model_filter=args.model,
        )
        if result is None:
            print(
                "No local telemetry yet — run guard_vllm for a few days, then retry."
            )
            sys.exit(0)

        loc_sources: List[Dict[str, Any]] = result["sources"]
        loc_savings: Optional[int]        = result.get("total_estimated_monthly_savings_usd")

        if args.as_json:
            out: Dict[str, Any] = {"sources": loc_sources}
            if loc_savings is not None:
                out["total_estimated_monthly_savings_usd"] = loc_savings
            print(json.dumps(out, indent=2))
        else:
            _print_table(loc_sources, loc_savings)
        return

    # ------------------------------------------------------------------
    # Cloud mode: call the memguard API
    # ------------------------------------------------------------------
    params   = {"lookback_days": str(lookback)}
    endpoint = "/v1/efficiency/fleet" if args.fleet else "/v1/efficiency"
    data     = _get(endpoint, params)

    # Validate response shape
    if "sources" not in data:
        print(f"error: unexpected API response: {data}", file=sys.stderr)
        sys.exit(3)

    sources: List[Dict[str, Any]] = data["sources"]

    # Client-side filtering
    if args.source_id:
        sources = [s for s in sources if args.source_id in str(s.get("source_id", ""))]
    if args.model:
        sources = [s for s in sources if args.model in str(s.get("model_name", ""))]

    total_savings: Optional[int] = data.get("total_estimated_monthly_savings_usd")

    if args.as_json:
        out2: Dict[str, Any] = {"sources": sources}
        if total_savings is not None:
            out2["total_estimated_monthly_savings_usd"] = total_savings
        print(json.dumps(out2, indent=2))
    else:
        _print_table(sources, total_savings)


if __name__ == "__main__":
    main()
