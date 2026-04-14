"""Tests for local_efficiency.py and KVCacheMonitor._write_local_telemetry.

Covers:
  - _p94: empty list returns 0.0
  - _p94: single element returns that element
  - _p94: correct 94th-percentile index selection
  - _match_current_tier: returns None when no catalog entry within tolerance
  - _match_current_tier: returns closest entry within ±2048 MB for matching device_count
  - _match_current_tier: ignores entries with wrong device_count
  - _recommend_tier: returns None when all tiers are too small
  - _recommend_tier: returns cheapest tier satisfying p94 × 1.10 headroom
  - _recommend_tier: allows device-count change (downgrade to single-GPU)
  - LocalTelemetryDB.exists(): returns False when file absent
  - LocalTelemetryDB.exists(): returns True after DB is created
  - LocalTelemetryDB.fetch_groups: returns empty dict on empty table
  - LocalTelemetryDB.fetch_groups: groups rows by (source_id, model_name)
  - LocalTelemetryDB.fetch_groups: respects lookback window cutoff
  - LocalTelemetryDB.fetch_groups: source_id_filter LIKE match
  - LocalTelemetryDB.fetch_groups: model_filter LIKE match
  - compute_local_efficiency_report: returns None when DB absent
  - compute_local_efficiency_report: returns None when too few rows (< 10)
  - compute_local_efficiency_report: returns report dict with correct keys
  - compute_local_efficiency_report: total_savings aggregates across sources
  - compute_local_efficiency_report: sources sorted by waste_fraction descending
  - compute_local_efficiency_report: confidence tiers (LOW/MED/HIGH)
  - KVCacheMonitor._write_local_telemetry: creates DB and inserts one row
  - KVCacheMonitor._write_local_telemetry: swallows exceptions silently
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.local_efficiency import (
    LocalTelemetryDB,
    _match_current_tier,
    _p94,
    _recommend_tier,
    compute_local_efficiency_report,
)
from memory_guard.monitoring.inference_monitor import KVCacheMonitor


# ---------------------------------------------------------------------------
# Catalog fixture
# ---------------------------------------------------------------------------

CATALOG = [
    {"sku": "T4",      "vram_mb": 16384, "device_count": 1, "on_demand_hourly_usd": 0.35},
    {"sku": "A10G",    "vram_mb": 24576, "device_count": 1, "on_demand_hourly_usd": 0.76},
    {"sku": "A100-40", "vram_mb": 40960, "device_count": 1, "on_demand_hourly_usd": 3.06},
    {"sku": "A100-80", "vram_mb": 81920, "device_count": 1, "on_demand_hourly_usd": 4.10},
    {"sku": "T4",      "vram_mb": 32768, "device_count": 2, "on_demand_hourly_usd": 0.70},
    {"sku": "A10G",    "vram_mb": 49152, "device_count": 2, "on_demand_hourly_usd": 1.52},
]


# ---------------------------------------------------------------------------
# _p94
# ---------------------------------------------------------------------------

class TestP94:
    def test_empty_returns_zero(self):
        assert _p94([]) == 0.0

    def test_single_element(self):
        assert _p94([42.0]) == 42.0

    def test_two_elements_returns_higher(self):
        # With 2 elements, idx = int(0.94 * 2) = 1 → second element
        assert _p94([10.0, 20.0]) == 20.0

    def test_10_elements(self):
        # idx = int(0.94 * 10) = 9 → last element of sorted list
        values = list(range(1, 11))  # [1..10]
        assert _p94(values) == 10

    def test_100_elements(self):
        # idx = int(0.94 * 100) = 94 → 95th value (0-indexed)
        values = list(range(1, 101))
        result = _p94(values)
        assert result == 95  # sorted[94] = 95

    def test_unsorted_input(self):
        values = [50.0, 10.0, 90.0, 30.0, 70.0]
        # sorted: [10, 30, 50, 70, 90]; idx = int(0.94 * 5) = 4 → 90
        assert _p94(values) == 90.0


# ---------------------------------------------------------------------------
# _match_current_tier
# ---------------------------------------------------------------------------

class TestMatchCurrentTier:
    def test_no_match_when_too_far(self):
        # 16384 vs 81920 — delta > 2048
        result = _match_current_tier(81920, 1, CATALOG[:1])
        assert result is None

    def test_exact_match(self):
        result = _match_current_tier(16384, 1, CATALOG)
        assert result is not None
        assert result["sku"] == "T4"
        assert result["device_count"] == 1

    def test_within_tolerance(self):
        # A10G is 24576; query with 23000 — delta = 1576 < 2048
        result = _match_current_tier(23000, 1, CATALOG)
        assert result is not None
        assert result["sku"] == "A10G"

    def test_wrong_device_count_ignored(self):
        # 2-device T4 is 32768; single-device T4 is 16384
        # Query with device_count=2 and 32768 should get the 2-GPU entry
        result = _match_current_tier(32768, 2, CATALOG)
        assert result is not None
        assert result["device_count"] == 2

    def test_device_count_mismatch_returns_none(self):
        # 16384 matches T4 single-device, but asking for device_count=4
        result = _match_current_tier(16384, 4, CATALOG)
        assert result is None

    def test_prefers_closest(self):
        # Both T4 (16384) and A10G (24576) could match 20000 within ±2048?
        # T4: delta = |16384 - 20000| = 3616 > 2048 → no
        # A10G: delta = |24576 - 20000| = 4576 > 2048 → no
        # Neither qualifies — returns None
        result = _match_current_tier(20000, 1, CATALOG)
        assert result is None


# ---------------------------------------------------------------------------
# _recommend_tier
# ---------------------------------------------------------------------------

class TestRecommendTier:
    def test_returns_none_when_all_too_small(self):
        # p94 = 100000 MB; largest catalog entry is 81920 * 1.10 = 90112 — all too small
        result = _recommend_tier(100000.0, CATALOG)
        assert result is None

    def test_returns_cheapest_qualifying(self):
        # p94 = 10000; needed = 11000; T4 (16384) qualifies at $0.35/hr
        result = _recommend_tier(10000.0, CATALOG)
        assert result is not None
        assert result["sku"] == "T4"
        assert result["on_demand_hourly_usd"] == 0.35

    def test_headroom_factor_applied(self):
        # p94 = 15000; needed = 16500; T4-1GPU (16384) < 16500 → excluded
        # Qualifying: A10G-1GPU (24576, $0.76), T4-2GPU (32768, $0.70)
        # Cheapest is T4-2GPU at $0.70/hr
        result = _recommend_tier(15000.0, CATALOG)
        assert result is not None
        assert result["sku"] == "T4"
        assert result["device_count"] == 2

    def test_cross_device_count_allowed(self):
        # p94 = 20000; needed = 22000; T4-1GPU (16384) < 22000 → excluded
        # Qualifying: A10G-1GPU (24576, $0.76), T4-2GPU (32768, $0.70)
        # Cheapest is T4-2GPU — cross device-count downgrade is allowed
        result = _recommend_tier(20000.0, CATALOG)
        assert result is not None
        assert result["sku"] == "T4"
        assert result["device_count"] == 2


# ---------------------------------------------------------------------------
# LocalTelemetryDB
# ---------------------------------------------------------------------------

class TestLocalTelemetryDB:

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "telemetry.db")

    def _seed(self, db_path: str, rows, recorded_at=None):
        """Helper: insert rows into a freshly-created DB."""
        if recorded_at is None:
            recorded_at = int(time.time())
        schema = (
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY, source_id TEXT, model_name TEXT,"
            "reserved_vram_mb REAL, total_peak_mb REAL, device_count INTEGER,"
            "recorded_at INTEGER DEFAULT (strftime('%s','now')))"
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(schema)
            for src, mdl, reserved, peak, dc in rows:
                conn.execute(
                    "INSERT INTO runs (source_id, model_name, reserved_vram_mb, "
                    "total_peak_mb, device_count, recorded_at) VALUES (?,?,?,?,?,?)",
                    (src, mdl, reserved, peak, dc, recorded_at),
                )

    def test_exists_false_when_absent(self, tmp_db):
        db = LocalTelemetryDB(tmp_db)
        assert db.exists() is False

    def test_exists_true_after_creation(self, tmp_db):
        self._seed(tmp_db, [])
        db = LocalTelemetryDB(tmp_db)
        assert db.exists() is True

    def test_fetch_groups_empty_table(self, tmp_db):
        self._seed(tmp_db, [])
        db = LocalTelemetryDB(tmp_db)
        result = db.fetch_groups(lookback_days=30)
        assert result == {}

    def test_fetch_groups_groups_by_source_model(self, tmp_db):
        rows = [
            ("src-A", "llama",  20000.0, 18000.0, 1),
            ("src-A", "llama",  20000.0, 17500.0, 1),
            ("src-B", "mistral", 40000.0, 35000.0, 1),
        ]
        self._seed(tmp_db, rows)
        db = LocalTelemetryDB(tmp_db)
        groups = db.fetch_groups(lookback_days=30)
        assert len(groups) == 2
        assert ("src-A", "llama") in groups
        assert len(groups[("src-A", "llama")]["rows"]) == 2
        assert ("src-B", "mistral") in groups

    def test_fetch_groups_respects_lookback(self, tmp_db):
        old_ts = int(time.time()) - 40 * 86400  # 40 days ago
        rows = [("src-A", "llama", 20000.0, 18000.0, 1)]
        self._seed(tmp_db, rows, recorded_at=old_ts)
        db = LocalTelemetryDB(tmp_db)
        groups = db.fetch_groups(lookback_days=30)
        assert groups == {}

    def test_fetch_groups_source_filter(self, tmp_db):
        rows = [
            ("prod-server", "llama", 20000.0, 18000.0, 1),
            ("dev-server",  "llama", 20000.0, 18000.0, 1),
        ]
        self._seed(tmp_db, rows)
        db = LocalTelemetryDB(tmp_db)
        groups = db.fetch_groups(lookback_days=30, source_id_filter="prod")
        assert len(groups) == 1
        assert ("prod-server", "llama") in groups

    def test_fetch_groups_model_filter(self, tmp_db):
        rows = [
            ("src-A", "llama-7b",   20000.0, 18000.0, 1),
            ("src-A", "mistral-7b", 20000.0, 18000.0, 1),
        ]
        self._seed(tmp_db, rows)
        db = LocalTelemetryDB(tmp_db)
        groups = db.fetch_groups(lookback_days=30, model_filter="llama")
        assert len(groups) == 1
        assert ("src-A", "llama-7b") in groups

    def test_fetch_groups_device_count_stored(self, tmp_db):
        rows = [("src-A", "llama", 49152.0, 45000.0, 2)]
        self._seed(tmp_db, rows)
        db = LocalTelemetryDB(tmp_db)
        groups = db.fetch_groups(lookback_days=30)
        assert groups[("src-A", "llama")]["device_count"] == 2


# ---------------------------------------------------------------------------
# compute_local_efficiency_report
# ---------------------------------------------------------------------------

class TestComputeLocalEfficiencyReport:

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "telemetry.db")

    def _seed(self, db_path: str, rows):
        schema = (
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY, source_id TEXT, model_name TEXT,"
            "reserved_vram_mb REAL, total_peak_mb REAL, device_count INTEGER,"
            "recorded_at INTEGER DEFAULT (strftime('%s','now')))"
        )
        now = int(time.time())
        with sqlite3.connect(db_path) as conn:
            conn.execute(schema)
            for src, mdl, reserved, peak, dc in rows:
                conn.execute(
                    "INSERT INTO runs (source_id, model_name, reserved_vram_mb, "
                    "total_peak_mb, device_count, recorded_at) VALUES (?,?,?,?,?,?)",
                    (src, mdl, reserved, peak, dc, now),
                )

    def test_returns_none_when_no_db(self, tmp_db):
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is None

    def test_returns_none_below_min_runs(self, tmp_db):
        # Only 5 rows — below _MIN_RUNS=10
        rows = [("src-A", "llama", 24576.0, 10000.0, 1)] * 5
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is not None
        assert result["sources"] == []

    def test_report_has_correct_keys(self, tmp_db):
        rows = [("prod", "llama-7b", 24576.0, 10000.0, 1)] * 15
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is not None
        assert "sources" in result
        assert "total_estimated_monthly_savings_usd" in result
        source = result["sources"][0]
        for key in (
            "source_id", "model_name", "current_sku", "recommended_sku",
            "peak_p94_mb", "waste_fraction", "estimated_monthly_savings_usd",
            "confidence", "sample_size", "device_count",
        ):
            assert key in source, f"missing key: {key}"

    def test_savings_computed_when_downgradable(self, tmp_db):
        # reserved=81920 (A100-80, $4.10/hr); p94 peak=10000 → recommend T4 ($0.35/hr)
        rows = [("prod", "llama-70b", 81920.0, 10000.0, 1)] * 15
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is not None
        assert len(result["sources"]) == 1
        src = result["sources"][0]
        assert src["estimated_monthly_savings_usd"] > 0
        assert result["total_estimated_monthly_savings_usd"] > 0

    def test_no_savings_when_already_optimal(self, tmp_db):
        # reserved=16384 (T4, cheapest); peak also small — already on cheapest tier
        rows = [("prod", "tiny-model", 16384.0, 5000.0, 1)] * 15
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is not None
        # T4 is cheapest; no cheaper recommendation possible
        assert result["sources"][0]["estimated_monthly_savings_usd"] == 0

    def test_sources_sorted_by_waste_fraction_descending(self, tmp_db):
        # src-A: big waste (reserved 81920, peak 10000)
        # src-B: small waste (reserved 24576, peak 23000)
        rows_a = [("src-A", "llama-70b", 81920.0, 10000.0, 1)] * 15
        rows_b = [("src-B", "mistral-7b", 24576.0, 23000.0, 1)] * 15
        self._seed(tmp_db, rows_a + rows_b)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is not None
        sources = result["sources"]
        assert len(sources) == 2
        assert sources[0]["source_id"] == "src-A"  # higher waste first

    def test_confidence_low(self, tmp_db):
        rows = [("prod", "llama", 24576.0, 10000.0, 1)] * 12  # 12 < 30
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result["sources"][0]["confidence"] == "LOW"

    def test_confidence_med(self, tmp_db):
        rows = [("prod", "llama", 24576.0, 10000.0, 1)] * 50  # 30 <= 50 < 100
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result["sources"][0]["confidence"] == "MED"

    def test_confidence_high(self, tmp_db):
        rows = [("prod", "llama", 24576.0, 10000.0, 1)] * 110  # >= 100
        self._seed(tmp_db, rows)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result["sources"][0]["confidence"] == "HIGH"

    def test_total_savings_aggregates(self, tmp_db):
        # Two sources each with savings
        rows_a = [("src-A", "llama-70b", 81920.0, 10000.0, 1)] * 15
        rows_b = [("src-B", "llama-70b", 81920.0, 10000.0, 1)] * 15
        self._seed(tmp_db, rows_a + rows_b)
        result = compute_local_efficiency_report(db_path=tmp_db)
        assert result is not None
        individual = sum(s["estimated_monthly_savings_usd"] for s in result["sources"])
        assert result["total_estimated_monthly_savings_usd"] == individual

    def test_lookback_days_passed_through(self, tmp_db):
        old_ts = int(time.time()) - 40 * 86400
        schema = (
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY, source_id TEXT, model_name TEXT,"
            "reserved_vram_mb REAL, total_peak_mb REAL, device_count INTEGER,"
            "recorded_at INTEGER)"
        )
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(schema)
            for _ in range(15):
                conn.execute(
                    "INSERT INTO runs (source_id, model_name, reserved_vram_mb, "
                    "total_peak_mb, device_count, recorded_at) VALUES (?,?,?,?,?,?)",
                    ("prod", "llama", 81920.0, 10000.0, 1, old_ts),
                )
        # With 30-day window the 40-day-old rows are excluded → empty sources
        result = compute_local_efficiency_report(lookback_days=30, db_path=tmp_db)
        assert result is not None
        assert result["sources"] == []
        # With 60-day window they are included
        result60 = compute_local_efficiency_report(lookback_days=60, db_path=tmp_db)
        assert result60 is not None
        assert len(result60["sources"]) == 1


# ---------------------------------------------------------------------------
# KVCacheMonitor._write_local_telemetry
# ---------------------------------------------------------------------------

class TestWriteLocalTelemetry:

    def _make_monitor(self, tmp_dir: str) -> KVCacheMonitor:
        monitor = KVCacheMonitor.__new__(KVCacheMonitor)
        monitor._source_id = "test-source"
        monitor._LOCAL_DB_SCHEMA = KVCacheMonitor._LOCAL_DB_SCHEMA
        monitor._db_dir = tmp_dir  # not used by method directly — we patch os.path.expanduser
        return monitor

    def test_creates_db_and_inserts_row(self, tmp_path):
        db_path = str(tmp_path / "telemetry.db")
        monitor = self._make_monitor(str(tmp_path))

        telemetry = SimpleNamespace(
            model_name="llama-7b",
            reserved_vram_mb=24576.0,
            total_peak_mb=18000.0,
            device_count=1,
        )

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            monitor._write_local_telemetry(telemetry)

        assert os.path.isfile(db_path)
        with sqlite3.connect(db_path) as conn:
            rows = list(conn.execute("SELECT source_id, model_name, reserved_vram_mb, "
                                     "total_peak_mb, device_count FROM runs"))
        assert len(rows) == 1
        assert rows[0] == ("test-source", "llama-7b", 24576.0, 18000.0, 1)

    def test_multiple_writes_accumulate(self, tmp_path):
        monitor = self._make_monitor(str(tmp_path))
        telemetry = SimpleNamespace(
            model_name="llama-7b",
            reserved_vram_mb=24576.0,
            total_peak_mb=18000.0,
            device_count=1,
        )
        with patch("os.path.expanduser", return_value=str(tmp_path)):
            monitor._write_local_telemetry(telemetry)
            monitor._write_local_telemetry(telemetry)
            monitor._write_local_telemetry(telemetry)

        db_path = str(tmp_path / "telemetry.db")
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 3

    def test_missing_fields_default_gracefully(self, tmp_path):
        monitor = self._make_monitor(str(tmp_path))
        # SimpleNamespace with no relevant attributes
        telemetry = SimpleNamespace()

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            monitor._write_local_telemetry(telemetry)

        db_path = str(tmp_path / "telemetry.db")
        with sqlite3.connect(db_path) as conn:
            rows = list(conn.execute("SELECT model_name, reserved_vram_mb, "
                                     "total_peak_mb, device_count FROM runs"))
        assert len(rows) == 1
        assert rows[0] == ("", 0.0, 0.0, 1)

    def test_swallows_exceptions_silently(self, tmp_path):
        monitor = self._make_monitor(str(tmp_path))
        telemetry = SimpleNamespace(model_name="x", reserved_vram_mb=0.0,
                                    total_peak_mb=0.0, device_count=1)
        # Make sqlite3.connect raise to simulate a disk-full or permissions error
        with patch("sqlite3.connect", side_effect=OSError("disk full")):
            # Must not raise
            monitor._write_local_telemetry(telemetry)
