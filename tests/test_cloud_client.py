"""Tests for cloud.py fleet-awareness logging (PR 12).

Covers:
  - download_policy() calls logger.info when fleet_contributors > 1
  - download_policy() does NOT call logger.info when fleet_contributors == 0
  - download_policy() does NOT call logger.info when fleet_contributors == 1 (solo)
  - download_policy() includes contributor count in logger.debug message
  - download_policy() returns data intact regardless of fleet_contributors value
  - _merge_cloud_policy() logs cold-fleet debug message when fleet_contributors == 0
  - _merge_cloud_policy() does NOT log cold-fleet message when fleet_contributors > 0
  - _merge_cloud_policy() includes contributor count in final merge debug message
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from memory_guard.bandit import BanditPolicy, _merge_cloud_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy_response(
    q_table: Optional[dict] = None,
    num_updates: int = 5,
    fleet_contributors: int = 0,
    fleet_last_rebuilt: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "q_table": q_table or {"state1": {"action1": 1.0}},
        "num_updates": num_updates,
        "fleet_contributors": fleet_contributors,
        "fleet_last_rebuilt": fleet_last_rebuilt or "2026-04-11T00:00:00.000Z",
    }


def _mock_httpx_get(data: dict, status_code: int = 200):
    """Return a mock httpx module whose get() returns a fake response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()

    httpx_mock = MagicMock()
    httpx_mock.get.return_value = resp
    return httpx_mock


# ---------------------------------------------------------------------------
# cloud.download_policy — fleet_contributors logging
# ---------------------------------------------------------------------------

class TestDownloadPolicyFleetLogging:
    def _call_with_response(self, response_data: dict):
        """Patch httpx and MEMGUARD_API_KEY, call download_policy, return data."""
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(response_data)
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                return cloud.download_policy()

    def test_returns_data_with_zero_contributors(self):
        data = _make_policy_response(fleet_contributors=0)
        result = self._call_with_response(data)
        assert result is not None
        assert result["fleet_contributors"] == 0

    def test_returns_data_with_multiple_contributors(self):
        data = _make_policy_response(fleet_contributors=42)
        result = self._call_with_response(data)
        assert result is not None
        assert result["fleet_contributors"] == 42

    def test_info_log_fires_when_contributors_gt_1(self, caplog):
        data = _make_policy_response(fleet_contributors=5)
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(data)
        with caplog.at_level(logging.INFO, logger="memory_guard.cloud"):
            with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
                with patch.dict("sys.modules", {"httpx": httpx_mock}):
                    cloud.download_policy()
        assert any("5 contributors" in r.message for r in caplog.records)

    def test_no_info_log_when_contributors_is_zero(self, caplog):
        data = _make_policy_response(fleet_contributors=0)
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(data)
        with caplog.at_level(logging.INFO, logger="memory_guard.cloud"):
            with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
                with patch.dict("sys.modules", {"httpx": httpx_mock}):
                    cloud.download_policy()
        # No INFO-level fleet message expected
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any("Fleet policy active" in m for m in info_msgs)

    def test_no_info_log_when_solo_contributor(self, caplog):
        """Single contributor (the user themselves) — no fleet info log yet."""
        data = _make_policy_response(fleet_contributors=1)
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(data)
        with caplog.at_level(logging.INFO, logger="memory_guard.cloud"):
            with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
                with patch.dict("sys.modules", {"httpx": httpx_mock}):
                    cloud.download_policy()
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any("Fleet policy active" in m for m in info_msgs)

    def test_debug_log_includes_contributor_count(self, caplog):
        data = _make_policy_response(fleet_contributors=3)
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(data)
        with caplog.at_level(logging.DEBUG, logger="memory_guard.cloud"):
            with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
                with patch.dict("sys.modules", {"httpx": httpx_mock}):
                    cloud.download_policy()
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("3 fleet contributor" in m for m in debug_msgs)

    def test_debug_log_singular_contributor_grammar(self, caplog):
        """'1 fleet contributor' not '1 fleet contributors'."""
        data = _make_policy_response(fleet_contributors=1)
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(data)
        with caplog.at_level(logging.DEBUG, logger="memory_guard.cloud"):
            with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
                with patch.dict("sys.modules", {"httpx": httpx_mock}):
                    cloud.download_policy()
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("1 fleet contributor" in m and "contributors" not in m
                   for m in debug_msgs)

    def test_info_log_includes_last_rebuilt(self, caplog):
        ts = "2026-04-11T12:00:00.000Z"
        data = _make_policy_response(fleet_contributors=10, fleet_last_rebuilt=ts)
        from memory_guard import cloud
        httpx_mock = _mock_httpx_get(data)
        with caplog.at_level(logging.INFO, logger="memory_guard.cloud"):
            with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
                with patch.dict("sys.modules", {"httpx": httpx_mock}):
                    cloud.download_policy()
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any(ts in m for m in info_msgs)

    def test_returns_none_when_no_api_key(self):
        from memory_guard import cloud
        with patch.dict("os.environ", {}, clear=True):
            result = cloud.download_policy()
        assert result is None

    def test_returns_none_on_404(self):
        from memory_guard import cloud
        resp = MagicMock()
        resp.status_code = 404
        httpx_mock = MagicMock()
        httpx_mock.get.return_value = resp
        with patch.dict("os.environ", {"MEMGUARD_API_KEY": "test-key-abcdefgh"}):
            with patch.dict("sys.modules", {"httpx": httpx_mock}):
                result = cloud.download_policy()
        assert result is None


# ---------------------------------------------------------------------------
# bandit._merge_cloud_policy — cold-fleet logging
# ---------------------------------------------------------------------------

class TestMergeCloudPolicyFleetLogging:
    def _policy_with_cloud(self, cloud_data: dict) -> BanditPolicy:
        """Return a fresh BanditPolicy after _merge_cloud_policy with fake cloud data."""
        policy = BanditPolicy()
        with patch("memory_guard.cloud.download_policy", return_value=cloud_data):
            _merge_cloud_policy(policy)
        return policy

    def test_cold_fleet_debug_logged_when_contributors_zero(self, caplog):
        data = _make_policy_response(fleet_contributors=0)
        with caplog.at_level(logging.DEBUG, logger="memory_guard.bandit"):
            self._policy_with_cloud(data)
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("cold start" in m.lower() or "first contributor" in m.lower()
                   for m in debug_msgs)

    def test_cold_fleet_message_not_logged_when_contributors_gt_0(self, caplog):
        data = _make_policy_response(fleet_contributors=7)
        with caplog.at_level(logging.DEBUG, logger="memory_guard.bandit"):
            self._policy_with_cloud(data)
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert not any("first contributor" in m for m in debug_msgs)

    def test_merge_debug_log_includes_contributor_count(self, caplog):
        data = _make_policy_response(fleet_contributors=4)
        with caplog.at_level(logging.DEBUG, logger="memory_guard.bandit"):
            self._policy_with_cloud(data)
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("4 fleet contributor" in m for m in debug_msgs)

    def test_merge_debug_singular_grammar(self, caplog):
        """'1 fleet contributor' not '1 fleet contributors'."""
        data = _make_policy_response(fleet_contributors=1)
        with caplog.at_level(logging.DEBUG, logger="memory_guard.bandit"):
            self._policy_with_cloud(data)
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("1 fleet contributor" in m and "contributors" not in m
                   for m in debug_msgs)

    def test_merge_succeeds_with_missing_fleet_contributors_key(self, caplog):
        """Policy response without fleet_contributors key — defaults to 0, no crash."""
        data = {
            "q_table": {"s1": {"a1": 1.0}},
            "num_updates": 3,
            # fleet_contributors intentionally absent
        }
        policy = self._policy_with_cloud(data)
        assert policy.num_states >= 0  # no exception raised

    def test_no_merge_when_download_returns_none(self):
        policy = BanditPolicy()
        with patch("memory_guard.cloud.download_policy", return_value=None):
            _merge_cloud_policy(policy)
        assert policy.num_states == 0  # nothing merged
