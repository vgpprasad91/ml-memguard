"""Shared pytest configuration for the memory-guard test suite.

v0.5.0 baseline (PR 8 audit — 2026-04-11)
------------------------------------------
Full suite: 591 passed, 1 skipped, 0 failed across 16 test files.

Skipped (intentional):
  tests/test_adapters_smoke.py — guarded by pytest.importorskip("torch") /
  pytest.importorskip("transformers").  Downloads distilgpt2; runs only in
  environments with ml-memguard[hf] installed and network access.

Adapter tests (huggingface, unsloth, vllm, sglang):
  All pass without optional deps because they use unittest.mock.MagicMock to
  simulate framework objects.  No importorskip guards are needed in those
  files — they are intentionally dependency-free.

Core tests (no optional deps — regressions here are release blockers):
  test_guard.py                  108 passed
  test_bandit.py                  37 passed
  test_bandit_state.py            63 passed
  test_reward.py                  26 passed
  test_kv_cache_monitor.py        26 passed
  test_inference_estimator.py     35 passed
  test_bandit_integration.py      37 passed
  test_bandit_recommend.py        22 passed   ← v0.5.0 PR 5
  test_watchdog.py                62 passed   ← v0.5.0 PR 6
  test_kvcache_critical.py        20 passed   ← v0.5.0 PR 7
"""
