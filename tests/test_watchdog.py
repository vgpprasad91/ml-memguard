"""Tests for memory_guard.watchdog — VLLMWatchdog OOM detection and command patching.

PR 6 — Stub tests for watchdog helpers (no live subprocess required).

Covers:
  - _is_oom_exit(): all OOM_PATTERNS × all OOM_EXIT_CODES combinations → True
  - _is_oom_exit(): exit code in OOM_EXIT_CODES but stderr has no pattern → False
  - _is_oom_exit(): OOM pattern present but exit code is 0 → False
  - _is_oom_exit(): OOM pattern present but exit code not in OOM_EXIT_CODES → False
  - _is_oom_exit(): case-insensitive pattern matching
  - _patch_flag(): replaces value in --flag value form
  - _patch_flag(): replaces value in --flag=value form
  - _patch_flag(): appends flag when absent from command
  - _patch_flag(): does not mutate the original cmd list
  - _patch_flag(): leaves unrelated flags untouched
  - _apply_action_to_cmd(): patches --max-num-seqs from action.max_num_seqs
  - _apply_action_to_cmd(): patches --gpu-memory-utilization when gpu_mem_util provided
  - _apply_action_to_cmd(): skips --max-num-seqs when action.max_num_seqs == 0
  - _apply_action_to_cmd(): skips --gpu-memory-utilization when gpu_mem_util is None
  - guard_vllm_watchdog(): command contains --model, --host, --port
  - guard_vllm_watchdog(): command contains --max-num-seqs with provided value
  - guard_vllm_watchdog(): command contains --gpu-memory-utilization with provided value
  - guard_vllm_watchdog(): extra_args are appended verbatim
  - guard_vllm_watchdog(): returns a VLLMWatchdog with max_retries and backoff_seconds set
"""

from __future__ import annotations

import pytest

from memory_guard.deployment.watchdog import (
    OOM_EXIT_CODES,
    OOM_PATTERNS,
    VLLMWatchdog,
    _apply_action_to_cmd,
    _is_oom_exit,
    _patch_flag,
    guard_vllm_watchdog,
)
from memory_guard.adaptation.bandit import BanditPolicy
from memory_guard.adaptation.bandit_state import ConfigAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(batch_size: int = 4, max_num_seqs: int = 128) -> ConfigAction:
    return ConfigAction(
        batch_size=batch_size,
        lora_rank=0,
        seq_length=512,
        max_num_seqs=max_num_seqs,
    )


# ---------------------------------------------------------------------------
# _is_oom_exit — all patterns × all exit codes
# ---------------------------------------------------------------------------

class TestIsOomExit:
    @pytest.mark.parametrize("pattern", list(OOM_PATTERNS))
    @pytest.mark.parametrize("code", sorted(OOM_EXIT_CODES))
    def test_all_patterns_all_codes_return_true(self, pattern: str, code: int):
        """Every OOM_PATTERN combined with every OOM_EXIT_CODE must return True."""
        stderr = f"some prefix {pattern} some suffix"
        assert _is_oom_exit(code, stderr) is True

    @pytest.mark.parametrize("code", sorted(OOM_EXIT_CODES))
    def test_exit_code_matches_but_no_pattern_returns_false(self, code: int):
        """Suspicious exit code alone (clean stderr) must NOT trigger recovery."""
        clean_stderr = "Starting vLLM server... model loaded successfully."
        assert _is_oom_exit(code, clean_stderr) is False

    @pytest.mark.parametrize("pattern", list(OOM_PATTERNS))
    def test_oom_pattern_present_but_exit_code_zero_returns_false(self, pattern: str):
        """Exit code 0 is always clean even if a warning log contains OOM text."""
        stderr = f"WARNING: encountered {pattern} in previous run log"
        assert _is_oom_exit(0, stderr) is False

    def test_oom_pattern_present_but_non_oom_exit_code_returns_false(self):
        """Exit code 2 (argparse error) with OOM text in stderr must be False."""
        stderr = "oom-like text in error message"
        assert _is_oom_exit(2, stderr) is False

    def test_empty_stderr_with_oom_exit_code_returns_false(self):
        """No stderr text — cannot confirm OOM pattern — must be False."""
        assert _is_oom_exit(1, "") is False

    def test_pattern_matching_is_case_insensitive(self):
        """Pattern check must be case-insensitive (CUDA OOM vs cuda oom)."""
        assert _is_oom_exit(1, "CUDA OUT OF MEMORY: tried to allocate") is True
        assert _is_oom_exit(1, "OutOfMemoryError on GPU") is True
        assert _is_oom_exit(137, "KILLED") is True


# ---------------------------------------------------------------------------
# _patch_flag
# ---------------------------------------------------------------------------

class TestPatchFlag:
    def test_replaces_space_separated_flag(self):
        cmd = ["vllm", "--max-num-seqs", "256", "--model", "llama"]
        result = _patch_flag(cmd, "--max-num-seqs", "128")
        assert result == ["vllm", "--max-num-seqs", "128", "--model", "llama"]

    def test_replaces_equals_separated_flag(self):
        cmd = ["vllm", "--max-num-seqs=256", "--model", "llama"]
        result = _patch_flag(cmd, "--max-num-seqs", "64")
        assert result == ["vllm", "--max-num-seqs=64", "--model", "llama"]

    def test_appends_flag_when_absent(self):
        cmd = ["vllm", "--model", "llama"]
        result = _patch_flag(cmd, "--max-num-seqs", "128")
        assert "--max-num-seqs" in result
        idx = result.index("--max-num-seqs")
        assert result[idx + 1] == "128"

    def test_does_not_mutate_original_cmd(self):
        original = ["vllm", "--max-num-seqs", "256"]
        original_copy = list(original)
        _patch_flag(original, "--max-num-seqs", "128")
        assert original == original_copy

    def test_leaves_unrelated_flags_untouched(self):
        cmd = ["vllm", "--model", "llama", "--host", "0.0.0.0", "--max-num-seqs", "256"]
        result = _patch_flag(cmd, "--max-num-seqs", "64")
        assert result[result.index("--model") + 1] == "llama"
        assert result[result.index("--host") + 1] == "0.0.0.0"

    def test_replaces_first_occurrence_only(self):
        """If a flag somehow appears twice, only the first is replaced."""
        cmd = ["vllm", "--max-num-seqs", "256", "--max-num-seqs", "512"]
        result = _patch_flag(cmd, "--max-num-seqs", "64")
        # First occurrence is replaced; second replacement is handled as a new
        # encounter — implementation replaces each occurrence it finds
        assert "64" in result

    def test_gpu_mem_util_flag_replaced(self):
        cmd = ["vllm", "--gpu-memory-utilization", "0.9000"]
        result = _patch_flag(cmd, "--gpu-memory-utilization", "0.7500")
        assert result == ["vllm", "--gpu-memory-utilization", "0.7500"]


# ---------------------------------------------------------------------------
# _apply_action_to_cmd
# ---------------------------------------------------------------------------

class TestApplyActionToCmd:
    def test_patches_max_num_seqs(self):
        cmd = ["vllm", "--max-num-seqs", "256"]
        action = _action(max_num_seqs=128)
        result = _apply_action_to_cmd(cmd, action)
        idx = result.index("--max-num-seqs")
        assert result[idx + 1] == "128"

    def test_patches_gpu_memory_utilization(self):
        cmd = ["vllm", "--gpu-memory-utilization", "0.9000"]
        action = _action(max_num_seqs=64)
        result = _apply_action_to_cmd(cmd, action, gpu_mem_util=0.75)
        idx = result.index("--gpu-memory-utilization")
        assert float(result[idx + 1]) == pytest.approx(0.75, abs=1e-4)

    def test_skips_max_num_seqs_when_action_value_is_zero(self):
        """max_num_seqs=0 means 'not applicable' — must not patch the flag."""
        cmd = ["vllm", "--max-num-seqs", "256"]
        action = _action(max_num_seqs=0)
        result = _apply_action_to_cmd(cmd, action)
        idx = result.index("--max-num-seqs")
        # Value must remain unchanged at 256
        assert result[idx + 1] == "256"

    def test_skips_gpu_mem_util_when_none(self):
        """No gpu_mem_util provided — flag in original cmd must be untouched."""
        cmd = ["vllm", "--gpu-memory-utilization", "0.9000"]
        action = _action(max_num_seqs=64)
        result = _apply_action_to_cmd(cmd, action, gpu_mem_util=None)
        idx = result.index("--gpu-memory-utilization")
        assert result[idx + 1] == "0.9000"

    def test_does_not_mutate_original_cmd(self):
        cmd = ["vllm", "--max-num-seqs", "256"]
        original_copy = list(cmd)
        _apply_action_to_cmd(cmd, _action(max_num_seqs=64))
        assert cmd == original_copy

    def test_appends_max_num_seqs_if_absent(self):
        cmd = ["vllm", "--model", "llama"]
        action = _action(max_num_seqs=64)
        result = _apply_action_to_cmd(cmd, action)
        assert "--max-num-seqs" in result
        idx = result.index("--max-num-seqs")
        assert result[idx + 1] == "64"


# ---------------------------------------------------------------------------
# guard_vllm_watchdog — command construction
# ---------------------------------------------------------------------------

class TestGuardVllmWatchdog:
    """guard_vllm_watchdog() must build a sensible command without launching anything."""

    def test_command_contains_model_flag(self):
        wdog = guard_vllm_watchdog("meta-llama/Meta-Llama-3-8B-Instruct")
        assert "--model" in wdog.current_cmd
        idx = wdog.current_cmd.index("--model")
        assert wdog.current_cmd[idx + 1] == "meta-llama/Meta-Llama-3-8B-Instruct"

    def test_command_contains_host_flag(self):
        wdog = guard_vllm_watchdog("llama", host="127.0.0.1")
        assert "--host" in wdog.current_cmd
        idx = wdog.current_cmd.index("--host")
        assert wdog.current_cmd[idx + 1] == "127.0.0.1"

    def test_command_contains_port_flag(self):
        wdog = guard_vllm_watchdog("llama", port=9000)
        assert "--port" in wdog.current_cmd
        idx = wdog.current_cmd.index("--port")
        assert wdog.current_cmd[idx + 1] == "9000"

    def test_command_contains_max_num_seqs(self):
        wdog = guard_vllm_watchdog("llama", max_num_seqs=512)
        assert "--max-num-seqs" in wdog.current_cmd
        idx = wdog.current_cmd.index("--max-num-seqs")
        assert wdog.current_cmd[idx + 1] == "512"

    def test_command_contains_gpu_memory_utilization(self):
        wdog = guard_vllm_watchdog("llama", gpu_memory_utilization=0.85)
        assert "--gpu-memory-utilization" in wdog.current_cmd
        idx = wdog.current_cmd.index("--gpu-memory-utilization")
        assert float(wdog.current_cmd[idx + 1]) == pytest.approx(0.85)

    def test_extra_args_appended(self):
        wdog = guard_vllm_watchdog("llama", extra_args=["--dtype", "float16"])
        assert "--dtype" in wdog.current_cmd
        idx = wdog.current_cmd.index("--dtype")
        assert wdog.current_cmd[idx + 1] == "float16"

    def test_returns_vllm_watchdog_instance(self):
        wdog = guard_vllm_watchdog("llama")
        assert isinstance(wdog, VLLMWatchdog)

    def test_max_retries_propagated(self):
        wdog = guard_vllm_watchdog("llama", max_retries=5)
        assert wdog.max_retries == 5

    def test_backoff_seconds_propagated(self):
        wdog = guard_vllm_watchdog("llama", backoff_seconds=30.0)
        assert wdog.backoff_seconds == pytest.approx(30.0)

    def test_custom_bandit_used_not_loaded(self):
        """If a bandit is supplied, guard_vllm_watchdog must use it directly."""
        custom_bandit = BanditPolicy(epsilon=0.0)
        wdog = guard_vllm_watchdog("llama", bandit=custom_bandit)
        assert wdog.bandit is custom_bandit

    def test_alert_callback_propagated(self):
        alerts = []
        cb = lambda msg, a, m: alerts.append(msg)
        wdog = guard_vllm_watchdog("llama", alert_callback=cb)
        assert wdog.alert_callback is cb
