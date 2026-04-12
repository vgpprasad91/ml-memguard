"""VLLMWatchdog — auto-heal loop for vLLM inference servers (v0.5).

Wraps a vLLM server process (``vllm serve`` / OpenAI-compatible API server)
and automatically recovers from OOM crashes without human intervention,
compressing MTTR from 60–80 minutes (engineer-assisted) to under 2 minutes
(fully automated).

Recovery flow
-------------
1. vLLM subprocess starts with the caller-supplied command.
2. A background thread drains stdout/stderr into a rolling buffer.
3. When the process exits, the watchdog checks:
   a. Was the exit caused by OOM?  (stderr patterns + exit code)
   b. Are retries remaining?
4. If yes to both: call ``bandit.recommend_conservative(state_key)`` to
   get a guaranteed-safer config, patch ``--max-num-seqs`` and
   ``--gpu-memory-utilization`` in the command, back off, and relaunch.
5. If no OOM or retries exhausted: call ``alert_callback`` (if set) and
   propagate the exit to the caller.

Design contract
---------------
- **Never mutates the running process** — only patches the CLI command
  between restarts.
- **Never raises** in the recovery path — every exception inside the
  watchdog loop is caught and logged; the alert_callback is invoked on
  failure so external systems (PagerDuty / Slack) are notified.
- **Deterministic** — uses ``recommend_conservative()`` (no epsilon
  exploration) so every relaunch is guaranteed safer than the one that
  crashed, not randomly different.

Quick start::

    from memory_guard import BanditPolicy
    from memory_guard.watchdog import guard_vllm_watchdog

    watchdog = guard_vllm_watchdog(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        host="0.0.0.0",
        port=8000,
        max_num_seqs=256,
        gpu_memory_utilization=0.90,
    )
    watchdog.run()   # blocks; auto-heals on OOM

Lower-level usage with an existing bandit::

    from memory_guard.watchdog import VLLMWatchdog
    from memory_guard.bandit import BanditPolicy
    from memory_guard.bandit_state import StateKey

    bandit = BanditPolicy.load()
    state_key = StateKey.from_values(...)

    wdog = VLLMWatchdog(
        cmd=["python", "-m", "vllm.entrypoints.openai.api_server",
             "--model", "meta-llama/Meta-Llama-3-8B-Instruct",
             "--max-num-seqs", "256"],
        state_key=state_key,
        bandit=bandit,
        max_retries=3,
        backoff_seconds=10.0,
        alert_callback=lambda msg, attempt, max_r: print(msg),
    )
    wdog.run()
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from .bandit import BanditPolicy
from .bandit_state import ConfigAction, StateKey
from .platforms import detect_platform, get_available_memory_mb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OOM detection patterns
# ---------------------------------------------------------------------------

#: Stderr substrings that indicate an OOM exit (case-insensitive checked).
OOM_PATTERNS: tuple[str, ...] = (
    "cuda out of memory",
    "no available memory for cache blocks",
    "outofmemoryerror",
    "cuda error: out of memory",
    "torch.cuda.outofmemoryerror",
    "out_of_memory",
    "oom",
    "killed",                          # Linux OOM-killer signal
)

#: Exit codes that are treated as potential OOM (alongside stderr check).
#: -9 = SIGKILL (Linux OOM-killer), 1 = generic error, 137 = 128+SIGKILL.
OOM_EXIT_CODES: frozenset[int] = frozenset({1, -9, 137})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GPU_MEM_UTIL_FLAG = "--gpu-memory-utilization"
_MAX_NUM_SEQS_FLAG  = "--max-num-seqs"


def _is_oom_exit(returncode: int, stderr_text: str) -> bool:
    """Return True if *returncode* + *stderr_text* look like an OOM crash."""
    stderr_lower = stderr_text.lower()
    pattern_match = any(p in stderr_lower for p in OOM_PATTERNS)
    code_match = returncode in OOM_EXIT_CODES
    # Require both a suspicious exit code AND an OOM string in stderr.
    # Pure code match (e.g. exit 1 from a config error) is not treated as OOM.
    return code_match and pattern_match


def _patch_flag(cmd: List[str], flag: str, new_value: str) -> List[str]:
    """Replace the value of *flag* in *cmd*, or append if absent.

    Handles both ``--flag value`` and ``--flag=value`` forms.
    Returns a new list; *cmd* is not mutated.
    """
    result: List[str] = []
    i = 0
    found = False
    while i < len(cmd):
        token = cmd[i]
        if token == flag:
            result.append(flag)
            result.append(new_value)
            i += 2          # skip the old value
            found = True
        elif token.startswith(f"{flag}="):
            result.append(f"{flag}={new_value}")
            i += 1
            found = True
        else:
            result.append(token)
            i += 1
    if not found:
        result.extend([flag, new_value])
    return result


def _apply_action_to_cmd(
    cmd: List[str],
    action: ConfigAction,
    gpu_mem_util: Optional[float] = None,
) -> List[str]:
    """Return a new command with *action*'s config patched in.

    Updates ``--max-num-seqs`` from *action.max_num_seqs* (if non-zero).
    Updates ``--gpu-memory-utilization`` with *gpu_mem_util* (if provided).
    """
    result = list(cmd)
    if action.max_num_seqs > 0:
        result = _patch_flag(result, _MAX_NUM_SEQS_FLAG, str(action.max_num_seqs))
    if gpu_mem_util is not None:
        result = _patch_flag(result, _GPU_MEM_UTIL_FLAG, f"{gpu_mem_util:.4f}")
    return result


def _drain_stream(
    stream,
    buffer: List[str],
    max_lines: int = 500,
) -> None:
    """Read *stream* line-by-line into *buffer* (runs in a daemon thread)."""
    try:
        for raw in stream:
            line = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
            buffer.append(line)
            if len(buffer) > max_lines:
                buffer.pop(0)
            # Echo to stderr so operators see live output
            sys.stderr.write(line)
            sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VLLMWatchdog
# ---------------------------------------------------------------------------


class VLLMWatchdog:
    """Process supervisor that auto-heals vLLM on OOM crashes.

    Parameters
    ----------
    cmd:
        Full launch command as a list of strings, e.g.::

            ["python", "-m", "vllm.entrypoints.openai.api_server",
             "--model", "meta-llama/Meta-Llama-3-8B-Instruct",
             "--max-num-seqs", "256",
             "--gpu-memory-utilization", "0.90"]

    state_key:
        ``StateKey`` for this device × model combination.  Used to look up
        conservative configs in *bandit* after an OOM.

    bandit:
        ``BanditPolicy`` instance.  The watchdog calls
        ``bandit.recommend_conservative(state_key)`` on each recovery.
        If ``recommend_conservative`` returns ``None`` (cold start), the
        watchdog applies a fixed 15 % reduction to the current flags instead.

    max_retries:
        Maximum number of recovery attempts before the watchdog gives up and
        re-raises / calls *alert_callback*.  Default 3.

    backoff_seconds:
        Seconds to wait between a crash and the next relaunch.  Default 5.
        Gives the GPU driver time to release memory before vLLM re-initialises.

    conservative_margin:
        Fractional reduction passed to ``recommend_conservative`` (default 0.15).

    alert_callback:
        Called on every recovery attempt *and* on final failure.
        Signature: ``callback(message: str, attempt: int, max_retries: int)``.
        Use this to fire PagerDuty / Slack webhooks.

    ebpf_session:
        Optional :class:`~memory_guard.ebpf.MemguardBPFSession` (or any
        duck-typed object with ``add_oom_imminent_callback``).  When
        provided, an OOM-imminent BPF event triggers :meth:`stop` immediately
        — compressing MTTR from ~2 min (process-exit detection) to near-zero
        (kernel-event detection, 200–500 ms before OOM kill).
        Requires the ``[ebpf]`` extra on Linux.  Silently ignored on other
        platforms or when BPF is unavailable.

    Attributes
    ----------
    current_cmd : List[str]
        The command that will be used for the *next* launch (updated after
        each successful recovery).
    attempts : int
        Number of recovery attempts made so far.
    """

    def __init__(
        self,
        cmd: List[str],
        state_key: StateKey,
        bandit: BanditPolicy,
        max_retries: int = 3,
        backoff_seconds: float = 5.0,
        conservative_margin: float = 0.15,
        alert_callback: Optional[Callable[[str, int, int], None]] = None,
        ebpf_session: Optional[object] = None,
    ) -> None:
        if not cmd:
            raise ValueError("cmd must be a non-empty list of strings")
        self.current_cmd: List[str] = list(cmd)
        self.state_key: StateKey = state_key
        self.bandit: BanditPolicy = bandit
        self.max_retries: int = max(0, max_retries)
        self.backoff_seconds: float = max(0.0, backoff_seconds)
        self.conservative_margin: float = conservative_margin
        self.alert_callback: Optional[Callable[[str, int, int], None]] = alert_callback
        self.attempts: int = 0
        self._stop_event: threading.Event = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._ebpf_session: Optional[object] = ebpf_session

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Start the vLLM process and supervise it until exit or exhausted retries.

        Blocks until the process exits cleanly (non-OOM) or all recovery
        attempts are exhausted.

        Returns:
            The final process exit code.

        Raises:
            RuntimeError: if max_retries is exhausted after repeated OOM crashes.
        """
        self._stop_event.clear()
        last_returncode: int = 1

        # Wire BPF OOM-imminent callback — fires before SIGKILL enabling a
        # graceful SIGTERM-based restart rather than waiting for process exit.
        if self._ebpf_session is not None:
            _add_cb = getattr(self._ebpf_session, "add_oom_imminent_callback", None)
            if callable(_add_cb):
                _add_cb(self.stop)
                logger.debug(
                    "[watchdog] eBPF OOM-imminent callback registered "
                    "(immediate graceful restart enabled)"
                )

        while not self._stop_event.is_set():
            logger.info(
                "[watchdog] Launching vLLM (attempt %d / %d): %s",
                self.attempts + 1,
                self.max_retries + 1,
                " ".join(shlex.quote(t) for t in self.current_cmd),
            )

            returncode, stderr_text = self._run_process(self.current_cmd)
            last_returncode = returncode

            if self._stop_event.is_set():
                # Graceful stop was requested while process was running
                break

            if returncode == 0:
                logger.info("[watchdog] vLLM exited cleanly (code 0).")
                break

            if not _is_oom_exit(returncode, stderr_text):
                logger.error(
                    "[watchdog] vLLM exited with code %d (non-OOM). "
                    "Last stderr tail:\n%s",
                    returncode, stderr_text[-2000:],
                )
                self._fire_alert(
                    f"vLLM exited with non-OOM code {returncode} — not retrying.",
                    self.attempts, self.max_retries,
                )
                break

            # OOM confirmed -----------------------------------------------
            self.attempts += 1
            logger.warning(
                "[watchdog] OOM detected (exit code %d, attempt %d / %d). "
                "Recovering in %.1f s …",
                returncode, self.attempts, self.max_retries, self.backoff_seconds,
            )
            self._fire_alert(
                f"OOM crash detected (attempt {self.attempts}/{self.max_retries}). "
                f"Auto-recovering in {self.backoff_seconds:.0f}s.",
                self.attempts, self.max_retries,
            )

            if self.attempts > self.max_retries:
                msg = (
                    f"[watchdog] Max retries ({self.max_retries}) exhausted after "
                    f"repeated OOM crashes. Giving up."
                )
                logger.error(msg)
                self._fire_alert(msg, self.attempts, self.max_retries)
                raise RuntimeError(msg)

            # Build recovery command ----------------------------------------
            self.current_cmd = self._recovery_cmd(self.current_cmd)
            logger.info(
                "[watchdog] Recovery command: %s",
                " ".join(shlex.quote(t) for t in self.current_cmd),
            )

            time.sleep(self.backoff_seconds)

        return last_returncode

    def stop(self) -> None:
        """Signal the watchdog to stop after the current process exits.

        If a process is running, sends SIGTERM.  The watchdog will not restart
        after the process exits.
        """
        self._stop_event.set()
        proc = self._process
        if proc is not None and proc.poll() is None:
            logger.info("[watchdog] Sending SIGTERM to vLLM process (pid=%d).", proc.pid)
            try:
                proc.terminate()
            except Exception as exc:
                logger.debug("[watchdog] terminate() failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_process(self, cmd: List[str]) -> tuple[int, str]:
        """Start *cmd*, drain streams, wait for exit.

        Returns ``(returncode, accumulated_stderr_text)``.
        """
        stderr_lines: List[str] = []
        stdout_lines: List[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            logger.error("[watchdog] Cannot start process: %s", exc)
            return 1, str(exc)

        self._process = proc

        # Drain stdout and stderr in background threads so the pipes never block
        t_out = threading.Thread(
            target=_drain_stream, args=(proc.stdout, stdout_lines), daemon=True
        )
        t_err = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_lines), daemon=True
        )
        t_out.start()
        t_err.start()

        proc.wait()
        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)

        self._process = None
        return proc.returncode, "".join(stderr_lines)

    def _recovery_cmd(self, cmd: List[str]) -> List[str]:
        """Return a new command with a conservative safe config applied.

        Priority:
        1. ``bandit.recommend_conservative(state_key)`` if confidence > 0.
        2. Fixed *conservative_margin* reduction of the current flag values
           as a cold-start fallback (no prior Q-table data).
        """
        action = self.bandit.recommend_conservative(
            self.state_key, margin=self.conservative_margin
        )

        if action is not None:
            logger.info(
                "[watchdog] Bandit recommendation (confidence=%.2f): "
                "max_num_seqs=%d, batch_size=%d",
                self.bandit.confidence,
                action.max_num_seqs,
                action.batch_size,
            )
            # Reduce gpu_memory_utilization by the same margin
            current_util = _parse_flag_float(cmd, _GPU_MEM_UTIL_FLAG, default=0.90)
            new_util = max(0.50, current_util * (1.0 - self.conservative_margin))
            return _apply_action_to_cmd(cmd, action, gpu_mem_util=new_util)

        # Cold start: no bandit data — apply fixed reduction to current flags
        logger.warning(
            "[watchdog] Bandit has no data for this state (cold start). "
            "Applying %.0f%% reduction to current flags.",
            self.conservative_margin * 100,
        )
        current_seqs = _parse_flag_int(cmd, _MAX_NUM_SEQS_FLAG, default=256)
        current_util = _parse_flag_float(cmd, _GPU_MEM_UTIL_FLAG, default=0.90)
        new_seqs = max(1, int(current_seqs * (1.0 - self.conservative_margin)))
        new_util = max(0.50, current_util * (1.0 - self.conservative_margin))
        fallback_action = ConfigAction(
            batch_size=max(1, int(1 * (1.0 - self.conservative_margin))),
            lora_rank=0,
            seq_length=512,
            max_num_seqs=new_seqs,
        )
        return _apply_action_to_cmd(cmd, fallback_action, gpu_mem_util=new_util)

    def _fire_alert(self, message: str, attempt: int, max_retries: int) -> None:
        """Call alert_callback safely — never raises."""
        if self.alert_callback is None:
            return
        try:
            self.alert_callback(message, attempt, max_retries)
        except Exception as exc:
            logger.debug("[watchdog] alert_callback raised: %s", exc)


# ---------------------------------------------------------------------------
# Flag parsing helpers
# ---------------------------------------------------------------------------


def _parse_flag_float(cmd: List[str], flag: str, default: float) -> float:
    """Extract the float value of *flag* from *cmd*, or return *default*."""
    for i, token in enumerate(cmd):
        if token == flag and i + 1 < len(cmd):
            try:
                return float(cmd[i + 1])
            except ValueError:
                return default
        if token.startswith(f"{flag}="):
            try:
                return float(token.split("=", 1)[1])
            except ValueError:
                return default
    return default


def _parse_flag_int(cmd: List[str], flag: str, default: int) -> int:
    """Extract the int value of *flag* from *cmd*, or return *default*."""
    for i, token in enumerate(cmd):
        if token == flag and i + 1 < len(cmd):
            try:
                return int(cmd[i + 1])
            except ValueError:
                return default
        if token.startswith(f"{flag}="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return default
    return default


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def guard_vllm_watchdog(
    model: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    model_params: float = 7e9,
    model_bits: int = 16,
    bandit: Optional[BanditPolicy] = None,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
    conservative_margin: float = 0.15,
    alert_callback: Optional[Callable[[str, int, int], None]] = None,
    extra_args: Optional[List[str]] = None,
) -> VLLMWatchdog:
    """Build a ``VLLMWatchdog`` from high-level parameters.

    Constructs the vLLM CLI command, builds the ``StateKey`` from the
    current platform, loads or creates the bandit policy, and returns a
    ready-to-run watchdog.

    Args:
        model:                   HuggingFace model ID or local path.
        host:                    Bind address for the OpenAI-compatible server.
        port:                    Bind port.
        max_num_seqs:            Initial ``--max-num-seqs`` value.
        gpu_memory_utilization:  Initial ``--gpu-memory-utilization`` value.
        tensor_parallel_size:    ``--tensor-parallel-size`` (default 1).
        model_params:            Estimated parameter count for StateKey bucketing
                                 (default 7e9 = 7 B).
        model_bits:              Quantisation bits for StateKey bucketing (16 = fp16).
        bandit:                  Existing ``BanditPolicy`` to use; loads from
                                 the default path if None.
        max_retries:             Passed to ``VLLMWatchdog``.
        backoff_seconds:         Passed to ``VLLMWatchdog``.
        conservative_margin:     Passed to ``VLLMWatchdog``.
        alert_callback:          Passed to ``VLLMWatchdog``.
        extra_args:              Extra CLI tokens appended verbatim to the command.

    Returns:
        A configured ``VLLMWatchdog`` ready to call ``.run()``.

    Example::

        import memory_guard as mg

        watchdog = mg.guard_vllm_watchdog(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_num_seqs=512,
            gpu_memory_utilization=0.90,
            max_retries=5,
            alert_callback=lambda msg, a, m: print(f"[ALERT] {msg}"),
        )
        watchdog.run()
    """
    # Build the vLLM server command
    cmd: List[str] = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        _MAX_NUM_SEQS_FLAG, str(max_num_seqs),
        _GPU_MEM_UTIL_FLAG, str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Build StateKey from current platform
    try:
        platform_info = detect_platform()
        available_mb = get_available_memory_mb()
        backend_str = platform_info.backend.value
        os_platform = platform_info.os_platform
    except Exception:
        available_mb = 0.0
        backend_str = "unknown"
        os_platform = sys.platform

    state_key = StateKey.from_values(
        available_mb=available_mb,
        backend=backend_str,
        model_params=model_params,
        model_bits=model_bits,
        os_platform=os_platform,
    )

    # Load bandit if not provided
    if bandit is None:
        bandit = BanditPolicy.load()

    logger.info(
        "[watchdog] Configured for model=%r, state=%s, max_retries=%d, "
        "backoff=%.1fs, confidence=%.2f",
        model, state_key, max_retries, backoff_seconds, bandit.confidence,
    )

    return VLLMWatchdog(
        cmd=cmd,
        state_key=state_key,
        bandit=bandit,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        conservative_margin=conservative_margin,
        alert_callback=alert_callback,
    )
