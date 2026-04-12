#!/usr/bin/env python3
"""vLLM + memguard-allocator: GPU allocation telemetry integration.

Demonstrates the four-line setup that layers the BSL memguard-allocator
allocator hooks under the OSS KVCacheMonitor, giving per-request OOM
telemetry (near-miss count, allocation velocity, fragmentation ratio)
alongside KV cache utilization.

Stack
-----
┌─────────────────────────────────────────────────┐
│  ml-memguard  (OSS / Apache-2.0 after 2030)     │  KVCacheMonitor
│    KVCacheMonitor — kv-block utilization         │  policy & callbacks
├─────────────────────────────────────────────────┤
│  memguard-allocator  (BSL-1.1)                  │  allocator hooks
│    CUDA / Metal ring-buffer telemetry            │  + ring buffer
├─────────────────────────────────────────────────┤
│  Cloud telemetry backend  (optional)             │  remote metrics
│    InferenceTelemetry upload                     │
└─────────────────────────────────────────────────┘

Prerequisites
-------------
    pip install ml-memguard
    pip install memguard-allocator[cuda]   # NVIDIA GPU
    # or
    pip install memguard-allocator[metal]  # Apple Silicon

Quick run (requires a live vLLM server at localhost:8000):

    python examples/vllm_allocator_integration.py

Environment variables
---------------------
    VLLM_URL          vLLM base URL (default: http://localhost:8000)
    MEMGUARD_THRESHOLD   OOM probability threshold (default: 0.70)
    DEMO_DURATION_S   How many seconds to run the demo (default: 30)
"""

from __future__ import annotations

import logging
import os
import time

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("vllm_allocator_demo")


# ---------------------------------------------------------------------------
# Step 0: Auto-detect total GPU / unified memory
# ---------------------------------------------------------------------------

def _gpu_total_mb() -> float:
    """Detect total GPU memory in MB (CUDA or Apple Silicon).

    Wraps ``memguard_allocator.detect_gpu_total_mb()`` with a
    user-visible log line so the number is visible in production logs.
    """
    try:
        import memguard_allocator
        mb = memguard_allocator.detect_gpu_total_mb()
        if mb > 0:
            logger.info("Auto-detected GPU memory: %.0f MB (%.1f GB)", mb, mb / 1024)
            return mb
    except ImportError:
        pass
    logger.warning(
        "Could not auto-detect GPU memory — fragmentation_ratio will be omitted. "
        "Install memguard-allocator: pip install memguard-allocator[cuda]"
    )
    return 0.0


# ---------------------------------------------------------------------------
# Step 1: Build the vLLM KV cache poll_fn
# ---------------------------------------------------------------------------

def _make_vllm_poll_fn(vllm_url: str):
    """Return a poll_fn that reads KV block stats from vLLM's /metrics endpoint.

    In production you can replace this with a direct block-manager query:

        bm = engine.scheduler.block_manager
        return lambda: (bm.get_num_used_gpu_blocks(),
                        bm.get_num_total_gpu_blocks())

    This demo uses the Prometheus /metrics endpoint so it works with any
    running vLLM process without importing vLLM.
    """
    import re
    import urllib.request

    _used_re  = re.compile(r'^vllm:gpu_cache_usage_perc\s+(\S+)', re.MULTILINE)
    _total_re = re.compile(r'^vllm:num_gpu_blocks\s+(\S+)', re.MULTILINE)

    def _poll() -> tuple[int, int]:
        try:
            with urllib.request.urlopen(f"{vllm_url}/metrics", timeout=3) as resp:
                body = resp.read().decode()
            used_pct = float(_used_re.search(body).group(1))   # 0.0 – 1.0
            total    = int(_total_re.search(body).group(1))
            used     = int(used_pct * total)
            return used, total
        except Exception:
            return 0, 1   # safe default — 0 % utilization

    return _poll


# ---------------------------------------------------------------------------
# Main: four-line allocator integration
# ---------------------------------------------------------------------------

def main() -> None:
    vllm_url     = os.environ.get("VLLM_URL",       "http://localhost:8000")
    threshold    = float(os.environ.get("MEMGUARD_THRESHOLD", "0.70"))
    demo_seconds = int(os.environ.get("DEMO_DURATION_S",     "30"))

    # ── Step 1: Start the allocator monitor ──────────────────────────────────
    #
    # On CUDA: installs mgcuda_malloc/mgcuda_free as PyTorch's current allocator.
    # On Apple Silicon: starts a background MLX polling thread (no ObjC swizzle).
    # Raises RuntimeError if memguard-allocator is not installed.
    #
    import memguard_allocator
    memguard_allocator.install()
    logger.info("memguard-allocator installed: %r", memguard_allocator._bridge)

    # ── Step 2: Build the extended_poll_fn ───────────────────────────────────
    #
    # detect_gpu_total_mb() reads torch.cuda.get_device_properties(0).total_memory
    # on CUDA or sysctl hw.memsize on Apple Silicon — no manual lookup needed.
    #
    from memguard_allocator.poll_fn import make_extended_poll_fn
    gpu_mb         = _gpu_total_mb()
    extended_fn    = make_extended_poll_fn(total_memory_mb=gpu_mb)

    # ── Step 3: Create the KVCacheMonitor ────────────────────────────────────
    #
    # poll_fn          → reads KV block utilization from vLLM
    # extended_poll_fn → reads allocator ring-buffer stats (near-miss, velocity,
    #                    fragmentation) and merges them into InferenceTelemetry
    #
    from memory_guard.inference_monitor import KVCacheMonitor
    monitor = KVCacheMonitor(
        poll_fn=_make_vllm_poll_fn(vllm_url),
        extended_poll_fn=extended_fn,
        poll_interval=5.0,
        on_warning=lambda u: logger.warning(
            "KV cache WARNING: %.1f%% — consider reducing upstream weight", u * 100
        ),
        on_shed_load=lambda u: logger.error(
            "KV cache SHED_LOAD: %.1f%% — stopping new traffic to this replica", u * 100
        ),
    )

    # ── Step 4: Run inside monitor.session() ─────────────────────────────────
    #
    # session() starts the background poll thread on enter and stops it on exit.
    # Wrap your actual serve_forever() call here in production.
    #
    logger.info(
        "Starting monitor (vllm=%s, threshold=%.2f, demo=%ds)",
        vllm_url, threshold, demo_seconds,
    )

    with monitor.session():
        try:
            time.sleep(demo_seconds)
        except KeyboardInterrupt:
            logger.info("Stopped by user")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    memguard_allocator.uninstall()
    logger.info("Done — allocator uninstalled")


# ---------------------------------------------------------------------------
# Minimal standalone demo (no live vLLM required)
# ---------------------------------------------------------------------------

def demo_no_vllm() -> None:
    """Run the full allocator integration against a simulated KV poll_fn.

    Useful for validating the wiring without a live vLLM process.
    """
    logger.info("Running standalone demo (simulated KV poll — no vLLM required)")

    import memguard_allocator
    from memguard_allocator.poll_fn import make_extended_poll_fn
    from memory_guard.inference_monitor import KVCacheMonitor

    # Simulated KV blocks: 4 of 10 used → 40 % utilization
    def _simulated_poll() -> tuple[int, int]:
        return 4, 10

    gpu_mb = _gpu_total_mb()

    # ── The four lines ────────────────────────────────────────────────────────
    memguard_allocator.install()                                         # 1
    extended_fn = make_extended_poll_fn(total_memory_mb=gpu_mb)         # 2
    monitor     = KVCacheMonitor(                                        # 3
        poll_fn=_simulated_poll,
        extended_poll_fn=extended_fn,
        poll_interval=2.0,
    )
    with monitor.session():                                              # 4
        logger.info("Monitor running for 6 seconds…")
        time.sleep(6)

    logger.info("Current utilization: %.1f%%", monitor.current_utilization * 100)
    logger.info(
        "Extended stats (last poll): near_miss=%s  velocity=%s  frag=%s",
        monitor._last_extended.get("near_miss_count",          "n/a"),
        monitor._last_extended.get("allocation_velocity_mbps", "n/a"),
        monitor._last_extended.get("fragmentation_ratio",      "n/a"),
    )

    memguard_allocator.uninstall()
    logger.info("Standalone demo complete")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv or os.environ.get("VLLM_URL") is None:
        # No VLLM_URL set — run the standalone demo
        demo_no_vllm()
    else:
        main()
