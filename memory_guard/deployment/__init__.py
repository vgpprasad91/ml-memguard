"""Deployment integration — Kubernetes, sidecar, and watchdog."""
from .watchdog import VLLMWatchdog, guard_vllm_watchdog
from .sidecar import MemGuardSidecar
from .k8s_policy import K8sPolicyWatcher

__all__ = [
    "VLLMWatchdog",
    "guard_vllm_watchdog",
    "MemGuardSidecar",
    "K8sPolicyWatcher",
]
