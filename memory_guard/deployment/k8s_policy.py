"""In-cluster Kubernetes API client for MemGuardPolicy CRD hot-reload.

Uses only stdlib (json, logging, os, ssl, threading, urllib) — no
kubernetes-client dependency required.

The sidecar calls this module to:
  1. Detect whether it is running inside a Kubernetes pod.
  2. Fetch the initial MemGuardPolicy spec on startup (blocking GET).
  3. Stream watch events for real-time hot-reload without a pod restart.

In-cluster authentication (Kubernetes SA token projection):
  Token : /var/run/secrets/kubernetes.io/serviceaccount/token
  CA    : /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
  Host  : KUBERNETES_SERVICE_HOST environment variable
  Port  : KUBERNETES_SERVICE_PORT environment variable
  NS    : /var/run/secrets/kubernetes.io/serviceaccount/namespace

API paths used:
  GET  /apis/memguard.io/v1alpha1/namespaces/{ns}/memguardpolicies/{name}
  GET  /apis/memguard.io/v1alpha1/namespaces/{ns}/memguardpolicies
         ?watch=true&fieldSelector=metadata.name%3D{name}

Hot-reload architecture:
  K8sPolicyWatcher.watch(on_policy) starts a daemon thread.  The thread
  keeps a long-lived streaming HTTP connection to the API server.  On every
  ADDED or MODIFIED event for the named policy, it calls on_policy(spec).
  The sidecar's on_policy callback updates sidecar._threshold and
  monitor instance attributes in-place; the Python GIL makes float/int
  assignments on the main sidecar thread visible immediately — no lock needed
  for these scalar values.

  The watch loop reconnects automatically after transient errors (network
  blip, API server restart) with a 5-second back-off.  call stop() to
  terminate cleanly.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import urllib.request
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Kubernetes in-cluster SA mount paths
_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_CA_PATH    = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_NS_PATH    = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# CRD API coordinates
_API_GROUP   = "memguard.io"
_API_VERSION = "v1alpha1"
_RESOURCE    = "memguardpolicies"


class K8sPolicyWatcher:
    """Watch a MemGuardPolicy CRD and call a callback when the spec changes.

    Parameters
    ----------
    policy_name:
        Name of the MemGuardPolicy resource (e.g. ``"default"``).
    namespace:
        Kubernetes namespace.  Defaults to the pod's own namespace read from
        the SA namespace file.  Explicit value overrides the file.
    """

    def __init__(
        self,
        policy_name: str = "default",
        namespace: str = "",
    ) -> None:
        self._policy_name = policy_name
        self._namespace   = namespace or self._read_namespace()
        self._token       = self._read_token()
        self._ssl_ctx     = self._build_ssl_context()
        self._api_base    = self._build_api_base()
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_in_cluster() -> bool:
        """Return ``True`` when running inside a Kubernetes pod.

        Checks for the presence of the SA token projection file — the most
        reliable in-cluster indicator that works regardless of environment
        variables or network connectivity.
        """
        return os.path.isfile(_TOKEN_PATH)

    def get(self) -> Optional[Dict[str, Any]]:
        """Fetch the MemGuardPolicy ``spec`` once (blocking GET).

        Returns the ``spec`` dict from the MemGuardPolicy object, or ``None``
        on any error (resource not found, permission denied, network failure).
        Logs the error at DEBUG level so the sidecar can fall back to its
        built-in defaults without crashing.
        """
        url = (
            f"{self._api_base}/namespaces/{self._namespace}"
            f"/{_RESOURCE}/{self._policy_name}"
        )
        try:
            obj = self._api_get(url)
            spec = obj.get("spec") or {}
            logger.info(
                "[memguard] MemGuardPolicy %s/%s fetched: %s",
                self._namespace, self._policy_name, spec,
            )
            return spec
        except Exception as exc:
            logger.debug(
                "[memguard] K8sPolicyWatcher.get %s/%s failed: %s",
                self._namespace, self._policy_name, exc,
            )
            return None

    def watch(self, on_policy: Callable[[Dict[str, Any]], None]) -> None:
        """Start a background daemon thread that streams policy change events.

        ``on_policy(spec)`` is called on every ADDED or MODIFIED watch event.
        The thread reconnects automatically after transient errors.  Call
        ``stop()`` to shut it down.

        Parameters
        ----------
        on_policy:
            Callable that receives the MemGuardPolicy ``spec`` dict.
            Called from the watcher daemon thread — must be thread-safe.
        """
        if self._thread is not None and self._thread.is_alive():
            return  # already watching
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            args=(on_policy,),
            daemon=True,
            name="memguard-policy-watcher",
        )
        self._thread.start()
        logger.debug(
            "[memguard] K8sPolicyWatcher started watching %s/%s",
            self._namespace, self._policy_name,
        )

    def stop(self) -> None:
        """Stop the background watch thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.debug("[memguard] K8sPolicyWatcher stopped")

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _watch_loop(self, on_policy: Callable[[Dict[str, Any]], None]) -> None:
        """Streaming watch loop.  Reconnects on transient errors with back-off."""
        # URL-encode the = in the fieldSelector value
        url = (
            f"{self._api_base}/namespaces/{self._namespace}"
            f"/{_RESOURCE}"
            f"?watch=true&fieldSelector=metadata.name%3D{self._policy_name}"
        )
        while not self._stop_event.is_set():
            try:
                req = urllib.request.Request(url, headers=self._auth_headers())
                # timeout=60: stream stays open for up to 60 s without a new line
                # before reconnecting (normal for low-event-rate policies)
                with urllib.request.urlopen(
                    req, context=self._ssl_ctx, timeout=60
                ) as resp:
                    for raw_line in resp:
                        if self._stop_event.is_set():
                            return
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug(
                                "[memguard] K8sPolicyWatcher: non-JSON line: %.100s",
                                line,
                            )
                            continue
                        ev_type = event.get("type", "")
                        if ev_type in ("ADDED", "MODIFIED"):
                            spec = event.get("object", {}).get("spec") or {}
                            logger.info(
                                "[memguard] MemGuardPolicy %s/%s %s — "
                                "hot-reloading spec: %s",
                                self._namespace, self._policy_name, ev_type, spec,
                            )
                            try:
                                on_policy(spec)
                            except Exception as exc:
                                logger.debug(
                                    "[memguard] on_policy callback raised: %s", exc
                                )
                        elif ev_type == "DELETED":
                            logger.info(
                                "[memguard] MemGuardPolicy %s/%s deleted — "
                                "retaining last applied settings",
                                self._namespace, self._policy_name,
                            )
                        elif ev_type == "ERROR":
                            logger.debug(
                                "[memguard] K8sPolicyWatcher ERROR event: %s",
                                event.get("object", {}),
                            )
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                logger.debug(
                    "[memguard] K8sPolicyWatcher stream error (%s) — "
                    "reconnecting in 5 s",
                    exc,
                )
                self._stop_event.wait(5)

    def _api_get(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers=self._auth_headers())
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))  # type: ignore[return-value]

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _build_api_base(self) -> str:
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        return f"https://{host}:{port}/apis/{_API_GROUP}/{_API_VERSION}"

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        if os.path.isfile(_CA_PATH):
            ctx.load_verify_locations(_CA_PATH)
        return ctx

    @staticmethod
    def _read_token() -> str:
        if os.path.isfile(_TOKEN_PATH):
            with open(_TOKEN_PATH) as fh:
                return fh.read().strip()
        return ""

    @staticmethod
    def _read_namespace() -> str:
        if os.path.isfile(_NS_PATH):
            with open(_NS_PATH) as fh:
                return fh.read().strip()
        return "default"
