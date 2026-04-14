"""Compatibility coverage for the neutral optional-integration surface."""

from __future__ import annotations

import importlib


def test_legacy_backends_module_aliases_integrations_module():
    integrations = importlib.import_module("memory_guard.integrations")
    backends = importlib.import_module("memory_guard.backends")

    assert backends is integrations
