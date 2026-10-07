"""Offline, scope-aligned event decision experiments.

Importing this package never imports a provider SDK or a VLM backend.
"""

from .contracts import CONTRACT_VERSION, FEATURE_NAMES, EventDecisionError

__all__ = ["CONTRACT_VERSION", "FEATURE_NAMES", "EventDecisionError"]

