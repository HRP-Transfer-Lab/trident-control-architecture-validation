"""Synthetic infrastructure fixtures for tests and pipeline dry-runs."""

from .fixtures import CORE_SYNTHETIC_FEATURES, make_synthetic_window_table
from .worlds import (
    STATIC_SYNTHETIC_WORLD_IDS,
    WORLD_MODEL_ALIGNMENT,
    make_all_static_synthetic_worlds,
    make_static_synthetic_world,
)

__all__ = [
    "CORE_SYNTHETIC_FEATURES",
    "STATIC_SYNTHETIC_WORLD_IDS",
    "WORLD_MODEL_ALIGNMENT",
    "make_all_static_synthetic_worlds",
    "make_static_synthetic_world",
    "make_synthetic_window_table",
]
