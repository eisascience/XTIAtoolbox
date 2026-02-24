"""
core/tasks/utils.py – shared utilities for task modules.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_tiatoolbox_version() -> str:
    try:
        import tiatoolbox  # type: ignore
        return str(getattr(tiatoolbox, "__version__", "unknown"))
    except ImportError:
        return "not installed"


def get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def ensure_output_dir(run_dir: Path) -> Path:
    out = run_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def roi_to_bounds(roi: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    """Convert ROI dict to (x, y, x2, y2) or None."""
    if roi is None:
        return None
    x, y = roi.get("x", 0), roi.get("y", 0)
    w, h = roi.get("width", 0), roi.get("height", 0)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def bounds_to_wsi_kwargs(bounds: tuple[int, int, int, int] | None) -> dict[str, Any]:
    """Convert bounds to tiatoolbox resolution/region kwargs."""
    if bounds is None:
        return {}
    x1, y1, x2, y2 = bounds
    return {
        "region": [x1, y1, x2, y2],
    }
