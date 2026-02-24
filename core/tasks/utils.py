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


def get_available_devices() -> list[str]:
    """Return available compute devices in preference order (best first).

    Possible values: "cuda", "mps", "cpu".
    "cpu" is always included as the last fallback.
    """
    devices: list[str] = []
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            devices.append("cuda")
        mps_ok = (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
        if mps_ok:
            devices.append("mps")
    except ImportError:
        pass
    devices.append("cpu")
    return devices


def normalize_device_choice(ui_choice: str) -> str:
    """Normalize a UI device label to a lowercase torch device string.

    "CPU"                 -> "cpu"
    "MPS (Apple Silicon)" -> "mps"
    "CUDA (if available)" -> "cuda"
    """
    _MAP: dict[str, str] = {
        "CPU": "cpu",
        "MPS (Apple Silicon)": "mps",
        "CUDA (if available)": "cuda",
    }
    return _MAP.get(ui_choice, ui_choice.lower())


def resolve_task_device(device: str | None) -> tuple[str, bool, str, str]:
    """Resolve a requested device string to task-runner parameters.

    Returns
    -------
    (device_str, on_gpu, device_used, fallback_reason)
        device_str      – normalised device string passed to the task
        on_gpu          – boolean for TIAToolbox's ``on_gpu`` parameter
        device_used     – the device that will actually be used
        fallback_reason – non-empty string if a fallback occurred
    """
    if device is None:
        device = get_available_devices()[0]

    device = device.lower()

    if device == "cuda":
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda", True, "cuda", ""
            return "cuda", False, "cpu", "CUDA requested but torch.cuda.is_available() is False"
        except ImportError:
            return "cuda", False, "cpu", "CUDA requested but torch is not installed"

    if device == "mps":
        try:
            import torch  # type: ignore
            mps_ok = (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
            if not mps_ok:
                return "mps", False, "cpu", (
                    "MPS requested but not available on this system "
                    "(requires macOS 12.3+ with Apple Silicon and a PyTorch MPS build)"
                )
        except ImportError:
            return "mps", False, "cpu", "MPS requested but torch is not installed"
        # MPS is available but TIAToolbox uses on_gpu for CUDA only; run on CPU
        # and record the fallback so the manifest clearly reflects what happened.
        return "mps", False, "cpu", (
            "MPS requested; TIAToolbox does not natively support MPS inference via on_gpu, "
            "running on CPU instead"
        )

    # Default: cpu
    return "cpu", False, "cpu", ""


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
