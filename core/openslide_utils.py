"""
core/openslide_utils.py – helpers for detecting missing OpenSlide and generating
platform-specific install guidance.
"""
from __future__ import annotations

import platform


_OPENSLIDE_INDICATORS = (
    "openslide",
    "couldn't locate openslide",
    "libopenslide",
    "openslide-bin",
)


def is_openslide_error(exc_str: str) -> bool:
    """Return True if the exception message indicates a missing OpenSlide library."""
    lower = exc_str.lower()
    return any(kw in lower for kw in _OPENSLIDE_INDICATORS)


def openslide_install_hint() -> str:
    """Return a plain-text install hint for OpenSlide appropriate to the current platform."""
    system = platform.system()
    if system == "Darwin":
        return (
            "OpenSlide native library not found. On macOS, install it with:\n"
            "  brew install openslide\n"
            "  pip install openslide-python\n"
            "Or, as a cross-platform alternative:\n"
            "  pip install openslide-bin openslide-python"
        )
    if system == "Linux":
        return (
            "OpenSlide native library not found. On Linux, install it with:\n"
            "  sudo apt-get install openslide-tools libopenslide-dev  (Debian/Ubuntu)\n"
            "  pip install openslide-python\n"
            "Or, as a cross-platform alternative:\n"
            "  pip install openslide-bin openslide-python"
        )
    return (
        "OpenSlide native library not found. Install it with:\n"
        "  pip install openslide-bin openslide-python\n"
        "See https://openslide.org/api/python/#installing for platform-specific instructions."
    )


def openslide_error_hint(exc_str: str) -> str | None:
    """If *exc_str* looks like a missing-OpenSlide error, return an install hint; else None."""
    if is_openslide_error(exc_str):
        return openslide_install_hint()
    return None


def format_roi_error(exc_str: str, default_prefix: str = "ROI extraction failed") -> str:
    """Return an install hint if the error is OpenSlide-related, else a prefixed error message."""
    hint = openslide_error_hint(exc_str)
    return hint if hint else f"{default_prefix}: {exc_str}"
