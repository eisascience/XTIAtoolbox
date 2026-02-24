"""
core/config.py – centralised configuration for the orchestrator.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace root – can be overridden via the XTIA_WORKSPACE environment var
# ---------------------------------------------------------------------------
_DEFAULT_WORKSPACE = Path(os.getcwd()) / "workspace"
WORKSPACE_ROOT: Path = Path(os.environ.get("XTIA_WORKSPACE", str(_DEFAULT_WORKSPACE)))

UPLOADS_DIR: Path = WORKSPACE_ROOT / "uploads"
RUNS_DIR: Path = WORKSPACE_ROOT / "runs"

# Registry file that persists file metadata across Streamlit sessions
REGISTRY_FILE: Path = WORKSPACE_ROOT / "registry.json"

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
WSI_EXTENSIONS: set[str] = {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
                              ".bif", ".tif", ".tiff", ".btf"}
SMALL_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALL_SUPPORTED_EXTENSIONS: set[str] = WSI_EXTENSIONS | SMALL_EXTENSIONS

# OME-TIFF is a sub-type of TIFF – identified by content, not extension.
OME_TIFF_MAGIC: bytes = b"OME-XML"

# ---------------------------------------------------------------------------
# Default task parameters
# ---------------------------------------------------------------------------
DEFAULT_PATCH_SIZE: int = 512
DEFAULT_STRIDE: int = 256
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_NUM_WORKERS: int = 0  # 0 = main thread (safe in Streamlit)

# Maximum ROI area (pixels²) to warn before full-slide runs
MAX_ROI_AREA_WARNING: int = 10_000 * 10_000  # 100 Mpx

# ---------------------------------------------------------------------------
# Ensure workspace directories exist
# ---------------------------------------------------------------------------
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
