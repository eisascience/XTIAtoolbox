"""
core/io.py – file I/O helpers: saving uploads, reading image metadata, thumbnails.
"""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    ALL_SUPPORTED_EXTENSIONS,
    UPLOADS_DIR,
    WSI_EXTENSIONS,
)
from .hashing import sha256_file
from .models import FileEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_wsi(path: Path) -> bool:
    return path.suffix.lower() in WSI_EXTENSIONS


def _probe_image(path: Path) -> tuple[int, int, float]:
    """Return (width, height, mpp).  mpp is 0 if unknown."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            # XResolution tag (unit: pixels per cm or inch) → mpp approximation
            mpp = 0.0
            meta = getattr(img, "info", {})
            xres = meta.get("dpi")
            if xres:
                dpi = xres[0] if isinstance(xres, tuple) else float(xres)
                if dpi > 0:
                    mpp = round(25400.0 / dpi, 4)  # inch→µm
            return w, h, mpp
    except Exception:
        return 0, 0, 0.0


def _probe_wsi(path: Path) -> tuple[int, int, float]:
    """Return (width, height, mpp) for a WSI using tiatoolbox."""
    try:
        from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
        reader = WSIReader.open(path)
        info = reader.info
        w, h = info.slide_dimensions
        mpp = float(info.mpp[0]) if info.mpp is not None else 0.0
        reader.close()
        return w, h, mpp
    except Exception as exc:
        logger.warning("WSI probe failed for %s: %s", path, exc)
        return 0, 0, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_upload(file_bytes: bytes, original_name: str) -> FileEntry:
    """Persist uploaded bytes to workspace/uploads/<file_id>/ and return a FileEntry."""
    file_id = uuid.uuid4().hex
    dest_dir = UPLOADS_DIR / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / original_name
    dest_path.write_bytes(file_bytes)

    sha = sha256_file(dest_path)
    suffix = Path(original_name).suffix.lower()
    is_wsi = suffix in WSI_EXTENSIONS

    if is_wsi:
        w, h, mpp = _probe_wsi(dest_path)
    else:
        w, h, mpp = _probe_image(dest_path)

    return FileEntry(
        file_id=file_id,
        original_name=original_name,
        stored_path=dest_path,
        sha256=sha,
        size_bytes=len(file_bytes),
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        is_wsi=is_wsi,
        width=w,
        height=h,
        mpp=mpp,
    )


def make_thumbnail(entry: FileEntry, max_size: int = 512) -> Image.Image | None:
    """Return a PIL thumbnail for display, or None on failure."""
    path = entry.stored_path
    try:
        if entry.is_wsi:
            from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
            reader = WSIReader.open(path)
            thumb = reader.slide_thumbnail(resolution=1.25, units="power")
            reader.close()
            img = Image.fromarray(thumb)
        else:
            img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img
    except Exception as exc:
        logger.warning("Thumbnail generation failed for %s: %s", path, exc)
        return None


def read_region(entry: FileEntry, x: int, y: int, width: int, height: int,
                level: int = 0) -> np.ndarray | None:
    """Read a rectangular region from a slide/image as an RGB numpy array."""
    path = entry.stored_path
    try:
        if entry.is_wsi:
            from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
            reader = WSIReader.open(path)
            region = reader.read_region((x, y), level, (width, height))
            reader.close()
            return np.array(region)[..., :3]
        else:
            with Image.open(path) as img:
                img = img.convert("RGB")
                region = img.crop((x, y, x + width, y + height))
                return np.array(region)
    except Exception as exc:
        logger.error("read_region failed: %s", exc)
        return None


def delete_upload(entry: FileEntry) -> None:
    """Remove upload directory from disk."""
    try:
        shutil.rmtree(entry.stored_path.parent, ignore_errors=True)
    except Exception as exc:
        logger.warning("delete_upload failed: %s", exc)


def validate_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALL_SUPPORTED_EXTENSIONS
