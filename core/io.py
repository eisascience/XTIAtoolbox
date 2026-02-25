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
    OME_TIFF_MAGIC,
    UPLOADS_DIR,
    WSI_EXTENSIONS,
)
from .hashing import sha256_file
from .models import FileEntry
from .openslide_utils import openslide_error_hint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_wsi(path: Path) -> bool:
    return path.suffix.lower() in WSI_EXTENSIONS


def _is_ome_tiff(path: Path) -> bool:
    """Return True if path is an OME-TIFF (by extension or OME-XML magic bytes)."""
    name = path.name.lower()
    if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
        return True
    if path.suffix.lower() in (".tif", ".tiff", ".btf"):
        try:
            with open(path, "rb") as fh:
                header = fh.read(4096)
            return OME_TIFF_MAGIC in header
        except Exception:
            pass
    return False


def _axes_to_rgb(data: np.ndarray, axes: str) -> np.ndarray:
    """Convert a numpy array with the given axes string to an RGB uint8 array.

    Supported axes (case-insensitive): YX, CYX, YXC, ZYX, ZYXC, ZCYX.
    T (time) and any leading singleton dimensions are stripped first.
    """
    axes = axes.upper()

    # Strip T axis – take the first frame
    while "T" in axes:
        idx = axes.index("T")
        data = np.take(data, 0, axis=idx)
        axes = axes[:idx] + axes[idx + 1:]

    # Strip Z axis – take the first slice
    while "Z" in axes:
        idx = axes.index("Z")
        data = np.take(data, 0, axis=idx)
        axes = axes[:idx] + axes[idx + 1:]

    # Now axes should reduce to one of: YX, YXC, CYX (or YXS, SYX)
    if axes in ("YX",):
        rgb = np.stack([data, data, data], axis=-1)
    elif axes in ("YXC", "YXS"):
        c = data.shape[2]
        if c == 1:
            ch = data[..., 0]
            rgb = np.stack([ch, ch, ch], axis=-1)
        elif c >= 3:
            rgb = data[..., :3]
        else:
            rgb = np.stack([data[..., 0]] * 3, axis=-1)
    elif axes in ("CYX", "SYX"):
        c = data.shape[0]
        if c == 1:
            ch = data[0]
            rgb = np.stack([ch, ch, ch], axis=-1)
        elif c >= 3:
            rgb = np.stack([data[0], data[1], data[2]], axis=-1)
        else:
            rgb = np.stack([data[0]] * 3, axis=-1)
    else:
        # Unknown axes: squeeze singletons and guess
        data = np.squeeze(data)
        if data.ndim == 2:
            rgb = np.stack([data, data, data], axis=-1)
        elif data.ndim == 3 and data.shape[2] in (3, 4):
            rgb = data[..., :3]
        elif data.ndim == 3 and data.shape[0] in (3, 4):
            rgb = np.stack([data[0], data[1], data[2]], axis=-1)
        else:
            rgb = np.zeros((*data.shape[:2], 3), dtype=np.uint8)

    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rmin, rmax = float(rgb.min()), float(rgb.max())
        if rmax > rmin:
            rgb = ((rgb.astype(np.float32) - rmin) / (rmax - rmin) * 255).astype(np.uint8)
        else:
            rgb = np.zeros(rgb.shape, dtype=np.uint8)
    return rgb


def _tiff_to_rgb(path: Path) -> np.ndarray:
    """Read a TIFF/OME-TIFF file and return an RGB uint8 numpy array.

    Uses tifffile to read the data and interprets the axes metadata so that
    multi-channel, Z-stack, or greyscale TIFFs are all returned as RGB.
    """
    import tifffile  # type: ignore

    with tifffile.TiffFile(str(path)) as tif:
        if tif.series:
            series = tif.series[0]
            axes = series.axes.upper()
            data = series.asarray()
        else:
            data = tif.asarray()
            # Guess axes from shape
            ndim = data.ndim
            if ndim == 2:
                axes = "YX"
            elif ndim == 3 and data.shape[2] in (1, 3, 4):
                axes = "YXC"
            elif ndim == 3:
                axes = "CYX"
            elif ndim == 4:
                axes = "ZYXC"
            else:
                axes = "YX"
    return _axes_to_rgb(data, axes)


def get_channel_info(path: Path) -> tuple[int, list[str]]:
    """Return ``(channel_count, default_channel_names)`` for an image file.

    For TIFF/OME-TIFF files the C axis is inspected via tifffile so that
    multi-channel fluorescence images are detected correctly.
    All other formats return ``(1, ["Ch 0"])``.
    """
    if path.suffix.lower() not in (".tif", ".tiff", ".btf"):
        return 1, ["Ch 0"]
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(str(path)) as tif:
            if tif.series:
                series = tif.series[0]
                axes = series.axes.upper()
                shape = series.shape
            else:
                data = tif.asarray()
                ndim = data.ndim
                if ndim == 2:
                    return 1, ["Ch 0"]
                elif ndim == 3 and data.shape[2] in (1, 3, 4):
                    axes, shape = "YXC", data.shape
                elif ndim == 3:
                    axes, shape = "CYX", data.shape
                else:
                    return 1, ["Ch 0"]
            if "C" in axes:
                n = shape[axes.index("C")]
                return n, [f"Ch {i}" for i in range(n)]
    except Exception as exc:
        logger.warning("get_channel_info failed for %s: %s", path, exc)
    return 1, ["Ch 0"]


def _extract_channel_as_gray(
    data: np.ndarray, axes: str, channel_idx: int
) -> np.ndarray:
    """Extract one channel plane from an ND array and return a 2-D (YX) array."""
    axes = axes.upper()
    for dim in ("T", "Z"):
        while dim in axes:
            idx = axes.index(dim)
            data = np.take(data, 0, axis=idx)
            axes = axes[:idx] + axes[idx + 1:]
    if "C" in axes:
        c_ax = axes.index("C")
        c_idx = min(channel_idx, data.shape[c_ax] - 1)
        data = np.take(data, c_idx, axis=c_ax)
        axes = axes[:c_ax] + axes[c_ax + 1:]
    if "S" in axes:
        s_ax = axes.index("S")
        data = np.take(data, 0, axis=s_ax)
        axes = axes[:s_ax] + axes[s_ax + 1:]
    data = np.squeeze(data)
    if data.ndim > 2:
        data = data[0] if data.shape[0] <= data.shape[-1] else data[..., 0]
    return data


def _normalize_to_uint8(
    data: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0
) -> np.ndarray:
    """Percentile-stretch an array to uint8 [0–255]."""
    data = data.astype(np.float32)
    lo = float(np.percentile(data, low_pct))
    hi = float(np.percentile(data, high_pct))
    if hi > lo:
        return np.clip((data - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.zeros(data.shape, dtype=np.uint8)


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
    # OME-TIFF files are treated as regular images unless proven pyramidal/WSI.
    is_wsi = suffix in WSI_EXTENSIONS and not _is_ome_tiff(dest_path)

    if is_wsi:
        w, h, mpp = _probe_wsi(dest_path)
    else:
        w, h, mpp = _probe_image(dest_path)

    channel_count, channel_names_default = get_channel_info(dest_path)

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
        channel_count=channel_count,
        channel_names=channel_names_default if channel_count > 1 else [],
    )


def make_thumbnail_with_error(
    entry: FileEntry, max_size: int = 512
) -> tuple[Image.Image | None, str | None]:
    """Return (thumbnail, error_message).

    On success: (PIL Image, None).
    On failure: (None, human-readable error string).  When OpenSlide is missing
    the error string contains platform-specific install guidance.
    """
    path = entry.stored_path
    try:
        if entry.is_wsi:
            from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
            reader = WSIReader.open(path)
            thumb = reader.slide_thumbnail(resolution=1.25, units="power")
            reader.close()
            img = Image.fromarray(thumb)
        elif path.suffix.lower() in (".tif", ".tiff", ".btf"):
            # Use tifffile for TIFF/OME-TIFF to handle multi-channel and
            # multi-axis data that PIL cannot decode without axes context.
            try:
                arr = _tiff_to_rgb(path)
                img = Image.fromarray(arr)
            except Exception:
                img = Image.open(path).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img, None
    except Exception as exc:
        exc_str = str(exc)
        logger.warning("Thumbnail generation failed for %s: %s", path, exc_str)
        hint = openslide_error_hint(exc_str)
        if hint:
            return None, hint
        return None, f"Thumbnail generation failed: {exc_str}"


def make_thumbnail(entry: FileEntry, max_size: int = 512) -> Image.Image | None:
    """Return a PIL thumbnail for display, or None on failure."""
    img, _ = make_thumbnail_with_error(entry, max_size=max_size)
    return img


def make_channel_thumbnail(
    entry: FileEntry,
    channel_idx: int = 0,
    max_size: int = 512,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> tuple[Image.Image | None, str | None]:
    """Return a grayscale-RGB thumbnail for a single channel of a TIFF image.

    Each channel is normalized with a percentile stretch (default 1–99 %).
    The result is a three-channel (H × W × 3) grayscale-RGB PIL Image so that
    Streamlit can display it without forced colorization.

    For non-TIFF files or single-channel images this falls back to
    ``make_thumbnail_with_error``.
    """
    path = entry.stored_path
    if path.suffix.lower() not in (".tif", ".tiff", ".btf") or entry.is_wsi:
        return make_thumbnail_with_error(entry, max_size=max_size)
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(str(path)) as tif:
            if tif.series:
                series = tif.series[0]
                axes = series.axes.upper()
                data = series.asarray()
            else:
                data = tif.asarray()
                ndim = data.ndim
                if ndim == 2:
                    axes = "YX"
                elif ndim == 3 and data.shape[2] in (1, 3, 4):
                    axes = "YXC"
                elif ndim == 3:
                    axes = "CYX"
                else:
                    axes = "YX"
        ch = _extract_channel_as_gray(data, axes, channel_idx)
        ch = _normalize_to_uint8(ch, low_pct, high_pct)
        rgb = np.stack([ch, ch, ch], axis=-1)
        img = Image.fromarray(rgb)
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img, None
    except Exception as exc:
        exc_str = str(exc)
        logger.warning(
            "make_channel_thumbnail failed for %s ch%d: %s", path, channel_idx, exc_str
        )
        return None, f"Channel thumbnail failed: {exc_str}"


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
