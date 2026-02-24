"""
core/viz.py – lightweight visualisation helpers (PIL / matplotlib).
These functions are used to render result previews inside Streamlit.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def overlay_geojson_on_image(
    base_image: Image.Image,
    geojson_path: Path,
    *,
    outline_color: str = "#FF0000",
    fill_alpha: int = 60,
) -> Image.Image:
    """Draw GeoJSON nuclei contours on top of a PIL image."""
    try:
        import geojson as gj  # type: ignore
        with open(geojson_path) as fh:
            fc = gj.load(fh)
    except Exception as exc:
        logger.warning("Could not load GeoJSON: %s", exc)
        return base_image

    img = base_image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for feat in fc.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon" and coords:
            ring = [(int(x), int(y)) for x, y in coords[0]]
            if len(ring) > 2:
                draw.polygon(ring, outline=outline_color)
        elif gtype == "Point" and coords:
            x, y = int(coords[0]), int(coords[1])
            r = 3
            draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=outline_color)

    return Image.alpha_composite(img, overlay).convert("RGB")


def load_preview_image(path: Path, max_size: int = 800) -> Image.Image | None:
    """Load any supported image/npy for preview."""
    try:
        if path.suffix == ".npy":
            arr = np.load(path)
            if arr.ndim == 2:
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
                return Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3:
                arr = arr[..., :3].astype(np.uint8)
                return Image.fromarray(arr)
            return None
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img
    except Exception as exc:
        logger.warning("load_preview_image failed for %s: %s", path, exc)
        return None


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
