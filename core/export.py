"""
core/export.py – helpers for packaging run outputs for download.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from .models import RunManifest


def make_run_zip(manifest: RunManifest) -> bytes:
    """Return a ZIP archive containing all outputs + manifest for a run."""
    from .config import RUNS_DIR

    run_dir = RUNS_DIR / manifest.run_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            zf.write(manifest_path, arcname="manifest.json")
        out_dir = run_dir / "outputs"
        if out_dir.exists():
            for f in sorted(out_dir.rglob("*")):
                if f.is_file():
                    arcname = str(f.relative_to(run_dir))
                    zf.write(f, arcname=arcname)
    return buf.getvalue()


def read_csv(path: Path) -> str:
    """Return CSV file contents as string."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def read_geojson(path: Path) -> dict:
    """Return parsed GeoJSON dict."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}
