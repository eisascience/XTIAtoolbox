"""
core/tasks/nuclei.py – nuclei instance segmentation via TIAToolbox HoVer-Net.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..models import FileEntry, ROI
from .utils import ensure_output_dir, get_device, roi_to_bounds

logger = logging.getLogger(__name__)

# Default HoVer-Net pretrained model
DEFAULT_MODEL = "hovernet_fast-pannuke"


def run_nuclei_detection(
    entry: FileEntry,
    run_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    roi: dict[str, Any] | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    on_gpu: bool | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    """
    Run nuclei instance segmentation and write outputs to *run_dir/outputs/*.

    Returns a dict with keys: geojson_path, csv_path, status, error.
    """
    if log_fn is None:
        log_fn = logger.info

    device = get_device() if on_gpu is None else ("cuda" if on_gpu else "cpu")
    out_dir = ensure_output_dir(run_dir)

    # ----------------------------------------------------------------
    # Prepare input: if ROI requested, extract a region first
    # ----------------------------------------------------------------
    bounds = roi_to_bounds(roi)
    input_path = entry.stored_path

    if bounds is not None and entry.is_wsi:
        log_fn(f"Extracting ROI {bounds} from WSI …")
        input_path = _extract_roi_as_image(entry, bounds, out_dir)
        if input_path is None:
            return {"status": "failed", "error": "ROI extraction failed", "geojson_path": None, "csv_path": None}

    # ----------------------------------------------------------------
    # Run NucleusInstanceSegmentor
    # ----------------------------------------------------------------
    log_fn(f"Running nuclei detection (model={model}, device={device}) …")
    try:
        from tiatoolbox.models.engine.nucleus_instance_segmentor import (  # type: ignore
            NucleusInstanceSegmentor,
        )
    except ImportError as exc:
        return {"status": "failed", "error": str(exc), "geojson_path": None, "csv_path": None}

    with tempfile.TemporaryDirectory() as tmp:
        try:
            segmentor = NucleusInstanceSegmentor(
                pretrained_model=model,
                num_loader_workers=num_workers,
                batch_size=batch_size,
                auto_generate_mask=False,
            )
            output = segmentor.predict(
                imgs=[str(input_path)],
                save_dir=tmp,
                on_gpu=(device == "cuda"),
                crash_on_exception=True,
            )
        except Exception as exc:
            logger.exception("NucleusInstanceSegmentor failed")
            return {"status": "failed", "error": str(exc), "geojson_path": None, "csv_path": None}

        # Collect results from tmp
        geojson_path, csv_path = _collect_nuclei_outputs(Path(tmp), out_dir, log_fn)

    log_fn("Nuclei detection completed.")
    return {
        "status": "completed",
        "error": "",
        "geojson_path": str(geojson_path) if geojson_path else None,
        "csv_path": str(csv_path) if csv_path else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_roi_as_image(entry: FileEntry, bounds: tuple[int, int, int, int],
                          out_dir: Path) -> Path | None:
    """Extract a ROI from a WSI and save as PNG."""
    try:
        from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
        x1, y1, x2, y2 = bounds
        reader = WSIReader.open(entry.stored_path)
        region = reader.read_bounds(bounds, resolution=0, units="level")
        reader.close()
        from PIL import Image
        img = Image.fromarray(region[..., :3])
        roi_path = out_dir / "roi_input.png"
        img.save(roi_path)
        return roi_path
    except Exception as exc:
        logger.error("ROI extraction failed: %s", exc)
        return None


def _collect_nuclei_outputs(
    tmp_dir: Path, out_dir: Path, log_fn: Any
) -> tuple[Path | None, Path | None]:
    """Move nuclei outputs from tmp to out_dir and return (geojson, csv) paths."""
    import glob as _glob
    geojson_path: Path | None = None
    csv_path: Path | None = None

    # tiatoolbox saves .dat files and optionally JSON; convert to GeoJSON+CSV
    dat_files = list(tmp_dir.rglob("*.0.dat")) + list(tmp_dir.rglob("*.dat"))
    for dat_f in dat_files:
        try:
            import pickle
            with open(dat_f, "rb") as fh:
                result = pickle.load(fh)
            geojson_path = _write_geojson(result, out_dir)
            csv_path = _write_csv(result, out_dir)
            log_fn(f"Parsed nuclei output from {dat_f.name}")
            break
        except Exception as exc:
            log_fn(f"Could not parse {dat_f}: {exc}")

    # Also copy any existing JSON/CSV files
    for ext in ("*.json", "*.geojson", "*.csv"):
        for f in tmp_dir.rglob(ext):
            dest = out_dir / f.name
            shutil.copy2(f, dest)
            if f.suffix in (".json", ".geojson") and geojson_path is None:
                geojson_path = dest
            elif f.suffix == ".csv" and csv_path is None:
                csv_path = dest

    return geojson_path, csv_path


def _write_geojson(result: dict, out_dir: Path) -> Path:
    """Convert tiatoolbox nuclei dict to GeoJSON FeatureCollection."""
    import geojson as gj  # type: ignore

    features = []
    for nuc_id, nuc in result.items():
        try:
            contour = nuc.get("contour", [])
            if len(contour) > 2:
                coords = [[float(p[0]), float(p[1])] for p in contour]
                coords.append(coords[0])
                geom = gj.Polygon([coords])
            else:
                cx, cy = nuc.get("centroid", [0, 0])
                geom = gj.Point([float(cx), float(cy)])
            props = {
                "id": str(nuc_id),
                "type": int(nuc.get("type", 0)),
                "prob": float(nuc.get("prob", 0.0)),
            }
            features.append(gj.Feature(geometry=geom, properties=props))
        except Exception:
            continue

    fc = gj.FeatureCollection(features)
    out_path = out_dir / "nuclei.geojson"
    with open(out_path, "w") as fh:
        gj.dump(fc, fh)
    return out_path


def _write_csv(result: dict, out_dir: Path) -> Path:
    """Write nuclei centroids to CSV."""
    import csv

    out_path = out_dir / "nuclei.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cx", "cy", "type", "prob"])
        for nuc_id, nuc in result.items():
            cx, cy = nuc.get("centroid", [0, 0])
            writer.writerow([nuc_id, cx, cy, nuc.get("type", 0), nuc.get("prob", 0.0)])
    return out_path
