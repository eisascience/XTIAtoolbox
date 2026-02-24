"""
core/tasks/patch_pred.py – patch-level classification/prediction via TIAToolbox PatchPredictor.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..models import FileEntry
from ..openslide_utils import format_roi_error
from .utils import ensure_output_dir, get_device, resolve_task_device, roi_to_bounds

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "resnet18-kather100k"


def run_patch_prediction(
    entry: FileEntry,
    run_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    patch_size: int = 224,
    stride: int = 224,
    roi: dict[str, Any] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    on_gpu: bool | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    """
    Run patch-level prediction and write outputs to *run_dir/outputs/*.

    Returns dict with keys: csv_path, overlay_path, status, error,
    device_used, device_fallback_reason, warnings.
    """
    if log_fn is None:
        log_fn = logger.info

    if on_gpu is not None and device is None:
        device = "cuda" if on_gpu else "cpu"
    _, tia_on_gpu, device_used, fallback_reason = resolve_task_device(device)
    run_warnings: list[str] = []
    if fallback_reason:
        run_warnings.append(fallback_reason)
        log_fn(f"WARNING: {fallback_reason}")

    out_dir = ensure_output_dir(run_dir)

    bounds = roi_to_bounds(roi)
    input_path = entry.stored_path

    if bounds is not None and entry.is_wsi:
        log_fn(f"Extracting ROI {bounds} from WSI for patch prediction …")
        input_path, roi_err = _extract_roi(entry, bounds, out_dir)
        if input_path is None:
            return {
                "status": "failed",
                "error": roi_err or "ROI extraction failed",
                "csv_path": None, "overlay_path": None,
                "device_used": device_used, "device_fallback_reason": fallback_reason,
                "warnings": run_warnings,
            }

    log_fn(f"Running patch prediction (model={model}, patch_size={patch_size}, device={device_used}) …")

    try:
        from tiatoolbox.models.engine.patch_predictor import (  # type: ignore
            IOPatchPredictorConfig,
            PatchPredictor,
        )
    except ImportError as exc:
        return {
            "status": "failed", "error": str(exc),
            "csv_path": None, "overlay_path": None,
            "device_used": device_used, "device_fallback_reason": fallback_reason,
            "warnings": run_warnings,
        }

    with tempfile.TemporaryDirectory() as tmp:
        try:
            predictor = PatchPredictor(
                pretrained_model=model,
                batch_size=batch_size,
                num_loader_workers=num_workers,
            )
            ioconfig = IOPatchPredictorConfig(
                input_resolutions=[{"resolution": 0.5, "units": "mpp"}],
                patch_input_shape=[patch_size, patch_size],
                stride_shape=[stride, stride],
            )
            output = predictor.predict(
                imgs=[str(input_path)],
                mode="tile" if not entry.is_wsi else "wsi",
                on_gpu=tia_on_gpu,
                ioconfig=ioconfig,
                save_dir=tmp,
                crash_on_exception=True,
            )
        except RuntimeError as exc:
            err_str = str(exc)
            run_warnings.append(f"RuntimeError on {device_used}: {err_str}")
            logger.exception("PatchPredictor failed")
            return {
                "status": "failed", "error": err_str,
                "csv_path": None, "overlay_path": None,
                "device_used": device_used, "device_fallback_reason": fallback_reason,
                "warnings": run_warnings,
            }
        except Exception as exc:
            logger.exception("PatchPredictor failed")
            return {
                "status": "failed", "error": str(exc),
                "csv_path": None, "overlay_path": None,
                "device_used": device_used, "device_fallback_reason": fallback_reason,
                "warnings": run_warnings,
            }

        csv_path, overlay_path = _collect_patch_outputs(Path(tmp), out_dir, log_fn)

    log_fn("Patch prediction completed.")
    return {
        "status": "completed",
        "error": "",
        "csv_path": str(csv_path) if csv_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "device_used": device_used,
        "device_fallback_reason": fallback_reason,
        "warnings": run_warnings,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_roi(entry: FileEntry, bounds: tuple[int, int, int, int], out_dir: Path) -> tuple[Path | None, str]:
    """Extract ROI from WSI and save as PNG.  Returns (path, error_str)."""
    try:
        from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
        reader = WSIReader.open(entry.stored_path)
        region = reader.read_bounds(bounds, resolution=0, units="level")
        reader.close()
        from PIL import Image
        img = Image.fromarray(region[..., :3])
        roi_path = out_dir / "roi_patch_input.png"
        img.save(roi_path)
        return roi_path, ""
    except Exception as exc:
        exc_str = str(exc)
        logger.error("ROI extraction error: %s", exc_str)
        return None, format_roi_error(exc_str)


def _collect_patch_outputs(
    tmp_dir: Path, out_dir: Path, log_fn: Any
) -> tuple[Path | None, Path | None]:
    csv_path: Path | None = None
    overlay_path: Path | None = None

    for f in tmp_dir.rglob("*.csv"):
        dest = out_dir / "patch_predictions.csv"
        shutil.copy2(f, dest)
        csv_path = dest
        break

    for f in tmp_dir.rglob("*.npy"):
        try:
            import numpy as _np
            arr = _np.load(f)
            dest_npy = out_dir / "patch_map.npy"
            _np.save(dest_npy, arr)
            if arr.ndim >= 2:
                from PIL import Image as _Img
                vis = arr if arr.ndim == 2 else arr[..., 0]
                norm = vis - vis.min()
                mx = norm.max()
                if mx > 0:
                    norm = (norm / mx * 255).astype(_np.uint8)
                ov_path = out_dir / "patch_overlay.png"
                _Img.fromarray(norm).convert("RGB").save(ov_path)
                overlay_path = ov_path
        except Exception as exc:
            log_fn(f"Could not convert patch map: {exc}")

    for ext in ("*.png", "*.jpg"):
        for f in tmp_dir.rglob(ext):
            dest = out_dir / f.name
            shutil.copy2(f, dest)
            if overlay_path is None:
                overlay_path = dest

    return csv_path, overlay_path
