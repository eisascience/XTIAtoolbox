"""
core/tasks/semantic.py – semantic segmentation via TIAToolbox SemanticSegmentor.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..models import FileEntry
from .utils import ensure_output_dir, get_device, resolve_task_device, roi_to_bounds

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fcn_resnet50_unet-bcss"


def run_semantic_segmentation(
    entry: FileEntry,
    run_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    roi: dict[str, Any] | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str | None = None,
    on_gpu: bool | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    """
    Run tissue semantic segmentation and write outputs to *run_dir/outputs/*.

    Returns a dict with keys: mask_path, overlay_path, status, error,
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
        log_fn(f"Extracting ROI {bounds} from WSI for segmentation …")
        input_path = _extract_roi(entry, bounds, out_dir)
        if input_path is None:
            return {
                "status": "failed",
                "error": "ROI extraction failed",
                "mask_path": None, "overlay_path": None,
                "device_used": device_used, "device_fallback_reason": fallback_reason,
                "warnings": run_warnings,
            }

    log_fn(f"Running semantic segmentation (model={model}, device={device_used}) …")

    try:
        from tiatoolbox.models.engine.semantic_segmentor import (  # type: ignore
            SemanticSegmentor,
        )
    except ImportError as exc:
        return {
            "status": "failed", "error": str(exc),
            "mask_path": None, "overlay_path": None,
            "device_used": device_used, "device_fallback_reason": fallback_reason,
            "warnings": run_warnings,
        }

    with tempfile.TemporaryDirectory() as tmp:
        try:
            segmentor = SemanticSegmentor(
                pretrained_model=model,
                num_loader_workers=num_workers,
                batch_size=batch_size,
            )
            output = segmentor.predict(
                imgs=[str(input_path)],
                save_dir=tmp,
                on_gpu=tia_on_gpu,
                crash_on_exception=True,
                mode="wsi" if entry.is_wsi else "tile",
            )
        except RuntimeError as exc:
            err_str = str(exc)
            run_warnings.append(f"RuntimeError on {device_used}: {err_str}")
            logger.exception("SemanticSegmentor failed")
            return {
                "status": "failed", "error": err_str,
                "mask_path": None, "overlay_path": None,
                "device_used": device_used, "device_fallback_reason": fallback_reason,
                "warnings": run_warnings,
            }
        except Exception as exc:
            logger.exception("SemanticSegmentor failed")
            return {
                "status": "failed", "error": str(exc),
                "mask_path": None, "overlay_path": None,
                "device_used": device_used, "device_fallback_reason": fallback_reason,
                "warnings": run_warnings,
            }

        mask_path, overlay_path = _collect_seg_outputs(Path(tmp), out_dir, input_path, log_fn)

    log_fn("Semantic segmentation completed.")
    return {
        "status": "completed",
        "error": "",
        "mask_path": str(mask_path) if mask_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "device_used": device_used,
        "device_fallback_reason": fallback_reason,
        "warnings": run_warnings,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_roi(entry: FileEntry, bounds: tuple[int, int, int, int], out_dir: Path) -> Path | None:
    try:
        from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
        reader = WSIReader.open(entry.stored_path)
        region = reader.read_bounds(bounds, resolution=0, units="level")
        reader.close()
        from PIL import Image
        img = Image.fromarray(region[..., :3])
        roi_path = out_dir / "roi_seg_input.png"
        img.save(roi_path)
        return roi_path
    except Exception as exc:
        logger.error("ROI extraction error: %s", exc)
        return None


def _collect_seg_outputs(
    tmp_dir: Path, out_dir: Path, input_path: Path, log_fn: Any
) -> tuple[Path | None, Path | None]:
    mask_path: Path | None = None
    overlay_path: Path | None = None

    for f in tmp_dir.rglob("*.npy"):
        dest = out_dir / "seg_mask.npy"
        shutil.copy2(f, dest)
        mask_path = dest
        # Generate a PNG overlay
        try:
            import numpy as _np
            from PIL import Image
            mask = _np.load(dest)
            # Normalise for display
            if mask.ndim == 2:
                norm = (mask - mask.min())
                mx = norm.max()
                if mx > 0:
                    norm = (norm / mx * 255).astype(_np.uint8)
                else:
                    norm = norm.astype(_np.uint8)
                overlay = Image.fromarray(norm, mode="L").convert("RGB")
                ov_path = out_dir / "seg_overlay.png"
                overlay.save(ov_path)
                overlay_path = ov_path
        except Exception as exc:
            log_fn(f"Could not generate overlay: {exc}")
        break

    for ext in ("*.png", "*.jpg", "*.tif"):
        for f in tmp_dir.rglob(ext):
            dest = out_dir / f.name
            shutil.copy2(f, dest)
            if overlay_path is None:
                overlay_path = dest

    return mask_path, overlay_path
