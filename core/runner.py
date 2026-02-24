"""
core/runner.py – top-level task runner that writes the run manifest.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import RUNS_DIR
from .hashing import sha256_file
from .models import FileEntry, RunManifest
from .tasks import TASK_MAP
from .tasks.utils import get_available_devices, get_device, get_python_version, get_tiatoolbox_version

logger = logging.getLogger(__name__)


def run_task(
    entry: FileEntry,
    task_name: str,
    task_params: dict[str, Any],
    roi: dict[str, Any] | None = None,
    device: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> RunManifest:
    """
    Execute *task_name* on *entry* and return a completed RunManifest.

    Parameters
    ----------
    entry : FileEntry
        The input file to process.
    task_name : str
        One of the keys in TASK_MAP.
    task_params : dict
        Task-specific keyword arguments (model, batch_size, etc.).
    roi : dict | None
        Optional ROI dict {x, y, width, height}.
    device : str | None
        Requested compute device: "cpu", "cuda", or "mps".  When None, the
        best available device is detected automatically.
    log_fn : callable | None
        A callable that accepts a string for UI logging (e.g. st.write).
    """
    if log_fn is None:
        log_fn = logger.info

    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {list(TASK_MAP)}")

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve requested device: fall back to best available when not specified.
    device_requested = device or get_available_devices()[0]

    manifest = RunManifest(
        run_id=run_id,
        task_name=task_name,
        task_params=task_params,
        input_files=[
            {
                "name": entry.original_name,
                "file_id": entry.file_id,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
        ],
        roi=roi,
        device=device_requested,
        device_requested=device_requested,
        tiatoolbox_version=get_tiatoolbox_version(),
        python_version=get_python_version(),
        status="running",
    )
    _write_manifest(run_dir, manifest)

    log_fn(f"[run:{run_id[:8]}] Starting task '{task_name}' on '{entry.original_name}' …")
    log_fn(f"[run:{run_id[:8]}] Requested device: {device_requested}")
    start = time.monotonic()

    try:
        task_fn = TASK_MAP[task_name]
        result = task_fn(
            entry,
            run_dir,
            roi=roi,
            device=device_requested,
            log_fn=log_fn,
            **task_params,
        )
        elapsed = time.monotonic() - start
        manifest.status = result.get("status", "completed")
        manifest.error = result.get("error", "")
        manifest.duration_seconds = round(elapsed, 2)

        # Record actual device used (may differ from requested if fallback occurred)
        manifest.device_used = result.get("device_used", device_requested)
        manifest.device_fallback_reason = result.get("device_fallback_reason", "")
        manifest.device = manifest.device_used  # keep legacy field consistent
        manifest.warnings = result.get("warnings", [])

        if manifest.device_fallback_reason:
            log_fn(
                f"[run:{run_id[:8]}] Device fallback: {device_requested} → "
                f"{manifest.device_used} ({manifest.device_fallback_reason})"
            )

        # Record output file paths relative to run_dir
        for key in ("geojson_path", "csv_path", "mask_path", "overlay_path", "patch_csv_path"):
            val = result.get(key)
            if val:
                rel = str(Path(val).relative_to(run_dir))
                manifest.outputs.append(rel)

        log_fn(f"[run:{run_id[:8]}] Task finished in {elapsed:.1f}s – status={manifest.status}")

    except Exception as exc:
        elapsed = time.monotonic() - start
        manifest.status = "failed"
        manifest.error = str(exc)
        manifest.duration_seconds = round(elapsed, 2)
        logger.exception("Task %s failed", task_name)
        log_fn(f"[run:{run_id[:8]}] ERROR: {exc}")

    _write_manifest(run_dir, manifest)
    return manifest


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    path = run_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest.as_dict(), fh, indent=2)


def list_runs() -> list[RunManifest]:
    """Return all persisted run manifests, newest first."""
    manifests = []
    for manifest_file in sorted(RUNS_DIR.glob("*/manifest.json"), reverse=True):
        try:
            with open(manifest_file, encoding="utf-8") as fh:
                data = json.load(fh)
            manifests.append(RunManifest.from_dict(data))
        except Exception as exc:
            logger.warning("Could not load manifest %s: %s", manifest_file, exc)
    return manifests


def get_run(run_id: str) -> RunManifest | None:
    path = RUNS_DIR / run_id / "manifest.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return RunManifest.from_dict(json.load(fh))
    except Exception as exc:
        logger.warning("Could not load run %s: %s", run_id, exc)
        return None
