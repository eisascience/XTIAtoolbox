"""
pages/5_Batch_Queue.py – queue multiple file × task jobs and run them sequentially.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import registry
from core.config import WORKSPACE_ROOT
from core.models import BatchJob
from core.runner import run_task

# ---------------------------------------------------------------------------
# Persistent batch queue stored in workspace
# ---------------------------------------------------------------------------
QUEUE_FILE = WORKSPACE_ROOT / "batch_queue.json"


def _load_queue() -> list[BatchJob]:
    if not QUEUE_FILE.exists():
        return []
    try:
        with open(QUEUE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return [BatchJob.from_dict(d) for d in data]
    except Exception:
        return []


def _save_queue(jobs: list[BatchJob]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_FILE, "w", encoding="utf-8") as fh:
        json.dump([j.as_dict() for j in jobs], fh, indent=2)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Batch Queue", page_icon="📋", layout="wide")
st.title("📋 Batch Queue")

TASK_LABELS = {
    "nuclei_detection": "Nuclei Detection (HoVer-Net)",
    "semantic_segmentation": "Semantic Segmentation (Tissue)",
    "patch_prediction": "Patch-Level Prediction",
}

# ---------------------------------------------------------------------------
# Add jobs to queue
# ---------------------------------------------------------------------------
st.header("Add jobs to queue")

entries = registry.list_files()
if not entries:
    st.warning("No files uploaded.  Go to **Upload & Manage** first.")
else:
    with st.form("add_batch_job"):
        col1, col2 = st.columns(2)
        with col1:
            selected_files = st.multiselect(
                "Files to process",
                [e.original_name for e in entries],
                default=[e.original_name for e in entries[:1]],
            )
        with col2:
            task_label = st.selectbox("Task", list(TASK_LABELS.values()))
            task_name = {v: k for k, v in TASK_LABELS.items()}[task_label]

        # Minimal shared parameters
        batch_size = st.slider("Batch size", 1, 32, 4, key="batch_bs")

        # Model select per task
        if task_name == "nuclei_detection":
            model = st.selectbox("Model", [
                "hovernet_fast-pannuke", "hovernet_fast-monusac",
                "hovernet-pannuke", "hovernet-monusac",
            ], key="batch_model_n")
        elif task_name == "semantic_segmentation":
            model = st.selectbox("Model", ["fcn_resnet50_unet-bcss", "unet-bcss"], key="batch_model_s")
        else:
            model = st.selectbox("Model", [
                "resnet18-kather100k", "resnet34-kather100k",
                "alexnet-kather100k",
            ], key="batch_model_p")

        use_roi = st.checkbox("Apply saved ROI for each file (if set)", value=True)

        submitted = st.form_submit_button("➕ Add to queue")
        if submitted:
            queue = _load_queue()
            added = 0
            for fname in selected_files:
                file_entry = next((e for e in entries if e.original_name == fname), None)
                if file_entry is None:
                    continue
                roi_data = None
                if use_roi:
                    roi_data = st.session_state.get(f"roi_{file_entry.file_id}")
                job = BatchJob(
                    file_id=file_entry.file_id,
                    task_name=task_name,
                    task_params={"model": model, "batch_size": batch_size, "num_workers": 0},
                    roi=roi_data,
                )
                queue.append(job)
                added += 1
            _save_queue(queue)
            st.success(f"Added {added} job(s) to the queue.")

# ---------------------------------------------------------------------------
# Queue table
# ---------------------------------------------------------------------------
st.divider()
st.header("Current queue")

queue = _load_queue()
file_map = {e.file_id: e.original_name for e in registry.list_files()}

if not queue:
    st.info("Queue is empty.")
else:
    for i, job in enumerate(queue):
        fname = file_map.get(job.file_id, f"<unknown:{job.file_id[:8]}>")
        status_icon = {
            "queued": "🕐",
            "running": "⏳",
            "completed": "✅",
            "failed": "❌",
        }.get(job.status, "❓")

        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown(
                f"{status_icon} **{fname}** → {TASK_LABELS.get(job.task_name, job.task_name)} "
                f"(job `{job.job_id[:8]}`)"
            )
            if job.status == "failed":
                st.error(f"Error: {job.error}")
            if job.run_id:
                st.caption(f"Run ID: {job.run_id[:8]}")
        with col2:
            if st.button("🗑️ Remove", key=f"rm_job_{job.job_id}"):
                queue = [j for j in queue if j.job_id != job.job_id]
                _save_queue(queue)
                st.rerun()
        with col3:
            if job.status == "failed" and st.button("↩️ Reset", key=f"reset_{job.job_id}"):
                job.status = "queued"
                job.error = ""
                _save_queue(queue)
                st.rerun()

    st.divider()

    # Clear completed jobs
    col_clear, col_run = st.columns(2)
    with col_clear:
        if st.button("🧹 Clear completed/failed"):
            queue = [j for j in queue if j.status not in ("completed", "failed")]
            _save_queue(queue)
            st.rerun()

    # Run all queued jobs
    with col_run:
        run_all = st.button("▶️ Run all queued jobs", type="primary")

    if run_all:
        queued_jobs = [j for j in queue if j.status == "queued"]
        if not queued_jobs:
            st.warning("No queued jobs to run.")
        else:
            progress = st.progress(0, text="Running batch …")
            log_placeholder = st.empty()
            log_lines: list[str] = []

            def _log(msg: str) -> None:
                log_lines.append(msg)
                log_placeholder.code("\n".join(log_lines[-40:]), language=None)

            for idx, job in enumerate(queued_jobs):
                file_entry = registry.get_file(job.file_id)
                if file_entry is None:
                    job.status = "failed"
                    job.error = f"File {job.file_id} not found in registry"
                    _save_queue(queue)
                    continue

                # Mark as running
                job.status = "running"
                _save_queue(queue)

                _log(f"--- Job {idx + 1}/{len(queued_jobs)}: {file_entry.original_name} / {job.task_name} ---")

                try:
                    manifest = run_task(
                        entry=file_entry,
                        task_name=job.task_name,
                        task_params=job.task_params,
                        roi=job.roi,
                        log_fn=_log,
                    )
                    job.status = manifest.status
                    job.run_id = manifest.run_id
                    job.error = manifest.error or ""
                except Exception as exc:
                    job.status = "failed"
                    job.error = str(exc)
                    _log(f"ERROR: {exc}")

                _save_queue(queue)
                progress.progress((idx + 1) / len(queued_jobs))

            progress.empty()
            st.success("Batch run finished.  Go to **Results & Downloads** to view outputs.")
            st.rerun()
