"""
pages/3_Run_Tasks.py – configure and execute TIAToolbox tasks interactively.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import registry
from core.runner import run_task
from core.tasks.utils import get_device, get_tiatoolbox_version

st.set_page_config(page_title="Run Tasks", page_icon="⚙️", layout="wide")
st.title("⚙️ Run Tasks")

# ---------------------------------------------------------------------------
# File selector
# ---------------------------------------------------------------------------
entries = registry.list_files()
if not entries:
    st.warning("No files uploaded.  Go to **Upload & Manage** first.")
    st.stop()

file_names = [e.original_name for e in entries]
active_idx = 0
if "active_file_id" in st.session_state:
    ids = [e.file_id for e in entries]
    if st.session_state.active_file_id in ids:
        active_idx = ids.index(st.session_state.active_file_id)

selected = st.selectbox("Select file to analyse", file_names, index=active_idx)
entry = next(e for e in entries if e.original_name == selected)
st.session_state["active_file_id"] = entry.file_id

roi_key = f"roi_{entry.file_id}"
current_roi = st.session_state.get(roi_key)

col_info, col_roi = st.columns(2)
with col_info:
    st.markdown(f"**File:** `{entry.original_name}`")
    st.markdown(f"**Type:** {'WSI' if entry.is_wsi else 'image'}")
    if entry.width and entry.height:
        st.markdown(f"**Dimensions:** {entry.width} × {entry.height} px")
with col_roi:
    if current_roi:
        st.success(f"ROI: x={current_roi['x']}, y={current_roi['y']}, "
                   f"w={current_roi['width']}, h={current_roi['height']}, "
                   f"level={current_roi.get('level', 0)}")
    else:
        if entry.is_wsi:
            st.warning("⚠️ No ROI set for this WSI. Full-slide analysis may be very slow.")
        else:
            st.info("No ROI set – full image will be analysed.")

st.divider()

# ---------------------------------------------------------------------------
# Task selector
# ---------------------------------------------------------------------------
TASK_LABELS = {
    "nuclei_detection": "Nuclei Detection (HoVer-Net)",
    "semantic_segmentation": "Semantic Segmentation (Tissue)",
    "patch_prediction": "Patch-Level Prediction",
}

st.header("Task configuration")
task_label = st.selectbox("Task", list(TASK_LABELS.values()))
task_name = {v: k for k, v in TASK_LABELS.items()}[task_label]

# ---------------------------------------------------------------------------
# Task-specific parameters
# ---------------------------------------------------------------------------
task_params: dict = {}

if task_name == "nuclei_detection":
    NUCLEI_MODELS = [
        "hovernet_fast-pannuke",
        "hovernet_fast-monusac",
        "hovernet-pannuke",
        "hovernet-monusac",
        "hovernet-consep",
    ]
    task_params["model"] = st.selectbox("HoVer-Net pretrained model", NUCLEI_MODELS)
    task_params["batch_size"] = st.slider("Batch size", 1, 32, 4)
    task_params["num_workers"] = st.slider("DataLoader workers (0 = main thread)", 0, 8, 0)

elif task_name == "semantic_segmentation":
    SEG_MODELS = [
        "fcn_resnet50_unet-bcss",
        "unet-bcss",
    ]
    task_params["model"] = st.selectbox("Segmentation model", SEG_MODELS)
    task_params["batch_size"] = st.slider("Batch size", 1, 32, 4)
    task_params["num_workers"] = st.slider("DataLoader workers", 0, 8, 0)

elif task_name == "patch_prediction":
    PATCH_MODELS = [
        "resnet18-kather100k",
        "resnet34-kather100k",
        "alexnet-kather100k",
        "densenet121-kather100k",
        "mobilenet_v2-kather100k",
        "mobilenet_v3_large-kather100k",
        "mobilenet_v3_small-kather100k",
        "resnext50_32x4d-kather100k",
        "googlenet-kather100k",
    ]
    task_params["model"] = st.selectbox("Patch model", PATCH_MODELS)
    task_params["patch_size"] = st.slider("Patch size (px)", 64, 512, 224, step=32)
    task_params["stride"] = st.slider("Stride (px)", 32, 512, 224, step=32)
    task_params["batch_size"] = st.slider("Batch size", 1, 128, 32)
    task_params["num_workers"] = st.slider("DataLoader workers", 0, 8, 0)

# GPU toggle
device_available = get_device()
use_gpu = st.checkbox(
    f"Use GPU (cuda) – {'available ✅' if device_available == 'cuda' else 'not available ❌'}",
    value=(device_available == "cuda"),
    disabled=(device_available != "cuda"),
)
task_params["on_gpu"] = use_gpu

st.caption(
    f"TIAToolbox version: **{get_tiatoolbox_version()}** | "
    f"Device: **{device_available}**"
)

st.divider()

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
if "run_log" not in st.session_state:
    st.session_state.run_log = []
if "last_run_manifest" not in st.session_state:
    st.session_state.last_run_manifest = None

run_btn = st.button("▶️ Run Task", type="primary")

log_placeholder = st.empty()
status_placeholder = st.empty()


def _append_log(msg: str) -> None:
    st.session_state.run_log.append(msg)


if run_btn:
    # Warn for full WSI without ROI
    if entry.is_wsi and not current_roi:
        st.warning(
            "⚠️ You are about to run on the FULL slide without a ROI.  "
            "This may take a very long time.  Proceed with caution."
        )

    st.session_state.run_log = []
    st.session_state.last_run_manifest = None

    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        log_placeholder.code("\n".join(log_lines), language=None)

    with st.spinner(f"Running {task_label} …"):
        try:
            manifest = run_task(
                entry=entry,
                task_name=task_name,
                task_params={k: v for k, v in task_params.items() if k != "on_gpu"},
                roi=current_roi,
                log_fn=_log,
            )
            st.session_state.last_run_manifest = manifest
            if manifest.status == "completed":
                status_placeholder.success(
                    f"✅ Run **{manifest.run_id[:8]}** completed in {manifest.duration_seconds}s"
                )
            else:
                status_placeholder.error(
                    f"❌ Run **{manifest.run_id[:8]}** failed: {manifest.error}"
                )
        except Exception as exc:
            status_placeholder.error(f"Run failed: {exc}")
            st.exception(exc)

    log_placeholder.code("\n".join(log_lines), language=None)

# Show last manifest
if st.session_state.last_run_manifest:
    m = st.session_state.last_run_manifest
    with st.expander("📋 Run manifest", expanded=True):
        import json
        st.json(m.as_dict())
    st.info("Go to **Results & Downloads** to view outputs and launch the viewer.")
