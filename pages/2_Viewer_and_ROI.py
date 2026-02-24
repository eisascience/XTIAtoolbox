"""
pages/2_Viewer_and_ROI.py – inspect slide thumbnail and define a region of interest (ROI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import registry
from core.config import MAX_ROI_AREA_WARNING
from core.io import make_thumbnail
from core.models import ROI
from core.viz import pil_to_bytes

st.set_page_config(page_title="Viewer & ROI", page_icon="🗺️", layout="wide")
st.title("🗺️ Viewer & ROI")

st.info(
    "This page lets you inspect a slide thumbnail and define a **Region of Interest (ROI)** "
    "to limit analysis to a manageable area.  "
    "For full interactive slide browsing, use the **Launch Viewer** button on the Results page."
)

# ---------------------------------------------------------------------------
# File selector
# ---------------------------------------------------------------------------
entries = registry.list_files()
if not entries:
    st.warning("No files found.  Go to **Upload & Manage** to upload files first.")
    st.stop()

file_names = [e.original_name for e in entries]
active_idx = 0
if "active_file_id" in st.session_state:
    ids = [e.file_id for e in entries]
    if st.session_state.active_file_id in ids:
        active_idx = ids.index(st.session_state.active_file_id)

selected = st.selectbox("Select file", file_names, index=active_idx)
entry = next(e for e in entries if e.original_name == selected)
st.session_state["active_file_id"] = entry.file_id

# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------
st.subheader(f"Thumbnail – {entry.original_name}")
with st.spinner("Generating thumbnail …"):
    thumb = make_thumbnail(entry, max_size=600)

if thumb is None:
    st.error("Could not generate thumbnail for this file.")
else:
    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(pil_to_bytes(thumb), caption="Slide thumbnail (downsampled)", use_container_width=True)
    with col_info:
        st.markdown(f"**File:** `{entry.original_name}`")
        st.markdown(f"**Type:** {'WSI' if entry.is_wsi else 'image'}")
        if entry.width and entry.height:
            st.markdown(f"**Full dimensions:** {entry.width} × {entry.height} px")
        if entry.mpp:
            st.markdown(f"**Resolution:** {entry.mpp} µm/px")

# ---------------------------------------------------------------------------
# ROI definition
# ---------------------------------------------------------------------------
st.subheader("Define Region of Interest (ROI)")

if entry.is_wsi:
    st.warning(
        "⚠️ This is a large WSI.  Running tasks on the full slide may take a long time and "
        "consume significant memory.  **It is strongly recommended** to set a ROI."
    )

# Load existing ROI from session state (scoped to file_id)
roi_key = f"roi_{entry.file_id}"
existing_roi: dict = st.session_state.get(roi_key, {})

use_roi = st.checkbox(
    "Enable ROI (restrict analysis to a sub-region)",
    value=bool(existing_roi),
    key=f"use_roi_{entry.file_id}",
)

roi_dict: dict | None = None
if use_roi:
    max_w = entry.width if entry.width else 10000
    max_h = entry.height if entry.height else 10000

    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("X (left edge, px)", min_value=0, max_value=max(max_w - 1, 0),
                             value=int(existing_roi.get("x", 0)), step=64, key=f"roi_x_{entry.file_id}")
        w = st.number_input("Width (px)", min_value=64, max_value=max_w,
                             value=int(existing_roi.get("width", min(2048, max_w))),
                             step=64, key=f"roi_w_{entry.file_id}")
    with col2:
        y = st.number_input("Y (top edge, px)", min_value=0, max_value=max(max_h - 1, 0),
                             value=int(existing_roi.get("y", 0)), step=64, key=f"roi_y_{entry.file_id}")
        h = st.number_input("Height (px)", min_value=64, max_value=max_h,
                             value=int(existing_roi.get("height", min(2048, max_h))),
                             step=64, key=f"roi_h_{entry.file_id}")

    level = st.number_input("Pyramid level (0 = full resolution)", min_value=0, max_value=8,
                             value=int(existing_roi.get("level", 0)), key=f"roi_lvl_{entry.file_id}")

    roi_dict = {"x": int(x), "y": int(y), "width": int(w), "height": int(h), "level": int(level)}

    area = int(w) * int(h)
    if area > MAX_ROI_AREA_WARNING:
        st.warning(
            f"⚠️ ROI area is {area:,} px² ({area / 1e6:.1f} Mpx).  "
            "This may be slow on CPU – consider reducing the region."
        )
    else:
        st.success(f"ROI area: {area:,} px²")

    if st.button("💾 Save ROI", key=f"save_roi_{entry.file_id}"):
        st.session_state[roi_key] = roi_dict
        st.success(f"ROI saved: x={x}, y={y}, w={w}, h={h}, level={level}")
else:
    # Clear stored ROI
    st.session_state.pop(roi_key, None)
    st.info("No ROI set – tasks will run on the full image.")

# Show current ROI in session
current_roi = st.session_state.get(roi_key)
if current_roi:
    st.caption(f"Current ROI: {current_roi}")
