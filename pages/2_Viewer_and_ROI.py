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
from core.channel_config import get_channel_percentiles
from core.config import MAX_ROI_AREA_WARNING
from core.io import get_channel_info, make_channel_thumbnail, make_thumbnail_with_error
from core.models import ROI
from core.viz import pil_to_bytes

st.set_page_config(page_title="Viewer & ROI", page_icon=None, layout="wide")
st.title("Viewer & ROI")

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
# Channel detection & naming
# ---------------------------------------------------------------------------
if "channel_names_by_file" not in st.session_state:
    st.session_state["channel_names_by_file"] = {}

# Lazily detect channel count if not yet stored in the registry entry
if entry.channel_count == 0 and not entry.is_wsi:
    n_ch, default_names = get_channel_info(entry.stored_path)
    if n_ch > 0:
        entry.channel_count = n_ch
        if not entry.channel_names:
            entry.channel_names = default_names
        registry.update_file(entry)
else:
    n_ch = entry.channel_count if entry.channel_count > 0 else 1
    default_names = [f"Ch {i}" for i in range(n_ch)]

# Resolve channel names: session state > registry > defaults
ch_names: list[str] = list(
    st.session_state["channel_names_by_file"].get(
        entry.file_id,
        entry.channel_names if entry.channel_names else default_names,
    )
)
# Ensure list is long enough for every channel
while len(ch_names) < n_ch:
    ch_names.append(f"Ch {len(ch_names)}")
st.session_state["channel_names_by_file"][entry.file_id] = ch_names

# ---------------------------------------------------------------------------
# Thumbnail + channel controls (for multi-channel TIFF)
# ---------------------------------------------------------------------------
st.subheader(f"Thumbnail – {entry.original_name}")

is_multichannel_tiff = (
    not entry.is_wsi
    and n_ch > 1
    and entry.stored_path.suffix.lower() in (".tif", ".tiff", ".btf")
)

selected_ch = 0
if is_multichannel_tiff:
    ch_labels = [f"{i}: {ch_names[i]}" for i in range(n_ch)]
    selected_ch_label = st.selectbox(
        "Displayed channel",
        ch_labels,
        key=f"disp_ch_{entry.file_id}",
    )
    selected_ch = ch_labels.index(selected_ch_label)

    col_rename, col_btn = st.columns([3, 1])
    with col_rename:
        new_name = st.text_input(
            "Rename channel",
            value=ch_names[selected_ch],
            key=f"rename_ch_{entry.file_id}_{selected_ch}",
        )
    with col_btn:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Apply", key=f"apply_name_{entry.file_id}_{selected_ch}"):
            ch_names[selected_ch] = new_name
            st.session_state["channel_names_by_file"][entry.file_id] = ch_names
            entry.channel_names = ch_names
            registry.update_file(entry)
            st.success(f"Channel {selected_ch} renamed to '{new_name}'.")
            st.rerun()

# Generate the thumbnail
with st.spinner("Generating thumbnail …"):
    if is_multichannel_tiff:
        # Read per-channel percentile settings from channel_configs if available
        ch_cfgs = st.session_state.get("channel_configs_by_file", {}).get(
            entry.file_id, {}
        )
        low_pct, high_pct = get_channel_percentiles(ch_cfgs, selected_ch)
        thumb, thumb_err = make_channel_thumbnail(
            entry, channel_idx=selected_ch, max_size=600,
            low_pct=low_pct, high_pct=high_pct,
        )
    else:
        thumb, thumb_err = make_thumbnail_with_error(entry, max_size=600)

if thumb is None:
    st.error(thumb_err or "Could not generate thumbnail for this file.")
else:
    col_img, col_info = st.columns([2, 1])
    with col_img:
        caption = (
            f"Channel {selected_ch}: {ch_names[selected_ch]}" if is_multichannel_tiff
            else "Slide thumbnail (downsampled)"
        )
        st.image(pil_to_bytes(thumb), caption=caption, use_container_width=True)
    with col_info:
        st.markdown(f"**File:** `{entry.original_name}`")
        st.markdown(f"**Type:** {'WSI' if entry.is_wsi else 'image'}")
        if entry.width and entry.height:
            st.markdown(f"**Full dimensions:** {entry.width} x {entry.height} px")
        if entry.mpp:
            st.markdown(f"**Resolution:** {entry.mpp} um/px")
        if n_ch > 1:
            st.markdown(f"**Channels:** {n_ch}")
            names_preview = ", ".join(ch_names[:5])
            if n_ch > 5:
                names_preview += f" … (+{n_ch - 5} more)"
            st.markdown(f"**Channel names:** {names_preview}")

# ---------------------------------------------------------------------------
# ROI definition
# ---------------------------------------------------------------------------
st.subheader("Define Region of Interest (ROI)")

if entry.is_wsi:
    st.warning(
        "This is a large WSI.  Running tasks on the full slide may take a long time and "
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
            f"ROI area is {area:,} px2 ({area / 1e6:.1f} Mpx).  "
            "This may be slow on CPU – consider reducing the region."
        )
    else:
        st.success(f"ROI area: {area:,} px²")

    if st.button("Save ROI", key=f"save_roi_{entry.file_id}"):
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

