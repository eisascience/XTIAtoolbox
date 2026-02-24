"""
pages/1_Upload_and_Manage.py – upload files and manage the workspace registry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure core package is importable when running from the pages/ subdirectory
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import registry
from core.io import delete_upload, make_thumbnail_with_error, save_upload, validate_extension
from core.viz import pil_to_bytes

st.set_page_config(page_title="Upload & Manage", page_icon="📁", layout="wide")
st.title("📁 Upload & Manage")

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
st.header("Upload files")

SUPPORTED_TYPES = [
    "svs", "ndpi", "mrxs", "scn", "tif", "tiff",
    "png", "jpg", "jpeg", "bmp",
]

uploaded_files = st.file_uploader(
    "Select one or more slide / image files",
    accept_multiple_files=True,
    type=SUPPORTED_TYPES,
    help="Supported: SVS, NDPI, MRXS, SCN, OME-TIFF, TIFF, PNG, JPEG, BMP",
)

if uploaded_files:
    progress = st.progress(0, text="Saving uploads …")
    for i, uf in enumerate(uploaded_files):
        if not validate_extension(uf.name):
            st.warning(f"Unsupported file type: {uf.name}")
            continue
        # Check for duplicates by name
        existing = [e for e in registry.list_files() if e.original_name == uf.name]
        if existing:
            st.info(f"'{uf.name}' already in workspace (file_id={existing[0].file_id[:8]}). Skipping.")
            continue
        try:
            entry = save_upload(uf.read(), uf.name)
            registry.add_file(entry)
            st.success(
                f"✅ Saved **{uf.name}** "
                f"({'WSI' if entry.is_wsi else 'image'}, "
                f"{entry.size_bytes / 1024:.1f} KB, "
                f"sha256={entry.sha256[:12]}…)"
            )
        except Exception as exc:
            st.error(f"Failed to save {uf.name}: {exc}")
        progress.progress((i + 1) / len(uploaded_files))
    progress.empty()

# ---------------------------------------------------------------------------
# File registry table
# ---------------------------------------------------------------------------
st.header("Workspace files")

entries = registry.list_files()
if not entries:
    st.info("No files uploaded yet.")
else:
    for entry in entries:
        with st.expander(
            f"{'🔬 WSI' if entry.is_wsi else '🖼 Image'} – **{entry.original_name}**  "
            f"(ID: {entry.file_id[:8]}…)",
            expanded=False,
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                thumb, thumb_err = make_thumbnail_with_error(entry, max_size=256)
                if thumb is not None:
                    st.image(pil_to_bytes(thumb), caption="Thumbnail", use_container_width=True)
                else:
                    st.warning(thumb_err or "Thumbnail unavailable.")

            with col2:
                st.markdown(f"**File ID:** `{entry.file_id}`")
                st.markdown(f"**Original name:** `{entry.original_name}`")
                st.markdown(f"**Stored path:** `{entry.stored_path}`")
                st.markdown(f"**Size:** {entry.size_bytes / 1024:.1f} KB")
                st.markdown(f"**Uploaded:** {entry.uploaded_at}")
                st.markdown(f"**SHA-256:** `{entry.sha256}`")
                if entry.width and entry.height:
                    st.markdown(f"**Dimensions:** {entry.width} × {entry.height} px")
                if entry.mpp:
                    st.markdown(f"**MPP:** {entry.mpp} µm/px")
                if entry.is_wsi:
                    st.warning("⚠️ Large WSI detected – use **Viewer & ROI** to set a region before running tasks.")

                # Tag editor
                tags_str = st.text_input(
                    "Tags (comma-separated)",
                    value=", ".join(entry.tags),
                    key=f"tags_{entry.file_id}",
                )
                if st.button("💾 Save tags", key=f"save_tags_{entry.file_id}"):
                    entry.tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    registry.update_file(entry)
                    st.success("Tags saved.")

                # Delete button
                if st.button("🗑️ Delete from workspace", key=f"del_{entry.file_id}"):
                    delete_upload(entry)
                    registry.remove_file(entry.file_id)
                    st.success(f"Deleted {entry.original_name}.")
                    st.rerun()

    # Set active file for interactive runs
    st.divider()
    st.subheader("Active file for interactive runs")
    file_names = [e.original_name for e in entries]
    active_idx = 0
    if "active_file_id" in st.session_state:
        ids = [e.file_id for e in entries]
        if st.session_state.active_file_id in ids:
            active_idx = ids.index(st.session_state.active_file_id)

    selected = st.selectbox("Select active file", file_names, index=active_idx)
    sel_entry = next(e for e in entries if e.original_name == selected)
    st.session_state["active_file_id"] = sel_entry.file_id
    st.info(f"Active file: **{sel_entry.original_name}** (ID: `{sel_entry.file_id[:8]}…`)")
