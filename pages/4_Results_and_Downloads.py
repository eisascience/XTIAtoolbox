"""
pages/4_Results_and_Downloads.py – browse run outputs, download files, launch the viewer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.config import RUNS_DIR
from core.export import make_run_zip, read_csv, read_geojson
from core.runner import get_run, list_runs
from core.viewer_launcher import launch_viewer, viewer_command_string
from core.viz import load_preview_image, overlay_geojson_on_image, pil_to_bytes

st.set_page_config(page_title="Results & Downloads", page_icon=None, layout="wide")
st.title("Results & Downloads")

# ---------------------------------------------------------------------------
# Run selector
# ---------------------------------------------------------------------------
manifests = list_runs()
if not manifests:
    st.info("No runs found yet.  Go to **Run Tasks** to execute a task.")
    st.stop()

run_labels = [
    f"{m.run_id[:8]} | {m.task_name} | {m.status} | {m.created_at[:19]}"
    for m in manifests
]

# Pre-select the last run if coming from Run Tasks page
default_idx = 0
if "last_run_manifest" in st.session_state and st.session_state.last_run_manifest:
    last_id = st.session_state.last_run_manifest.run_id
    ids = [m.run_id for m in manifests]
    if last_id in ids:
        default_idx = ids.index(last_id)

chosen_label = st.selectbox("Select run", run_labels, index=default_idx)
chosen_idx = run_labels.index(chosen_label)
manifest = manifests[chosen_idx]

run_dir = RUNS_DIR / manifest.run_id
out_dir = run_dir / "outputs"

# ---------------------------------------------------------------------------
# Manifest details
# ---------------------------------------------------------------------------
with st.expander("Run manifest", expanded=False):
    st.json(manifest.as_dict())

st.markdown(
    f"**Run ID:** `{manifest.run_id}` | "
    f"**Task:** {manifest.task_name} | "
    f"**Status:** {manifest.status} | "
    f"**Duration:** {manifest.duration_seconds}s | "
    f"**Device:** {manifest.device}"
)

if manifest.status == "failed":
    st.error(f"Run failed: {manifest.error}")

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
st.divider()
st.header("Output files")

output_files = sorted(out_dir.rglob("*")) if out_dir.exists() else []
output_files = [f for f in output_files if f.is_file()]

if not output_files:
    st.info("No output files found for this run.")
else:
    for f in output_files:
        col1, col2, col3 = st.columns([3, 1, 1])
        rel = str(f.relative_to(run_dir))
        with col1:
            st.markdown(f"`{rel}` ({f.stat().st_size / 1024:.1f} KB)")
        with col2:
            with open(f, "rb") as fh:
                st.download_button(
                    "Download",
                    data=fh.read(),
                    file_name=f.name,
                    key=f"dl_{f}",
                )
        with col3:
            if f.suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"):
                if st.button("Preview", key=f"prev_{f}"):
                    st.session_state[f"preview_{manifest.run_id}"] = str(f)

    # Render preview if requested
    preview_key = f"preview_{manifest.run_id}"
    if preview_key in st.session_state:
        pf = Path(st.session_state[preview_key])
        st.subheader(f"Preview: {pf.name}")
        img = load_preview_image(pf, max_size=800)
        if img:
            # Overlay GeoJSON if available
            gjson = out_dir / "nuclei.geojson"
            if gjson.exists() and pf.suffix in (".png", ".jpg", ".jpeg"):
                try:
                    img = overlay_geojson_on_image(img, gjson)
                    st.caption("GeoJSON nuclei outlines overlaid in red.")
                except Exception:
                    pass
            st.image(pil_to_bytes(img), caption=pf.name, use_container_width=True)
        else:
            st.warning("Could not load preview.")

    # Download all as ZIP
    st.divider()
    zip_bytes = make_run_zip(manifest)
    st.download_button(
        "Download all outputs as ZIP",
        data=zip_bytes,
        file_name=f"run_{manifest.run_id[:8]}.zip",
        mime="application/zip",
    )

# ---------------------------------------------------------------------------
# Nuclei CSV preview
# ---------------------------------------------------------------------------
csv_candidates = list(out_dir.glob("*.csv")) if out_dir.exists() else []
if csv_candidates:
    st.divider()
    st.subheader("CSV preview")
    import pandas as pd

    for csv_f in csv_candidates[:1]:
        try:
            df = pd.read_csv(csv_f)
            st.dataframe(df.head(200), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not load {csv_f.name}: {exc}")

# ---------------------------------------------------------------------------
# GeoJSON stats
# ---------------------------------------------------------------------------
geojson_candidates = list(out_dir.glob("*.geojson")) if out_dir.exists() else []
if geojson_candidates:
    gj = read_geojson(geojson_candidates[0])
    n_feats = len(gj.get("features", []))
    st.success(f"GeoJSON: **{n_feats}** features detected.")

# ---------------------------------------------------------------------------
# Launch viewer
# ---------------------------------------------------------------------------
st.divider()
st.header("Launch TIAToolbox Viewer")
st.markdown(
    "The TIAToolbox Bokeh viewer provides interactive slide exploration. "
    "It opens at **http://localhost:5006** in your browser."
)

cmd_str = viewer_command_string(run_dir)
st.code(cmd_str, language="bash")

if st.button("Launch Viewer", type="primary"):
    ok, msg = launch_viewer(run_dir)
    if ok:
        st.success(msg)
        st.markdown("[Open viewer →](http://localhost:5006)")
    else:
        st.warning(msg)
