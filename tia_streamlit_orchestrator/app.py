"""
app.py – entry point for the TIA Streamlit Orchestrator.

Launch with:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="TIA Streamlit Orchestrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 TIA Streamlit Orchestrator")
st.markdown(
    """
    Welcome to the **TIA Streamlit Orchestrator** – a GUI for running
    [TIAToolbox](https://tia-toolbox.readthedocs.io/) computational pathology tasks.

    ### Quick start

    1. **Upload & Manage** → upload slide or image files.
    2. **Viewer & ROI** → inspect a thumbnail and draw a region of interest.
    3. **Run Tasks** → configure and execute nuclei detection, segmentation, or patch prediction.
    4. **Results & Downloads** → browse outputs, download files, launch the TIAToolbox viewer.
    5. **Batch Queue** → queue multiple file × task jobs and run them sequentially.

    Use the sidebar to navigate between pages.
    """
)

st.info(
    "ℹ️ This app **does not** re-implement a WSI viewer.  "
    "For interactive slide exploration use the **Launch Viewer** button on the Results page, "
    "which starts the built-in TIAToolbox Bokeh server."
)
