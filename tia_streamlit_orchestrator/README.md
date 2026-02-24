# TIA Streamlit Orchestrator

A Streamlit-based GUI for orchestrating [TIAToolbox](https://tia-toolbox.readthedocs.io/) computational pathology tasks.
This app does **not** re-implement a whole-slide-image viewer; instead it drives TIAToolbox tasks, saves structured
outputs, and launches the built-in TIAToolbox Bokeh viewer when you need to inspect results.

---

## Features

- **Upload & Manage** – Upload one or many slide files (SVS, NDPI, MRXS, OME-TIFF, PNG, JPEG, TIFF).
- **Viewer & ROI** – Define regions of interest (ROI) for safe, targeted analysis of large WSI.
- **Run Tasks** – Configure and execute nuclei detection, semantic segmentation, or patch-level prediction.
- **Results & Downloads** – Browse run outputs, download GeoJSON / CSV / thumbnails, view run manifests.
- **Batch Queue** – Queue multiple files × task combinations and run them sequentially.

Every run writes a **manifest JSON** capturing parameters, tool versions, timestamps, device, and input-file SHA-256 hashes.

---

## Installation

### Requirements
- Python 3.9 – 3.11 (recommended: 3.10)
- macOS or Linux
- CPU-only works out of the box; CUDA GPU is used automatically when available.

### Setup (local)

```bash
# 1. Clone the repository
git clone https://github.com/eisascience/XTIAtoolbox.git
cd XTIAtoolbox/tia_streamlit_orchestrator

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

### Optional – GPU (CUDA)

Install PyTorch with CUDA support *before* installing the requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Workspace layout

All run artefacts are stored under `./workspace/` relative to the directory you launch `streamlit run` from:

```
workspace/
  uploads/
    <file_id>/
      <original_filename>
  runs/
    <run_id>/
      manifest.json        # parameters + versions + hashes
      outputs/             # task-specific results
        nuclei.geojson
        nuclei.csv
        thumbnail.png
        …
```

---

## Launching the TIAToolbox Viewer

After a successful run the **Results & Downloads** page shows a **Launch Viewer** button.
Clicking it prints the `tiatoolbox visualize` command that points at the run output directory,
and (when possible) spawns the Bokeh server in a background process.

---

## Supported tasks

| Task | TIAToolbox API used | Notes |
|---|---|---|
| Nuclei detection | `NucleusInstanceSegmentor` | HoVer-Net |
| Semantic segmentation | `SemanticSegmentor` | Configurable model |
| Patch-level prediction | `PatchPredictor` | Configurable model |

---

## License

See repository root `LICENSE` for terms.
