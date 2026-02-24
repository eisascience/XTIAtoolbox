# XTIAtoolbox - TIA Streamlit Orchestrator

A Streamlit-based GUI for orchestrating [TIAToolbox](https://tia-toolbox.readthedocs.io/)
computational pathology tasks.
This app does **not** re-implement a whole-slide-image viewer; instead it drives TIAToolbox
tasks, saves structured outputs, and launches the built-in TIAToolbox Bokeh viewer when you
need to inspect results.

---

## Features

- **Upload & Manage** - Upload one or many slide files (SVS, NDPI, MRXS, OME-TIFF, PNG, JPEG, TIFF).
- **Viewer & ROI** - Define regions of interest (ROI) for safe, targeted analysis of large WSI.
- **Run Tasks** - Configure and execute nuclei detection, semantic segmentation, or patch-level
  prediction.  Choose CPU, CUDA, or MPS (Apple Silicon) as the compute device.
- **Results & Downloads** - Browse run outputs, download GeoJSON / CSV / thumbnails, view run
  manifests.
- **Batch Queue** - Queue multiple files x task combinations and run them sequentially.

Every run writes a **manifest JSON** capturing parameters, tool versions, timestamps,
device requested / device used, and input-file SHA-256 hashes.

---

## Installation

### Requirements

- Python 3.9 - 3.11 (recommended: 3.10)
- Conda (Miniconda or Anaconda) - recommended for reliable dependency resolution with TIA Toolbox
- macOS or Linux
- CPU-only works out of the box; GPU support follows upstream TIA Toolbox guidance.

### Setup (local)

```bash
# 1. Clone the repository
git clone https://github.com/eisascience/XTIAtoolbox.git
cd XTIAtoolbox

# 2. Create and activate a conda environment
conda create -n xtiatoolbox python=3.10 -y
conda activate xtiatoolbox

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install TIA Toolbox
#    For CPU-only: pip install tiatoolbox
#    For GPU/CUDA or other options, follow the upstream installation guide:
#    https://tia-toolbox.readthedocs.io/en/latest/installation.html
pip install tiatoolbox

# 5. Install app dependencies
pip install -r requirements.txt

# 6. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

### Verify TIA Toolbox installation

```python
python -c "import tiatoolbox; print(tiatoolbox.__version__)"
```

### GPU / CUDA / PyTorch

GPU, CUDA, and PyTorch installation specifics depend on your hardware and driver version.
Follow the [TIA Toolbox installation guide](https://tia-toolbox.readthedocs.io/en/latest/installation.html)
for up-to-date instructions.

---

## Workspace layout

All run artefacts are stored under `./workspace/` relative to the directory you
launch `streamlit run` from:

```
workspace/
  uploads/
    <file_id>/
      <original_filename>
  runs/
    <run_id>/
      manifest.json        # parameters + versions + hashes + device info
      outputs/             # task-specific results
        nuclei.geojson
        nuclei.csv
        thumbnail.png
        ...
```

### Run manifest device fields

| Field | Description |
|---|---|
| `device_requested` | Device selected in the UI (`"cpu"`, `"cuda"`, `"mps"`) |
| `device_used` | Device that was actually used after any fallback |
| `device_fallback_reason` | Human-readable explanation if a fallback occurred |
| `warnings` | List of runtime warnings (e.g. fallback messages) |

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

## Troubleshooting

### OpenSlide not found (OME-TIFF / WSI thumbnails and ROI extraction)

When loading `.tif` / `.tiff` / `.svs` / `.ndpi` / `.mrxs` files, XTIAtoolbox uses
[TIAToolbox](https://tia-toolbox.readthedocs.io/) which in turn depends on the
[OpenSlide](https://openslide.org/) native library.  If the library is missing you
will see an error similar to:

```
Couldn't locate OpenSlide dylib. Try `pip install openslide-bin`.
```

**macOS**

```bash
brew install openslide
pip install openslide-python
```

**Linux (Debian / Ubuntu)**

```bash
sudo apt-get install openslide-tools libopenslide-dev
pip install openslide-python
```

**Cross-platform alternative (no system package manager required)**

```bash
pip install openslide-bin openslide-python
```

After installation, restart the Streamlit app.  Thumbnail generation and ROI
extraction for WSI files should then work as expected.

---

## License

See repository root `LICENSE` for terms.
