# XTIAtoolbox — TIA Streamlit Orchestrator

A Streamlit-based GUI for orchestrating [TIAToolbox](https://tia-toolbox.readthedocs.io/)
computational pathology tasks.  
This app does **not** re-implement a whole-slide-image viewer; instead it drives TIAToolbox
tasks, saves structured outputs, and launches the built-in TIAToolbox Bokeh viewer when you
need to inspect results.

---

## ⚠️ Migration note (repository restructure)

The application files previously lived inside a nested `tia_streamlit_orchestrator/`
sub-folder.  They now live **at the repository root**:

```
XTIAtoolbox/
  app.py                  ← Streamlit entry point
  requirements.txt
  pages/
  core/
  workspace/              ← created at first run (not committed)
```

If you had an existing workspace, it remains at `./workspace` relative to wherever
you run `streamlit run app.py` from.  No data migration is needed — just run from
the repository root instead of from inside `tia_streamlit_orchestrator/`.

---

## Features

- **Upload & Manage** – Upload one or many slide files (SVS, NDPI, MRXS, OME-TIFF, PNG, JPEG, TIFF).
- **Viewer & ROI** – Define regions of interest (ROI) for safe, targeted analysis of large WSI.
- **Run Tasks** – Configure and execute nuclei detection, semantic segmentation, or patch-level
  prediction.  Choose CPU, CUDA, or MPS (Apple Silicon) as the compute device.
- **Results & Downloads** – Browse run outputs, download GeoJSON / CSV / thumbnails, view run
  manifests.
- **Batch Queue** – Queue multiple files × task combinations and run them sequentially.

Every run writes a **manifest JSON** capturing parameters, tool versions, timestamps,
device requested / device used, and input-file SHA-256 hashes.

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
cd XTIAtoolbox

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

## Apple Silicon GPU acceleration (macOS M1 / M2 / M3)

PyTorch ships a **Metal Performance Shaders (MPS)** backend for Apple Silicon that
provides hardware acceleration without CUDA.

### Requirements

- macOS 12.3 (Monterey) or later
- Apple Silicon Mac (M1, M2, M3, …)
- PyTorch ≥ 1.13 built with MPS support (included in the default `pip` wheel)

### Install

```bash
# Standard pip install is sufficient – no extra index URL needed
pip install -r requirements.txt
```

Verify MPS is detected:

```python
import torch
print(torch.backends.mps.is_available())   # should print True
print(torch.backends.mps.is_built())       # should print True
```

### Selecting MPS in the UI

On the **Run Tasks** and **Batch Queue** pages, open the **Compute device** drop-down.
If MPS is detected on your system, **MPS (Apple Silicon)** will appear as an option.

> **Note:** TIAToolbox's built-in task engines (`NucleusInstanceSegmentor`,
> `SemanticSegmentor`, `PatchPredictor`) currently do not use MPS via their
> `on_gpu` flag — they run on CPU when MPS is selected.  
> The manifest will record `"device_requested": "mps"` and `"device_used": "cpu"`
> along with a human-readable `"device_fallback_reason"` so the behaviour is
> always transparent.  Future TIAToolbox releases may add native MPS support.

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
        …
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

## License

See repository root `LICENSE` for terms.
