# Lung Nodule Detection (BIOSTAT821 Final MIP)

A modular Python-based pipeline for detecting lung nodules from chest CT scans using deep learning. It includes tools for DICOM loading, preprocessing, segmentation inference, post-processing analysis, and result visualization.


## Team Members

| Name         | Role                                 |
|--------------|--------------------------------------|
| Ziye Tian    | Imaging Preprocessing & Testing      |
| Yunqian Liu  | Deep Learning & Pipeline Integration |

Course: **BIOSTAT 821** 


## Table of Contents

- [ General Info](#-general-info)
- [ Project Structure](#-project-structure)
- [ Requirements and Setup](#-requirements-and-setup)
- [ Workflow](#-workflow)
  - [ 1. Load and Preprocess DICOM](#-1-load-and-preprocess-dicom)
  - [ 2. Segment Lung Nodules](#-2-segment-lung-nodules)
  - [ 3. Analyze and Export Nodule Info](#-3-analyze-and-export-nodule-info)
  - [ 4. Visualize Slice with Mask](#-4-visualize-slice-with-mask)
- [ Testing](#-testing)
- [ Module Descriptions](#-module-descriptions)
- [ Citation](#-citation-if-used)

---

## Project Structure

```
BIOSTAT821-Final-MIP/
├── medimgx/
│   ├── inference.py         # Nodule segmentation model wrapper
│   ├── io_1.py              # DICOM I/O utilities
│   ├── mask.py              # Nodule analysis and volume calculation
│   ├── predict_nodule.py   # Main pipeline: from DICOM to output
│   ├── preprocessing.py     # Windowing, normalization, and clipping
│   ├── resample.py          # Image resampling to isotropic spacing
│   └── visualize.py         # Overlay slice visualization
├── tests/  
│   └── test_function.py                 # Pytest-based test cases
├── README.md
└── requirements.txt
```


## Module Descriptions

### `inference.py`
Wraps a PyTorch segmentation model (e.g., 3D UNet) with a unified API for predicting lung nodules.

### `io_1.py`
**DICOM I/O Utilities**
- Loads a series of `.dcm` files into a 3D NumPy array.
- Extracts spacing (slice thickness and pixel spacing) for downstream resampling.

### `mask.py`
Analyzes the predicted binary mask:
- Computes number, size, volume, and centroid for each detected nodule.
- Returns structured report and CSV-compatible data.

### `predict_nodule.py`
**Main end-to-end pipeline**:
- Loads and preprocesses DICOM series.
- Applies model segmentation.
- Post-processes masks and extracts features.
- Optionally exports CSV and overlay image.

### `preprocessing.py`
- Applies Hounsfield unit windowing.
- Normalizes intensities to [0, 1].
- Optionally clips range.

### `resample.py`
- Resamples the input volume to isotropic voxel spacing (1mm x 1mm x 1mm).
- Preserves spatial orientation.

### `visualize.py`
- Generates overlay visualization between CT slice and segmentation.
- Used in the `run_pipeline` function to save central slice as `.png`.

---

## Requirements and Setup

This repository assumes the following:

### You have:
- A folder of `.dcm` files (e.g., from [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/))
- A PyTorch `.pt` model trained for nodule segmentation
- Python 3.8+

### Installation
```bash
pip install -r requirements.txt
```

Alternatively:
```bash
pip install git+https://github.com/yumm123/BIOSTAT821-Final-MIP.git@main
```

## Run Inference Pipeline
Use the run_pipeline function:
```python
from medimgx.predict_nodule import run_pipeline
from pathlib import Path

nodules = run_pipeline(
    dicom_dir=Path("sample_data/lung_CT_series"),
    model_path=Path("lung_nodule_segmentor.pt"),
    output_dir=Path("results"),
    save_image=True,
    csv_path=Path("results/report.csv")
)
```
Run with:
```bash
python -m medimgx.predict_nodule \
  -i sample_data/lung_series \
  -m lung_nodule_segmentor.pt \
  -o results/ \
  --csv results/report.csv
```

- `-i`: Input directory containing `.dcm` files
- `-m`: Path to model file (`.pt`)
- `-o`: Output directory for saving overlay images
- `--csv`: Optional path for CSV report


## Tests

All unit tests are in the `tests/` directory. Run with:
```bash
PYTHONPATH=. pytest
```
For the information of modules. Run the code:
```bash
PYTHONPATH=. pytest -q -s
```
If want to use the real lung CT series to test, please set up this environment variables first:
```blash
export REAL_CT_DIR=/path/to/dicom_folder
export REAL_MODEL_PATH=/path/to/model.pt
```

## Example Output
```
The nodules: not exists
The number of nodules: 0
The volume of the nodules: []
The path of the csv file is /private/var/folders/f4/8q9172sn1mg8sf_0h9mq_txc0000gn/T/pytest-of-liuyunqian/pytest-23/test_run_pipeline_0/out/dummy.csv
Overlay saved: /private/var/folders/f4/8q9172sn1mg8sf_0h9mq_txc0000gn/T/pytest-of-liuyunqian/pytest-23/test_run_pipeline_0/out/nodule_overlay.png
```
