# medimgx: Medical Image Processing and Lung Nodule Detection Toolkit


Python-based workflow for detecting pulmonary nodules from chest CT scans using image preprocessing and deep learning. 

## 👥 Team Members


| Name            | Role                   |
|-----------------|------------------------|
| Ziye Tian       | Preprocessing pipeline |
| Yunqian Liu     | Model training & eval  |


Course: **BIOSTAT 821** 


## Table of contents
* [General Info](#general-info)
* [Structure](#structure)
* [Requirements and Setup](#requirements-and-setup)
* [Workflow](#workflow)
   1. [Image Loading and Preprocessing](#1-image-loading-and-preprocessing)
   2. [Nodule Segmentation and Inference](#2-nodule-segmentation-and-inference)
   3. [Nodule Volume Quantification](#3-nodule-volume-quantification)




## General Info


Lung nodules are a frequent and important finding in chest computed tomography (CT) scans, especially in the context of lung cancer screening. Early detection and precise measurement of nodules are critical for clinical decision-making. In this project, we present `medimgx`, a lightweight Python library for reading 3D chest CT images, applying clinical windowing, running pretrained segmentation models, and quantifying pulmonary nodules.


This toolkit supports DICOM-format image series and includes a fully executable pipeline for preprocessing, inference, and visualization. It is designed to be modular, testable, and reusable for future research in medical imaging.


## Structure


```
medimgx/
├── io.py                 # DICOM loader and metadata parser
├── preprocessing.py      # Windowing and normalization
├── resample.py           # Image resampling to 1x1x1 mm³
├── inference.py          # Load model and predict masks
├── mask.py               # Post-processing and nodule labeling
├── visualize.py          # Display CT slices with overlaid masks


examples/
├── predict_nodule.py     # Main pipeline entry point


tests/
├── test_io.py            # Unit tests for I/O modules


.github/
└── workflows/ci.yml      # GitHub Actions (testing and linting)


sample_data/
├── lung_series/          # Input DICOM folder (user-provided)


README.md
requirements.txt
```


## Requirements and Setup


This repository requires a local setup with medical image data and a pretrained segmentation model. The sample input is a chest CT series in **DICOM format**, typically consisting of axial slices from a lung scan.


You will need:
A DICOM CT scan (e.g., from [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
A pretrained PyTorch segmentation model (e.g., `lung_nodule_segmentor.pt`)
Python 3.8+ with required packages listed in `requirements.txt`


 **Note**: This project does **not** include any patient data or pretrained model by default.


### 1. Download and Prepare Input Data


Download a chest CT scan in DICOM format and place the slice files in the following directory:


```
sample_data/lung_series/
```


If using public data (e.g., LIDC-IDRI), extract a single patient study with axial slices in `.dcm` format.


### 2. Pretrained Model


Place your PyTorch model (e.g., a 3D UNet trained for lung nodule segmentation) in the root directory:


```
lung_nodule_segmentor.pt
```


> You can train your own model or request a demo model from the authors.



### 3. Install Dependencies


To install Python library:
```bash
pip install git+https://github.com/yumm123/BIOSTAT821-Final-MIP.git@main
```




## Workflow


The following steps describe the end-to-end pipeline for loading CT images, predicting lung nodules, and computing volume statistics.


### 1. Image Loading and Preprocessing


~~~
python examples/predict_nodule.py
~~~


**Arguments**: (configured in script or CLI)


- `dicom_path` : Path to folder of DICOM slices
- `model_path` : Path to pretrained PyTorch model


**Output**:


- Preprocessed CT volume (normalized and resampled)
- Visualization of CT slice and predicted nodule overlay


**Details**:


DICOM slices are loaded and stacked into a 3D volume. Clinical lung windowing is applied (center = -600, width = 1500), followed by normalization and resampling to standard voxel size.


---


### 2. Nodule Segmentation and Inference


**Arguments**:


- `img_tensor` : Preprocessed 3D tensor `[1, 1, D, H, W]`
- `model_path` : Path to `.pt` PyTorch model


**Output**:


- Binary mask (`1 = nodule`, `0 = background`)


**Details**:


The input CT volume is passed to a pretrained 3D UNet segmentation model. A threshold of 0.5 is applied to output probabilities to produce a binary mask of suspected nodule regions.


---


### 3. Nodule Volume Quantification


**Arguments**:


- `mask_pred` : Binary mask from segmentation
- `spacing` : 3D voxel spacing in mm


**Output**:


- Number of nodules
- Volume of each in mm³
- Visualization of labeled slice


**Details**:


Connected components are labeled using `scipy.ndimage.label`. Each detected nodule's volume is computed by multiplying voxel count by the voxel volume. Results are printed and optionally exported.


Example output:


```
✅ Lung nodules detected
📌 Number of nodules: 3
📏 Nodule volumes (mm³): [52.3, 88.1, 35.4]
