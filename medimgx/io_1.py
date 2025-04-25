"""DICOM Loader and Metadata Parser."""

import os

import numpy as np
import pydicom
from numpy.typing import NDArray


def load_dicom(
    folder_path: str,
) -> tuple[NDArray[np.int16], tuple[float, float, float]]:
    """Load a series of DICOM slices into a 3D volume and extract voxel spacing.

    Args:
        folder_path: The path of the folder which loading into the function.

    Returns:
        volume: 3D image volume with shape, the shape is (D, H, W).
        spacing : Spacing between voxels in (z, y, x) format.
    """  # noqa: E501
    DICOM_files = [
        pydicom.dcmread(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    if not DICOM_files:
        raise ValueError("There is not DICOM files found in this folder.")

    DICOM_files.sort(key=lambda x: getattr(x, "InstanceNumber", 0))
    volume = np.stack([dcm.pixel_array for dcm in DICOM_files]).astype(np.int16)  # noqa: E501

    try:
        spacing_x, spacing_y = map(float, DICOM_files[0].PixelSpacing)
        spacing_z = float(DICOM_files[0].SliceThickness)
    except AttributeError:
        positions = [
            float(getattr(d, "ImagePositionPatient", [0, 0, i])[2])
            for i, d in enumerate(DICOM_files)
        ]
        spacing_z = float(np.abs(np.mean(np.diff(positions))))
        spacing_y, spacing_x = 1.0, 1.0

    spacing = (spacing_z, spacing_y, spacing_x)
    return volume, spacing
