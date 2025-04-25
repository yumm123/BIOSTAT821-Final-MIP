"""Resample CT volume to isotropic resolution (1.0mm³)."""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom


def resample_volume(
    volume: NDArray[np.float32], spacing: tuple[float, float, float]
) -> NDArray[np.float32]:
    """Resample a 3D CT volume to isotropic voxel spacing (1.0mm³)."""
    target_spacing = (1.0, 1.0, 1.0)
    zoom_factors = tuple(s / t for s, t in zip(spacing, target_spacing))
    resampled: NDArray[np.float32] = zoom(volume, zoom=zoom_factors, order=1).astype(
        np.float32
    )
    return resampled
