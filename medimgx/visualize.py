"""Visualization utilities for displaying CT slices with optional mask overlays."""  # noqa: E501

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def ct_mask(  # noqa: D417
    volume: NDArray[np.float32],
    mask: Optional[NDArray[np.uint8]] = None,
    slice_index: Optional[int] = None,
    figure_size: tuple[float, float] = (8.0, 8.0),
    title: str = "CT Slice",
    alpha: float = 0.4,
    path: Optional[str] = None,
    dpi: int = 100,
) -> NDArray[np.float32]:
    """Plot a single CT slice with optional segmentation mask overlay.

    Args:
        volume : The volume of the CT.
        mask : The mask based on the CT.
        slice_index : Index of the slice to display.
        figure_size : The size of the figure.
        title : The title of the CT.
        alpha : Transparency level of the overlay.
        path :  The path which save the CT slice.
        dip : The dip to save the CT slice.

    Returns:
        The 2D CT slice image array.
    """
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    if slice_index is None:
        slice_index = volume.shape[0] // 2

    image: NDArray[np.float32] = volume[slice_index, :, :].astype(np.float32)

    plt.figure(figsize=figure_size)
    plt.imshow(image, cmap="gray")

    if mask is not None:
        if mask.shape != volume.shape:
            raise ValueError("Mask and volume must have same shape")
        overlay = mask[slice_index, :, :]
        from numpy import ma

        plt.imshow(ma.masked_where(overlay == 0, overlay), cmap="jet", alpha=alpha)  # type: ignore  # noqa: E501

    plt.title(f"{title} (Slice {slice_index})")
    plt.axis("off")

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=dpi)

    plt.show()
    return image
