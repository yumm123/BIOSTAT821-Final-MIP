"""CT lung window preprocessing utilities."""

import numpy as np
from numpy.typing import NDArray

__all__ = ["apply_window", "normalize_image", "clip_range"]


def apply_window(
    image: NDArray[np.float32], center: int, width: int
) -> NDArray[np.float32]:
    """Apply lung windowing to CT image volume.

    Args:
        image: Raw CT image volume.
        center: Window center (default -600 for lung).
        width: Window width (default 1500 for lung).

    Returns:
        clip: Windowed image with intensity values clipped.
    """
    lower = center - width // 2
    upper = center + width // 2
    clip = np.clip(image, lower, upper).astype(np.float32)
    return clip


def normalize_image(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize image to [0, 1] using min-max normalization.

    Args:
        image: CT or windowed image.

    Returns:
        nor_image: Normalized image volume.
    """
    min_val = np.min(image)
    max_val = np.max(image)

    nor_image = (image - min_val) / (max_val - min_val + 1e-5).astype(np.float32)
    return nor_image


def clip_range(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Ensure image values are clipped to [0, 1].

    Args:
        image: Normalized image.

    Returns:
        ran_image: Image with all values clipped to range [0, 1].
    """
    ran_image = np.clip(image, 0, 1).astype(np.float32)
    return ran_image
