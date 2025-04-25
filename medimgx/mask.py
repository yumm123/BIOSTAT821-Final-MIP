"""Post-processing and quantitative analysis of nodule masks."""

import math
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label


class NoduleInfo(TypedDict):
    """TypedDict structure for storing nodule metadata."""

    id: int
    volume_mm3: float
    centroid_voxels: tuple[float, float, float]
    voxel_count: int


class NoduleAnalyzer:
    """Analyze binary nodule masks for volume and centroid."""

    def __init__(self, voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """Initialize analyzer with voxel spacing (z, y, x) in mm."""
        self.voxel_volume = math.prod(voxel_spacing)

    def analyze_mask(
        self, binary_mask: NDArray[np.uint8], min_volume: float = 10.0
    ) -> tuple[list[NoduleInfo], NDArray[np.uint8]]:
        """Identify and measure lung nodules in a binary mask."""
        labeled_mask, num_features = label(binary_mask)
        regions: list[NoduleInfo] = []

        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            voxel_count = int(np.sum(region_mask))
            volume_mm3 = voxel_count * self.voxel_volume

            if volume_mm3 >= min_volume:
                z, y, x = np.where(region_mask)
                centroid: tuple[float, float, float] = (
                    round(float(np.mean(z)), 2),
                    round(float(np.mean(y)), 2),
                    round(float(np.mean(x)), 2),
                )

                regions.append(
                    {
                        "id": i,
                        "volume_mm3": round(volume_mm3, 2),
                        "centroid_voxels": centroid,
                        "voxel_count": voxel_count,
                    }
                )

        regions.sort(key=lambda x: x["volume_mm3"], reverse=True)
        return regions, labeled_mask

    def generate_report(self, nodules: list[NoduleInfo]) -> str:
        """Create a summary string from nodule list."""
        report = [f"Found {len(nodules)} lung nodule(s)", "Volume distribution (mm³):"]
        for i, n in enumerate(nodules, 1):
            z_coord = int(n["centroid_voxels"][0])
            report.append(f"{i}. {n['volume_mm3']} mm³ at slice {z_coord}")
        return "\n".join(report)
