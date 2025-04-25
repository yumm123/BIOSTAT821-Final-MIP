"""End-to-end lung nodule detection pipeline from DICOM CT scan."""

import csv
from pathlib import Path

import numpy as np

from medimgx.inference import NoduleSegmentor
from medimgx.io_1 import load_dicom
from medimgx.mask import NoduleAnalyzer, NoduleInfo
from medimgx.preprocessing import apply_window, normalize_image
from medimgx.resample import resample_volume
from medimgx.visualize import ct_mask


def run_pipeline(
    dicom_dir: Path,
    model_path: Path,
    output_dir: Path,
    save_image: bool,
    csv_path: Path,
) -> list[NoduleInfo]:
    """Run complete lung nodule detection pipeline.

    Args:
        dicom_dir: Path to DICOM directory.
        model_path: Path to model file.
        output_dir: Output directory path.
        save_image: Whether to save visualization.
        csv_path: The path for save the csv file.

    Returns:
        nodules: The information of the nodules.
    """
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM folder not found: {dicom_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading DICOM series...")
    volume, spacing = load_dicom(str(dicom_dir))

    windowed = apply_window(volume.astype(np.float32), center=-600, width=1500)
    normalized = normalize_image(windowed)
    resampled = resample_volume(normalized, spacing)
    segmentor = NoduleSegmentor(str(model_path))
    mask = segmentor.predict(resampled.astype(np.float32))
    analyzer = NoduleAnalyzer(voxel_spacing=(1.0, 1.0, 1.0))
    nodules, _ = analyzer.analyze_mask(mask)
    exists = bool(nodules)
    print("The nodules:" + ("exists" if exists else "not exists"))
    print(f"The number of nodules: {len(nodules)}")
    volumes = [n["volume_mm3"] for n in nodules]
    vol_str = ", ".join(f"{v} mm³" for v in volumes)
    print(f"The volume of the nodules: [{vol_str}]")

    output_dir.mkdir(parents=True, exist_ok=True)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "volume_mm3",
                    "centroid_z",
                    "centroid_y",
                    "centroid_x",
                    "voxel_count",
                ]
            )
            for n in nodules:
                z, y, x = n["centroid_voxels"]
                writer.writerow([n["id"], n["volume_mm3"], z, y, x, n["voxel_count"]])
        print(f"The path of the csv file is {csv_path}")

    if save_image:
        mid_slice = resampled.shape[0] // 2
        out_path = output_dir / "nodule_overlay.png"
        ct_mask(
            volume=resampled,
            mask=mask,
            slice_index=mid_slice,
            figure_size=(6, 6),
            title="Lung Nodule Detection",
            alpha=0.4,
            path=str(out_path),
            dpi=150,
        )
        print(f"Overlay saved: {out_path}")

    return nodules
