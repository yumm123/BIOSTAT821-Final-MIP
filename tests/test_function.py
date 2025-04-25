"""Test the function of the MIP."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from numpy.typing import NDArray

import medimgx.predict_nodule as pnk
from medimgx.inference import NoduleSegmentor
from medimgx.io_1 import load_dicom
from medimgx.mask import NoduleAnalyzer
from medimgx.predict_nodule import run_pipeline
from medimgx.preprocessing import apply_window, clip_range, normalize_image
from medimgx.resample import resample_volume


@pytest.fixture
def mock_volume_spacing() -> tuple[NDArray[np.float32], tuple[float, float, float]]:
    """Generate a mock 3D volume and spacing."""
    volume = np.random.uniform(-1000, 1000, size=(20, 64, 64)).astype(np.float32)
    spacing = (2.0, 0.7, 0.7)
    return volume, spacing


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    """Create and save a dummy PyTorch model for tests of inference."""
    model = nn.Identity()
    path = tmp_path / "dummy_model.pt"
    torch.save(model, str(path))
    return path


@pytest.fixture
def make_fake_dicom() -> Any:
    """Create a mock function that returns a fake DICOM object."""
    counter = {"n": 1}

    def _fake(_: Any) -> MagicMock:
        dcm = MagicMock()
        n = counter["n"]
        dcm.pixel_array = np.full((4, 5), n, dtype=np.int16)
        dcm.InstanceNumber = n
        dcm.PixelSpacing = [0.8, 0.6]
        dcm.SliceThickness = 2.0
        counter["n"] += 1
        return dcm

    return _fake


@patch("medimgx.io_1.os.listdir")
@patch("medimgx.io_1.pydicom.dcmread")
def test_load_dicom_success(
    mock_dcmread: Any,
    mock_listdir: Any,
    make_fake_dicom: Any
) -> None:
    """Test load_dicom with valid files."""
    mock_listdir.return_value = ["a.dcm", "b.dcm", "c.dcm"]
    mock_dcmread.side_effect = make_fake_dicom

    volume, spacing = load_dicom("dummy_folder")

    assert volume.shape == (3, 4, 5)
    np.testing.assert_array_equal(volume[0], 1)
    np.testing.assert_array_equal(volume[1], 2)
    np.testing.assert_array_equal(volume[2], 3)

    expect_spacing = (2.0, 0.6, 0.8)  
    assert spacing == expect_spacing


@patch("medimgx.io_1.os.listdir", return_value=[])
def test_load_dicom_empty_folder(mock_listdir: Any) -> None:
    """Test load_dicom raises error on empty folder."""
    with pytest.raises(ValueError, match="not DICOM files"):
        load_dicom("empty_folder")



def test_apply_window(
    mock_volume_spacing: tuple[NDArray[np.float32], tuple[float, float, float]]
) -> None:
    """Test apply_window limits image range correctly."""
    volume = mock_volume_spacing[0]
    window = apply_window(volume.astype(np.float32), -600, 1500)
    low, high = -600 - 750, -600 + 750

    assert window.dtype == np.float32
    assert window.shape == volume.shape
    assert window.min() >= low
    assert window.max() <= high
    assert isinstance(window, np.ndarray)


def test_normalize_image(
    mock_volume_spacing: tuple[NDArray[np.float32], tuple[float, float, float]]
) -> None:
    """Test normalize_image scales intensities to [0, 1]."""
    volume = mock_volume_spacing[0]
    window = apply_window(volume.astype(np.float32), -600, 1500)
    normed = normalize_image(window)

    assert normed.dtype == np.float32
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0 + 1e-6


def test_clip(
    mock_volume_spacing: tuple[np.ndarray[Any, Any], tuple[float, float, float]]
) -> None:
    """Test clip_range truncates values into [0, 1]."""
    volume = mock_volume_spacing[0]
    window = apply_window(volume.astype(np.float32), -600, 1500)
    normed = normalize_image(window)
    clip = clip_range(normed)

    assert clip.min() >= 0.0
    assert clip.max() <= 1.0


def test_resampling(
    mock_volume_spacing: tuple[np.ndarray[Any, Any], tuple[float, float, float]]
) -> None:
    """Test resample_volume outputs expected shape and dtype."""
    volume, spacing = mock_volume_spacing
    window = apply_window(volume.astype(np.float32), -600, 1500)
    normed = normalize_image(window)
    clip = clip_range(normed)
    resampled = resample_volume(clip, spacing)

    assert resampled.ndim == 3
    assert resampled.dtype == np.float32


def test_nodule_analyzer_and_mask() -> None:
    """Test NoduleAnalyzer functionality."""
    mask = np.zeros((10, 32, 32), dtype=np.uint8)
    mask[4:6, 10:12, 10:12] = 1

    analyzer = NoduleAnalyzer(voxel_spacing=(1, 1, 1))
    nodules, _ = analyzer.analyze_mask(mask, min_volume=1.0)
    report = analyzer.generate_report(nodules)

    assert len(nodules) == 1
    assert "mm³" in report


def test_run_pipeline_(
    mock_volume_spacing: tuple[np.ndarray[Any, Any], 
                               tuple[float, float, float]],
    dummy_model_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    csv_path: Optional[Path] = None 
) -> None:
    """Test run_pipeline with dummy model and mock volume."""
    volume, spacing = mock_volume_spacing
    monkeypatch.setattr(pnk, "load_dicom", lambda _: (volume, spacing))

    class DummySeg(NoduleSegmentor):
        def __init__(self, mp: Path, device: Optional[str] = None):
            super().__init__(str(mp), device)

        def predict(
            self,
            volume: np.ndarray[Any, Any],
            threshold: float = 0.5,
            batch_size: int = 4
        ) -> np.ndarray[Any, Any]:
            m = np.zeros_like(volume, dtype=np.uint8)
            mid = volume.shape[0] // 2
            m[mid, 32, 32] = 1
            return m

    monkeypatch.setattr(pnk, "NoduleSegmentor", DummySeg)

    dicom_dir = tmp_path / "dummy"
    dicom_dir.mkdir() 
    outdir = tmp_path / "out"
    csv_path = outdir / "results.csv"

    run_pipeline(
        dicom_dir=dicom_dir,
        model_path=dummy_model_path,
        output_dir=outdir,
        save_image=True,
        csv_path=csv_path
    )

    assert (outdir / "nodule_overlay.png").exists()



@pytest.fixture
def real_ct_dir() -> Path:
    """Return path from REAL_CT_DIR or skip test."""
    dir_path = os.environ.get("REAL_CT_DIR")
    if not dir_path:
        pytest.skip("REAL_CT_DIR not set, skipping real CT integration test")
    return Path(dir_path)


@pytest.fixture
def real_model_path() -> Path:
    """Return path from REAL_MODEL_PATH or skip test."""
    model_path = os.environ.get("REAL_MODEL_PATH")
    if not model_path:
        pytest.skip("REAL_MODEL_PATH not set, skipping real CT integration test")
    return Path(model_path)


def test_real_lung_ct_integration(tmp_path: Path) -> None:
    """Integration test with environment variables (optional real data)."""
    ct_dir = os.environ.get("REAL_CT_DIR")
    model_path = os.environ.get("REAL_MODEL_PATH")
    if not ct_dir or not model_path:
        pytest.skip("REAL_CT_DIR or REAL_MODEL_PATH not set")

    csv_out = tmp_path / "real.csv"  

    nodules = run_pipeline(
        dicom_dir=Path(ct_dir),
        model_path=Path(model_path),
        output_dir=tmp_path,
        save_image=False,
        csv_path=csv_out, 
    )
   

    print("Integration detected nodules:", nodules)
    assert isinstance(nodules, list)
